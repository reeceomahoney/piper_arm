"""Distributional value function using SmolVLM + expert backbone.

Architecture mirrors SmolVLA's VLM + action expert pattern but replaces noisy-action
suffix tokens with a single learned value query that cross-attends to the VLM prefix.
The value head predicts returns as a categorical distribution over discrete bins
(RECAP-style from pi0.6).
"""

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from lerobot.policies.smolvla.modeling_smolvla import (
    make_att_2d_masks,
    pad_tensor,
    pad_vector,
    resize_with_pad,
)
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor, nn


@dataclass
class ValueConfig:
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    expert_width_multiplier: float = 0.5
    resize_imgs_with_padding: tuple[int, int] = (256, 256)
    max_state_dim: int = 8
    n_bins: int = 201
    tokenizer_max_length: int = 48
    prefix_length: int = -1
    add_image_special_tokens: bool = False
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False


class ValueModel(nn.Module):
    def __init__(self, config: ValueConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=True,
            attention_mode="cross_attn",
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device="cpu",
        )

        vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size

        # State projection (same as SmolVLA)
        self.state_proj = nn.Linear(config.max_state_dim, vlm_hidden)

        # Single learnable value query token (replaces noisy-action suffix)
        self.value_query = nn.Parameter(torch.randn(1, 1, expert_hidden) * 0.02)

        # Value head: expert_hidden → logits over bins
        self.value_head = nn.Sequential(
            nn.Linear(expert_hidden, expert_hidden),
            nn.SiLU(),
            nn.Linear(expert_hidden, config.n_bins),
        )

        # Bin centers for computing expected value
        self.register_buffer("bin_centers", torch.linspace(-1.0, 0.0, config.n_bins))

        # Special tokens for image wrapping
        self.fake_image_token = (
            self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        )
        self.global_image_token = (
            self.vlm_with_expert.processor.tokenizer.global_image_token_id
        )
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.add_image_special_tokens = config.add_image_special_tokens
        self.prefix_length = config.prefix_length

        self._set_requires_grad()

    def _set_requires_grad(self):
        """Freeze vision encoder. VLM + expert + heads stay trainable."""
        # Vision encoder: frozen
        vision_model = self.vlm_with_expert.get_vlm_model().vision_model
        vision_model.eval()
        for p in vision_model.parameters():
            p.requires_grad = False

        # VLM language model: trainable (train_expert_only=False already handles this)
        # Expert + value_query + value_head + state_proj: trainable by default

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep vision encoder in eval mode
        self.vlm_with_expert.get_vlm_model().vision_model.eval()
        return self

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed images, language, and state into prefix tokens.

        Duplicated from VLAFlowMatching.embed_prefix to stay decoupled from the
        installed lerobot action-policy internals.
        """
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks: list[int] = []

        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(
                            device=self.vlm_with_expert.vlm.device
                        )
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_start_token.device,
                )
                att_masks += [0] * image_start_mask.shape[-1]
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_end_token.device,
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * image_end_mask.shape[1]

        # Language tokens
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        # State
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        embs_cat = torch.cat(embs, dim=1)
        pad_masks_cat = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(
            att_masks, dtype=torch.bool, device=pad_masks_cat.device
        )
        att_masks_t = att_masks_t[None, :]

        # Pad to prefix_length if needed
        seq_len = pad_masks_cat.shape[1]
        if self.prefix_length > 0 and seq_len < self.prefix_length:
            embs_cat = pad_tensor(embs_cat, self.prefix_length, pad_value=0)
            pad_masks_cat = pad_tensor(pad_masks_cat, self.prefix_length, pad_value=0)
            att_masks_t = pad_tensor(att_masks_t, self.prefix_length, pad_value=0)

        att_masks_t = att_masks_t.expand(bsize, -1)
        return embs_cat, pad_masks_cat, att_masks_t

    def prepare_images(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Resize, pad, and normalize images for SigLIP."""
        img_keys = sorted(k for k in batch if k.startswith("observation.images."))
        images = []
        img_masks = []
        for key in img_keys:
            img = batch[key]
            img = img[:, -1] if img.ndim == 5 else img
            w, h = self.config.resize_imgs_with_padding
            img = resize_with_pad(img, w, h, pad_value=-1)
            img = img * 2.0 - 1.0
            images.append(img)
            mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Pad state to max_state_dim."""
        state = batch[OBS_STATE]
        state = state[:, -1] if state.ndim > 2 else state
        return pad_vector(state, self.config.max_state_dim)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass returning logits over value bins.

        Args:
            batch: Preprocessed batch dict with observation.images.*, observation.state,
                   observation.language.tokens, observation.language.attention_mask.

        Returns:
            logits: (B, n_bins) — raw logits over bin_centers.
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # 1. Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )

        # 2. Build suffix from value query
        bsize = prefix_embs.shape[0]
        device = prefix_embs.device
        suffix_embs = self.value_query.expand(bsize, -1, -1).to(
            dtype=prefix_embs.dtype, device=device
        )
        suffix_pad_masks = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        suffix_att_masks = torch.ones(
            bsize, 1, dtype=prefix_att_masks.dtype, device=device
        )

        # 3. Concatenate masks and build 2D attention
        pad_masks_cat = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_cat = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks_cat, att_masks_cat)
        position_ids = torch.cumsum(pad_masks_cat, dim=1) - 1

        # 4. Forward through VLM + expert
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=cast(torch.LongTensor, position_ids),
            past_key_values=None,
            inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, suffix_embs]),
            use_cache=False,
            fill_kv_cache=False,
        )

        # 5. Extract single suffix token → value head
        suffix_out = suffix_out[:, -1:].to(dtype=torch.float32)
        logits = self.value_head(suffix_out.squeeze(1))  # (B, n_bins)
        return logits

    def predict_value(self, logits: Tensor) -> Tensor:
        """Expected value from logits via softmax + dot with bin centers."""
        probs = F.softmax(logits, dim=-1)
        return (probs * cast(Tensor, self.bin_centers)).sum(dim=-1)

    @staticmethod
    def returns_to_bins(returns: Tensor, n_bins: int = 201) -> Tensor:
        """Convert return values in [-1, 0] to one-hot bin targets.

        Args:
            returns: (B,) float tensor of return values in [-1, 0].
            n_bins: number of discrete bins.

        Returns:
            One-hot targets: (B, n_bins).
        """
        returns = returns.clamp(-1.0, 0.0)
        # Map [-1, 0] → [0, n_bins-1]
        bin_indices = ((returns + 1.0) * (n_bins - 1)).long()
        bin_indices = bin_indices.clamp(0, n_bins - 1)
        return F.one_hot(bin_indices, num_classes=n_bins).float()
