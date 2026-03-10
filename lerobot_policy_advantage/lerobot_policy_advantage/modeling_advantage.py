"""Advantage-conditioned policy model.

Wraps SmolVLA's VLAFlowMatching and injects a learned advantage embedding
between the prefix (images+language+state) and suffix (noisy actions+time).
During training, reads pre-computed `advantage_label` from the batch.
During inference, always injects a positive advantage token.
"""

import logging
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor, nn

from .configuration_advantage import AdvantageConfig


class AdvantageEmbedding(nn.Module):
    """Learned embedding for binary advantage indicator (negative=0, positive=1)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(2, hidden_size)

    def forward(self, labels: Tensor) -> Tensor:
        """Embed binary advantage labels.

        Args:
            labels: (B,) int tensor with values 0 or 1.

        Returns:
            (B, 1, hidden_size) embedding.
        """
        return self.embedding(labels).unsqueeze(1)


class AdvantagePolicy(PreTrainedPolicy):
    """Advantage-conditioned policy wrapping SmolVLA's VLAFlowMatching."""

    config_class = AdvantageConfig
    name = "advantage"

    def __init__(self, config: AdvantageConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config: AdvantageConfig = config

        smolvla_config = config.to_smolvla_config()
        self.model = VLAFlowMatching(smolvla_config)

        vlm_hidden = self.model.vlm_with_expert.config.text_config.hidden_size
        self.adv_embedding = AdvantageEmbedding(vlm_hidden)

        if config.smolvla_checkpoint is not None:
            self.load_smolvla_weights(config.smolvla_checkpoint)

        self.reset()

    def load_smolvla_weights(self, smolvla_path: str):
        """Load VLAFlowMatching weights from a SmolVLA checkpoint."""
        from safetensors.torch import load_file

        checkpoint_dir = Path(smolvla_path)
        if not checkpoint_dir.exists():
            logging.info(f"Downloading SmolVLA checkpoint from {smolvla_path}...")
            checkpoint_dir = Path(snapshot_download(smolvla_path))

        weights = load_file(str(checkpoint_dir / "model.safetensors"))

        state = self.state_dict()
        loaded = 0
        for key, tensor in weights.items():
            if key in state and state[key].shape == tensor.shape:
                state[key] = tensor
                loaded += 1

        self.load_state_dict(state)
        logging.info(
            f"Loaded {loaded}/{len(weights)} SmolVLA weights from {smolvla_path}"
        )

    def reset(self):
        self.queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self):
        return self.parameters()

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ):
        """Training forward with advantage token injection.

        Reads `advantage_label` from batch (pre-computed integer 0 or 1).
        Injects advantage embedding between prefix and suffix with dropout.
        Returns (loss, loss_dict).
        """
        model = self.model

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        # Flow matching: sample noise and time
        if noise is None:
            noise = model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 1. Embed prefix (images + language + state)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        bsize = prefix_embs.shape[0]
        device = prefix_embs.device

        if self.config.use_advantage_tokens:
            # Read pre-computed advantage labels
            adv_labels = batch["advantage_label"].long().to(device)

            # 2. Embed advantage token
            adv_embs = self.adv_embedding(adv_labels).to(dtype=prefix_embs.dtype)
            adv_att_masks = torch.ones(
                bsize, 1, dtype=prefix_att_masks.dtype, device=device
            )

            # Apply advantage dropout (mask out advantage_dropout fraction)
            dropout_mask = (
                torch.rand(bsize, device=device) > self.config.advantage_dropout
            )
            adv_pad_masks = dropout_mask.unsqueeze(1)  # (B, 1)

            # 3. Concatenate prefix + advantage
            prefix_embs = torch.cat([prefix_embs, adv_embs], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, adv_pad_masks], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, adv_att_masks], dim=1)

            adv_labels_for_log = adv_labels
        else:
            adv_labels_for_log = None

        # 4. Embed suffix (noisy actions + time)
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time)

        # 5. Full sequence: prefix [+ advantage] + suffix
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = (torch.cumsum(pad_masks, dim=1) - 1).long()

        # 6. Forward through VLM + expert
        (_, suffix_out), _ = model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,  # type: ignore[invalid-argument-type]
            past_key_values=None,
            inputs_embeds=[prefix_embs.float(), suffix_embs.float()],  # type: ignore[invalid-argument-type]
            use_cache=False,
            fill_kv_cache=False,
        )

        # 7. Extract action predictions and compute loss
        suffix_out = suffix_out[:, -model.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        losses = losses[:, :, : self.config.max_action_dim]

        loss_dict = {}
        if adv_labels_for_log is not None:
            loss_dict["pct_positive"] = adv_labels_for_log.float().mean().item()
        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Select a single action. Uses queue-based action chunking."""
        self.eval()
        self.queues = populate_queues(self.queues, batch, exclude_keys=[ACTION])

        if len(self.queues[ACTION]) == 0:
            actions = self.get_action_chunk(batch, noise)
            self.queues[ACTION].extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]
            )

        return self.queues[ACTION].popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Generate full action chunk with positive advantage token injected."""
        self.eval()
        self.queues = populate_queues(self.queues, batch, exclude_keys=[ACTION])
        return self.get_action_chunk(batch, noise)

    def get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Internal: compute action chunk with advantage injection."""
        for k in batch:
            if k in self.queues and k != ACTION:
                batch[k] = torch.stack(list(self.queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.sample_actions_with_advantage(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]  # type: ignore[unresolved-attribute]
        actions = actions[:, :, :original_action_dim]
        return actions

    def sample_actions_with_advantage(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ) -> Tensor:
        """Sample actions with positive advantage token injected."""
        model = self.model
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
            noise = model.sample_noise(shape, device)

        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        if self.config.use_advantage_tokens:
            # Inject advantage token (always positive=1 at inference)
            pos = torch.ones(bsize, dtype=torch.long, device=device)
            adv_embs = self.adv_embedding(pos).to(dtype=prefix_embs.dtype)
            adv_att = torch.ones(bsize, 1, dtype=prefix_att_masks.dtype, device=device)
            adv_pad = torch.ones(bsize, 1, dtype=prefix_pad_masks.dtype, device=device)

            prefix_embs = torch.cat([prefix_embs, adv_embs], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, adv_pad], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, adv_att], dim=1)

        # KV cache forward
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = (torch.cumsum(prefix_pad_masks, dim=1) - 1).long()
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,  # type: ignore[invalid-argument-type]
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # type: ignore[invalid-argument-type]
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )

        # Denoising loop
        num_steps = model.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(
                bsize
            )
            v_t = model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing: resize, pad, normalize to [-1, 1]."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features missing from batch. (batch: {batch.keys()}) "
                f"(image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(
                    img, *self.config.resize_imgs_with_padding, pad_value=0
                )
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        state = (
            batch[OBS_STATE][:, -1, :]
            if batch[OBS_STATE].ndim > 2
            else batch[OBS_STATE]
        )
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
