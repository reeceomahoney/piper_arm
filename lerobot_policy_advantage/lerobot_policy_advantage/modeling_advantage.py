"""Advantage-conditioned policy model.

Extends SmolVLAPolicy with advantage conditioning injected between the prefix
(images+language+state) and suffix (noisy actions+time) during training, and
always set to positive (advantage=1) during inference. Supports a learned
embedding (default) or text-token embedding (use_text_advantage=True).
"""

import math
from typing import cast

import torch
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import Tensor, nn
from transformers import AutoTokenizer

from .configuration_advantage import AdvantageConfig

ADVANTAGE_POSITIVE = "Advantage: positive"
ADVANTAGE_NEGATIVE = "Advantage: negative"


class AdvantageVLAFlowMatching(VLAFlowMatching):
    """VLAFlowMatching with advantage conditioning in the prefix."""

    def __init__(self, config: AdvantageConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: AdvantageConfig = config
        self.advantage_dropout = config.advantage_dropout
        self.num_adv_tokens = config.num_adv_tokens

        if config.use_text_advantage:
            tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
            assert tokenizer is not None

            pos_ids = tokenizer.encode(ADVANTAGE_POSITIVE, add_special_tokens=False)
            neg_ids = tokenizer.encode(ADVANTAGE_NEGATIVE, add_special_tokens=False)

            max_len = max(len(pos_ids), len(neg_ids))
            pad_id = tokenizer.pad_token_id or 0
            pos_ids += [pad_id] * (max_len - len(pos_ids))
            neg_ids += [pad_id] * (max_len - len(neg_ids))

            self.register_buffer(
                "adv_tokens",
                torch.tensor([neg_ids, pos_ids], dtype=torch.long),
            )
        else:
            vlm_hidden_size = self.vlm_with_expert.vlm.config.text_config.hidden_size
            self.adv_embedding = nn.Embedding(2, vlm_hidden_size)

    def embed_advantage(self, adv_labels):
        """Embed advantage labels into prefix embeddings.

        Args:
            adv_labels: (batch,) integer labels, 0=negative, 1=positive.

        Returns:
            (adv_embs, adv_pad_masks, adv_att_masks) ready for prefix concat.
        """
        bsize = adv_labels.shape[0]

        if self.config.use_text_advantage:
            # Text-token approach: embed via VLM language embedding layer
            adv_token_ids = self.adv_tokens[adv_labels]  # (B, seq_len)
            adv_embs = self.vlm_with_expert.embed_language_tokens(adv_token_ids)
            embed_dim = adv_embs.shape[-1]
            adv_embs = adv_embs * math.sqrt(embed_dim)
            seq_len = adv_token_ids.shape[1]
        else:
            # Learned embedding approach: single embedding repeated num_adv_tokens times
            adv_emb = self.adv_embedding(adv_labels)  # (B, hidden_dim)
            embed_dim = adv_emb.shape[-1]
            adv_emb = adv_emb * math.sqrt(embed_dim)
            adv_embs = adv_emb.unsqueeze(1).expand(bsize, self.num_adv_tokens, -1)
            seq_len = self.num_adv_tokens

        # Advantage dropout: zero out embeddings for classifier-free guidance
        if self.training and self.advantage_dropout > 0:
            keep = (
                torch.rand(bsize, device=adv_embs.device) > self.advantage_dropout
            ).float()
            adv_embs = adv_embs * keep[:, None, None]

        adv_pad_masks = torch.ones(
            bsize, seq_len, dtype=torch.bool, device=adv_embs.device
        )
        # att_mask=1: causal attention (last tokens in prefix)
        adv_att_masks = torch.ones(
            bsize, seq_len, dtype=torch.bool, device=adv_embs.device
        )

        return adv_embs, adv_pad_masks, adv_att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        adv_labels=None,
    ) -> Tensor:
        """Training forward with advantage conditioning in prefix."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        if adv_labels is not None:
            adv_embs, adv_pad_masks, adv_att_masks = self.embed_advantage(adv_labels)
            # Append advantage as the last tokens in the prefix
            prefix_embs = torch.cat([prefix_embs, adv_embs], dim=1)
            prefix_pad_masks = torch.cat([prefix_pad_masks, adv_pad_masks], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, adv_att_masks], dim=1)

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=cast(torch.LongTensor, position_ids),
            past_key_values=None,
            inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, suffix_embs]),
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = torch.nn.functional.mse_loss(u_t, v_t, reduction="none")
        return losses

    def _build_prefix_kv_cache(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        """Build KV cache from prefix embeddings."""
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=cast(torch.LongTensor, position_ids),
            past_key_values=None,
            inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, None]),
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        return past_key_values

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ) -> Tensor:
        """Inference with classifier-free guidance over advantage conditioning."""
        bsize = state.shape[0]
        device = state.device
        guidance_scale = self.config.guidance_scale

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # Conditional prefix: append positive advantage embeddings
        pos_labels = torch.ones(bsize, dtype=torch.long, device=device)
        adv_embs, adv_pad_masks, adv_att_masks = self.embed_advantage(pos_labels)

        cond_embs = torch.cat([prefix_embs, adv_embs], dim=1)
        cond_pad = torch.cat([prefix_pad_masks, adv_pad_masks], dim=1)
        cond_att = torch.cat([prefix_att_masks, adv_att_masks], dim=1)

        cond_kv = self._build_prefix_kv_cache(cond_embs, cond_pad, cond_att)

        # Unconditional prefix: append zeroed-out advantage embeddings
        if guidance_scale != 1.0:
            uncond_adv_embs = torch.zeros_like(adv_embs)
            uncond_embs = torch.cat([prefix_embs, uncond_adv_embs], dim=1)
            uncond_pad = cond_pad
            uncond_att = cond_att
            uncond_kv = self._build_prefix_kv_cache(uncond_embs, uncond_pad, uncond_att)

        # Denoising loop
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(
                bsize
            )

            v_cond = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=cond_pad,
                past_key_values=cond_kv,
                timestep=time_tensor,
            )

            if guidance_scale == 1.0:
                v_t = v_cond
            else:
                v_uncond = self.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=uncond_pad,
                    past_key_values=uncond_kv,
                    timestep=time_tensor,
                )
                v_t = v_uncond + guidance_scale * (v_cond - v_uncond)

            x_t = x_t + dt * v_t

        return x_t


class AdvantagePolicy(SmolVLAPolicy):
    """Advantage-conditioned policy extending SmolVLA."""

    config_class = AdvantageConfig
    name = "advantage"

    def __init__(self, config: AdvantageConfig, **kwargs):
        super(SmolVLAPolicy, self).__init__(config, **kwargs)
        config.validate_features()
        self.config: AdvantageConfig = config
        self.init_rtc_processor()
        self.model: AdvantageVLAFlowMatching = AdvantageVLAFlowMatching(
            config, rtc_processor=self.rtc_processor
        )
        self.reset()

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ):
        """Training forward: resolve advantage labels then delegate to model."""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        # Resolve advantage labels as (B,) integer tensor
        device = actions.device
        if self.config.fixed_advantage:
            bsize = actions.shape[0]
            adv_labels = torch.ones(bsize, dtype=torch.long, device=device)
        else:
            adv_labels = batch["observation.language.advantage_label"]
            adv_labels = adv_labels.to(dtype=torch.long, device=device).squeeze(-1)

        losses = AdvantageVLAFlowMatching.forward(
            self.model,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            adv_labels=adv_labels,
        )

        assert self.config.action_feature is not None
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        losses = losses[:, :, : self.config.max_action_dim]

        loss_dict = {}
        if not self.config.fixed_advantage:
            loss_dict["pct_positive"] = adv_labels.float().mean().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
