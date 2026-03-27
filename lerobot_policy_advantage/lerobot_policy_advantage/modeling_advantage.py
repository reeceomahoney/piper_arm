"""Advantage-conditioned policy model.

Extends SmolVLAPolicy with a learned advantage embedding injected between the
prefix (images+language+state) and suffix (noisy actions+time) during training,
and always set to positive (advantage=1) during inference.
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
from torch import Tensor
from transformers import AutoTokenizer

from .configuration_advantage import AdvantageConfig

ADVANTAGE_POSITIVE = "Advantage: positive"
ADVANTAGE_NEGATIVE = "Advantage: negative"


class AdvantageVLAFlowMatching(VLAFlowMatching):
    """VLAFlowMatching with text-based advantage conditioning in the prefix."""

    # (2, seq_len) stacked token IDs: index 0 = negative, index 1 = positive
    adv_tokens: Tensor

    def __init__(self, config: AdvantageConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config: AdvantageConfig = config
        self.advantage_dropout = config.advantage_dropout

        tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
        assert tokenizer is not None

        pos_ids = tokenizer.encode(ADVANTAGE_POSITIVE, add_special_tokens=False)
        neg_ids = tokenizer.encode(ADVANTAGE_NEGATIVE, add_special_tokens=False)

        # Pad to same length so we can stack as (2, seq_len)
        max_len = max(len(pos_ids), len(neg_ids))
        pad_id = tokenizer.pad_token_id or 0
        pos_ids += [pad_id] * (max_len - len(pos_ids))
        neg_ids += [pad_id] * (max_len - len(neg_ids))

        self.register_buffer(
            "adv_tokens",
            torch.tensor([neg_ids, pos_ids], dtype=torch.long),
        )

    def embed_advantage(self, adv_tokens):
        """Embed advantage tokens via the VLM language embedding layer.

        Args:
            adv_tokens: (batch, seq_len) token IDs.

        Returns:
            (adv_embs, adv_pad_masks, adv_att_masks) ready for prefix concat.
        """
        adv_embs = self.vlm_with_expert.embed_language_tokens(adv_tokens)
        embed_dim = adv_embs.shape[-1]
        adv_embs = adv_embs * math.sqrt(embed_dim)

        bsize = adv_tokens.shape[0]
        seq_len = adv_tokens.shape[1]

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
        adv_tokens=None,
    ) -> Tensor:
        """Training forward with advantage language tokens in prefix."""
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

        if adv_tokens is not None:
            adv_embs, adv_pad_masks, adv_att_masks = self.embed_advantage(adv_tokens)
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

        # Conditional prefix: append positive advantage tokens
        pos_tokens = self.adv_tokens[1].unsqueeze(0).expand(bsize, -1)
        adv_embs, adv_pad_masks, adv_att_masks = self.embed_advantage(pos_tokens)

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
        """Training forward: resolve advantage tokens then delegate to model."""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        # Resolve advantage tokens via index into (2, seq_len) buffer
        device = actions.device
        if self.config.fixed_advantage:
            bsize = actions.shape[0]
            indices = torch.ones(bsize, dtype=torch.long, device=device)
        else:
            adv_labels = batch["observation.language.advantage_label"]
            indices = adv_labels.to(dtype=torch.long, device=device)
        adv_tokens = (
            self.model.adv_tokens[indices][:, -1, :]
            if indices.ndim > 1
            else self.model.adv_tokens[indices]
        )

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
            adv_tokens=adv_tokens,
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
            loss_dict["pct_positive"] = indices.float().mean().item()

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
