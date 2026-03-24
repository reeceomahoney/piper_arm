"""VLM prefix embedding extraction for PI05 and SmolVLA policies.

Provides mean-pooled embeddings over image tokens from the VLM prefix,
used for Mahalanobis distance-based OOD detection.
"""

from typing import Union, cast

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)
from lerobot.policies.smolvla.modeling_smolvla import (
    make_att_2d_masks as smolvla_make_att_2d_masks,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


@torch.no_grad()
def embed_prefix_pooled(
    policy: Union[PI05Policy, SmolVLAPolicy], batch: dict
) -> torch.Tensor:
    """Run a batch through the VLM prefix and return mean-pooled embeddings.

    Supports both PI05 and SmolVLA policies. Only image tokens are included
    in the pooling; language and state tokens are excluded.

    Args:
        batch: Already preprocessed observation dict on device.

    Returns:
        (B, hidden_dim) mean-pooled over image tokens.
    """
    if isinstance(policy, PI05Policy):
        prefix_out, prefix_pad_masks = _embed_prefix_pi05(policy, batch)
    elif isinstance(policy, SmolVLAPolicy):
        prefix_out, prefix_pad_masks = _embed_prefix_smolvla(policy, batch)
    else:
        raise TypeError(f"Unsupported policy type: {type(policy)}")

    mask = prefix_pad_masks.unsqueeze(-1).float()
    pooled = (prefix_out.float() * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled


@torch.no_grad()
def embed_prefix_tokens(
    policy: Union[PI05Policy, SmolVLAPolicy], batch: dict
) -> torch.Tensor:
    """Run a batch through the VLM prefix and return raw image token activations.

    Supports both PI05 and SmolVLA policies. Only image tokens are returned;
    language and state tokens are excluded.

    Args:
        batch: Already preprocessed observation dict on device.

    Returns:
        (B, n_img_tokens, hidden_dim) float tensor of image token activations.
    """
    if isinstance(policy, PI05Policy):
        prefix_out, img_mask = _embed_prefix_pi05(policy, batch)
    elif isinstance(policy, SmolVLAPolicy):
        prefix_out, img_mask = _embed_prefix_smolvla(policy, batch)
    else:
        raise TypeError(f"Unsupported policy type: {type(policy)}")

    # All samples share the same image token count (fixed camera setup),
    # so use the first sample's mask to determine which positions are image tokens.
    token_indices = torch.where(img_mask[0])[0]
    return prefix_out[:, token_indices, :].float()


def _embed_prefix_pi05(policy: PI05Policy, batch: dict):
    """PI05: images + language -> PaliGemma prefix forward (4D attention masks).

    Returns prefix_out and a pooling mask covering image tokens only
    (language token positions are masked out).
    """
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    prefix_att_2d_masks_4d = prefix_att_2d_masks_4d.to(dtype=prefix_embs.dtype)

    (prefix_out, _), _ = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=cast(torch.LongTensor, prefix_position_ids),
        past_key_values=None,
        inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, None]),
        use_cache=False,
    )

    # Pooling mask: image tokens only (prefix layout: [img...][lang...])
    n_lang = lang_tokens.shape[1]
    n_img = prefix_embs.shape[1] - n_lang
    vis_mask = prefix_pad_masks.clone()
    vis_mask[:, n_img:] = False

    return prefix_out, vis_mask


def _embed_prefix_smolvla(policy: SmolVLAPolicy, batch: dict):
    """SmolVLA: images + language + state -> SmolVLM prefix forward (3D masks).

    Returns prefix_out and a pooling mask covering image tokens only
    (language and state token positions are masked out).
    """
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = smolvla_make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    (prefix_out, _), _ = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=cast(torch.LongTensor, prefix_position_ids),
        past_key_values=None,
        inputs_embeds=cast(list[torch.FloatTensor], [prefix_embs, None]),
        use_cache=False,
        fill_kv_cache=True,
    )

    # Pooling mask: image tokens only
    # Prefix layout: [img_special+img...][lang...][state...][padding...]
    # State tokens are the only positions with prefix_att_masks == 1
    n_lang = lang_tokens.shape[1]
    state_positions = torch.where(prefix_att_masks[0] == 1)[0]
    first_state = state_positions[0].item()
    lang_start = first_state - n_lang

    img_mask = prefix_pad_masks.clone()
    img_mask[:, lang_start:] = False

    return prefix_out, img_mask
