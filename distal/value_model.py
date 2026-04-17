#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available
from torch import Tensor

PI05_VLM_KEY_PREFIX = "paligemma_with_expert.paligemma."


def collect_images(batch: dict[str, Any], image_size: int) -> Tensor:
    """Extract camera images from batch (per-camera or combined format)."""
    image_keys = sorted(k for k in batch if k.startswith(f"{OBS_IMAGES}."))
    if image_keys:
        img_list = []
        for key in image_keys:
            img = batch[key]
            if img.ndim == 5:
                img = img[:, -1]
            img_list.append(img)
        images = torch.stack(img_list, dim=1)
    elif "observation.images" in batch:
        images = batch["observation.images"]
        if images.ndim == 4:
            images = images.unsqueeze(1)
    else:
        raise ValueError("No image keys found in batch")

    batch_size, n_cams, C, H, W = images.shape
    target_h, target_w = image_size, image_size
    if target_h != H or target_w != W:
        flat = images.reshape(batch_size * n_cams, C, H, W)
        flat = F.interpolate(
            flat, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        images = flat.reshape(batch_size, n_cams, C, target_h, target_w)
    return images


def load_pretrained_vlm_weights(model: nn.Module, pretrained_path: str) -> None:
    """Load VLM weights from pi0.5 checkpoint into the PaliGemma backbone."""
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    logging.info(f"Loading pretrained VLM weights from {pretrained_path}")

    resolved_file = cached_file(pretrained_path, "model.safetensors")
    if resolved_file is None:
        raise FileNotFoundError(
            f"Could not find model.safetensors in {pretrained_path}"
        )
    full_state_dict = load_file(resolved_file)

    vlm_state_dict: dict[str, Tensor] = {}
    for key, value in full_state_dict.items():
        if not key.startswith(PI05_VLM_KEY_PREFIX):
            continue
        new_key = key[len("paligemma_with_expert.") :]
        vlm_state_dict[new_key] = value

    lm_head_key = "paligemma.lm_head.weight"
    embed_key = "paligemma.model.language_model.embed_tokens.weight"
    if lm_head_key in vlm_state_dict and embed_key not in vlm_state_dict:
        vlm_state_dict[embed_key] = vlm_state_dict[lm_head_key].clone()

    missing, unexpected = model.load_state_dict(vlm_state_dict, strict=False)

    expected_missing = [
        k for k in missing if k.startswith(("value_head.", "value_bin_support"))
    ]
    truly_missing = [k for k in missing if k not in expected_missing]

    loaded_count = len(vlm_state_dict) - len(unexpected)
    logging.info(
        f"Pretrained VLM weights: loaded {loaded_count} tensors, "
        f"{len(expected_missing)} expected-missing (value head), "
        f"{len(truly_missing)} unexpectedly missing, "
        f"{len(unexpected)} unexpected."
    )
    if truly_missing:
        logging.warning(f"Unexpectedly missing keys: {truly_missing[:10]}")
    if unexpected:
        logging.warning(f"Unexpected keys (not loaded): {unexpected[:10]}")


if _transformers_available:
    from transformers import CONFIG_MAPPING
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
    )
else:
    CONFIG_MAPPING = None
    PaliGemmaForConditionalGeneration = None


def _load_get_gemma_config():
    try:
        module = import_module("lerobot.policies.pi05.modeling_pi05")
    except ModuleNotFoundError:
        module = import_module("lerobot.policies.pi05_full.modeling_pi05")
    return module.get_gemma_config


get_gemma_config = _load_get_gemma_config()


@PreTrainedConfig.register_subclass("recap_value")
@dataclass
class RECAPValueConfig(PreTrainedConfig):
    """Configuration for the standalone RECAP value network."""

    paligemma_variant: str = "gemma_300m"
    precision: str = "float32"
    image_size: int = 224
    tokenizer_name: str = "google/paligemma-3b-pt-224"
    hidden_dim: int = 768
    num_value_bins: int = 50
    v_min: float = -1.0
    v_max: float = 0.0
    freeze_vision_encoder: bool = False
    freeze_backbone: bool = False
    num_unfrozen_backbone_layers: int = 0
    num_vlm_layers: int = 18
    value_head_depth: int = 1
    dropout: float = 0.1
    vlm_pretrained_path: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig()

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        return None


class RECAPValueNetwork(PreTrainedPolicy):
    config_class = RECAPValueConfig
    name = "recap_value"
    config: RECAPValueConfig
    value_bin_support: Tensor

    def __init__(self, config: RECAPValueConfig):
        super().__init__(config)
        if PaliGemmaForConditionalGeneration is None or CONFIG_MAPPING is None:
            raise ImportError(
                "transformers is required to instantiate RECAPValueNetwork."
            )

        self.config = config
        gemma_config = get_gemma_config(config.paligemma_variant)

        paligemma_config_hf = CONFIG_MAPPING["paligemma"]()
        paligemma_config_hf._vocab_size = 257152  # noqa: SLF001
        paligemma_config_hf.image_token_index = 257152
        paligemma_config_hf.text_config.hidden_size = gemma_config.width
        paligemma_config_hf.text_config.intermediate_size = gemma_config.mlp_dim
        paligemma_config_hf.text_config.num_attention_heads = gemma_config.num_heads
        paligemma_config_hf.text_config.head_dim = gemma_config.head_dim
        paligemma_config_hf.text_config.num_hidden_layers = gemma_config.depth
        paligemma_config_hf.text_config.num_key_value_heads = gemma_config.num_kv_heads
        paligemma_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        paligemma_config_hf.text_config.torch_dtype = "float32"
        paligemma_config_hf.text_config.vocab_size = 257152
        paligemma_config_hf.vision_config.image_size = config.image_size
        paligemma_config_hf.vision_config.intermediate_size = 4304
        paligemma_config_hf.vision_config.projection_dim = gemma_config.width
        paligemma_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        paligemma_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config_hf)  # ty: ignore[invalid-argument-type]

        if config.precision == "bfloat16":
            self.paligemma = self.paligemma.to(dtype=torch.bfloat16)  # ty: ignore[missing-argument]
        elif config.precision == "float32":
            self.paligemma = self.paligemma.to(dtype=torch.float32)  # ty: ignore[missing-argument]
        else:
            raise ValueError(f"Invalid precision: {config.precision}")

        lm_inner = self.paligemma.model.language_model
        if hasattr(lm_inner, "model"):
            lm_inner = lm_inner.model
        if config.num_vlm_layers > 0:
            total_layers = len(lm_inner.layers)
            if config.num_vlm_layers > total_layers:
                raise ValueError(
                    f"num_vlm_layers={config.num_vlm_layers} exceeds "
                    f"model depth {total_layers}"
                )
            lm_inner.layers = lm_inner.layers[: config.num_vlm_layers]
            logging.info(
                f"Using first {len(lm_inner.layers)} PaliGemma text layers "
                "for value network"
            )

        if config.freeze_backbone:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False
            if config.num_unfrozen_backbone_layers > 0:
                num_layers = len(lm_inner.layers)
                if config.num_unfrozen_backbone_layers > num_layers:
                    raise ValueError(
                        "num_unfrozen_backbone_layers="
                        f"{config.num_unfrozen_backbone_layers} "
                        f"exceeds available layers {num_layers}"
                    )
                unfrozen_layers = lm_inner.layers[
                    -config.num_unfrozen_backbone_layers :
                ]
                for layer in unfrozen_layers:
                    layer.train()
                    for param in layer.parameters():
                        param.requires_grad = True
                logging.info(
                    "Unfreezing last "
                    f"{config.num_unfrozen_backbone_layers}/{num_layers} "
                    "backbone transformer layers"
                )
        elif config.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False

        self.register_buffer(
            "value_bin_support",
            torch.linspace(
                config.v_min, config.v_max, config.num_value_bins, dtype=torch.float32
            ),
            persistent=True,
        )

        # Value head as per Bryson Jones implementation
        # https://github.com/brysonjones/open-value/blob/main/src/open_value_estimator/value_estimator.py
        value_head_layers: list[nn.Module] = []
        for i in range(config.value_head_depth):
            value_head_layers.append(
                nn.Linear(
                    gemma_config.width if i == 0 else config.hidden_dim,
                    config.hidden_dim,
                )
            )
            value_head_layers.append(nn.GELU())
        value_head_layers.append(nn.Linear(config.hidden_dim, config.num_value_bins))
        self.value_head = nn.Sequential(*value_head_layers)

        if config.vlm_pretrained_path:
            load_pretrained_vlm_weights(self, config.vlm_pretrained_path)

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}

    def reset(self) -> None:
        pass

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("RECAPValueNetwork does not produce actions.")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("RECAPValueNetwork does not produce actions.")

    def predict_value(self, batch: dict[str, Any]) -> Tensor:
        """Return expected scalar value V(s) for each item in ``batch``."""
        outputs = self.compute_outputs(batch)
        return outputs["expected_value"].squeeze(-1)

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Compute cross-entropy loss over value bins.

        Expects ``batch["target_bin"]`` to contain integer bin targets.
        """
        outputs = self.compute_outputs(batch)
        loss = F.cross_entropy(outputs["value_logits"], batch["target_bin"])
        return loss, outputs

    def compute_outputs(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Forward pass returning logits over value bins.

        Args:
            batch: Preprocessed batch with images, language tokens and mask.

        Returns:
            Dictionary with value_logits, value_probs and expected_value.
        """
        device = next(self.parameters()).device
        images = collect_images(batch, self.config.image_size)
        batch_size, n_cams = images.shape[:2]

        # Paligemma SigLIP image encoder
        flat_images = images.reshape(batch_size * n_cams, *images.shape[2:]).to(device)
        image_outputs = self.paligemma.model.get_image_features(flat_images)
        flat_img_emb = image_outputs.pooler_output
        if flat_img_emb.ndim == 2:
            flat_img_emb = flat_img_emb.unsqueeze(1)
        img_token_len = flat_img_emb.shape[1]
        img_emb = flat_img_emb.reshape(
            batch_size, n_cams * img_token_len, flat_img_emb.shape[-1]
        )
        img_mask = torch.ones(
            batch_size, img_emb.shape[1], dtype=torch.bool, device=device
        )

        # Language instruction embedding
        input_ids = batch[OBS_LANGUAGE_TOKENS].to(device)
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
        lang_emb = self.paligemma.model.language_model.embed_tokens(input_ids)
        text_mask = attention_mask.bool()

        # Concat image + language embeddings
        full_embs = torch.cat((img_emb, lang_emb), dim=1)
        full_mask = torch.cat((img_mask, text_mask), dim=1)

        position_ids = torch.cumsum(full_mask, dim=1) - 1
        position_ids = position_ids.masked_fill(~full_mask, 0).long()

        # Forward pass. The property is called language_model but it is in fact a vlm
        vlm = self.paligemma.model.language_model
        text_dtype = next(vlm.parameters()).dtype
        text_model_inputs = {
            "inputs_embeds": full_embs.to(dtype=text_dtype),
            "attention_mask": full_mask,
            "use_cache": False,
        }
        try:
            text_model_inputs["position_ids"] = position_ids
            hidden_states = vlm.forward(**text_model_inputs).last_hidden_state
        except TypeError:
            text_model_inputs.pop("position_ids", None)
            hidden_states = vlm.forward(**text_model_inputs).last_hidden_state

        seq_lengths = full_mask.sum(dim=1) - 1
        last_token_hidden_state = hidden_states[
            torch.arange(batch_size, device=device), seq_lengths.long()
        ]

        # Feed last hidden state into the value head
        value_logits = self.value_head(last_token_hidden_state.float())
        value_probs = torch.softmax(value_logits, dim=-1)
        expected_value = (value_probs * self.value_bin_support).sum(
            dim=-1, keepdim=True
        )

        return {
            "value_logits": value_logits,
            "value_probs": value_probs,
            "expected_value": expected_value,
        }


def build_value_preprocessor(
    dataset: Any,
    paligemma_variant: str,
    model_precision: str,
    device: str,
) -> Any:
    """Build a pi05-compatible preprocessor for the value network."""
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.feature_utils import dataset_to_policy_features
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        k: f for k, f in features.items() if f.type is FeatureType.ACTION
    }
    input_features = {k: f for k, f in features.items() if k not in output_features}

    policy_cfg = PI05Config(
        input_features=input_features,
        output_features=output_features,
        paligemma_variant=paligemma_variant,
        dtype=model_precision,
        device=device,
    )
    preprocessor, _ = make_pi05_pre_post_processors(
        config=policy_cfg,
        dataset_stats=dataset.meta.stats,
    )
    return preprocessor
