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


@torch.compiler.disable
def build_mask_and_position_ids(
    img_mask: Tensor, text_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Build the full attention mask and position ids, forced to run eager.

    Kept out of torch.compile: Inductor's split-scan codegen for this cumsum
    emits a kernel that triggers a CUDA illegal memory access.
    """
    full_mask = torch.cat((img_mask, text_mask), dim=1)
    position_ids = torch.cumsum(full_mask, dim=1) - 1
    position_ids = position_ids.masked_fill(~full_mask, 0).long()
    return full_mask, position_ids


if _transformers_available:
    from transformers import AutoModel, SiglipVisionModel
else:
    AutoModel = None
    SiglipVisionModel = None


@PreTrainedConfig.register_subclass("recap_value")
@dataclass
class RECAPValueConfig(PreTrainedConfig):
    """Configuration for the standalone RECAP value network."""

    text_backbone: str = "google/gemma-3-270m"
    vision_tower: str = "google/siglip-so400m-patch14-224"
    precision: str = "bfloat16"
    image_size: int = 224
    hidden_dim: int = 640
    num_value_bins: int = 201
    v_min: float = -1.0
    v_max: float = 0.0
    value_head_depth: int = 1
    gradient_checkpointing: bool = False
    compile_model: bool = False

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
        if AutoModel is None or SiglipVisionModel is None:
            raise ImportError(
                "transformers is required to instantiate RECAPValueNetwork."
            )

        self.config = config
        self.vision_tower = SiglipVisionModel.from_pretrained(config.vision_tower)
        self.language_model = AutoModel.from_pretrained(config.text_backbone)

        vision_hidden = self.vision_tower.config.hidden_size
        text_hidden = self.language_model.config.hidden_size

        self.multi_modal_projector = nn.Linear(vision_hidden, text_hidden, bias=True)

        if config.precision == "bfloat16":
            dtype = torch.bfloat16
        elif config.precision == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid precision: {config.precision}")
        self.vision_tower = self.vision_tower.to(dtype=dtype)  # ty: ignore[missing-argument]
        self.language_model = self.language_model.to(dtype=dtype)
        self.multi_modal_projector = self.multi_modal_projector.to(dtype=dtype)

        if config.gradient_checkpointing:
            self.vision_tower.gradient_checkpointing_enable()
            self.language_model.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing enabled for vision tower and LM")

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
                    text_hidden if i == 0 else config.hidden_dim,
                    config.hidden_dim,
                )
            )
            value_head_layers.append(nn.GELU())
        value_head_layers.append(nn.Linear(config.hidden_dim, config.num_value_bins))
        self.value_head = nn.Sequential(*value_head_layers)

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.forward = torch.compile(  # ty: ignore[invalid-assignment]
                self.forward, mode="max-autotune"
            )
            self.predict_value = torch.compile(  # ty: ignore[invalid-assignment]
                self.predict_value, mode="max-autotune"
            )
            logging.info("Compiled forward and predict_value (mode=max-autotune)")

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

    def forward(  # ty: ignore[invalid-method-override]
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
        images = collect_images(batch, self.config.image_size)
        batch_size, n_cams = images.shape[:2]
        device = images.device

        # SigLIP vision encoder → linear projection into text hidden dim
        vision_dtype = next(self.vision_tower.parameters()).dtype
        flat_images = images.reshape(batch_size * n_cams, *images.shape[2:]).to(
            dtype=vision_dtype
        )
        vision_outputs = self.vision_tower(pixel_values=flat_images)
        flat_img_emb = self.multi_modal_projector(vision_outputs.last_hidden_state)
        img_token_len = flat_img_emb.shape[1]
        img_emb = flat_img_emb.reshape(
            batch_size, n_cams * img_token_len, flat_img_emb.shape[-1]
        )
        img_mask = torch.ones(
            batch_size, img_emb.shape[1], dtype=torch.bool, device=device
        )

        # Language instruction embedding
        input_ids = batch[OBS_LANGUAGE_TOKENS]
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK]
        lang_emb = self.language_model.embed_tokens(input_ids)
        text_mask = attention_mask.bool()

        # Concat image + language embeddings
        full_embs = torch.cat((img_emb, lang_emb), dim=1)
        full_mask, position_ids = build_mask_and_position_ids(img_mask, text_mask)

        text_dtype = next(self.language_model.parameters()).dtype
        hidden_states = self.language_model(
            inputs_embeds=full_embs.to(dtype=text_dtype),
            attention_mask=full_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

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
    tokenizer_name: str,
    model_precision: str,
    device: str,
) -> Any:
    """Build a pi05-compatible preprocessor that tokenizes with ``tokenizer_name``."""
    from lerobot.configs.types import FeatureType
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.processor import TokenizerProcessorStep
    from lerobot.utils.feature_utils import dataset_to_policy_features

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        k: f for k, f in features.items() if f.type is FeatureType.ACTION
    }
    input_features = {k: f for k, f in features.items() if k not in output_features}

    policy_cfg = PI05Config(
        input_features=input_features,
        output_features=output_features,
        dtype=model_precision,
        device=device,
    )
    preprocessor, _ = make_pi05_pre_post_processors(
        config=policy_cfg,
        dataset_stats=dataset.meta.stats,
    )

    # Swap the PaliGemma tokenizer step for ``tokenizer_name`` so the produced
    # OBS_LANGUAGE_TOKENS line up with the value network's text backbone vocab.
    new_steps = []
    for step in preprocessor.steps:
        if isinstance(step, TokenizerProcessorStep):
            step = TokenizerProcessorStep(
                tokenizer_name=tokenizer_name,
                max_length=step.max_length,
                padding_side=step.padding_side,
                padding=step.padding,
            )
        new_steps.append(step)
    preprocessor.steps = new_steps
    return preprocessor
