"""Distributional value function using PaliGemma (gemma_2b) backbone.

Builds a PaliGemma model sized to the pi0.5 gemma_2b variant, truncated to the
first N text layers, and initializes it from `lerobot/pi05_base` weights so the
VLM features match the base policy. All backbone parameters are frozen except
the last K text layers. State is discretized into 256 bins and appended to the
task prompt; bidirectional attention is enabled over the prefix. Value is
extracted from the last valid token hidden state and fed through an MLP head
that predicts returns as a categorical distribution over discrete bins.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)
from lerobot.policies.pi05.modeling_pi05 import get_gemma_config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from torch import Tensor, nn
from transformers import CONFIG_MAPPING
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)

PI05_VLM_KEY_PREFIX = "paligemma_with_expert.paligemma."


def load_pretrained_vlm_weights(model: nn.Module, pretrained_path: str) -> None:
    """Load VLM weights from a pi0.5 checkpoint into `model.paligemma`.

    Subsets `paligemma_with_expert.paligemma.*` keys from the checkpoint's
    safetensors, strips the outer prefix so keys start with `paligemma.`, and
    copies `paligemma.lm_head.weight` to the embed_tokens slot if missing.
    Ported from lerobot PR #3245 `recap_utils.load_pretrained_vlm_weights`.
    """
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
        k for k in missing if k.startswith(("value_head.", "bin_centers"))
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


@PreTrainedConfig.register_subclass("value")
@dataclass
class ValueConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    image_size: int = 224
    num_vlm_layers: int = 10
    tokenizer_name: str = "google/paligemma-3b-pt-224"
    tokenizer_max_length: int = 200
    freeze_vision_encoder: bool = True
    freeze_backbone: bool = True
    num_unfrozen_backbone_layers: int = 3
    pretrained_path: str | None = "lerobot/pi05_base"
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    n_bins: int = 201
    hl_gauss_sigma: float = 0.0
    value_head_depth: int = 1
    value_head_hidden_dim: int = 768

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
        }
    )

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def validate_features(self) -> None:
        pass


def make_value_pre_post_processors(
    config: ValueConfig,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre/post processors for the value model.

    Uses Pi0.5-style state discretization (256 bins) and tokenizes with the
    Gemma 3 tokenizer to match the value model's language backbone.
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features=config.input_features or {},
            norm_map=config.normalization_mapping,  # type: ignore[arg-type]
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=32),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device or "cpu"),
    ]

    output_steps: list[ProcessorStep] = [
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        ),
    )


class MLPHead(nn.Module):
    """MLP projection head with GELU activations.

    With depth=1: Linear(in_dim, hidden_dim) -> GELU -> Linear(hidden_dim, out_dim)
    With depth>1: repeated Linear -> GELU blocks, final Linear to out_dim.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ValueFunction(PreTrainedPolicy):
    name = "value"
    config_class = ValueConfig
    config: ValueConfig

    def __init__(self, config: ValueConfig, **kwargs):
        super().__init__(config)

        gemma_cfg = get_gemma_config(config.paligemma_variant)
        pg_cfg = CONFIG_MAPPING["paligemma"]()
        pg_cfg._vocab_size = 257152
        pg_cfg.image_token_index = 257152
        pg_cfg.text_config.hidden_size = gemma_cfg.width
        pg_cfg.text_config.intermediate_size = gemma_cfg.mlp_dim
        pg_cfg.text_config.num_attention_heads = gemma_cfg.num_heads
        pg_cfg.text_config.num_key_value_heads = gemma_cfg.num_kv_heads
        pg_cfg.text_config.head_dim = gemma_cfg.head_dim
        pg_cfg.text_config.num_hidden_layers = gemma_cfg.depth
        pg_cfg.text_config.hidden_activation = "gelu_pytorch_tanh"
        pg_cfg.text_config.torch_dtype = "float32"
        pg_cfg.text_config.vocab_size = 257152
        pg_cfg.vision_config.image_size = config.image_size
        pg_cfg.vision_config.intermediate_size = 4304
        pg_cfg.vision_config.projection_dim = gemma_cfg.width
        pg_cfg.vision_config.projector_hidden_act = "gelu_fast"
        pg_cfg.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=pg_cfg)  # ty: ignore[invalid-argument-type]
        target_dtype = (
            torch.bfloat16 if config.precision == "bfloat16" else torch.float32
        )
        self.paligemma = self.paligemma.to(dtype=target_dtype)  # ty: ignore[missing-argument]

        # Truncate text layers to the first `num_vlm_layers`.
        lm_inner = self.paligemma.model.language_model
        if hasattr(lm_inner, "model"):
            lm_inner = lm_inner.model
        total_layers = len(lm_inner.layers)
        if config.num_vlm_layers > total_layers:
            raise ValueError(
                f"num_vlm_layers={config.num_vlm_layers} "
                f"exceeds model depth {total_layers}"
            )
        lm_inner.layers = lm_inner.layers[: config.num_vlm_layers]

        # Bidirectional attention over the prefix.
        self.paligemma.model.language_model.config.use_bidirectional_attention = True

        gemma_hidden: int = gemma_cfg.width

        # Value head
        hidden_dim = config.value_head_hidden_dim
        self.value_head = MLPHead(
            gemma_hidden, hidden_dim, config.n_bins, depth=config.value_head_depth
        )

        # Bin centers for computing expected value
        self.bin_centers: Tensor
        self.register_buffer("bin_centers", torch.linspace(-1.0, 0.0, config.n_bins))

        if config.pretrained_path:
            load_pretrained_vlm_weights(self, config.pretrained_path)

        self.set_requires_grad()

    def backbone_text_layers(self) -> nn.ModuleList:
        lm_inner = self.paligemma.model.language_model
        if hasattr(lm_inner, "model"):
            lm_inner = lm_inner.model
        return lm_inner.layers

    def set_requires_grad(self):
        layers = self.backbone_text_layers()
        if self.config.freeze_backbone:
            self.paligemma.eval()
            for p in self.paligemma.parameters():
                p.requires_grad = False
            if self.config.num_unfrozen_backbone_layers > 0:
                n = self.config.num_unfrozen_backbone_layers
                if n > len(layers):
                    raise ValueError(
                        f"num_unfrozen_backbone_layers={n} "
                        f"exceeds available layers {len(layers)}"
                    )
                for layer in list(layers)[-n:]:
                    layer.train()
                    for p in layer.parameters():
                        p.requires_grad = True
        elif self.config.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for p in self.paligemma.model.vision_tower.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        print(
            f"Value model: {trainable:,} trainable / "
            f"{total:,} total ({frozen:,} frozen)"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.paligemma.eval()
            if self.config.num_unfrozen_backbone_layers > 0 and mode:
                n = self.config.num_unfrozen_backbone_layers
                for layer in list(self.backbone_text_layers())[-n:]:
                    layer.train()
        elif self.config.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        return self

    def prepare_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        """Resize and normalize images for PaliGemma's SigLIP tower."""
        img_keys = sorted(k for k in batch if k.startswith("observation.images."))
        size = self.config.image_size
        images = []
        for key in img_keys:
            img = batch[key]
            img = img[:, -1] if img.ndim == 5 else img
            img = F.interpolate(
                img, size=(size, size), mode="bilinear", align_corners=False
            )
            img = img * 2.0 - 1.0  # [0,1] -> [-1,1] for SigLIP
            images.append(img)
        return images

    def compute_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass returning logits over value bins.

        Encodes each camera image through PaliGemma's vision tower (which
        already projects patch tokens to the Gemma token dimension), embeds
        pre-tokenized text, concatenates, and processes through Gemma with
        bidirectional attention. Extracts the last valid token for the value
        head.
        """
        images = self.prepare_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        device = lang_tokens.device
        bsize = lang_tokens.shape[0]

        vlm = self.paligemma.model.language_model
        param_dtype = next(vlm.parameters()).dtype

        all_img_embeds = []
        all_img_masks = []
        vision_tower = self.paligemma.model.vision_tower
        projector = self.paligemma.model.multi_modal_projector
        for img in images:
            vision_out = vision_tower(pixel_values=img.to(param_dtype))
            patch_embeds = vision_out.last_hidden_state
            img_emb = projector(patch_embeds)
            if img_emb.ndim == 2:
                img_emb = img_emb.unsqueeze(1)
            all_img_embeds.append(img_emb)
            num_patches = img_emb.shape[1]
            all_img_masks.append(
                torch.ones(bsize, num_patches, dtype=lang_masks.dtype, device=device)
            )

        text_embeds = vlm.embed_tokens(lang_tokens)
        text_embeds = text_embeds * math.sqrt(text_embeds.shape[-1])

        combined_embeds = torch.cat(
            [e.to(text_embeds.dtype) for e in all_img_embeds] + [text_embeds], dim=1
        )
        combined_mask = torch.cat(all_img_masks + [lang_masks], dim=1)

        position_ids = torch.cumsum(combined_mask, dim=1) - 1
        position_ids = position_ids.masked_fill(combined_mask == 0, 0).long()

        vlm_inputs = {
            "inputs_embeds": combined_embeds.to(dtype=param_dtype),
            "attention_mask": combined_mask,
            "use_cache": False,
        }
        try:
            hidden_states = vlm(
                **vlm_inputs, position_ids=position_ids
            ).last_hidden_state
        except TypeError:
            hidden_states = vlm(**vlm_inputs).last_hidden_state

        seq_lengths = combined_mask.long().sum(dim=1) - 1
        last_hidden = hidden_states[torch.arange(bsize, device=device), seq_lengths]

        logits = self.value_head(last_hidden.to(dtype=torch.float32))
        return logits

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass: compute cross-entropy loss over value bins.

        Expects batch to contain a "returns" key with (B,) float tensor
        in [-1, 0]. When hl_gauss_sigma > 0, uses soft Gaussian targets
        (HL-Gauss) instead of one-hot targets.
        """
        logits = self.compute_logits(batch)
        returns = batch["returns"]

        if self.config.hl_gauss_sigma > 0:
            targets = self.returns_to_hl_gauss(
                returns, self.config.n_bins, self.config.hl_gauss_sigma
            )
        else:
            targets = self.returns_to_bins(returns, self.config.n_bins)

        loss = F.cross_entropy(logits, targets)

        with torch.no_grad():
            pred_values = self.logits_to_value(logits)
            mae = (pred_values - returns).abs().mean().item()

        return loss, {"mae": mae}

    def predict_value(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute expected value for a batch of observations."""
        logits = self.compute_logits(batch)
        return self.logits_to_value(logits)

    def logits_to_value(self, logits: Tensor) -> Tensor:
        """Expected value from logits via softmax + dot with bin centers."""
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    @staticmethod
    def returns_to_bins(returns: Tensor, n_bins: int = 50) -> Tensor:
        """Convert return values in [-1, 0] to one-hot bin targets."""
        returns = returns.clamp(-1.0, 0.0)
        bin_indices = ((returns + 1.0) * (n_bins - 1)).long()
        bin_indices = bin_indices.clamp(0, n_bins - 1)
        return F.one_hot(bin_indices, num_classes=n_bins).float()

    def returns_to_hl_gauss(self, returns: Tensor, n_bins: int, sigma: float) -> Tensor:
        """Convert returns to soft Gaussian targets over bins."""
        returns = returns.clamp(-1.0, 0.0)
        bin_centers = self.bin_centers
        diff = returns.unsqueeze(-1) - bin_centers.unsqueeze(0)
        log_probs = -0.5 * (diff / sigma) ** 2
        return F.softmax(log_probs, dim=-1)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("ValueFunction does not produce actions.")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("ValueFunction does not produce actions.")

    def reset(self):
        pass

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}
