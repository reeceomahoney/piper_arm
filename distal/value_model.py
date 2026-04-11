"""Distributional value function using SigLIP + Gemma backbone.

Uses a SigLIP vision encoder for image features and a Gemma language model
for joint processing of visual tokens and text (task description + discretized
state). Pi0.5-style architecture: vision features projected and scaled by
sqrt(embed_dim), state discretized into 256 bins as text, bidirectional
attention in Gemma. Value extracted from last token hidden state, fed through
an MLP value head that predicts returns as a categorical distribution over
discrete bins.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)
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
from transformers import Gemma3ForCausalLM, SiglipVisionModel


@PreTrainedConfig.register_subclass("value")
@dataclass
class ValueConfig(PreTrainedConfig):
    vision_model_name: str = "google/siglip-base-patch16-224"
    lm_model_name: str = "google/gemma-3-270m"
    n_bins: int = 201
    tokenizer_max_length: int = 48
    freeze_vision_encoder: bool = True
    hl_gauss_sigma: float = 0.0
    value_head_depth: int = 1
    value_head_hidden_dim: int = 768

    # Training presets
    optimizer_lr: float = 3e-4
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
            tokenizer_name=config.lm_model_name,
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

        # Vision encoder (SigLIP)
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            config.vision_model_name
        )
        vision_hidden = self.vision_encoder.config.hidden_size
        self.image_size = self.vision_encoder.config.image_size

        # Language model (Gemma with bidirectional attention)
        self.language_model = Gemma3ForCausalLM.from_pretrained(
            config.lm_model_name,
            torch_dtype=torch.bfloat16,
        )
        self.language_model.model.config.use_bidirectional_attention = True
        gemma_hidden: int = self.language_model.config.hidden_size  # type: ignore[assignment]

        # Vision projection: SigLIP features -> Gemma token dimension
        self.vision_proj = nn.Linear(vision_hidden, gemma_hidden)
        self.vision_scale = math.sqrt(gemma_hidden)

        # Value head
        hidden_dim = config.value_head_hidden_dim
        self.value_head = MLPHead(
            gemma_hidden, hidden_dim, config.n_bins, depth=config.value_head_depth
        )

        # Bin centers for computing expected value
        self.bin_centers: Tensor
        self.register_buffer("bin_centers", torch.linspace(-1.0, 0.0, config.n_bins))

        self.set_requires_grad()

    def set_requires_grad(self):
        """Freeze vision encoder only. Everything else is trainable."""
        if self.config.freeze_vision_encoder:
            for p in self.vision_encoder.parameters():
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
        self.vision_encoder.eval()
        return self

    def prepare_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        """Resize, normalize images for SigLIP."""
        img_keys = sorted(k for k in batch if k.startswith("observation.images."))
        images = []
        for key in img_keys:
            img = batch[key]
            img = img[:, -1] if img.ndim == 5 else img
            img = F.interpolate(
                img,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            img = img * 2.0 - 1.0  # [0,1] -> [-1,1] for SigLIP
            images.append(img)
        return images

    def compute_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Forward pass returning logits over value bins.

        Encodes images through SigLIP, projects to Gemma dimension, embeds
        pre-tokenized text, concatenates, and processes through Gemma with
        bidirectional attention. Extracts last valid token for value head.
        """
        images = self.prepare_images(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        device = lang_tokens.device
        bsize = lang_tokens.shape[0]

        # Encode images through SigLIP + projection
        all_img_embeds = []
        all_img_masks = []
        for img in images:
            with torch.no_grad():
                vision_out = self.vision_encoder(pixel_values=img)
            patch_embeds = vision_out.last_hidden_state
            img_emb = self.vision_proj(patch_embeds) * self.vision_scale
            all_img_embeds.append(img_emb)
            num_patches = img_emb.shape[1]
            all_img_masks.append(
                torch.ones(bsize, num_patches, dtype=lang_masks.dtype, device=device)
            )

        # Embed pre-tokenized text
        text_embeds = self.language_model.model.embed_tokens(lang_tokens)
        text_embeds = text_embeds * math.sqrt(text_embeds.shape[-1])

        # Concatenate: [img1_tokens, img2_tokens, ..., text_tokens]
        combined_embeds = torch.cat(
            [e.to(text_embeds.dtype) for e in all_img_embeds] + [text_embeds], dim=1
        )
        combined_mask = torch.cat(all_img_masks + [lang_masks], dim=1)

        # Forward through Gemma with bidirectional attention
        outputs = self.language_model.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        # Pool: last valid token per sequence
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
