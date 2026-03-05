"""Advantage-conditioned policy configuration.

Wraps SmolVLAConfig fields and adds advantage-specific parameters.
Registered as the "advantage" policy type for LeRobot plugin discovery.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("advantage")
@dataclass
class AdvantageConfig(PreTrainedConfig):
    # Advantage-specific
    advantage_dropout: float = 0.3
    smolvla_checkpoint: str | None = None
    stats_repo_id: str | None = None

    # SmolVLA fields (mirrored so lerobot-train can set them via YAML)
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    max_state_dim: int = 32
    max_action_dim: int = 32

    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0
    adapt_to_pi_aloha: bool = False
    use_delta_joint_actions_aloha: bool = False

    tokenizer_max_length: int = 48
    num_steps: int = 10
    use_cache: bool = True

    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False

    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"

    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    min_period: float = 4e-3
    max_period: float = 4.0

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) "
                f"must be <= chunk_size ({self.chunk_size})."
            )

    def to_smolvla_config(self) -> SmolVLAConfig:
        """Build a SmolVLAConfig from the shared fields."""
        return SmolVLAConfig(
            n_obs_steps=self.n_obs_steps,
            chunk_size=self.chunk_size,
            n_action_steps=self.n_action_steps,
            normalization_mapping=self.normalization_mapping,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            resize_imgs_with_padding=self.resize_imgs_with_padding,
            empty_cameras=self.empty_cameras,
            adapt_to_pi_aloha=self.adapt_to_pi_aloha,
            use_delta_joint_actions_aloha=self.use_delta_joint_actions_aloha,
            tokenizer_max_length=self.tokenizer_max_length,
            num_steps=self.num_steps,
            use_cache=self.use_cache,
            freeze_vision_encoder=self.freeze_vision_encoder,
            train_expert_only=self.train_expert_only,
            train_state_proj=self.train_state_proj,
            optimizer_lr=self.optimizer_lr,
            optimizer_betas=self.optimizer_betas,
            optimizer_eps=self.optimizer_eps,
            optimizer_weight_decay=self.optimizer_weight_decay,
            optimizer_grad_clip_norm=self.optimizer_grad_clip_norm,
            scheduler_warmup_steps=self.scheduler_warmup_steps,
            scheduler_decay_steps=self.scheduler_decay_steps,
            scheduler_decay_lr=self.scheduler_decay_lr,
            vlm_model_name=self.vlm_model_name,
            load_vlm_weights=self.load_vlm_weights,
            add_image_special_tokens=self.add_image_special_tokens,
            attention_mode=self.attention_mode,
            prefix_length=self.prefix_length,
            pad_language_to=self.pad_language_to,
            num_expert_layers=self.num_expert_layers,
            num_vlm_layers=self.num_vlm_layers,
            self_attn_every_n_layers=self.self_attn_every_n_layers,
            expert_width_multiplier=self.expert_width_multiplier,
            min_period=self.min_period,
            max_period=self.max_period,
            input_features=self.input_features,
            output_features=self.output_features,
            device=self.device,
        )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
