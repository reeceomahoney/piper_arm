"""Shared reward and return helpers for value training and label generation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

REWARD_CONTEXT_FILENAME = "reward_context.safetensors"


# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RewardContext:
    """Normalized rewards and returns for a dataset."""

    reward_type: str
    gamma: float
    normalization_constant: float
    failure_penalty: float
    rewards: np.ndarray
    returns: np.ndarray


# ── Serialization ────────────────────────────────────────────────────────────


def save_reward_context(
    checkpoint_dir: Path | str,
    reward_context: RewardContext,
    num_frames: int,
) -> Path:
    """Save a reward-context sidecar next to a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    failure_penalty_scale = (
        reward_context.failure_penalty / reward_context.normalization_constant
        if reward_context.normalization_constant > 0
        else 0.0
    )
    arrays = {
        "rewards": reward_context.rewards.astype(np.float32),
        "returns": reward_context.returns.astype(np.float32),
        "normalization_constant": np.array(
            [reward_context.normalization_constant], dtype=np.float32
        ),
        "failure_penalty": np.array([reward_context.failure_penalty], dtype=np.float32),
        "failure_penalty_scale": np.array([failure_penalty_scale], dtype=np.float32),
        "gamma": np.array([reward_context.gamma], dtype=np.float32),
        "reward_type": np.frombuffer(
            reward_context.reward_type.encode(), dtype=np.uint8
        ),
        "num_frames": np.array([num_frames], dtype=np.int32),
    }

    output_path = checkpoint_dir / REWARD_CONTEXT_FILENAME
    save_file(arrays, str(output_path))
    return output_path


def load_reward_context(checkpoint_ref: str | Path, *, dataset=None) -> RewardContext:
    """Load and validate a saved reward context from a local checkpoint or Hub repo."""
    path = Path(checkpoint_ref)
    reward_path: Path | None = None

    if path.exists():
        candidate = path / REWARD_CONTEXT_FILENAME if path.is_dir() else path
        if candidate.exists():
            reward_path = candidate
    else:
        from huggingface_hub import hf_hub_download

        try:
            reward_path = Path(
                hf_hub_download(
                    str(checkpoint_ref), REWARD_CONTEXT_FILENAME, repo_type="model"
                )
            )
        except Exception:
            reward_path = None

    if reward_path is None:
        raise FileNotFoundError(
            "Missing reward_context.safetensors next to the value checkpoint. "
            "Re-run `distal.train_value` to produce a checkpoint with saved reward "
            "context before computing advantage labels."
        )

    data = load_file(str(reward_path))
    reward_type = data["reward_type"].tobytes().decode()
    num_frames = int(data["num_frames"][0])

    expected_num_frames = getattr(dataset, "num_frames", None)
    if expected_num_frames is not None and num_frames != expected_num_frames:
        raise ValueError(
            f"Saved reward context does not match the current dataset: "
            f"checkpoint_num_frames={num_frames}, "
            f"dataset_num_frames={expected_num_frames}"
        )

    return RewardContext(
        reward_type=reward_type,
        gamma=float(data["gamma"][0]),
        normalization_constant=float(data["normalization_constant"][0]),
        failure_penalty=float(data["failure_penalty"][0]),
        rewards=data["rewards"].astype(np.float64),
        returns=data["returns"].astype(np.float64),
    )


# ── Building contexts ────────────────────────────────────────────────────────


def build_reward_context(
    cfg,
    episode_index: np.ndarray,
    success: np.ndarray,
    steps_remaining: np.ndarray | None = None,
    max_episode_length: int | None = None,
    dataset=None,
    device: str | None = None,
    normalization_constant: float | None = None,
) -> RewardContext:
    """Build a reward context using cfg.reward_type to select the reward path."""
    if cfg.reward_type == "maha":
        policy_path = getattr(cfg, "base_policy", None) or getattr(
            cfg, "pretrained_path", None
        )
        if dataset is None:
            raise ValueError("dataset is required when cfg.reward_type='maha'")
        if policy_path is None:
            raise ValueError(
                "cfg must provide `base_policy` or `pretrained_path` for maha rewards"
            )
        maha = _compute_maha_distances_from_dataset(
            dataset,
            policy_path=policy_path,
            device=device or getattr(cfg, "device", "cpu"),
            stats_repo_id=cfg.stats_repo_id,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        return _build_maha_context(
            maha,
            success,
            episode_index,
            cfg.gamma,
            cfg.failure_penalty_scale,
            normalization_constant=normalization_constant,
        )

    if steps_remaining is None or max_episode_length is None:
        raise ValueError(
            "steps_remaining and max_episode_length are required when "
            "cfg.reward_type='steps'"
        )
    return _build_steps_context(
        steps_remaining,
        success,
        episode_index,
        max_episode_length,
        cfg.gamma,
        cfg.failure_penalty_scale,
    )


def _build_steps_context(
    steps_remaining: np.ndarray,
    success: np.ndarray,
    episode_index: np.ndarray,
    max_episode_length: int,
    gamma: float,
    failure_penalty_scale: float,
) -> RewardContext:
    """Build normalized rewards/returns for step-based supervision."""
    normalization_constant = _steps_normalization_constant(max_episode_length, gamma)
    failure_penalty = normalization_constant * failure_penalty_scale

    rewards = np.zeros(len(steps_remaining), dtype=np.float64)
    rewards[steps_remaining > 0] = -1.0 / normalization_constant

    failed_terminal = (steps_remaining == 0) & (~success.astype(bool))
    rewards[failed_terminal] -= failure_penalty_scale

    returns = _discounted_cumsum_by_episode(rewards, episode_index, gamma)
    returns = np.clip(returns, -1.0, 0.0)

    return RewardContext(
        reward_type="steps",
        gamma=gamma,
        normalization_constant=normalization_constant,
        failure_penalty=failure_penalty,
        rewards=rewards,
        returns=returns,
    )


def _compute_maha_distances_from_dataset(
    dataset,
    policy_path: str,
    device: str,
    stats_repo_id: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Load a policy and Gaussian stats, compute per-frame Mahalanobis distances."""
    from huggingface_hub import hf_hub_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.numpy import load_file

    from distal.compute_maha_stats import compute_maha_distances

    stats_file = hf_hub_download(
        stats_repo_id, "stats.safetensors", repo_type="dataset"
    )
    data = load_file(stats_file)
    gauss_mean = data["mean"]
    gauss_cov_inv = data["cov_inv"]
    print(f"Loaded Gaussian stats from {stats_repo_id}, dim={gauss_mean.shape[0]}")

    print("Loading policy for Mahalanobis distance computation...")
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = device
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
    policy.eval()

    policy_preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
    )

    return compute_maha_distances(
        policy,
        policy_preprocessor,
        dataset,
        gauss_mean,
        gauss_cov_inv,
        batch_size,
        num_workers,
    )


def _build_maha_context(
    maha: np.ndarray,
    success: np.ndarray,
    episode_index: np.ndarray,
    gamma: float,
    failure_penalty_scale: float,
    normalization_constant: float | None = None,
) -> RewardContext:
    """Build normalized rewards/returns for Mahalanobis-distance supervision."""
    maha_f64 = maha.astype(np.float64)

    if normalization_constant is None:
        raw_returns = _discounted_cumsum_by_episode(-maha_f64, episode_index, gamma)
        normalization_constant = float(np.abs(raw_returns).max())

    if normalization_constant <= 0:
        normalization_constant = 1.0

    failure_penalty = normalization_constant * failure_penalty_scale
    rewards = -maha_f64 / normalization_constant

    terminal = _episode_terminal_mask(episode_index)
    failed_terminal = terminal & (~success.astype(bool))
    rewards[failed_terminal] -= failure_penalty_scale

    returns = _discounted_cumsum_by_episode(rewards, episode_index, gamma)
    returns = np.clip(returns, -1.0, 0.0)

    return RewardContext(
        reward_type="maha",
        gamma=gamma,
        normalization_constant=normalization_constant,
        failure_penalty=failure_penalty,
        rewards=rewards,
        returns=returns,
    )


# ── Advantages ───────────────────────────────────────────────────────────────


def compute_nstep_advantages(
    values: np.ndarray,
    rewards: np.ndarray,
    returns: np.ndarray,
    episode_index: np.ndarray,
    n_step: int,
    gamma: float,
) -> np.ndarray:
    """Compute n-step TD advantages with MC fallback near episode end.

    When n_step=0, uses pure MC advantages: A(s_t) = G_t - V(s_t).
    """
    num_frames = len(values)

    if n_step == 0:
        return returns - values

    advantages = np.zeros(num_frames, dtype=np.float64)

    discounts = gamma ** np.arange(n_step)
    gamma_n = gamma**n_step

    for i in range(num_frames):
        target = i + n_step
        in_episode = target < num_frames and episode_index[target] == episode_index[i]

        if in_episode:
            discounted_rewards = np.dot(discounts, rewards[i:target])
            advantages[i] = discounted_rewards + gamma_n * values[target] - values[i]
        else:
            advantages[i] = returns[i] - values[i]

    return advantages


# ── Helpers ──────────────────────────────────────────────────────────────────


def _discounted_cumsum_by_episode(
    rewards: np.ndarray,
    episode_index: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute discounted returns with episode boundaries respected."""
    n = len(rewards)
    returns = np.zeros(n, dtype=np.float64)

    cumsum = 0.0
    for i in range(n - 1, -1, -1):
        if i == n - 1 or episode_index[i] != episode_index[i + 1]:
            cumsum = 0.0
        cumsum = float(rewards[i]) + gamma * cumsum
        returns[i] = cumsum

    return returns


def _episode_terminal_mask(episode_index: np.ndarray) -> np.ndarray:
    """Return a boolean mask for terminal frames."""
    terminal = np.zeros(len(episode_index), dtype=bool)
    if len(terminal) == 0:
        return terminal

    terminal[-1] = True
    terminal[:-1] = episode_index[:-1] != episode_index[1:]
    return terminal


def _steps_normalization_constant(
    max_episode_length: int,
    gamma: float,
) -> float:
    """Return the step-reward normalization constant before failure penalties."""
    if gamma == 1.0:
        return float(max_episode_length)

    discounted_steps = (1 - gamma**max_episode_length) / (1 - gamma)
    return float(discounted_steps)
