"""Advantage-conditioned policy fine-tuning (RECAP-style).

Uses a trained value model to compute per-timestep advantages, binarizes them,
and conditions the SmolVLA policy on a binary advantage indicator during training.
At inference, always condition on "positive" to bias toward high-advantage actions.
30% dropout of the advantage token enables classifier-free guidance.

Usage:
    python piper_arm/train_advantage.py --config_path configs/advantage.yaml
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import draccus
import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import Tensor, nn
from torch.utils.data import DataLoader

from piper_arm.train_value import TrainValueConfig, compute_returns  # noqa: F401
from piper_arm.value_model import ValueConfig, ValueModel


@dataclass
class TrainAdvantageConfig:
    policy_repo_id: str = "reece-omahoney/smolvla-libero-16-chunk"
    value_checkpoint: str = "outputs/value/checkpoint_final.pt"
    dataset_repo_id: str = "reece-omahoney/libero-10-maha"
    dataset_root: str | None = None

    advantage_percentile: float = 0.3
    advantage_dropout: float = 0.3
    c_fail: float = 1000.0

    batch_size: int = 32
    total_steps: int = 50_000

    log_interval: int = 50
    save_interval: int = 5000
    eval_freq: int = 5_000
    eval_n_episodes: int = 10
    eval_batch_size: int = 1
    output_dir: str = "outputs/advantage"
    wandb_project: str | None = "piper-advantage"
    wandb_run_name: str | None = None
    push_to_hub: bool = False
    hub_repo_id: str = "reece-omahoney/smolvla-libero-advantage"

    num_workers: int = 4


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


@contextmanager
def advantage_inference(policy: SmolVLAPolicy, adv_embedding: AdvantageEmbedding):
    """Patch policy to inject advantage token (always=1) during inference."""
    model = policy.model
    original_sample_actions = model.sample_actions

    @wraps(original_sample_actions)
    def patched_sample_actions(
        images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ):
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
            noise = model.sample_noise(shape, device)

        # Embed prefix as usual
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # Inject advantage token (always positive=1)
        pos = torch.ones(bsize, dtype=torch.long, device=device)
        adv_embs = adv_embedding(pos).to(dtype=prefix_embs.dtype)
        att_dtype = prefix_att_masks.dtype
        pad_dtype = prefix_pad_masks.dtype
        adv_att = torch.ones(bsize, 1, dtype=att_dtype, device=device)
        adv_pad = torch.ones(bsize, 1, dtype=pad_dtype, device=device)

        prefix_embs = torch.cat([prefix_embs, adv_embs], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, adv_pad], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, adv_att], dim=1)

        # KV cache forward
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
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

    model.sample_actions = patched_sample_actions
    try:
        yield
    finally:
        model.sample_actions = original_sample_actions


def load_value_model(
    checkpoint_path: str, device: torch.device
) -> tuple[ValueModel, int]:
    """Load a trained value model from checkpoint.

    Returns:
        (model, max_episode_length) — model in eval mode with no grad.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: ValueConfig = ckpt["config"].value
    model = ValueModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _prepare_value_inputs(
    batch: dict[str, Tensor],
    value_model: ValueModel,
    tokenizer,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor, Tensor]:
    """Prepare a batch for the value model.

    Handles image preprocessing (resize, normalize), state padding,
    and language tokenization.

    Returns:
        (images, img_masks, lang_tokens, lang_masks, state)
    """
    img_keys = sorted(k for k in batch if k.startswith("observation.images."))
    images = []
    img_masks = []
    for key in img_keys:
        img = batch[key]
        img = img[:, -1] if img.ndim == 5 else img
        img = img.to(device)
        w, h = value_model.config.resize_imgs_with_padding
        img = resize_with_pad(img, w, h, pad_value=-1)
        img = img * 2.0 - 1.0
        images.append(img)
        img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))

    state = batch["observation.state"]
    state = state[:, -1] if state.ndim > 2 else state
    state = state.to(device)
    state = pad_vector(state, value_model.config.max_state_dim)

    task_texts = batch["task"]
    tokenized = tokenizer(
        task_texts,
        padding="longest",
        truncation=True,
        max_length=value_model.config.tokenizer_max_length,
        return_tensors="pt",
    )
    lang_tokens = tokenized["input_ids"].to(device)
    lang_masks = tokenized["attention_mask"].bool().to(device)

    return images, img_masks, lang_tokens, lang_masks, state


def compute_advantage_thresholds(
    dataset: LeRobotDataset,
    value_model: ValueModel,
    c_fail: float,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Compute per-task advantage thresholds (30th percentile).

    Iterates over the full dataset, computes V(s) and ground-truth returns,
    then finds the advantage percentile threshold per task.
    """
    all_steps = dataset.hf_dataset["steps_remaining"]
    max_ep_len = max(s.item() for s in all_steps) + 1
    tokenizer = value_model.vlm_with_expert.processor.tokenizer

    loader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    task_advantages: dict[str, list[float]] = {}

    for batch in loader:
        images, img_masks, lang_tokens, lang_masks, state = _prepare_value_inputs(
            batch, value_model, tokenizer, device
        )

        returns = compute_returns(
            batch["steps_remaining"].to(device),
            batch["success"].to(device),
            max_ep_len,
            c_fail,
        )

        with torch.no_grad():
            logits = value_model(images, img_masks, lang_tokens, lang_masks, state)
            values = value_model.predict_value(logits)

        advantages = (returns - values).cpu().tolist()

        for i, task in enumerate(batch["task"]):
            if task not in task_advantages:
                task_advantages[task] = []
            task_advantages[task].append(advantages[i])

    # Compute percentile threshold per task
    thresholds: dict[str, float] = {}
    for task, advs in task_advantages.items():
        advs_t = torch.tensor(advs)
        thresholds[task] = torch.quantile(advs_t, 0.3).item()
        print(
            f"  Task: {task[:60]:60s}  threshold={thresholds[task]:.4f}  n={len(advs)}"
        )

    return thresholds


def advantage_forward(
    policy: SmolVLAPolicy,
    adv_embedding: AdvantageEmbedding,
    batch: dict[str, Tensor],
    adv_labels: Tensor,
    adv_dropout: float,
    device: torch.device,
) -> Tensor:
    """Forward pass with advantage token injected between prefix and suffix.

    Args:
        policy: SmolVLAPolicy with loaded weights.
        adv_embedding: AdvantageEmbedding module.
        batch: Training batch from DataLoader.
        adv_labels: (B,) int tensor, 0=negative, 1=positive.
        adv_dropout: Probability of masking out the advantage token.
        device: Target device.

    Returns:
        Scalar MSE loss on flow matching velocity.
    """
    model = policy.model  # VLAFlowMatching

    # Prepare inputs using policy's preprocessing
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = policy.prepare_action(batch)

    # Sample noise and time for flow matching
    noise = model.sample_noise(actions.shape, actions.device)
    time = model.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # 1. Embed prefix (images + language + state)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    bsize = prefix_embs.shape[0]

    # 2. Embed advantage token
    adv_embs = adv_embedding(adv_labels.to(device))  # (B, 1, vlm_hidden)
    adv_embs = adv_embs.to(dtype=prefix_embs.dtype)

    # Advantage att_mask=1: prefix can't see it, but suffix (actions) can
    adv_att_masks = torch.ones(bsize, 1, dtype=prefix_att_masks.dtype, device=device)

    # Advantage pad_mask: apply dropout (mask out 30% of the time)
    dropout_mask = torch.rand(bsize, device=device) > adv_dropout
    adv_pad_masks = dropout_mask.unsqueeze(1)  # (B, 1)

    # 3. Concatenate prefix + advantage
    prefix_adv_embs = torch.cat([prefix_embs, adv_embs], dim=1)
    prefix_adv_pad_masks = torch.cat([prefix_pad_masks, adv_pad_masks], dim=1)
    prefix_adv_att_masks = torch.cat([prefix_att_masks, adv_att_masks], dim=1)

    # 4. Embed suffix (noisy actions + time)
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time)

    # 5. Full sequence: prefix + advantage + suffix
    pad_masks = torch.cat([prefix_adv_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_adv_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    # 6. Forward through VLM + expert
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_adv_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )

    # 7. Extract action predictions and compute loss
    suffix_out = suffix_out[:, -model.config.chunk_size :]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = model.action_out_proj(suffix_out)

    losses = F.mse_loss(u_t, v_t, reduction="none")
    # Remove padding dimensions
    losses = losses[:, :, : policy.config.max_action_dim]
    return losses.mean()


@draccus.wrap()  # type: ignore[misc]
def main(cfg: TrainAdvantageConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──
    if cfg.wandb_project:
        import wandb

        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))  # type: ignore[arg-type]

    # ── Dataset ──
    # Load policy config first to get chunk_size for delta_timestamps
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_repo_id)
    fps = LeRobotDataset(repo_id=cfg.dataset_repo_id, episodes=[0]).fps
    delta_timestamps = {
        "action": [i / fps for i in policy_cfg.action_delta_indices],
    }
    ds_kwargs: dict = {
        "repo_id": cfg.dataset_repo_id,
        "delta_timestamps": delta_timestamps,
    }
    if cfg.dataset_root:
        ds_kwargs["root"] = cfg.dataset_root
    dataset = LeRobotDataset(**ds_kwargs)

    # Auto-detect max episode length
    all_steps = dataset.hf_dataset["steps_remaining"]
    max_episode_length = max(s.item() for s in all_steps) + 1
    print(f"Max episode length: {max_episode_length}")
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # ── Value model (frozen) ──
    print("Loading value model...")
    value_model = load_value_model(cfg.value_checkpoint, device)

    # ── Compute advantage thresholds ──
    print("Computing per-task advantage thresholds...")
    thresholds = compute_advantage_thresholds(dataset, value_model, cfg.c_fail, device)

    # ── Policy ──
    print(f"Loading policy from {cfg.policy_repo_id}...")
    policy = SmolVLAPolicy.from_pretrained(cfg.policy_repo_id)
    policy = policy.to(device)
    policy.train()

    # ── Advantage embedding ──
    vlm_hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
    adv_embedding = AdvantageEmbedding(vlm_hidden).to(device)

    # ── Optimizer & scheduler (SmolVLA presets) ──
    smolvla_cfg = SmolVLAConfig()
    optimizer_cfg = smolvla_cfg.get_optimizer_preset()
    scheduler_cfg = smolvla_cfg.get_scheduler_preset()

    params = [p for p in policy.parameters() if p.requires_grad]
    params += list(adv_embedding.parameters())
    optimizer = optimizer_cfg.build(params)
    scheduler = scheduler_cfg.build(optimizer, num_training_steps=cfg.total_steps)

    # ── DataLoader ──
    loader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    policy_cfg.pretrained_path = Path(cfg.policy_repo_id)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
    )

    # ── Pre-compute value predictions for advantage labels ──
    # We need the value model's tokenizer for threshold computation, but during
    # training the policy uses its own lang_tokens from the dataset directly.
    # For per-batch advantage labels, we use the value model on-the-fly.
    value_tokenizer = value_model.vlm_with_expert.processor.tokenizer

    # ── Eval environment (LIBERO) ──
    eval_env = None
    env_preprocessor = None
    env_postprocessor = None
    if cfg.eval_freq > 0:
        print("Setting up LIBERO eval environment...")
        env_cfg = LiberoEnvConfig("libero_10", fps=10, task_ids=[9])
        eval_env = make_env(env_cfg, n_envs=cfg.eval_batch_size)
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg, policy_cfg
        )

    # ── Training loop ──
    data_iter = cycle(loader)

    for step in range(1, cfg.total_steps + 1):
        batch = next(data_iter)
        extra_keys = {
            k: batch[k]
            for k in ("maha_distance", "steps_remaining", "success")
            if k in batch
        }
        batch = preprocessor(batch)
        batch.update(
            {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in extra_keys.items()
            }
        )

        # Compute advantage labels for this batch
        with torch.no_grad():
            adv_labels = _compute_batch_advantage_labels(
                batch,
                value_model,
                value_tokenizer,
                thresholds,
                max_episode_length,
                cfg.c_fail,
                device,
            )

        loss = advantage_forward(
            policy, adv_embedding, batch, adv_labels, cfg.advantage_dropout, device
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, optimizer_cfg.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # ── Logging ──
        if step % cfg.log_interval == 0:
            pct_positive = adv_labels.float().mean().item()
            log = {
                "loss": loss.item(),
                "pct_positive": pct_positive,
                "lr": scheduler.get_last_lr()[0],
                "step": step,
            }
            lr_str = f"{log['lr']:.2e}"
            print(
                f"[step {step:>6d}] loss={log['loss']:.4f}"
                f"  pct_pos={pct_positive:.2f}  lr={lr_str}"
            )
            if cfg.wandb_project:
                wandb.log(log, step=step)  # type: ignore[possibly-undefined]

        # ── Checkpointing ──
        if step % cfg.save_interval == 0:
            _save_checkpoint(output_dir, step, policy, adv_embedding, optimizer, cfg)

        # ── Eval ──
        if eval_env and cfg.eval_freq > 0 and step % cfg.eval_freq == 0:
            print(f"\n[step {step}] Running evaluation...")
            policy.eval()
            with torch.no_grad(), advantage_inference(policy, adv_embedding):
                eval_info = eval_policy_all(
                    envs=eval_env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval_n_episodes,
                    videos_dir=output_dir / "eval" / f"videos_step_{step}",
                    max_episodes_rendered=2,
                    start_seed=0,
                )
            policy.train()

            overall = eval_info["overall"]
            pc_success = overall["pc_success"]
            avg_reward = overall["avg_sum_reward"]
            eval_s = overall["eval_s"]
            print(
                f"[step {step}] eval: success={pc_success:.1f}%"
                f"  avg_reward={avg_reward:.3f}  eval_s={eval_s:.1f}s"
            )
            if cfg.wandb_project:
                eval_log: dict = {
                    "eval/pc_success": pc_success,
                    "eval/avg_sum_reward": avg_reward,
                    "eval/eval_s": eval_s,
                }
                # Log per-task success rates
                for task_info in eval_info.get("per_task", []):
                    task_id = task_info["task_id"]
                    metrics = task_info["metrics"]
                    successes = metrics["successes"]
                    if successes:
                        eval_log[f"eval/task_{task_id}/pc_success"] = (
                            sum(successes) / len(successes) * 100
                        )
                        eval_log[f"eval/task_{task_id}/avg_sum_reward"] = sum(
                            metrics["sum_rewards"]
                        ) / len(metrics["sum_rewards"])
                # Log eval video
                video_paths = overall.get("video_paths", [])
                if video_paths:
                    eval_log["eval/video"] = wandb.Video(  # type: ignore[possibly-undefined]
                        video_paths[0], fps=fps, format="mp4"
                    )
                wandb.log(eval_log, step=step)  # type: ignore[possibly-undefined]

    # Close eval environments
    if eval_env:
        close_envs(eval_env)

    # Save final checkpoint
    _save_checkpoint(
        output_dir, cfg.total_steps, policy, adv_embedding, optimizer, cfg, final=True
    )
    print(f"Training complete. Final checkpoint: {output_dir / 'checkpoint_final.pt'}")

    # ── Push to Hub ──
    if cfg.push_to_hub:
        _push_to_hub(
            policy, adv_embedding, preprocessor, postprocessor, cfg.hub_repo_id
        )


def _push_to_hub(
    policy: SmolVLAPolicy,
    adv_embedding: AdvantageEmbedding,
    preprocessor,
    postprocessor,
    repo_id: str,
):
    """Save policy + advantage embedding and push to HuggingFace Hub.

    Mirrors lerobot_train's push logic: saves the policy via save_pretrained,
    adds the advantage embedding, then uploads everything.
    """
    from tempfile import TemporaryDirectory

    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = api.create_repo(repo_id=repo_id, private=True, exist_ok=True).repo_id

    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        saved_path = Path(tmp) / repo_id
        policy.save_pretrained(saved_path)

        # Save advantage embedding alongside the policy weights
        torch.save(
            adv_embedding.state_dict(),
            saved_path / "advantage_embedding.pt",
        )

        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=saved_path,
            commit_message="Upload advantage-conditioned policy",
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.pt"],
            ignore_patterns=["*.tmp", "*.log"],
        )
        print(f"Model pushed to {commit_info.repo_url.url}")

    # Push preprocessor/postprocessor if they were created (eval was enabled)
    if preprocessor is not None:
        preprocessor.push_to_hub(repo_id)
    if postprocessor is not None:
        postprocessor.push_to_hub(repo_id)


def _compute_batch_advantage_labels(
    batch: dict[str, Tensor],
    value_model: ValueModel,
    tokenizer,
    thresholds: dict[str, float],
    max_episode_length: int,
    c_fail: float,
    device: torch.device,
) -> Tensor:
    """Compute binary advantage labels for a batch.

    Returns:
        (B,) int tensor: 1 if advantage > task threshold, 0 otherwise.
    """
    images, img_masks, lang_tokens, lang_masks, state = _prepare_value_inputs(
        batch, value_model, tokenizer, device
    )

    returns = compute_returns(
        batch["steps_remaining"],
        batch["success"],
        max_episode_length,
        c_fail,
    )

    logits = value_model(images, img_masks, lang_tokens, lang_masks, state)
    values = value_model.predict_value(logits)
    advantages = returns - values

    # Binarize using per-task thresholds
    task_texts = batch["task"]
    labels = torch.zeros(len(task_texts), dtype=torch.long, device=device)
    for i, task in enumerate(task_texts):
        threshold = thresholds.get(task, 0.0)
        if advantages[i].item() > threshold:
            labels[i] = 1

    return labels


def _save_checkpoint(
    output_dir: Path,
    step: int,
    policy: SmolVLAPolicy,
    adv_embedding: AdvantageEmbedding,
    optimizer,
    cfg: TrainAdvantageConfig,
    final: bool = False,
):
    name = "checkpoint_final.pt" if final else f"checkpoint_{step}.pt"
    ckpt_path = output_dir / name
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "adv_embedding_state_dict": adv_embedding.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
