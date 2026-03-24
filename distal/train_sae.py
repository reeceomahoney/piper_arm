"""Train a Sparse Autoencoder on VLM prefix token activations for OOD detection."""

import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import draccus
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, inside_slurm
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from distal.embedding import embed_prefix_tokens
from distal.sae import SAEConfig, SparseAutoencoder


@dataclass
class TrainSAEConfig:
    policy_path: str = "reece-omahoney/adv-libero-base"
    dataset_repo_id: str = "reece-omahoney/libero-10"
    hub_repo_id: str = "reece-omahoney/sae-libero"
    output_dir: str | None = None

    # SAE hyperparams
    expansion_factor: int = 1
    l1_penalty: float = 0.3
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 20
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    val_fraction: float = 0.1

    # Infrastructure
    device: str = "cuda"
    embedding_batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    use_amp: bool = True
    wandb_project: str | None = "distal-sae"
    push_to_hub: bool = True
    token_cache_path: str = "outputs/sae/token_cache.pt"


def extract_all_tokens(
    policy: PI05Policy | SmolVLAPolicy,
    preprocessor,
    dataset: LeRobotDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract image token activations from the dataset.

    Returns:
        (N, n_img_tokens, hidden_dim) float32 tensor.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    all_tokens = []
    for batch in tqdm(loader, desc="Extracting tokens", disable=inside_slurm()):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = preprocessor(batch)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = embed_prefix_tokens(policy, batch)
        all_tokens.append(tokens.float().cpu())

    return torch.cat(all_tokens, dim=0)


@draccus.wrap()
def main(cfg: TrainSAEConfig):
    init_logging()
    register_third_party_plugins()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    now = datetime.now()
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = (
            Path("outputs/sae") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if cfg.wandb_project:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.hub_repo_id.split("/")[-1],
            config=vars(cfg),
            dir=str(output_dir),
        )

    # Extract all token activations (or load from cache)
    cache_path = Path(cfg.token_cache_path)
    if cache_path.exists():
        print(f"Loading cached tokens from {cache_path}")
        all_tokens = torch.load(cache_path, weights_only=True)
    else:
        dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id)
        print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

        policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
        policy_cfg.pretrained_path = Path(cfg.policy_path)
        policy_cfg.device = str(device)

        policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
        assert isinstance(policy, (PI05Policy, SmolVLAPolicy))
        policy.eval()

        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
        )

        all_tokens = extract_all_tokens(
            policy,
            preprocessor,
            dataset,
            cfg.embedding_batch_size,
            cfg.num_workers,
            device,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_tokens, cache_path)
        print(f"Saved token cache to {cache_path}")
    n_samples, n_tokens, token_dim = all_tokens.shape
    print(f"Extracted {n_samples} samples, {n_tokens} tokens, dim={token_dim}")

    # Flatten to individual tokens for per-token SAE training
    all_individual_tokens = all_tokens.reshape(-1, token_dim)
    n_total = all_individual_tokens.shape[0]
    print(f"Flattened to {n_total} individual tokens")

    # Train/val split
    indices = np.random.permutation(n_total)
    val_size = int(n_total * cfg.val_fraction)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_dataset = TensorDataset(all_individual_tokens[train_idx])
    val_dataset = TensorDataset(all_individual_tokens[val_idx])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Construct SAE (per-token: input_dim = token_dim)
    sae_config = SAEConfig(
        input_dim=token_dim,
        expansion_factor=cfg.expansion_factor,
        l1_penalty=cfg.l1_penalty,
    )
    model = SparseAutoencoder(sae_config).to(device)
    print(
        f"SAE: input_dim={sae_config.input_dim}, feature_dim={sae_config.feature_dim}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if cfg.use_amp and device.type == "cuda"
        else nullcontext()
    )

    for epoch in range(1, cfg.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_l1_sum = 0.0
        train_steps = 0

        for (batch_tokens,) in train_loader:
            batch_tokens = batch_tokens.to(device)
            with autocast:
                loss, info = model.compute_loss(batch_tokens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            train_loss_sum += info["loss"]
            train_mse_sum += info["mse"]
            train_l1_sum += info["l1"]
            train_steps += 1

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_l1_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for (batch_tokens,) in val_loader:
                batch_tokens = batch_tokens.to(device)
                with autocast:
                    _, info = model.compute_loss(batch_tokens)
                val_loss_sum += info["loss"]
                val_mse_sum += info["mse"]
                val_l1_sum += info["l1"]
                val_steps += 1

        train_loss = train_loss_sum / train_steps
        train_mse = train_mse_sum / train_steps
        train_l1 = train_l1_sum / train_steps
        val_loss = val_loss_sum / max(val_steps, 1)
        val_mse = val_mse_sum / max(val_steps, 1)
        val_l1 = val_l1_sum / max(val_steps, 1)

        print(
            f"[epoch {epoch:>3d}] train_loss={train_loss:.6f} "
            f"train_mse={train_mse:.6f} train_l1={train_l1:.6f} | "
            f"val_loss={val_loss:.6f} val_mse={val_mse:.6f} val_l1={val_l1:.6f}"
        )

        if cfg.wandb_project:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/mse": train_mse,
                    "train/l1": train_l1,
                    "val/loss": val_loss,
                    "val/mse": val_mse,
                    "val/l1": val_l1,
                    "epoch": epoch,
                },
                step=epoch,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_pretrained(output_dir / "checkpoint_best")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best checkpoint for final save / push
    model = SparseAutoencoder.from_pretrained(output_dir / "checkpoint_best")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Saved best checkpoint to {output_dir / 'checkpoint_best'}")

    if cfg.push_to_hub:
        model.push_to_hub(cfg.hub_repo_id)
        print(f"Pushed to https://huggingface.co/{cfg.hub_repo_id}")


if __name__ == "__main__":
    main()
