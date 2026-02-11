"""Train linear probes on VLM token embeddings.

Extracts representations after 0, 4, 8, 12, and 16 layers of the VLM encoder
from the pretrained SmolVLA model and trains a linear probe for each target at
each depth. Targets include contact_state (binary classification), object_pos,
gripper_pos (3D regression), and gripper_to_obj_dist (scalar regression).

Use --random_weights to run the same experiment with randomly initialized
weights as a control baseline.
"""

import argparse

import torch
import torch.nn as nn
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    make_att_2d_masks,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from prettytable import PrettyTable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PRETRAINED_PATH = "reece-omahoney/smolvla-libero-256"
DATASET_REPO = "reece-omahoney/libero-affordances"
DEVICE = "cuda"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
PATIENCE = 7
LAYER_DEPTHS = [0, 4, 8, 12, 16]

TARGET_CONFIG = {
    "contact_state": {"output_dim": 1, "task": "classification"},
    "object_pos": {"output_dim": 3, "task": "regression"},
    "gripper_pos": {"output_dim": 3, "task": "regression"},
    "gripper_to_obj_dist": {"output_dim": 1, "task": "regression"},
}


def load_policy(
    device: str, random_weights: bool = False
) -> tuple[SmolVLAPolicy, callable]:
    if random_weights:
        pretrained = SmolVLAPolicy.from_pretrained(PRETRAINED_PATH)
        config = pretrained.config
        config.load_vlm_weights = False
        policy = SmolVLAPolicy(config)
        del pretrained
    else:
        policy = SmolVLAPolicy.from_pretrained(PRETRAINED_PATH)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=PRETRAINED_PATH
    )
    return policy, preprocessor


def mean_pool(embs: torch.Tensor, pad_masks: torch.Tensor) -> torch.Tensor:
    """Mean-pool over sequence dim, masking padded tokens."""
    pad_masks_f = pad_masks.float().unsqueeze(-1)
    return (embs * pad_masks_f).sum(dim=1) / pad_masks_f.sum(dim=1).clamp(min=1)


def forward_n_layers(
    policy: SmolVLAPolicy,
    embs: torch.Tensor,
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
    n_layers: int,
) -> torch.Tensor:
    """Run prefix embeddings through the first n VLM layers. Returns hidden states."""
    vlm_with_expert = policy.model.vlm_with_expert
    text_model = vlm_with_expert.get_vlm_model().text_model

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    original_layers = text_model.layers
    original_num = vlm_with_expert.num_vlm_layers
    text_model.layers = original_layers[:n_layers]
    vlm_with_expert.num_vlm_layers = n_layers
    try:
        (prefix_out, _), _ = vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=[embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
    finally:
        text_model.layers = original_layers
        vlm_with_expert.num_vlm_layers = original_num

    return prefix_out


def extract_all_embeddings(
    policy: SmolVLAPolicy,
    preprocessor: callable,
    dataset: LeRobotDataset,
    device: str,
    layer_depths: list[int],
    targets: list[str],
) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    """Extract mean-pooled embeddings at multiple layer depths and labels for all targets."""
    all_embeddings = {n: [] for n in layer_depths}
    all_labels = {t: [] for t in targets}

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    for raw_batch in tqdm(dataloader, desc="Extracting embeddings"):
        for t in targets:
            t_labels = raw_batch[t]
            if t_labels.ndim == 2 and t_labels.shape[-1] == 1:
                t_labels = t_labels.squeeze(-1)
            all_labels[t].append(t_labels)

        batch = preprocessor(raw_batch)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

            embs, pad_masks, att_masks = policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state
            )

            for n in layer_depths:
                if n == 0:
                    pooled = mean_pool(embs, pad_masks)
                else:
                    hidden = forward_n_layers(policy, embs, pad_masks, att_masks, n)
                    pooled = mean_pool(hidden, pad_masks)
                all_embeddings[n].append(pooled.cpu())

    labels_by_target = {t: torch.cat(v, dim=0) for t, v in all_labels.items()}
    embeddings = {n: torch.cat(v, dim=0) for n, v in all_embeddings.items()}

    n_samples = len(next(iter(labels_by_target.values())))
    print(f"Samples: {n_samples}")
    for t, labels in labels_by_target.items():
        cfg = TARGET_CONFIG[t]
        if cfg["task"] == "classification":
            print(f"  {t}: positive rate {labels.mean():.3f}")
        else:
            print(f"  {t}: mean {labels.mean(dim=0)}, std {labels.std(dim=0)}")
    for n, e in embeddings.items():
        print(f"  n_layers={n:2d}: {e.shape}")

    return embeddings, labels_by_target


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


def train_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int,
    output_dim: int,
    task: str,
    device: str,
) -> float:
    """Train a linear probe. Returns best val accuracy (classification) or MAE (regression)."""
    is_cls = task == "classification"
    metric_name = "acc" if is_cls else "mae"

    n = len(embeddings)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_loader = DataLoader(
        TensorDataset(embeddings[train_idx], labels[train_idx]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(embeddings[val_idx], labels[val_idx]),
        batch_size=BATCH_SIZE,
    )

    model = LinearProbe(hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss() if is_cls else nn.MSELoss()

    best_val_loss, best_val_metric, patience_counter = float("inf"), 0.0, 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_metric, train_total = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
            if is_cls:
                train_metric += ((pred > 0).float() == y).sum().item()
            else:
                train_metric += (pred - y).abs().sum().item()
            train_total += len(x)

        model.eval()
        val_loss, val_metric, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * len(x)
                if is_cls:
                    val_metric += ((pred > 0).float() == y).sum().item()
                else:
                    val_metric += (pred - y).abs().sum().item()
                val_total += len(x)

        train_loss /= train_total
        val_loss /= val_total
        train_metric /= train_total
        val_metric /= val_total

        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} {metric_name}: {train_metric:.4f} | "
            f"Val loss: {val_loss:.4f} {metric_name}: {val_metric:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metric = val_metric
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    return best_val_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Reinitialize all model weights randomly (control baseline)",
    )
    args = parser.parse_args()

    tag = "random" if args.random_weights else "pretrained"
    targets = list(TARGET_CONFIG.keys())

    policy, preprocessor = load_policy(DEVICE, random_weights=args.random_weights)
    dataset = LeRobotDataset(DATASET_REPO)
    hidden_dim = policy.model.vlm_with_expert.config.text_config.hidden_size

    embeddings_by_depth, labels_by_target = extract_all_embeddings(
        policy, preprocessor, dataset, DEVICE, LAYER_DEPTHS, targets
    )

    del policy
    torch.cuda.empty_cache()

    all_results = {}
    for target in targets:
        cfg = TARGET_CONFIG[target]
        all_results[target] = {}
        for n in LAYER_DEPTHS:
            print(f"\n{'=' * 60}")
            print(f"Training probe ({tag}, {target}): n_layers={n}")
            print(f"{'=' * 60}")
            all_results[target][n] = train_probe(
                embeddings_by_depth[n],
                labels_by_target[target],
                hidden_dim,
                cfg["output_dim"],
                cfg["task"],
                DEVICE,
            )

    # Print summary table
    metric_names = {
        t: "acc" if TARGET_CONFIG[t]["task"] == "classification" else "mae"
        for t in targets
    }
    table = PrettyTable()
    table.field_names = ["n_layers"] + [f"{t} ({metric_names[t]})" for t in targets]
    for n in LAYER_DEPTHS:
        table.add_row([n] + [f"{all_results[t][n]:.4f}" for t in targets])
    print(f"\nSummary ({tag})\n")
    print(table)


if __name__ == "__main__":
    main()
