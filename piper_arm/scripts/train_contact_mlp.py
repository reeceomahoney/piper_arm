"""Train linear probes on VLM token embeddings to predict contact state.

Extracts representations after 0, 4, 8, 12, and 16 layers of the VLM encoder
from the pretrained SmolVLA model and trains a binary contact-state probe at
each depth.

  n=0  -> input token embeddings (output of embed_prefix())
  n>0  -> hidden states after n transformer layers

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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PRETRAINED_PATH = "reece-omahoney/smolvla-libero"
DATASET_REPO = "reece-omahoney/libero-affordances"
DEVICE = "cuda"
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
PATIENCE = 7
LAYER_DEPTHS = [0, 4, 8, 12, 16]


def load_policy(
    device: str, random_weights: bool = False
) -> tuple[SmolVLAPolicy, callable]:
    policy = SmolVLAPolicy.from_pretrained(PRETRAINED_PATH)
    if random_weights:
        print("Reinitializing all weights randomly...")
        for module in policy.modules():
            if hasattr(module, "reset_parameters"):
                print(f"  Resetting {module.__class__.__name__}")
                module.reset_parameters()
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
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Extract mean-pooled embeddings at multiple layer depths in one pass."""
    all_embeddings = {n: [] for n in layer_depths}
    all_labels = []

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    for raw_batch in tqdm(dataloader, desc="Extracting embeddings"):
        contact_labels = raw_batch["contact_state"].squeeze(-1)
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

        all_labels.append(contact_labels)

    labels = torch.cat(all_labels, dim=0)
    embeddings = {n: torch.cat(v, dim=0) for n, v in all_embeddings.items()}

    print(f"Samples: {len(labels)}, Contact rate: {labels.mean():.3f}")
    for n, e in embeddings.items():
        print(f"  n_layers={n:2d}: {e.shape}")

    return embeddings, labels


class ContactProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int,
    device: str,
) -> tuple[float, float]:
    """Train a linear probe. Returns (best_val_loss, best_val_acc)."""
    n = len(embeddings)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(embeddings[train_idx], labels[train_idx])
    val_ds = TensorDataset(embeddings[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ContactProbe(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
            train_correct += ((logits > 0).float() == y).sum().item()
            train_total += len(x)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(x)
                val_correct += ((logits > 0).float() == y).sum().item()
                val_total += len(x)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    return best_val_loss, best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Reinitialize all model weights randomly (control baseline)",
    )
    args = parser.parse_args()

    tag = "random" if args.random_weights else "pretrained"
    print(f"Mode: {tag}")

    policy, preprocessor = load_policy(DEVICE, random_weights=args.random_weights)
    dataset = LeRobotDataset(DATASET_REPO)
    hidden_dim = policy.model.vlm_with_expert.config.text_config.hidden_size

    embeddings_by_depth, labels = extract_all_embeddings(
        policy, preprocessor, dataset, DEVICE, LAYER_DEPTHS
    )

    del policy
    torch.cuda.empty_cache()

    results = {}
    for n in LAYER_DEPTHS:
        print(f"\n{'=' * 60}")
        print(f"Training probe ({tag}): n_layers={n}")
        print(f"{'=' * 60}")
        best_loss, best_acc = train_probe(
            embeddings_by_depth[n], labels, hidden_dim, DEVICE
        )
        results[n] = (best_loss, best_acc)

    print(f"\n{'=' * 60}")
    print(f"Summary ({tag})")
    print(f"{'=' * 60}")
    print(f"{'n_layers':>10} {'val_loss':>10} {'val_acc':>10}")
    for n, (loss, acc) in results.items():
        print(f"{n:>10} {loss:>10.4f} {acc:>10.4f}")


if __name__ == "__main__":
    main()
