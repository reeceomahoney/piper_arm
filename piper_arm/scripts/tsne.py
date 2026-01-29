"""Visualize encoder tokens from ACT policies using t-SNE."""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

CONFIGS = [
    (
        "reece-omahoney/act-aloha-transfer-cube-finetuned",
        "lerobot/aloha_sim_transfer_cube_human",
        "Fine-tuned",
    ),
    (
        "reece-omahoney/act-aloha-transfer-cube",
        "lerobot/aloha_sim_transfer_cube_human",
        "From scratch",
    ),
]


def load_policy_and_preprocessor(repo: str, device: str):
    policy = ACTPolicy.from_pretrained(repo)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=repo)
    return policy, preprocessor


def extract_encoder_output(policy, batch) -> torch.Tensor:
    encoder_output = []

    def hook(module, input, output):
        encoder_output.append(output.detach().cpu())

    handle = policy.model.encoder.register_forward_hook(hook)
    with torch.no_grad():
        policy.predict_action_chunk(batch)
    handle.remove()

    return encoder_output[0].squeeze(1)


def get_encoder_output(policy_repo: str, dataset_repo: str, device: str) -> np.ndarray:
    print(f"Loading policy from {policy_repo}...")
    policy, preprocessor = load_policy_and_preprocessor(policy_repo, device)

    print(f"Loading sample from {dataset_repo}...")
    batch = preprocessor(LeRobotDataset(dataset_repo, episodes=[0])[0])

    print("Extracting encoder output...")
    encoder_out = extract_encoder_output(policy, batch)
    print(f"Encoder output shape: {encoder_out.shape}")

    return encoder_out.numpy()


def main():
    outputs, names = [], []
    for policy_repo, dataset_repo, name in CONFIGS:
        print(f"\n=== {name} ===")
        outputs.append(get_encoder_output(policy_repo, dataset_repo, "cuda"))
        names.append(name)

    print("\nComputing joint t-SNE...")
    combined = np.vstack(outputs)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    combined_2d = tsne.fit_transform(combined)

    # Split back into separate embeddings
    split_indices = np.cumsum([len(o) for o in outputs[:-1]])
    embeddings_2d = np.split(combined_2d, split_indices)

    ax = plt.subplots(figsize=(8, 6))[1]
    colors = ["tab:blue", "tab:orange"]

    for emb, color in zip(embeddings_2d, colors):
        ax.scatter(emb[0, 0], emb[0, 1], c=color, s=100, marker="^", alpha=0.9)
        ax.scatter(emb[1, 0], emb[1, 1], c=color, s=100, marker="s", alpha=0.9)
        ax.scatter(emb[2:, 0], emb[2:, 1], c=color, s=15, alpha=0.5)

    line_fn = partial(Line2D, [0], [0], color="w", markersize=10)
    ax.legend(
        handles=[
            line_fn(marker="^", markerfacecolor="gray", label="Latent"),
            line_fn(marker="s", markerfacecolor="gray", label="Robot State"),
            line_fn(marker="o", markerfacecolor="gray", label="Image Patches"),
            line_fn(marker="o", markerfacecolor="tab:blue", label="Fine-tuned"),
            line_fn(marker="o", markerfacecolor="tab:orange", label="From scratch"),
        ],
        loc="best",
    )
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title("Encoder Tokens (by type)")

    plt.tight_layout()
    # plt.savefig("tsne.png", dpi=150)
    print("\nSaved to tsne.png")
    plt.show()


if __name__ == "__main__":
    main()
