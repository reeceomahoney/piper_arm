"""Sparse Autoencoder for OOD detection on VLM prefix token activations.

Operates per-token: encoder/decoder take individual token vectors (token_dim),
not the full flattened sequence. For OOD scoring, per-token reconstruction
errors are averaged across the sequence.
"""

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file


@dataclass
class SAEConfig:
    input_dim: int = 0
    expansion_factor: int = 1
    l1_penalty: float = 0.3

    @property
    def feature_dim(self) -> int:
        return self.input_dim * self.expansion_factor


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.feature_dim)
        self.decoder = nn.Linear(config.feature_dim, config.input_dim)
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode individual token vectors.

        Args:
            x: (B, input_dim) individual tokens.

        Returns:
            (reconstruction, features) both (B, D) tensors.
        """
        features = self.activation(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE reconstruction loss + L1 sparsity penalty.

        Args:
            x: (B, input_dim) individual tokens.

        Returns:
            (loss_tensor, {"mse": ..., "l1": ..., "loss": ...})
        """
        reconstruction, features = self.forward(x)
        mse = F.mse_loss(reconstruction, x)
        l1 = features.abs().mean()
        loss = mse + self.config.l1_penalty * l1
        return loss, {"mse": mse.item(), "l1": l1.item(), "loss": loss.item()}

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample OOD score from a sequence of tokens.

        Computes per-token MSE then averages across the sequence.

        Args:
            x: (B, n_tokens, token_dim) sequence of token activations.

        Returns:
            (B,) tensor of mean reconstruction errors.
        """
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        recon, _ = self.forward(flat)
        per_token_mse = ((recon - flat) ** 2).mean(dim=-1)
        return per_token_mse.reshape(b, t).mean(dim=-1)

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(path / "model.safetensors"))
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "SparseAutoencoder":
        path = Path(path)
        with open(path / "config.json") as f:
            config = SAEConfig(**json.load(f))
        model = cls(config)
        state_dict = load_file(str(path / "model.safetensors"))
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_hub(cls, repo_id: str) -> "SparseAutoencoder":
        config_path = hf_hub_download(repo_id, "config.json", repo_type="model")
        hf_hub_download(repo_id, "model.safetensors", repo_type="model")
        local_dir = Path(config_path).parent
        with open(local_dir / "config.json") as f:
            config = SAEConfig(**json.load(f))
        model = cls(config)
        state_dict = load_file(str(local_dir / "model.safetensors"))
        model.load_state_dict(state_dict)
        return model

    def push_to_hub(self, repo_id: str) -> None:
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save_pretrained(tmp)
            api.upload_folder(
                folder_path=tmp,
                repo_id=repo_id,
                repo_type="model",
            )
