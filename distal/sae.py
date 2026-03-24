"""Sparse Autoencoder for OOD detection on VLM prefix token activations."""

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
        """Encode and decode input activations.

        Args:
            x: (B, n_tokens, token_dim) or (B, input_dim).

        Returns:
            (reconstruction, features) where reconstruction matches input shape.
        """
        input_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)

        features = self.activation(self.encoder(x))
        reconstruction = self.decoder(features)

        if len(input_shape) == 3:
            reconstruction = reconstruction.reshape(input_shape)

        return reconstruction, features

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE reconstruction loss + L1 sparsity penalty.

        Returns:
            (loss_tensor, {"mse": ..., "l1": ..., "loss": ...})
        """
        reconstruction, features = self.forward(x)
        if x.ndim == 3:
            x_flat = x.reshape(x.shape[0], -1)
        else:
            x_flat = x
        recon_flat = reconstruction.reshape(x_flat.shape)

        mse = F.mse_loss(recon_flat, x_flat)
        l1 = features.abs().mean()
        loss = mse + self.config.l1_penalty * l1
        return loss, {"mse": mse.item(), "l1": l1.item(), "loss": loss.item()}

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error (OOD score).

        Args:
            x: (B, n_tokens, token_dim) or (B, input_dim).

        Returns:
            (B,) tensor of per-sample MSE values.
        """
        reconstruction, _ = self.forward(x)
        if x.ndim == 3:
            x_flat = x.reshape(x.shape[0], -1)
        else:
            x_flat = x
        recon_flat = reconstruction.reshape(x_flat.shape)
        return ((recon_flat - x_flat) ** 2).mean(dim=-1)

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
