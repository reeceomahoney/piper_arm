"""Train an image VAE using SmolVLM's frozen SigLIP vision encoder.

The VAE uses the SigLIP vision encoder from a pretrained SmolVLM model to encode
images into patch tokens. Small LLaMA-style transformers serve as the VAE encoder
and decoder, with a CNN pixel decoder reconstructing raw pixels at 256x256.

Architecture:
    Image (3,256,256) → normalize [-1,1]
    → Frozen SigLIP → (B, 256, vision_dim)  [256 patches, 16×16 grid]
    → VAE Encoder (6-layer transformer) → mu, logvar → z
    → VAE Decoder (6-layer transformer) → (B, 256, hidden_dim)
    → Reshape to (B, C, 16, 16)
    → CNN Pixel Decoder (4× upsample) → (B, 3, 256, 256)

Usage:
    python train_vae.py --repo-id reece-omahoney/libero --epochs 50 --wandb
"""

import argparse
import math
import os

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText

# ──────────────────────────────────────────────────────────────────────
# LLaMA-style transformer blocks
# ──────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class TransformerBlock(nn.Module):
    """
    LLaMA-style block: RMSNorm → MHA → residual → RMSNorm → SiLU-gated MLP → residual.
    """

    def __init__(self, hidden_dim: int, n_heads: int, intermediate_size: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # SiLU-gated MLP with pre-norm
        h = self.mlp_norm(x)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        h = self.down_proj(h)
        return x + h


class TransformerStack(nn.Module):
    """Stack of LLaMA-style transformer blocks with learned positional embeddings."""

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int = 512,
        n_heads: int = 8,
        intermediate_size: int = 1376,
        n_layers: int = 6,
    ):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, n_heads, intermediate_size)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_emb
        for layer in self.layers:
            x = layer(x)
        return x


# ──────────────────────────────────────────────────────────────────────
# Image VAE (LightningModule)
# ──────────────────────────────────────────────────────────────────────


class ImageVAE(L.LightningModule):
    def __init__(
        self,
        vision_model: nn.Module,
        vision_dim: int,
        camera_key: str = "observation.images.image",
        latent_dim: int = 256,
        hidden_dim: int = 512,
        n_heads: int = 8,
        intermediate_size: int = 1376,
        n_layers: int = 6,
        n_tokens: int = 256,
        kl_weight: float = 1e-4,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vision_model"])
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_tokens = n_tokens

        # Frozen pretrained vision encoder
        self.vision_model = vision_model
        self.vision_model.eval()
        for p in self.vision_model.parameters():
            p.requires_grad = False

        # ── Encoder ──
        self.enc_proj = nn.Linear(vision_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = TransformerStack(
            seq_len=n_tokens + 1,  # +1 for CLS
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
            n_layers=n_layers,
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder ──
        self.dec_latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.dec_queries = nn.Parameter(torch.randn(1, n_tokens, hidden_dim) * 0.02)
        self.decoder = TransformerStack(
            seq_len=n_tokens + 1,  # +1 for latent token
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
            n_layers=n_layers,
        )

        # ── CNN Pixel Decoder ──
        # Reshape decoder output: (B, n_tokens, hidden_dim) → (B, hidden_dim, 16, 16)
        # Then 4× ConvTranspose2d (2× upsample each): 16→32→64→128→256
        cnn_channels = [hidden_dim, 256, 128, 64, 3]
        cnn_layers = []
        for i in range(len(cnn_channels) - 1):
            cnn_layers.append(
                nn.ConvTranspose2d(
                    cnn_channels[i],
                    cnn_channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if i < len(cnn_channels) - 2:
                cnn_layers.append(nn.SiLU())
        cnn_layers.append(nn.Sigmoid())
        self.pixel_decoder = nn.Sequential(*cnn_layers)

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run frozen vision encoder. Input: (B, 3, 256, 256) normalized to [-1,1]."""
        return self.vision_model(
            pixel_values=pixel_values.to(
                dtype=self.vision_model.embeddings.patch_embedding.weight.dtype
            ),
        ).last_hidden_state.float()

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """VAE encoder: patch tokens → mu, logvar."""
        x = self.enc_proj(tokens)  # (B, 64, hidden_dim)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, 65, hidden_dim)
        x = self.encoder(x)
        cls_out = x[:, 0]  # (B, hidden_dim)
        mu = self.mu_head(cls_out)
        logvar = self.logvar_head(cls_out)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """VAE decoder: z → reconstructed image (B, 3, 256, 256) in [0, 1]."""
        B = z.size(0)
        latent_token = self.dec_latent_proj(z).unsqueeze(1)  # (B, 1, hidden_dim)
        queries = self.dec_queries.expand(B, -1, -1)  # (B, n_tokens, hidden_dim)
        x = torch.cat([latent_token, queries], dim=1)  # (B, n_tokens+1, hidden_dim)
        x = self.decoder(x)
        x = x[:, 1:]  # discard latent token, keep spatial tokens
        grid_size = int(math.sqrt(self.n_tokens))
        x = x.permute(0, 2, 1).view(B, self.hidden_dim, grid_size, grid_size)
        return self.pixel_decoder(x)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encode_image(pixel_values)
        mu, logvar = self.encode(tokens)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def training_step(self, batch, batch_idx):
        images = batch[self.hparams.camera_key]
        targets = images
        pixel_values = (images - 0.5) / 0.5

        recon, mu, logvar = self(pixel_values)
        recon_loss = F.mse_loss(recon, targets)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.hparams.kl_weight * kl_loss

        self.log_dict(
            {
                "train/loss": loss,
                "train/recon_loss": recon_loss,
                "train/kl_loss": kl_loss,
            },
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def train(self, mode=True):
        super().train(mode)
        self.vision_model.eval()
        return self


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train image VAE with SmolVLM vision encoder"
    )
    parser.add_argument("--repo-id", type=str, default="reece-omahoney/libero")
    parser.add_argument("--camera-key", type=str, default="observation.images.image")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    # ── Load frozen vision encoder ──
    VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    print(f"Loading VLM: {VLM_MODEL}")
    vlm = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=torch.bfloat16)
    vision_model = vlm.model.vision_model
    vision_dim = vlm.config.vision_config.hidden_size
    del vlm
    print(f"Vision encoder dim: {vision_dim}")

    # ── Model ──
    model = ImageVAE(
        vision_model=vision_model,
        vision_dim=vision_dim,
        camera_key=args.camera_key,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Data ──
    dataset = LeRobotDataset(repo_id=args.repo_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Trainer ──
    output_dir = "outputs/vae"
    logger = WandbLogger(project="image-vae", save_dir=output_dir)
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        every_n_epochs=1,
        save_top_k=-1,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        default_root_dir=output_dir,
        logger=logger,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
