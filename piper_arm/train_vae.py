"""Train an action-chunk VAE conditioned on frozen SmolVLA VLM features.

The VAE encodes and reconstructs action chunks (sequences of future actions),
conditioned on frozen SmolVLA prefix features (image + language + state) via
cross-attention in both the encoder and decoder.

Architecture:
    VLM context:
        (image, language, state) → SmolVLA embed_prefix → VLM forward pass
        → (B, seq_len, text_hidden_dim) → linear projection → (B, seq_len, hidden_dim)

    Encoder:
        Action chunk (B, chunk_size, action_dim) → project to hidden_dim
        → prepend CLS token → positional embeddings
        → self-attention → cross-attention (to VLM context)
        → CLS output → mu_head, logvar_head → z

    Decoder:
        z → project to hidden_dim → prepend to learnable queries
        → positional embeddings → self-attention → cross-attention (to VLM context)
        → discard latent token → action_head → (B, chunk_size, action_dim)

    All attention blocks use RMSNorm and SiLU-gated MLPs.

Usage:
    python -m piper_arm.train_vae --repo-id reece-omahoney/libero --epochs 50
"""

import argparse
import os
from datetime import datetime

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────────────
# Transformer blocks with cross-attention
# ──────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SelfAttentionBlock(nn.Module):
    """RMSNorm → MHA (self) → residual → RMSNorm → SiLU-gated MLP → residual."""

    def __init__(self, hidden_dim: int, n_heads: int, intermediate_size: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.mlp_norm(x)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        h = self.down_proj(h)
        return x + h


class CrossAttentionBlock(nn.Module):
    """RMSNorm → MHA (cross) → residual → RMSNorm → SiLU-gated MLP → residual."""

    def __init__(self, hidden_dim: int, n_heads: int, intermediate_size: int):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.ctx_norm = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        c = self.ctx_norm(context)
        h, _ = self.attn(h, c, c)
        x = x + h
        h = self.mlp_norm(x)
        h = F.silu(self.gate_proj(h)) * self.up_proj(h)
        h = self.down_proj(h)
        return x + h


class EncoderDecoderStack(nn.Module):
    """1 self-attention layer + 1 cross-attention layer with positional embeddings."""

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int = 512,
        n_heads: int = 8,
        intermediate_size: int = 1376,
    ):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.self_attn = SelfAttentionBlock(hidden_dim, n_heads, intermediate_size)
        self.cross_attn = CrossAttentionBlock(hidden_dim, n_heads, intermediate_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_emb
        x = self.self_attn(x)
        x = self.cross_attn(x, context)
        return x


# ──────────────────────────────────────────────────────────────────────
# Action Chunk VAE (LightningModule)
# ──────────────────────────────────────────────────────────────────────


class ActionChunkVAE(L.LightningModule):
    def __init__(
        self,
        policy: SmolVLAPolicy,
        action_dim: int,
        chunk_size: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        n_heads: int = 8,
        intermediate_size: int = 1376,
        kl_weight: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["policy"])
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.kl_weight = kl_weight
        self.preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=policy.config.pretrained_path,
        )

        # Frozen SmolVLA policy (used only for feature extraction)
        self.policy = policy
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False

        # Project text-model features to hidden_dim for cross-attention
        self.vlm_proj = nn.Linear(
            policy.model.vlm_with_expert.config.text_config.hidden_size, hidden_dim
        )

        # ── Encoder ──
        self.enc_proj = nn.Linear(action_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.encoder = EncoderDecoderStack(
            seq_len=chunk_size + 1,  # +1 for CLS
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder ──
        self.dec_latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.dec_queries = nn.Parameter(torch.randn(1, chunk_size, hidden_dim) * 0.02)
        self.decoder = EncoderDecoderStack(
            seq_len=chunk_size + 1,  # +1 for latent token
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def encode(
        self, actions: torch.Tensor, vlm_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """VAE encoder: action chunk + VLM context → mu, logvar."""
        x = self.enc_proj(actions)  # (B, chunk_size, hidden_dim)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, chunk_size+1, hidden_dim)
        x = self.encoder(x, vlm_context)
        cls_out = x[:, 0]  # (B, hidden_dim)
        mu = self.mu_head(cls_out)
        logvar = self.logvar_head(cls_out)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, vlm_context: torch.Tensor) -> torch.Tensor:
        """VAE decoder: z + VLM context → reconstructed action chunk."""
        B = z.size(0)
        latent_token = self.dec_latent_proj(z).unsqueeze(1)  # (B, 1, hidden_dim)
        queries = self.dec_queries.expand(B, -1, -1)  # (B, chunk_size, hidden_dim)
        x = torch.cat([latent_token, queries], dim=1)  # (B, chunk_size+1, hidden_dim)
        x = self.decoder(x, vlm_context)
        x = x[:, 1:]  # discard latent token, keep action tokens
        return self.action_head(x)  # (B, chunk_size, action_dim)

    def forward(self, batch: dict):
        # preprocess
        batch = self.preprocessor(batch)
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        actions = self.policy.prepare_action(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # prefix embedding
        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # vlm forward pass
        (prefix_out, _), _ = self.policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )

        # vae forward pass
        vlm_context = self.vlm_proj(prefix_out.to(self.vlm_proj.weight.dtype))
        mu, logvar = self.encode(actions, vlm_context)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, vlm_context)

        return recon, actions, mu, logvar

    def training_step(self, batch, _):
        recon, actions, mu, logvar = self(batch)
        recon_loss = F.mse_loss(recon, actions)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_weight * kl_loss

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
            lr=1e-4,
            weight_decay=1e-5,
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
        self.policy.eval()
        return self


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train action-chunk VAE with VLM cross-attention"
    )
    parser.add_argument("--repo-id", type=str, default="reece-omahoney/libero")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="reece-omahoney/smolvla-libero-256",
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    # ── Data ──
    print(f"Loading dataset from repo: {args.repo_id}")
    # Use delta_timestamps to load action chunks of chunk_size future steps
    delta_timestamps = {
        "action": [i / 10.0 for i in range(args.chunk_size)],
    }
    dataset = LeRobotDataset(repo_id=args.repo_id, delta_timestamps=delta_timestamps)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    action_dim = dataset.meta.features["action"]["shape"][0]

    # ── Load frozen SmolVLA policy ──
    print(f"Loading SmolVLA: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.eval()

    # ── Model ──
    model = ActionChunkVAE(
        policy=policy,
        action_dim=action_dim,
        chunk_size=args.chunk_size,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,
    )

    # ── Trainer ──
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    output_dir = f"outputs/action_vae/{timestamp}"
    logger = WandbLogger(project="action-chunk-vae", save_dir=output_dir)
    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        every_n_epochs=10,
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
