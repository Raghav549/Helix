"""HELIX image generation pipeline — from scratch, trainable from random init.

A small latent diffusion model (LDM) on CPU:
  * VAE (encoder/decoder) compresses images to a latent space (reconstruction + KL)
  * Small UNet in latent space predicts noise given latent + timestep
  * DDPM forward/noise schedule + reverse (denoising) sampler
  * Optional text conditioning via projected LLM token embeddings

Everything is trained from random initialization; no pretrained image weights are
used. Kept intentionally small (32x32 images, tiny latent) so a full training
run completes on an ordinary CPU in minutes. A hypothesis to benchmark — image
quality at this scale is prototype-level, documented honestly.

Design decisions (each a hypothesis, measured not assumed):
  * Latent diffusion: train VAE + diffusion separately (stable, CPU-feasible)
  * UNet with residual blocks + time + (optional) text conditioning
  * Cosine noise schedule (improved DDPM)
  * Pixel-space MSE reconstruction for VAE, KL regularization
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ============================================================================
# Config
# ============================================================================

@dataclass
class ImageGenConfig:
    image_size: int = 32            # 32x32 RGB (CPU-feasible)
    image_channels: int = 3
    latent_size: int = 8            # image_size / 4
    latent_channels: int = 4
    downsample_factor: int = 4

    # VAE
    vae_channels: int = 64
    vae_blocks: int = 2
    kl_weight: float = 1e-4

    # UNet (diffusion in latent space)
    unet_channels: int = 64
    unet_blocks: int = 2
    time_embed_dim: int = 128
    unet_heads: int = 4

    # Diffusion
    num_timesteps: int = 300
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"        # cosine | linear

    # Text conditioning (optional, projected LLM embeddings)
    cond_dim: int = 64              # condition embedding dim
    use_text_conditioning: bool = False
    cond_token_dim: int = 64        # dim of incoming text token embedding

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-3
    max_iters: int = 2000
    seed: int = 1337
    device: str = "cpu"
    dtype: str = "float32"


# ============================================================================
# Noise schedule
# ============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Improved DDPM / Nichol & Dhariwal)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, start: float, end: float) -> torch.Tensor:
    return torch.linspace(start, end, timesteps)


class DiffusionSchedule:
    def __init__(self, config: ImageGenConfig) -> None:
        self.config = config
        T = config.num_timesteps
        if config.schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T, config.beta_start, config.beta_end)
        self.register = {}
        self.register["betas"] = betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register["alphas"] = alphas
        self.register["alphas_cumprod"] = alphas_cumprod
        self.register["sqrt_alphas_cumprod"] = torch.sqrt(alphas_cumprod)
        self.register["sqrt_one_minus_alphas_cumprod"] = torch.sqrt(
            1.0 - alphas_cumprod
        )

    def to(self, device: str) -> "DiffusionSchedule":
        for k, v in self.register.items():
            self.register[k] = v.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        sqrt_ac = self.register["sqrt_alphas_cumprod"].to(x0.device)[t, None, None, None]
        sqrt_one_minus_ac = self.register["sqrt_one_minus_alphas_cumprod"].to(
            x0.device
        )[t, None, None, None]
        return sqrt_ac * x0 + sqrt_one_minus_ac * noise


# ============================================================================
# Building blocks
# ============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding -> MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions (CPU-feasible at small resolution)."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        att = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.heads), dim=-1
        )
        out = torch.matmul(att, v)  # (B, heads, H*W, head_dim)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.proj(out)


class TimeConditionedSequential(nn.Sequential):
    """Sequential that forwards t_emb to ResidualBlocks, plain x to others."""

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for module in self:
            if isinstance(module, ResidualBlock):
                x = module(x, t_emb)
            else:
                x = module(x)
        return x


class ConditionedUNet(nn.Module):
    """Small UNet predicting noise in latent space, conditioned on time + text."""

    def __init__(self, config: ImageGenConfig) -> None:
        super().__init__()
        c = config.unet_channels
        self.time_mlp = TimeEmbedding(config.time_embed_dim)

        if config.use_text_conditioning:
            self.text_proj = nn.Linear(config.cond_token_dim, config.time_embed_dim)
        else:
            self.text_proj = None

        # Encoder
        self.init_conv = nn.Conv2d(config.latent_channels, c, 3, padding=1)
        self.down1 = TimeConditionedSequential(
            ResidualBlock(c, c, config.time_embed_dim),
            ResidualBlock(c, c, config.time_embed_dim),
        )
        self.down1_pool = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.down2 = TimeConditionedSequential(
            ResidualBlock(c * 2, c * 2, config.time_embed_dim),
            ResidualBlock(c * 2, c * 2, config.time_embed_dim),
            AttentionBlock(c * 2, config.unet_heads),
        )
        self.down2_pool = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.mid = TimeConditionedSequential(
            ResidualBlock(c * 4, c * 4, config.time_embed_dim),
            AttentionBlock(c * 4, config.unet_heads),
            ResidualBlock(c * 4, c * 4, config.time_embed_dim),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.up2_blocks = TimeConditionedSequential(
            ResidualBlock(c * 4, c * 2, config.time_embed_dim),
            ResidualBlock(c * 2, c * 2, config.time_embed_dim),
        )
        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.up1_blocks = TimeConditionedSequential(
            ResidualBlock(c * 2, c, config.time_embed_dim),
            ResidualBlock(c, c, config.time_embed_dim),
        )
        self.out_conv = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(),
                                      nn.Conv2d(c, config.latent_channels, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        if text_emb is not None and self.text_proj is not None:
            # Fuse text condition into the timestep embedding (CPU-feasible).
            t_emb = t_emb + self.text_proj(text_emb.mean(dim=1))

        h = self.init_conv(x)
        h1 = self.down1(h, t_emb)  # (B, c, L, L)
        h = self.down1_pool(h1)  # (B, 2c, L/2, L/2)
        h2 = self.down2(h, t_emb)  # (B, 2c, L/2, L/2)
        h = self.down2_pool(h2)  # (B, 4c, L/4, L/4)
        h = self.mid(h, t_emb)
        h = self.up2(h)  # (B, 2c, L/2, L/2)
        h = self.up2_blocks(torch.cat([h, h2], dim=1), t_emb)
        h = self.up1(h)  # (B, c, L, L)
        h = self.up1_blocks(torch.cat([h, h1], dim=1), t_emb)
        return self.out_conv(h)


# ============================================================================
# VAE
# ============================================================================

class VAEEncoder(nn.Module):
    def __init__(self, in_ch: int, latent_ch: int, base: int = 64) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 3, padding=1), nn.SiLU(),
        )
        self.mu = nn.Conv2d(base * 4, latent_ch, 1)
        self.logvar = nn.Conv2d(base * 4, latent_ch, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(x)
        return self.mu(h), self.logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_ch: int, out_ch: int, base: int = 64) -> None:
        super().__init__()
        self.init = nn.Conv2d(latent_ch, base * 4, 3, padding=1)
        self.blocks = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(base, out_ch, 3, padding=1),
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.init(z)
        return self.blocks(h)


class VAE(nn.Module):
    def __init__(self, config: ImageGenConfig) -> None:
        super().__init__()
        self.encoder = VAEEncoder(
            config.image_channels, config.latent_channels, config.vae_channels
        )
        self.decoder = VAEDecoder(
            config.latent_channels, config.image_channels, config.vae_channels
        )
        self.latent_channels = config.latent_channels

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ============================================================================
# Latent Diffusion Model
# ============================================================================

class LatentDiffusion(nn.Module):
    """VAE + diffusion in latent space."""

    def __init__(self, config: ImageGenConfig) -> None:
        super().__init__()
        self.config = config
        self.vae = VAE(config)
        self.unet = ConditionedUNet(config)
        self.schedule = DiffusionSchedule(config)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(images)

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    def forward(
        self,
        images: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training forward: encode -> add noise -> predict noise."""
        z0 = self.encode_images(images)  # (B, latent_ch, L, L)
        B = z0.size(0)
        device = z0.device
        t = torch.randint(0, self.config.num_timesteps, (B,), device=device)
        noise = torch.randn_like(z0)
        zt = self.schedule.q_sample(z0, t, noise)
        noise_pred = self.unet(zt, t, text_emb)

        # VAE reconstruction loss (on a separate pass)
        recon, mu, logvar = self.vae(images)
        recon_loss = F.mse_loss(recon, images)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).mean()
        vae_loss = recon_loss + self.config.kl_weight * kl_loss

        # Diffusion loss
        diff_loss = F.mse_loss(noise_pred, noise)

        return {
            "loss": diff_loss + self.config.kl_weight * 0.0 + vae_loss * 0.1,
            "diff_loss": diff_loss,
            "vae_loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        num_images: int = 1,
        text_emb: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Reverse diffusion sampling from pure noise in latent space."""
        T = timesteps or self.config.num_timesteps
        L = self.config.latent_size
        z = torch.randn(num_images, self.config.latent_channels, L, L, device=device)
        schedule = self.schedule.to(device)

        for i in reversed(range(T)):
            t = torch.full((num_images,), i, device=device, dtype=torch.long)
            noise_pred = self.unet(z, t, text_emb)
            betas = schedule.register["betas"].to(device)
            alphas = schedule.register["alphas"].to(device)
            ac = schedule.register["alphas_cumprod"].to(device)
            beta = betas[i]
            alpha = alphas[i]
            alpha_cum = ac[i]
            if i > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            # DDPM reverse step (mean-only, simplified variance)
            z = (
                1
                / torch.sqrt(alpha)
                * (z - (1 - alpha) / torch.sqrt(1 - alpha_cum) * noise_pred)
                + torch.sqrt(beta) * noise
            )
        return self.decode_latents(z)


# ============================================================================
# Training
# ============================================================================

def train_image_gen(
    config: ImageGenConfig,
    images: torch.Tensor,  # (N, C, H, W) in [-1, 1]
    text_embeddings: Optional[torch.Tensor] = None,  # (N, num_tokens, cond_token_dim)
    out_dir: str = "out/image_gen",
) -> dict:
    """Train VAE + diffusion from random init. Returns metrics."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = LatentDiffusion(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    schedule = model.schedule.to(device)

    N = images.size(0)
    log_path = os.path.join(out_dir, "log.jsonl")
    start = time.perf_counter()
    best = float("inf")

    model.train()
    for it in range(config.max_iters):
        idx = torch.randint(0, N, (config.batch_size,))
        batch = images[idx].to(device)
        tb = None
        if text_embeddings is not None:
            tb = text_embeddings[idx].to(device)

        opt.zero_grad(set_to_none=True)
        out = model(batch, tb)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 100 == 0 or it == config.max_iters - 1:
            entry = {"iter": it, "loss": round(float(loss.item()), 6),
                     "diff_loss": round(float(out["diff_loss"].item()), 6),
                     "vae_loss": round(float(out["vae_loss"].item()), 6)}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[image_gen] iter {it}: loss {loss.item():.4f} "
                  f"diff {out['diff_loss'].item():.4f} vae {out['vae_loss'].item():.4f}")
            if float(loss.item()) < best:
                best = float(loss.item())
                torch.save(model.state_dict(), os.path.join(out_dir, "ckpt-best.pt"))

    dt = time.perf_counter() - start
    torch.save(model.state_dict(), os.path.join(out_dir, "ckpt-final.pt"))
    metrics = {
        "stage": "image_gen",
        "config": asdict(config),
        "num_images": N,
        "max_iters": config.max_iters,
        "elapsed_s": round(dt, 3),
        "best_loss": round(best, 6),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[image_gen] done in {dt:.1f}s. best_loss={best:.4f}")
    return metrics


# ============================================================================
# Inference helpers
# ============================================================================

def load_image_gen_model(config: ImageGenConfig, ckpt_path: str, device: str = "cpu"):
    model = LatentDiffusion(config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def generate_images(
    model: LatentDiffusion,
    num_images: int = 1,
    text_emb: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> List[Image.Image]:
    model.eval()
    out = model.sample(num_images=num_images, text_emb=text_emb, device=device)
    # out: (B, 3, 32, 32) in [-1, 1] -> PIL images
    images = []
    for i in range(out.size(0)):
        img = (out[i] * 0.5 + 0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).round().astype(np.uint8)
        images.append(Image.fromarray(img, mode="RGB"))
    return images
