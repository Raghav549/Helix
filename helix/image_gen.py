"""HELIX image generation pipeline — from scratch, trainable from random init.

A small latent diffusion model (LDM) on CPU:
  * VAE (encoder/decoder) compresses images to a latent space (reconstruction + KL)
  * Small UNet in latent space predicts noise given latent + timestep
  * DDPM forward/noise schedule + reverse (denoising) sampler
  * Optional text conditioning via a small, independently trained byte/GRU encoder

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

    # Text conditioning (optional byte/GRU encoder or caller-provided embeddings)
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
    vae_pretrain_iters: int = 0

    def __post_init__(self):
        if min(self.image_size, self.image_channels, self.latent_size, self.latent_channels,
               self.vae_channels, self.unet_channels, self.unet_heads, self.time_embed_dim,
               self.cond_token_dim, self.batch_size, self.max_iters) < 1:
            raise ValueError("image dimensions and training counts must be positive")
        if self.image_size != self.latent_size * 4 or self.latent_size % 4:
            raise ValueError("image_size must be 4 * latent_size, divisible by 16")
        if self.unet_channels % 8 or self.unet_channels % self.unet_heads:
            raise ValueError("UNet channels must be divisible by 8 and by heads")
        if self.time_embed_dim % 2 or self.num_timesteps < 2 or self.vae_pretrain_iters < 0:
            raise ValueError("invalid diffusion dimensions or steps")
        if self.schedule not in ("cosine", "linear"):
            raise ValueError("schedule must be cosine or linear")
        if not 0 < self.beta_start < self.beta_end < 1 or self.learning_rate <= 0 or self.kl_weight < 0:
            raise ValueError("invalid diffusion schedule or training hyperparameters")
        if (self.downsample_factor != 4 or self.vae_blocks != 2 or self.unet_blocks != 2 or
                self.cond_dim != 64 or self.dtype != "float32"):
            raise ValueError("this image architecture fixes downsample_factor=4, vae_blocks=2, unet_blocks=2, cond_dim=64 and dtype=float32")


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
        q, k, v = [qkv[:, i].transpose(-2, -1) for i in range(3)]
        # Attention is over spatial positions, not channel dimensions.
        out = F.scaled_dot_product_attention(q, k, v)
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

class PromptEncoder(nn.Module):
    """Trainable UTF-8 byte encoder; no external embeddings or weights."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(257, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, prompts: List[str], device) -> torch.Tensor:
        if not prompts or any(not isinstance(p, str) or len(p.encode("utf-8")) > 1024 for p in prompts):
            raise ValueError("prompts must be nonempty UTF-8 strings under 1024 bytes")
        encoded = [[b + 1 for b in p.encode("utf-8")[:128]] or [1] for p in prompts]
        ids = torch.zeros(len(encoded), max(map(len, encoded)), dtype=torch.long, device=device)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row, device=device)
        output, _ = self.gru(self.embedding(ids))
        ends = torch.tensor([len(row)-1 for row in encoded], device=device)
        return output[torch.arange(len(encoded), device=device), ends].unsqueeze(1)


class LatentDiffusion(nn.Module):
    """VAE + diffusion in latent space."""

    def __init__(self, config: ImageGenConfig) -> None:
        super().__init__()
        self.config = config
        self.vae = VAE(config)
        self.unet = ConditionedUNet(config)
        self.text_encoder = PromptEncoder(config.cond_token_dim) if config.use_text_conditioning else None
        self.schedule = DiffusionSchedule(config)
        self.register_buffer("latent_mean", torch.zeros(1, config.latent_channels, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, config.latent_channels, 1, 1))

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return (self.vae.encode(images) - self.latent_mean) / self.latent_std

    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z * self.latent_std + self.latent_mean)

    def forward(
        self,
        images: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
    ) -> dict:
        """Training forward: encode -> add noise -> predict noise."""
        if prompts is not None:
            if self.text_encoder is None:
                raise ValueError("model was not configured for text conditioning")
            if len(prompts) != images.size(0):
                raise ValueError("one prompt per image required")
            text_emb = self.text_encoder(prompts, images.device)
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
        kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        vae_loss = recon_loss + self.config.kl_weight * kl_loss

        # Diffusion loss
        diff_loss = F.mse_loss(noise_pred, noise)

        return {
            "loss": diff_loss + vae_loss * 0.1,
            "diff_loss": diff_loss,
            "vae_loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def sample(self, num_images: int = 1, text_emb: Optional[torch.Tensor] = None,
               timesteps: Optional[int] = None, device: str = "cpu",
               prompts: Optional[List[str]] = None, seed: Optional[int] = None,
               init_images: Optional[torch.Tensor] = None, strength: float = 0.6) -> torch.Tensor:
        """DDPM sampling, optionally initialized from a noised input image.

        `timesteps` uses the first T schedule steps (not an accelerated full
        schedule). For normal generation use the full configured schedule.
        """
        T = timesteps if timesteps is not None else self.config.num_timesteps
        if not 1 <= T <= self.config.num_timesteps or not 1 <= num_images <= 16:
            raise ValueError("invalid timesteps or image count")
        if prompts is not None:
            if self.text_encoder is None:
                raise ValueError("checkpoint is unconditional and cannot use prompts")
            if len(prompts) != num_images:
                raise ValueError("one prompt per image required")
            text_emb = self.text_encoder(prompts, device)
        gen = torch.Generator(device=device)
        if seed is None: gen.seed()
        else: gen.manual_seed(seed)
        L = self.config.latent_size
        schedule = self.schedule.to(device)
        if init_images is None:
            z = torch.randn(num_images, self.config.latent_channels, L, L, device=device, generator=gen)
            start = T - 1
        else:
            if not 0.0 < strength <= 1.0 or init_images.size(0) != num_images:
                raise ValueError("strength must be in (0,1] and image count must match")
            z0 = self.encode_images(init_images.to(device))
            start = max(0, min(T - 1, int(strength * (T - 1))))
            noise = torch.randn(z0.shape, device=device, dtype=z0.dtype, generator=gen)
            z = schedule.q_sample(z0, torch.full((num_images,), start, device=device, dtype=torch.long), noise)
        betas, alphas, ac = (schedule.register[k] for k in ("betas", "alphas", "alphas_cumprod"))
        for i in range(start, -1, -1):
            t = torch.full((num_images,), i, device=device, dtype=torch.long)
            noise_pred = self.unet(z, t, text_emb)
            beta, alpha, alpha_cum = betas[i], alphas[i], ac[i]
            noise = torch.randn(z.shape, device=device, dtype=z.dtype, generator=gen) if i else torch.zeros_like(z)
            z = (z - (1 - alpha) / torch.sqrt(1 - alpha_cum) * noise_pred) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
        return self.decode_latents(z)


# ============================================================================
# Training
# ============================================================================

def train_image_gen(
    config: ImageGenConfig,
    images: torch.Tensor,  # (N, C, H, W) in [-1, 1]
    text_embeddings: Optional[torch.Tensor] = None,  # (N, num_tokens, cond_token_dim)
    out_dir: str = "out/image_gen",
    captions: Optional[List[str]] = None,
) -> dict:
    """Train VAE + diffusion from random init. Returns metrics."""
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device
    from helix.runtime import configure_threads
    configure_threads()

    model = LatentDiffusion(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    schedule = model.schedule.to(device)

    N = images.size(0)
    if N < 1 or images.ndim != 4 or tuple(images.shape[1:]) != (config.image_channels, config.image_size, config.image_size):
        raise ValueError("images have invalid shape")
    if config.use_text_conditioning and captions is None and text_embeddings is None:
        raise ValueError("conditioned training requires captions or explicit text embeddings")
    if captions is not None and (not config.use_text_conditioning or len(captions) != N):
        raise ValueError("conditioned training needs one caption per image")
    log_path = os.path.join(out_dir, "log.jsonl")
    start = time.perf_counter()
    best = float("inf")

    model.train()
    # Optional VAE warmup freezes the latent representation during diffusion.
    for it in range(config.vae_pretrain_iters):
        idx = torch.randint(0, N, (config.batch_size,))
        batch = images[idx].to(device)
        recon, mu, logvar = model.vae(batch)
        vae_loss = F.mse_loss(recon, batch) + config.kl_weight * (-.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
        opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        opt.step()
    if config.vae_pretrain_iters:
        # Match the latent prior's location and scale to unit Gaussian noise.
        # Estimate only on training images; store the statistics in checkpoints.
        with torch.no_grad():
            latent_batches = [model.vae.encode(images[i:i+config.batch_size].to(device))
                              for i in range(0, N, config.batch_size)]
            latents = torch.cat(latent_batches, dim=0)
            model.latent_mean.copy_(latents.mean(dim=(0, 2, 3), keepdim=True))
            model.latent_std.copy_(latents.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4))
        for parameter in model.vae.parameters():
            parameter.requires_grad_(False)
    for it in range(config.max_iters):
        idx = torch.randint(0, N, (config.batch_size,))
        batch = images[idx].to(device)
        tb = None
        if text_embeddings is not None:
            tb = text_embeddings[idx].to(device)

        opt.zero_grad(set_to_none=True)
        batch_captions = [captions[int(i)] for i in idx] if captions is not None else None
        out = model(batch, tb, prompts=batch_captions)
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite image loss")
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
        "initialization": "random",
        "text_conditioned": bool(captions is not None or text_embeddings is not None),
        "vae_pretrain_iters": config.vae_pretrain_iters,
        "parameters": sum(p.numel() for p in model.parameters()),
        "latent_mean": model.latent_mean.flatten().detach().cpu().tolist(),
        "latent_std": model.latent_std.flatten().detach().cpu().tolist(),
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
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def generate_images(
    model: LatentDiffusion,
    num_images: int = 1,
    text_emb: Optional[torch.Tensor] = None,
    device: str = "cpu",
    prompts: Optional[List[str]] = None,
    seed: Optional[int] = None,
    init_images: Optional[torch.Tensor] = None,
    strength: float = 0.6,
) -> List[Image.Image]:
    model.eval()
    out = model.sample(num_images=num_images, text_emb=text_emb, device=device,
                       prompts=prompts, seed=seed, init_images=init_images, strength=strength)
    # out: (B, 3, 32, 32) in [-1, 1] -> PIL images
    images = []
    for i in range(out.size(0)):
        img = (out[i] * 0.5 + 0.5).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).round().astype(np.uint8)
        images.append(Image.fromarray(img, mode="RGB"))
    return images
