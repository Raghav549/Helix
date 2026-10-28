"""Tests for the HELIX image generation pipeline."""

import os
import tempfile

import numpy as np
import torch

from helix.image_gen import (
    ImageGenConfig,
    LatentDiffusion,
    VAE,
    ConditionedUNet,
    DiffusionSchedule,
    train_image_gen,
    generate_images,
    load_image_gen_model,
)


def _fake_images(n: int = 8, size: int = 32) -> torch.Tensor:
    return torch.rand(n, 3, size, size) * 2 - 1


def test_vae_roundtrip():
    cfg = ImageGenConfig(image_size=32, latent_size=8, latent_channels=4)
    vae = VAE(cfg)
    x = _fake_images(2, 32)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (2, 4, 8, 8)
    assert logvar.shape == (2, 4, 8, 8)
    z = vae.encode(x)
    out = vae.decode(z)
    assert out.shape == x.shape


def test_unet_forward():
    cfg = ImageGenConfig(image_size=32, latent_size=8, latent_channels=4)
    unet = ConditionedUNet(cfg)
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([10, 50])
    out = unet(x, t)
    assert out.shape == (2, 4, 8, 8)


def test_unet_with_text_conditioning():
    cfg = ImageGenConfig(
        image_size=32, latent_size=8, latent_channels=4,
        use_text_conditioning=True, cond_token_dim=64,
    )
    unet = ConditionedUNet(cfg)
    x = torch.randn(2, 4, 8, 8)
    t = torch.tensor([10, 50])
    text = torch.randn(2, 5, 64)
    out = unet(x, t, text)
    assert out.shape == (2, 4, 8, 8)


def test_diffusion_schedule_q_sample():
    cfg = ImageGenConfig(num_timesteps=50)
    sched = DiffusionSchedule(cfg)
    x0 = torch.randn(2, 4, 8, 8)
    t = torch.tensor([5, 20])
    noise = torch.randn_like(x0)
    xt = sched.q_sample(x0, t, noise)
    assert xt.shape == x0.shape
    assert torch.isfinite(xt).all()


def test_latent_diffusion_forward_shapes():
    cfg = ImageGenConfig(image_size=32, num_timesteps=50)
    model = LatentDiffusion(cfg)
    images = _fake_images(4, 32)
    out = model(images)
    assert "loss" in out and "diff_loss" in out and "vae_loss" in out
    assert out["loss"].item() > 0
    assert torch.isfinite(out["loss"]).all()


def test_latent_diffusion_sample():
    cfg = ImageGenConfig(image_size=32, num_timesteps=50)
    model = LatentDiffusion(cfg).eval()
    out = model.sample(num_images=2, device="cpu")
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_train_image_gen_smoke(tmp_path):
    cfg = ImageGenConfig(
        image_size=32, num_timesteps=50, max_iters=20, batch_size=4,
        learning_rate=1e-3,
    )
    images = _fake_images(16, 32)
    metrics = train_image_gen(cfg, images, out_dir=str(tmp_path / "image_gen"))
    assert metrics["stage"] == "image_gen"
    assert metrics["max_iters"] == 20
    assert os.path.exists(os.path.join(tmp_path / "image_gen", "ckpt-final.pt"))
    assert os.path.exists(os.path.join(tmp_path / "image_gen", "metrics.json"))


def test_load_and_generate(tmp_path):
    cfg = ImageGenConfig(image_size=32, num_timesteps=30, max_iters=10,
                         batch_size=4)
    images = _fake_images(8, 32)
    out_dir = str(tmp_path / "image_gen2")
    train_image_gen(cfg, images, out_dir=out_dir)
    model = load_image_gen_model(cfg, os.path.join(out_dir, "ckpt-final.pt"))
    imgs = generate_images(model, num_images=2, device="cpu")
    assert len(imgs) == 2
    for im in imgs:
        assert im.size == (32, 32)
        assert im.mode == "RGB"
