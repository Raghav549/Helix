"""Tests for the HELIX image understanding system."""

import os
import tempfile

import numpy as np
import torch

from helix.config import ModelConfig
from helix.image_understanding import (
    ImageUnderstandingConfig,
    PatchEmbedding,
    VisionTransformer,
    ImageProjection,
    MultimodalHelix,
    train_image_understanding,
    load_multimodal_model,
)


def _fake_images(n: int = 4, size: int = 32) -> torch.Tensor:
    return torch.rand(n, 3, size, size) * 2 - 1


def test_patch_embedding():
    pe = PatchEmbedding(32, 4, 3, 64)
    x = _fake_images(2, 32)
    out = pe(x)
    assert out.shape == (2, 64, 64)


def test_vision_transformer():
    cfg = ImageUnderstandingConfig()
    vit = VisionTransformer(cfg)
    x = _fake_images(2, 32)
    out = vit(x)
    assert out.shape == (2, 65, 64)  # cls + 64 patches
    assert torch.isfinite(out).all()


def test_image_projection():
    proj = ImageProjection(64, 128)
    x = torch.randn(2, 65, 64)
    out = proj(x)
    assert out.shape == (2, 65, 128)


def test_multimodal_forward():
    llm_cfg = ModelConfig(vocab_size=200, block_size=32, n_layer=1, n_head=2,
                          n_embd=64, n_kv_head=1, dropout=0.0)
    vis_cfg = ImageUnderstandingConfig(image_size=32, patch_size=4, vit_dim=64,
                                       vit_layers=1, vit_heads=2)
    model = MultimodalHelix(llm_cfg, vis_cfg).eval()
    images = _fake_images(2, 32)
    ids = torch.randint(0, 200, (2, 8))
    targets = torch.randint(0, 200, (2, 8))
    logits, loss = model(images, ids, targets)
    assert logits.shape == (2, 8, 200)
    assert loss is not None and loss.item() > 0
    # without image
    logits2, loss2 = model(None, ids, targets)
    assert logits2.shape == (2, 8, 200)


def test_multimodal_generate():
    llm_cfg = ModelConfig(vocab_size=200, block_size=32, n_layer=1, n_head=2,
                          n_embd=32, n_kv_head=1, dropout=0.0)
    vis_cfg = ImageUnderstandingConfig(image_size=16, patch_size=4, vit_dim=32,
                                       vit_layers=1, vit_heads=1, max_image_tokens=16)
    model = MultimodalHelix(llm_cfg, vis_cfg).eval()
    images = _fake_images(1, 16)
    ids = torch.tensor([[1, 2, 3]])
    out = model.generate(images, ids, max_new_tokens=4, temperature=1.0, top_k=2, device="cpu")
    assert out.shape == (1, 3 + 4)


def test_train_smoke(tmp_path):
    llm_cfg = ModelConfig(vocab_size=200, block_size=16, n_layer=1, n_head=2,
                          n_embd=32, n_kv_head=1, dropout=0.0)
    vis_cfg = ImageUnderstandingConfig(image_size=16, patch_size=4, vit_dim=32,
                                       vit_layers=1, vit_heads=1)
    images = _fake_images(8, 16)
    text_ids = torch.randint(0, 200, (8, 10))
    targets = torch.randint(0, 200, (8, 10))
    metrics = train_image_understanding(
        vis_cfg, llm_cfg, images, text_ids, targets,
        out_dir=str(tmp_path / "iu"), max_iters=20, batch_size=4, lr=1e-3,
    )
    assert metrics["stage"] == "image_understanding"
    assert os.path.exists(os.path.join(tmp_path / "iu", "ckpt-final.pt"))


def test_load_multimodal(tmp_path):
    llm_cfg = ModelConfig(vocab_size=200, block_size=16, n_layer=1, n_head=2,
                          n_embd=32, n_kv_head=1, dropout=0.0)
    vis_cfg = ImageUnderstandingConfig(image_size=16, patch_size=4, vit_dim=32,
                                       vit_layers=1, vit_heads=1)
    images = _fake_images(4, 16)
    text_ids = torch.randint(0, 200, (4, 8))
    targets = torch.randint(0, 200, (4, 8))
    train_image_understanding(vis_cfg, llm_cfg, images, text_ids, targets,
                              out_dir=str(tmp_path / "iu2"), max_iters=10,
                              batch_size=4, lr=1e-3)
    model = load_multimodal_model(llm_cfg, vis_cfg,
                                  os.path.join(tmp_path / "iu2", "ckpt-final.pt"))
    assert model.training is False
    ids = torch.tensor([[1, 2, 3]])
    out = model.generate(None, ids, max_new_tokens=3, device="cpu")
    assert out.shape == (1, 6)
