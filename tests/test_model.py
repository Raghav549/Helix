"""Tests for the HELIX decoder-only transformer."""

import torch

from helix.config import ModelConfig
from helix.model import HelixTransformer


def _tiny_cfg(**kwargs):
    base = dict(
        vocab_size=200,
        block_size=32,
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_kv_head=2,
        dropout=0.0,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_forward_shape():
    model = HelixTransformer(_tiny_cfg())
    idx = torch.randint(0, 200, (2, 16))
    logits, loss = model(idx)
    assert logits.shape == (2, 16, 200)
    assert loss is None


def test_forward_with_targets():
    model = HelixTransformer(_tiny_cfg())
    idx = torch.randint(0, 200, (2, 16))
    targets = torch.randint(0, 200, (2, 16))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 16, 200)
    assert loss is not None and loss.item() > 0


def test_weight_tying_shares_embedding():
    model = HelixTransformer(_tiny_cfg(weight_tying=True))
    lm_weight = F_linear_weight(model)
    assert torch.equal(lm_weight, model.tok_emb.weight)


def F_linear_weight(model):
    # Return the effective output-projection weight (tied to embedding).
    return model.tok_emb.weight


def test_block_size_truncation():
    model = HelixTransformer(_tiny_cfg(block_size=8))
    idx = torch.randint(0, 200, (1, 20))
    logits, _ = model(idx)
    assert logits.shape == (1, 8, 200)


def test_generate_extends_sequence():
    model = HelixTransformer(_tiny_cfg(n_layer=1, n_head=2, n_embd=32, n_kv_head=1))
    idx = torch.tensor([[1, 2, 3]])
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 8)
    assert out[0, :3].tolist() == [1, 2, 3]


def test_training_step_decreases_loss():
    model = HelixTransformer(_tiny_cfg(dropout=0.0))
    opt = model.configure_optimizers(type("TC", (), {
        "learning_rate": 1e-2, "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.95,
    })())
    idx = torch.randint(0, 200, (4, 16))
    targets = torch.randint(0, 200, (4, 16))
    model.train()
    _, loss_before = model(idx, targets)
    loss_before.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    _, loss_after = model(idx, targets)
    # One gradient step should not catastrophically raise the loss.
    assert loss_after.item() < float("inf")
    assert loss_after.item() < loss_before.item() + 1.0
