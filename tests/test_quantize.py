"""Tests for int8 quantization: roundtrip error and forward fidelity."""

import torch

from helix.config import ModelConfig
from helix.model import HelixTransformer
from helix.quantize import (
    quantize_tensor_int8, dequantize, quantize_model, QuantizedLinear,
)


def test_quantize_dequantize_roundtrip_small_error():
    w = torch.randn(64, 128) * 0.5
    w_int8, scale = quantize_tensor_int8(w)
    w_hat = dequantize(w_int8, scale)
    rel_err = (w_hat - w).abs().max().item() / max(w.abs().max().item(), 1e-6)
    assert rel_err < 0.05, f"relative error too high: {rel_err}"


def test_quantized_linear_shapes():
    lin = torch.nn.Linear(16, 32)
    q = QuantizedLinear(lin)
    x = torch.randn(2, 5, 16)
    out = q(x)
    assert out.shape == (2, 5, 32)
    assert q.w_int8.dtype == torch.int8


def test_quantize_model_replaces_linear_layers():
    model = HelixTransformer(ModelConfig(
        vocab_size=200, block_size=32, n_layer=2, n_head=2, n_embd=32,
        n_kv_head=1, dropout=0.0,
    ))
    q = quantize_model(model)
    # No nn.Linear should remain (embedding is not Linear; lm_head is tied/None).
    linears = [m for m in q.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 0, f"{len(linears)} Linear layers remain"
    quantized = [m for m in q.modules() if isinstance(m, QuantizedLinear)]
    assert len(quantized) >= 4, "expected several QuantizedLinear layers"


def test_quantized_model_forward_close_to_original():
    torch.manual_seed(1)
    model = HelixTransformer(ModelConfig(
        vocab_size=200, block_size=32, n_layer=1, n_head=2, n_embd=32,
        n_kv_head=1, dropout=0.0,
    )).eval()
    q = quantize_model(model).eval()
    idx = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        logits_ref, _ = model(idx)
        logits_q, _ = q(idx)
    # Outputs should be close (int8 quantization error is small).
    max_diff = (logits_q - logits_ref).abs().max().item()
    assert max_diff < 0.5, f"quantized forward diverges too much: {max_diff}"
