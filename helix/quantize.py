"""Dynamic int8 quantization for CPU inference weights.

Hypothesis: quantifying Linear weights to int8 reduces weight memory and may
speed up matmul on CPUs with int8 BLAS/oneDNN. Not assumed faster — benchmark
to verify, and prefer the unquantized model if there is no measured benefit.

Implementation: per-tensor symmetric int8 quantization. Forward dequantizes to
float32 for the matmul (correct on every CPU; real int8 speed requires int8
BLAS/oneDNN support). Documented honestly as a memory-reduction technique with
unproven speedup on this hardware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_tensor_int8(w: torch.Tensor):
    """Per-tensor symmetric int8 quantization.

    Returns (w_int8, scale) where w ≈ w_int8 * scale.
    """
    w = w.detach().to(torch.float32)
    scale = w.abs().max() / 127.0
    if scale == 0 or not torch.isfinite(scale):
        scale = torch.tensor(1.0)
    w_int8 = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return w_int8, scale.float()


def dequantize(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w_int8.to(torch.float32) * scale


class QuantizedLinear(nn.Module):
    """A Linear layer whose weight is stored as int8 + per-tensor scale."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w_int8, scale = quantize_tensor_int8(linear.weight)
        self.register_buffer("w_int8", w_int8)
        self.register_buffer("scale", scale)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(torch.float32))
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize(self.w_int8, self.scale)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, quantized=int8"


def quantize_linear_weight(linear: nn.Linear) -> QuantizedLinear:
    return QuantizedLinear(linear)


def quantize_model(model: nn.Module) -> nn.Module:
    """Return a copy of `model` with Linear layers replaced by QuantizedLinear.

    Embedding, RMSNorm, and buffers are left in float32. For weight-tied
    models there is no separate lm_head, so nothing to quantize there.
    """
    import copy

    q = copy.deepcopy(model)
    for name, module in list(q.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, child = name.rpartition(".")
            parent = q.get_submodule(parent_name) if parent_name else q
            setattr(parent, child, QuantizedLinear(module))
    return q
