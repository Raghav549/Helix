"""Alternative HELIX architecture blocks for ablation studies.

Independently implemented. All blocks share the forward signature
(x, cos, sin) -> x so the model can be built from any mixture:

  * "transformer"  — RoPE + causal softmax (GQA) attention + SwiGLU  (baseline,
                     reuses helix.model.Block so the baseline is identical)
  * "linear_attn"  — causal linear attention (O(T) memory/compute) + SwiGLU
  * "recurrent"    — GRU-like recurrent state (O(1) per-step state) + SwiGLU

These are hypotheses to benchmark: linear attention and recurrent state are
CPU-friendly (no T^2 attention matrix) but may trade quality per token against
the quadratic transformer. The ablation runner (helix/ablate.py) measures
which wins on validation loss per training token and per second.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy imports of helix.model primitives break the model <-> arch_blocks import
# cycle. RMSNorm / SwiGLU / apply_rope are defined in helix.model and re-used
# so the linear/recurrent blocks share identical primitives with the baseline.


def _rmsnorm(config, eps):
    from helix.model import RMSNorm
    return RMSNorm(config.n_embd, eps)


def _swiglu(config):
    from helix.model import SwiGLU
    return SwiGLU(config)


def _apply_rope(q, cos, sin):
    from helix.model import apply_rope
    return apply_rope(q, cos, sin)


class LinearAttention(nn.Module):
    """Causal linear attention (Katharopoulos et al., 2020).

    softmax(q k^T) v is approximated by a cumulative-sum formulation that is
    O(T) in both memory and time instead of O(T^2). CPU-friendly for long
    contexts; quality per token is the hypothesis under test.

    Forward: out_i = (q_i @ sum_{j<=i} k_j v_j^T) / (q_i @ sum_{j<=i} k_j)
    """

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.eps = 1e-6

        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _expand_kv(self, k, v, B, T):
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(2).expand(B, self.n_kv_head, rep, T, self.head_dim).reshape(
                B, self.n_head, T, self.head_dim
            )
            v = v.unsqueeze(2).expand(B, self.n_kv_head, rep, T, self.head_dim).reshape(
                B, self.n_head, T, self.head_dim
            )
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        k, v = self._expand_kv(k, v, B, T)  # (B,h,T,d)

        # Causal linear attention via cumulative sums: O(T) time/memory.
        kv = torch.einsum("bhtd,bhte->bhtde", k, v)        # (B,h,T,d,d)
        kv_cum = torch.cumsum(kv, dim=2)                    # sum over j<=i
        k_cum = torch.cumsum(k, dim=2)                      # (B,h,T,d)

        num = torch.einsum("bhtd,bhtde->bhte", q, kv_cum)   # (B,h,T,e)
        den = torch.einsum("bhtd,bhtd->bht", q, k_cum).unsqueeze(-1) + self.eps

        y = num / den                                       # (B,h,T,e)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class LinearAttnBlock(nn.Module):
    """Linear attention + SwiGLU, pre-norm, same signature as TransformerBlock."""

    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = _rmsnorm(config, config.eps)
        self.attn = LinearAttention(config)
        self.norm2 = _rmsnorm(config, config.eps)
        self.ffn = _swiglu(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class RecurrentBlock(nn.Module):
    """GRU-like recurrent block.

    Maintains a hidden state h updated token-by-token:
        z = sigmoid(W_z [x, h]); r = sigmoid(W_r [x, h])
        h_tilde = tanh(W_h [x, r*h]); h = (1-z)*h + z*h_tilde
    O(1) per-step state, O(T) total — the CPU-friendly recurrent alternative.
    No attention matrix at all. Quality per token vs transformer is the
    hypothesis under test.

    For batch/sequence parallelism during training we unroll over T with a
    loop (correct but not the fastest); inference can step one token at a time.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.eps = config.eps

        combined = 2 * config.n_embd  # [x, h]
        self.w_z = nn.Linear(combined, config.n_embd, bias=config.bias)
        self.w_r = nn.Linear(combined, config.n_embd, bias=config.bias)
        self.w_h = nn.Linear(combined, config.n_embd, bias=config.bias)
        self.norm = _rmsnorm(config, config.eps)
        self.ffn = _swiglu(config)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        h = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            inp = torch.cat([xt, h], dim=-1)
            z = torch.sigmoid(self.w_z(inp))
            r = torch.sigmoid(self.w_r(inp))
            h_tilde = torch.tanh(self.w_h(torch.cat([xt, r * h], dim=-1)))
            h = (1.0 - z) * h + z * h_tilde
            outs.append(h)
        y = torch.stack(outs, dim=1)  # (B, T, C)
        y = self.resid_dropout(y)
        y = self.norm(y + x)  # residual + norm
        y = y + self.ffn(self.norm(y))
        return y


def build_block(config, block_type: str) -> nn.Module:
    if block_type == "transformer":
        from helix.model import Block
        return Block(config)
    if block_type == "linear_attn":
        return LinearAttnBlock(config)
    if block_type == "recurrent":
        return RecurrentBlock(config)
    raise ValueError(f"unknown block_type: {block_type}")
