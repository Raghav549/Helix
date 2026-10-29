"""
HELIX core model: a decoder-only transformer with long-context extensions.

Backward-compatible with the original nanoGPT-adapted baseline:
  * decoder-only causal language model
  * Rotary Positional Embeddings (RoPE)
  * RMSNorm (pre-normalization)
  * SwiGLU feed-forward network
  * Grouped-Query Attention (GQA)
  * Weight tying between input embedding and output projection
  * block_type is a comma-separated list cycled across layers via helix.arch_blocks

New long-context / peak-intelligence capabilities (all optional, defaults keep
the original behavior unchanged):
  * NTK-aware RoPE scaling (rope_scaling='ntk' / 'linear') to extend context
    beyond the training block size without retraining position embeddings.
  * Sliding-window attention (window_size) for O(T*w) long-context inference.
  * max_position_embeddings to size the RoPE table independently of block_size.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from helix.config import ModelConfig
from helix.kvcache import KVCache
from helix.arch_blocks import build_block


# ---------------------------------------------------------------------------
# RoPE with NTK / linear scaling
# ---------------------------------------------------------------------------

def _ntk_scaling_factor(head_dim: int, rope_scaling: str, ntk_alpha: float,
                        block_size: int, max_position_embeddings: int) -> float:
    """NTK-aware scaling factor for extending context.

    Following NTK-by-parts: scale rope_base by (new_ctx / old_ctx)^(n/(n-2))
    where n = head_dim. Only applied when rope_scaling == 'ntk' and ntk_alpha > 1.
    """
    if rope_scaling != "ntk" or ntk_alpha <= 1.0 or head_dim <= 2:
        return 1.0
    old_ctx = max(block_size, 1)
    new_ctx = max(max_position_embeddings, old_ctx + 1)
    ratio = new_ctx / old_ctx
    return float(ratio ** (head_dim / (head_dim - 2.0)) * ntk_alpha)


class RotaryEmbedding(nn.Module):
    """RoPE tables with optional NTK/linear scaling for long-context extension."""

    def __init__(self, head_dim: int, base: float = 10000.0,
                 max_position_embeddings: int = 128,
                 rope_scaling: str = "none", ntk_alpha: float = 1.0,
                 block_size: int = 128) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.ntk_alpha = ntk_alpha
        self.block_size = block_size
        inv_freq = self._build_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        base = float(self.base)
        if self.rope_scaling == "linear":
            ratio = self.max_position_embeddings / max(self.block_size, 1)
            base = base * ratio
        elif self.rope_scaling == "ntk":
            base = base * _ntk_scaling_factor(
                self.head_dim, self.rope_scaling, self.ntk_alpha,
                self.block_size, self.max_position_embeddings)
        return 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))

    def forward(self, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin of shape (1, 1, T, head_dim) for positions [0, T)."""
        t = torch.arange(T, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to the last dimension of x.

    x: (..., T, head_dim); cos/sin: (1, 1, T, head_dim)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


# ---------------------------------------------------------------------------
# Primitives reused by helix.arch_blocks / helix.image_understanding / tests
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE and grouped-query K/V heads.

    Supports an optional sliding-window mask (window_size) so each query only
    attends to the most recent `window_size` keys — O(T*w) instead of O(T^2).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.window_size = config.window_size
        self.block_size = config.block_size

        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # full causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )
        # sliding-window mask buffer
        if self.window_size is not None and self.window_size < self.block_size:
            w = self.window_size
            idx = torch.arange(self.block_size)
            sw = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= w
            sw = sw & torch.tril(torch.ones(self.block_size, self.block_size, dtype=torch.bool))
            self.register_buffer(
                "window_mask",
                sw.view(1, 1, self.block_size, self.block_size),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(2).expand(B, self.n_kv_head, rep, T, self.head_dim).reshape(
                B, self.n_head, T, self.head_dim
            )
            v = v.unsqueeze(2).expand(B, self.n_kv_head, rep, T, self.head_dim).reshape(
                B, self.n_head, T, self.head_dim
            )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if self.window_size is not None and hasattr(self, "window_mask"):
            mask = self.window_mask[:, :, :T, :T]
        else:
            mask = self.causal_mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def forward_incremental(
        self,
        x: torch.Tensor,
        cache: KVCache,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Process new tokens x using the KV cache for previous positions.

        x: (B, T_new, C) new tokens at positions [start_pos, start_pos+T_new).
        cache: per-layer KV cache holding rotated K/V for positions [0, start_pos).
        cos/sin: RoPE for the *new* positions only, shape (1, 1, T_new, head_dim).
        Returns y of shape (B, T_new, C) and updates the cache in place.
        """
        B, T_new, C = x.size()

        q = self.c_q(x).view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T_new, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T_new, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Store rotated K/V in the cache.
        cache.update(layer_idx, k, v)
        k_full, v_full = cache.get(layer_idx)  # (B, nkv, S, hs)
        S = k_full.size(2)

        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k_full = k_full.unsqueeze(2).expand(
                B, self.n_kv_head, rep, S, self.head_dim
            ).reshape(B, self.n_head, S, self.head_dim)
            v_full = v_full.unsqueeze(2).expand(
                B, self.n_kv_head, rep, S, self.head_dim
            ).reshape(B, self.n_head, S, self.head_dim)

        att = (q @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Causal mask: query at position start_pos+i may attend to keys 0..start_pos+i.
        q_pos = torch.arange(start_pos, start_pos + T_new, device=x.device).view(T_new, 1)
        k_pos = torch.arange(0, S, device=x.device).view(1, S)
        mask = (k_pos <= q_pos).view(1, 1, T_new, S)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v_full
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Pre-normalization transformer block: attention + SwiGLU FFN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd, config.eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd, config.eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_incremental(
        self,
        x: torch.Tensor,
        cache: KVCache,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        x = x + self.attn.forward_incremental(
            self.norm1(x), cache, start_pos, cos, sin, layer_idx
        )
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class HelixTransformer(nn.Module):
    """Decoder-only language model trained from random initialization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.n_embd // config.n_head
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        block_types = [t.strip() for t in config.block_type.split(",")]
        self.blocks = nn.ModuleList(
            [build_block(config, block_types[i % len(block_types)]) for i in range(config.n_layer)]
        )
        self.norm_f = RMSNorm(config.n_embd, config.eps)
        if config.weight_tying:
            self.lm_head: Optional[nn.Linear] = None
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

        # RoPE sized to max_position_embeddings (>= block_size) for long-context.
        self.rope = RotaryEmbedding(
            self.head_dim,
            base=config.rope_base,
            max_position_embeddings=int(config.max_position_embeddings),
            rope_scaling=config.rope_scaling,
            ntk_alpha=config.ntk_alpha,
            block_size=config.block_size,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"HELIXTransformer: {n_params/1e6:.2f}M params, block_size={config.block_size}, "
              f"layers={config.n_layer}, heads={config.n_head}, embd={config.n_embd}, "
              f"rope_scaling={config.rope_scaling}, window={config.window_size}")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rope(T, device)

    def _rope_positions(self, start: int, end: int, device: torch.device
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE cos/sin for absolute positions [start, end)."""
        cos, sin = self.rope(end, device)
        return cos[:, :, start:end, :], sin[:, :, start:end, :]

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = self.config.block_size
            if targets is not None:
                targets = targets[:, -self.config.block_size:]

        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope(T, device)

        if self.config.grad_checkpoint and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, cos, sin)

        x = self.norm_f(x)

        if self.config.weight_tying:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)  # type: ignore[union-attr]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def incremental_forward(
        self,
        idx: torch.Tensor,
        cache: KVCache,
        start_pos: int,
    ) -> Tuple[torch.Tensor, KVCache]:
        """Incremental forward: process tokens at [start_pos, start_pos+T)."""
        device = idx.device
        B, T = idx.size()
        if T < 1 or start_pos < 0 or start_pos + T > self.config.block_size:
            raise ValueError("incremental context exceeds the model window")
        if any(cache.seq_len(i) != start_pos for i in range(self.config.n_layer)):
            raise ValueError("start_pos must equal each layer's cache length")
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_positions(start_pos, start_pos + T, device)

        for i, block in enumerate(self.blocks):
            if hasattr(block, "forward_incremental"):
                x = block.forward_incremental(x, cache, start_pos, cos, sin, i)
            else:
                # arch_blocks LinearAttnBlock / RecurrentBlock: no KV cache.
                x = block(x, cos, sin)

        x = self.norm_f(x)
        if self.config.weight_tying:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return logits, cache

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation (full-context recompute each step, or KV cached)."""
        self.eval()
        B, T0 = idx.shape
        cache = None
        if use_cache:
            cache = KVCache(
                n_layers=self.config.n_layer, batch_size=B, max_seq=self.config.block_size,
                n_kv_head=self.config.n_kv_head, head_dim=self.head_dim,
                device=idx.device, dtype=torch.float32,
            )

        for step in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            if use_cache and cache is not None and cache.seq_len(0) > 0:
                # feed only the newest token
                step_input = idx_cond[:, -1:]
                start_pos = cache.seq_len(0) - 1
                logits, cache = self.incremental_forward(step_input, cache, start_pos)
                step_logits = logits[:, -1, :]
            else:
                logits, _ = self(idx_cond)
                step_logits = logits[:, -1, :]
                if use_cache and cache is not None:
                    # prime the cache with the full context
                    self.incremental_forward(idx_cond, cache, 0)

            step_logits = step_logits / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                top_k = min(top_k, step_logits.size(-1))
                v, _ = torch.topk(step_logits, top_k)
                step_logits = torch.where(step_logits < v[:, [-1]], float("-inf"), step_logits)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                step_logits = step_logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(step_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
        return idx

    def configure_optimizers(self, train_cfg) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay, "weight_decay": train_cfg.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
        )
        return optimizer

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Alias for external callers that previously imported HELIXModel from here.
HELIXModel = HelixTransformer


def build_model(config: ModelConfig) -> HelixTransformer:
    return HelixTransformer(config)


def load_checkpoint(path: str, config: ModelConfig, device: str = "cpu") -> HelixTransformer:
    model = HelixTransformer(config).to(device)
    state = torch.load(path, map_location=device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    return model
