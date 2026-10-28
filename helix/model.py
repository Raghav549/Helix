"""HELIX core model: an independently implemented decoder-only transformer.

Architecture choices for the CPU-first baseline (all validated against the
nanoGPT reference implementation and adapted here):
  * decoder-only causal language model
  * Rotary Positional Embeddings (RoPE)
  * RMSNorm (pre-normalization)
  * SwiGLU feed-forward network
  * Grouped-Query Attention (GQA): many query heads share fewer K/V heads
  * Weight tying between the input embedding and the output projection
  * Optional activation checkpointing for training

No pretrained weights are used; everything is trained from random initialization.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from helix.kvcache import KVCache

from helix.config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to the last dimension of x.

    x: (..., T, head_dim); cos/sin: (1, 1, T, head_dim)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(T, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE and grouped-query K/V heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T, hs)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, nkv, T, hs)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand shared K/V heads to match query heads (grouped-query attention).
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.unsqueeze(2).expand(B, self.n_kv_head, rep, T, self.head_dim).reshape(
                B, self.n_head, T, self.head_dim
            )
            v = v.unsqueeze(2).expand(B, self.n_kv_head, rep, T, self.head_dim).reshape(
                B, self.n_head, T, self.head_dim
            )

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, hs)
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

        q = self.c_q(x).view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T_new, hs)
        k = self.c_k(x).view(B, T_new, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, nkv, T_new, hs)
        v = self.c_v(x).view(B, T_new, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Rotate the new queries/keys by their absolute positions. Cached K/V were
        # already rotated when they were inserted, so no re-rotation needed there.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Store rotated K/V in the cache.
        cache.update(layer_idx, k, v)
        k_full, v_full = cache.get(layer_idx)  # (B, nkv, S, hs)
        S = k_full.size(2)

        # Expand shared K/V heads to match query heads (grouped-query attention).
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

        y = att @ v_full  # (B, nh, T_new, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


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


class HelixTransformer(nn.Module):
    """Decoder-only language model trained from random initialization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.n_embd // config.n_head
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd, config.eps)
        if config.weight_tying:
            self.lm_head: Optional[nn.Linear] = None
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        rope = RotaryEmbedding(self.head_dim, self.config.rope_base)
        rope.inv_freq = rope.inv_freq.to(device)
        return rope(T, device)

    def _rope_positions(
        self, start: int, end: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE cos/sin for absolute positions [start, end)."""
        rope = RotaryEmbedding(self.head_dim, self.config.rope_base)
        rope.inv_freq = rope.inv_freq.to(device)
        return rope(end - start, device)

    def incremental_forward(
        self,
        idx: torch.Tensor,
        cache: KVCache,
        start_pos: int,
    ) -> Tuple[torch.Tensor, KVCache]:
        """Incremental forward: process tokens at [start_pos, start_pos+T).

        Uses the KV cache for all earlier positions. Returns logits for the
        input tokens and the updated cache.
        """
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = self.config.block_size
            start_pos = max(0, start_pos - (idx.size(1) - T))

        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_positions(start_pos, start_pos + T, device)

        if self.config.grad_checkpoint and self.training:
            # Not used during generation (eval mode), but kept for completeness.
            for i, block in enumerate(self.blocks):
                x = block.forward_incremental(x, cache, start_pos, cos, sin, i)
        else:
            for i, block in enumerate(self.blocks):
                x = block.forward_incremental(x, cache, start_pos, cos, sin, i)

        x = self.norm_f(x)
        if self.config.weight_tying:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)  # type: ignore[union-attr]
        return logits, cache

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size :]
            T = self.config.block_size
            if targets is not None:
                targets = targets[:, -self.config.block_size :]

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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation (full-context recompute each step)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)
                logits = torch.where(logits < v[:, [-1]], float("-inf"), logits)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
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
