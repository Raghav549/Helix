"""Key-value cache for incremental (autoregressive) generation.

Stores per-layer, already RoPE-rotated K and V so generation can attend to
previously computed keys/values without recomputing the whole prompt each step.

This is a CPU-first optimization *hypothesis to benchmark* — it is not assumed
to be faster. KV caching trades memory (the cache) for compute (no prompt
recompute). Benchmark it against full recompute before adopting.
"""

from __future__ import annotations

import torch


class KVCache:
    """Layer-wise KV cache with pre-allocated buffers.

    One cache per transformer block. Each holds rotated K and V of shape
    (batch, n_kv_head, max_seq, head_dim) plus a sequence-length cursor.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        max_seq: int,
        n_kv_head: int,
        head_dim: int,
        device,
        dtype=torch.float32,
    ) -> None:
        self.n_layers = n_layers
        self.max_seq = int(max_seq)
        self.caches = []
        for _ in range(n_layers):
            self.caches.append(
                {
                    "k": torch.zeros(
                        batch_size, n_kv_head, max_seq, head_dim,
                        device=device, dtype=dtype,
                    ),
                    "v": torch.zeros(
                        batch_size, n_kv_head, max_seq, head_dim,
                        device=device, dtype=dtype,
                    ),
                    "seq": 0,
                }
            )

    def seq_len(self, layer_idx: int) -> int:
        return int(self.caches[layer_idx]["seq"])

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append rotated k/v of shape (B, n_kv_head, T, head_dim)."""
        c = self.caches[layer_idx]
        s = c["seq"]
        t = k.size(2)
        if s + t > self.max_seq:
            raise RuntimeError(
                f"KV cache overflow: seq {s} + {t} > max {self.max_seq}"
            )
        c["k"][:, :, s : s + t, :] = k
        c["v"][:, :, s : s + t, :] = v
        c["seq"] = s + t

    def get(self, layer_idx: int):
        """Return (k, v) up to current seq: (B, n_kv_head, S, head_dim)."""
        c = self.caches[layer_idx]
        s = c["seq"]
        return c["k"][:, :, :s, :], c["v"][:, :, :s, :]

    def reset(self) -> None:
        for c in self.caches:
            c["seq"] = 0
