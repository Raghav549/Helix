"""Configuration dataclasses for HELIX.

All values are plain Python so they serialize to JSON. A starter CPU preset is
kept in configs/helix-tiny.json. Nothing here assumes an accelerator exists.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from helix.storage import REPOSITORY_ROOT, resolve_data_root
from typing import Optional


@dataclass
class ModelConfig:
    """Decoder-only transformer architecture."""

    vocab_size: int = 50304          # BPE vocab target (incl. special tokens)
    block_size: int = 256            # max training sequence length (tokens)
    n_layer: int = 4                 # transformer blocks
    n_head: int = 8                  # query heads
    n_embd: int = 256                # embedding / model dimension
    n_kv_head: int = 2               # key/value heads (grouped-query attention)
    dropout: float = 0.1
    bias: bool = False               # linear layer biases
    eps: float = 1e-5                # RMSNorm epsilon
    rope_base: float = 10000.0       # RoPE base frequency
    weight_tying: bool = True        # tie input embedding and output projection
    grad_checkpoint: bool = False    # activation checkpointing (train only)
    block_type: str = "transformer"  # transformer | linear_attn | recurrent |
                                     # "transformer,linear_attn" for hybrid (cycled)

    # Long-context extensions (all optional; defaults preserve original behavior)
    window_size: Optional[int] = None          # sliding-window attention (None = full)
    rope_scaling: str = "none"                 # 'none' | 'linear' | 'ntk'
    ntk_alpha: float = 1.0                     # NTK scaling factor
    max_position_embeddings: Optional[int] = None  # RoPE table length (default block_size)

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head,
               self.n_embd, self.n_kv_head) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        allowed = {"transformer", "linear_attn", "recurrent"}
        types = {t.strip() for t in self.block_type.split(",")}
        if not types or not types <= allowed:
            raise ValueError(f"block_type must use {allowed}, got {self.block_type!r}")
        if self.rope_scaling not in ("none", "linear", "ntk"):
            raise ValueError(f"rope_scaling must be 'none'|'linear'|'ntk', got {self.rope_scaling!r}")
        if self.window_size is not None and self.window_size < 1:
            raise ValueError("window_size must be a positive int or None")
        if self.ntk_alpha < 1.0:
            raise ValueError("ntk_alpha must be >= 1.0")
        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.block_size
        if self.max_position_embeddings < self.block_size:
            raise ValueError("max_position_embeddings must be >= block_size")


@dataclass
class TrainConfig:
    """Training hyperparameters and runtime behavior."""

    batch_size: int = 16
    block_size: int = 256
    max_iters: int = 500
    eval_interval: int = 100
    eval_iters: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    lr_decay_iters: int = 500
    min_lr: float = 3e-5
    warmup_iters: int = 100
    seed: int = 1337
    device: str = "cpu"
    dtype: str = "float32"           # float32 | float16 | bfloat16 (CPU-supported)
    log_interval: int = 25
    save_interval: int = 250
    out_dir: str = "out"
    resume_from: Optional[str] = None


@dataclass
class DataConfig:
    """Data pipeline locations."""

    train_file: str = "data/train.npy"
    val_file: str = "data/val.npy"
    vocab_file: str = "data/vocab.json"
    merges_file: str = "data/merges.json"
    corpus_file: str = "data/seed_corpus.txt"
    train_frac: float = 0.9


@dataclass
class HelixConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> HelixConfig:
    """Load configuration and project-relative external storage paths.

    The checked-in JSON path stays in Git; heavy checkpoints, tokenized
    corpora and experiment outputs intentionally resolve under
    ``HELIX_DATA_ROOT``. An explicit absolute path is never moved.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = HelixConfig()
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "train" in raw:
        cfg.train = TrainConfig(**raw["train"])
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])

    def storage_to_external(value: str) -> str:
        candidate = Path(os.path.expanduser(value))
        if candidate.is_absolute():
            return candidate.as_posix()
        external = resolve_data_root() / candidate
        if external.exists():
            return str(external)
        # Small authored corpora remain checked in; generated corpora and all
        # other artifacts live outside the repository. An existing repository
        # path is therefore selected only for corpus text, never for data.
        local = REPOSITORY_ROOT / candidate
        if candidate.name == 'seed_corpus.txt' and local.is_file():
            return str(local)
        return str(external)

    cfg.data.train_file = storage_to_external(cfg.data.train_file)
    cfg.data.val_file = storage_to_external(cfg.data.val_file)
    cfg.data.vocab_file = storage_to_external(cfg.data.vocab_file)
    cfg.data.merges_file = storage_to_external(cfg.data.merges_file)
    cfg.data.corpus_file = storage_to_external(cfg.data.corpus_file)
    cfg.train.out_dir = storage_to_external(cfg.train.out_dir)
    if cfg.train.resume_from:
        cfg.train.resume_from = storage_to_external(cfg.train.resume_from)
    return cfg


def save_config(cfg: HelixConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": asdict(cfg.model),
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
