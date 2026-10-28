"""Configuration dataclasses for HELIX.

All values are plain Python so they serialize to JSON. A starter CPU preset is
kept in configs/helix-tiny.json. Nothing here assumes an accelerator exists.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
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

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        allowed = {"transformer", "linear_attn", "recurrent"}
        types = {t.strip() for t in self.block_type.split(",")}
        if not types or not types <= allowed:
            raise ValueError(f"block_type must use {allowed}, got {self.block_type!r}")


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
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = HelixConfig()
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "train" in raw:
        cfg.train = TrainConfig(**raw["train"])
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
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
