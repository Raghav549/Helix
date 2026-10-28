"""Data pipeline: tokenize a corpus, split train/val, and serve batches.

The baseline uses a small seed corpus shipped in data/seed_corpus.txt. Later
stages will plug in the extensible corpus-discovery/ingestion system described
in docs/DATA_PIPELINE.md, with provenance tracking per document.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch

from helix.tokenizer import ByteLevelBPETokenizer


def build_from_corpus(
    corpus_file: str,
    vocab_file: str,
    merges_file: str,
    vocab_size: int,
    train_frac: float,
    train_out: str,
    val_out: str,
) -> Tuple[ByteLevelBPETokenizer, np.ndarray, np.ndarray]:
    """Train the BPE tokenizer on the corpus and write train/val token arrays."""
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Treat non-empty paragraph blocks as separate documents so the tokenizer
    # sees document boundaries and end-of-text markers.
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]

    tok = ByteLevelBPETokenizer()
    tok.train(docs, vocab_size)
    tok.save(vocab_file, merges_file)

    ids = tok.encode(text, add_eos=True)
    arr = np.array(ids, dtype=np.int64)
    n = len(arr)
    n_train = max(1, int(n * train_frac))
    train_ids = arr[:n_train]
    val_ids = arr[n_train:] if n_train < n else arr[-1:]

    os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_out) or ".", exist_ok=True)
    np.save(train_out, train_ids)
    np.save(val_out, val_ids)
    return tok, train_ids, val_ids


def load_ids(path: str) -> np.ndarray:
    return np.load(path)


def get_batch(
    split: str,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_ids if split == "train" else val_ids
    # Robust to tiny validation splits: use an effective block no larger than
    # the data allows; tile cyclically if the data is shorter than one block.
    if len(data) <= block_size:
        reps = max(1, block_size // max(len(data), 1) + 2)
        data = np.tile(data, reps)[: block_size + 1]
    effective_block = min(block_size, len(data) - 1)
    max_start = len(data) - effective_block
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + effective_block].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + effective_block].astype(np.int64)) for i in ix]
    )
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y
