"""Deterministic document-level split, train-only BPE and memory-mapped batches."""
from __future__ import annotations
import hashlib
import json
import os
from typing import Tuple
import numpy as np
import torch
from helix.tokenizer import ByteLevelBPETokenizer


def build_from_corpus(corpus_file: str, vocab_file: str, merges_file: str,
                      vocab_size: int, train_frac: float, train_out: str,
                      val_out: str) -> Tuple[ByteLevelBPETokenizer, np.ndarray, np.ndarray]:
    if not 0 < train_frac < 1:
        raise ValueError('train_frac must be between 0 and 1')
    with open(corpus_file, 'rb') as f:
        raw = f.read()
    text = raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    docs = [d.strip() for d in text.split('\n\n') if d.strip()]
    if not docs:
        raise ValueError('corpus is empty')
    # Hold out documents before training the tokenizer. For one document, split
    # characters (rather than train BPE on the validation portion).
    if len(docs) == 1:
        cut = max(1, min(len(text) - 1, int(len(text) * train_frac)))
        if len(text) < 2:
            raise ValueError('corpus needs at least two characters')
        train_docs, val_docs = [text[:cut]], [text[cut:]]
    else:
        ntrain = max(1, min(len(docs) - 1, int(len(docs) * train_frac)))
        train_docs, val_docs = docs[:ntrain], docs[ntrain:]
    tok = ByteLevelBPETokenizer()
    tok.train(train_docs, vocab_size)
    tok.save(vocab_file, merges_file)
    def encode_docs(items):
        return np.array([token for doc in items for token in tok.encode(doc, add_eos=True)], dtype=np.int64)
    train_ids, val_ids = encode_docs(train_docs), encode_docs(val_docs)
    for path, arr in ((train_out, train_ids), (val_out, val_ids)):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            np.save(f, arr)
    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                digest.update(chunk)
        return digest.hexdigest()
    metadata = {
        'corpus_sha256': hashlib.sha256(raw).hexdigest(),
        'train_documents': len(train_docs), 'val_documents': len(val_docs),
        'train_tokens': int(len(train_ids)), 'val_tokens': int(len(val_ids)),
        'tokenizer_trained_on': 'train only', 'vocab_size': len(tok),
        'vocab_target': vocab_size, 'train_fraction': train_frac,
        'file_sha256': {name: file_hash(path)
                        for name, path in (('train', train_out), ('val', val_out),
                                           ('vocab', vocab_file), ('merges', merges_file))},
    }
    with open(train_out + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    return tok, train_ids, val_ids


def load_ids(path: str) -> np.ndarray:
    return np.load(path, mmap_mode='r', allow_pickle=False)


def get_batch(split: str, train_ids: np.ndarray, val_ids: np.ndarray,
              batch_size: int, block_size: int, device: str, generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    data = train_ids if split == 'train' else val_ids
    if len(data) == 0 or batch_size < 1 or block_size < 1:
        raise ValueError('dataset, batch_size and block_size must be nonempty')
    if len(data) <= block_size:
        data = np.tile(data, block_size // len(data) + 2)[:block_size + 1]
    max_start = len(data) - block_size
    ix = torch.randint(max_start, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(np.array(data[i:i + block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.array(data[i + 1:i + 1 + block_size], dtype=np.int64)) for i in ix])
    if str(device).startswith('cuda'):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
