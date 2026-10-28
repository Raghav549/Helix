"""Byte-level BPE tokenizer, implemented from scratch.

Design notes
------------
* Vocabulary begins with the 256 byte values plus reserved special tokens, so
  every byte sequence is representable and decode is always lossless at the
  byte level (UTF-8 decode uses errors="replace").
* Merges are learned by frequency over the training corpus, exactly as in the
  original BPE / GPT-2 family. No external tokenizer library is required.
* Special tokens are handled literally during training and encoding so they
  never get split by merges.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


def _bytes_of(token_id: int, vocab: Dict[int, bytes]) -> bytes:
    return vocab[token_id]


class ByteLevelBPETokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[int, bytes]] = None,
        merges: Optional[List[Tuple[int, int, int]]] = None,
        special_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.special_set = set(self.special_tokens)
        # vocab: id -> bytes
        self.vocab: Dict[int, bytes] = dict(vocab or {})
        # merges: list of (id1, id2, new_id) in learning order
        self.merges: List[Tuple[int, int, int]] = list(merges or [])
        self.pair_to_new_id: Dict[Tuple[int, int], int] = {}
        self.rank: Dict[Tuple[int, int], int] = {}
        for i, (id1, id2, new_id) in enumerate(self.merges):
            self.pair_to_new_id[(id1, id2)] = new_id
            self.rank[(id1, id2)] = i

        self._next_id = 0
        if self.vocab:
            self._next_id = max(self.vocab) + 1
        else:
            self._init_default_vocab()

    def _init_default_vocab(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._next_id = 256
        for tok in self.special_tokens:
            self.vocab[self._next_id] = tok.encode("utf-8")
            self._next_id += 1

    # ---------------------------------------------------------------- training
    @staticmethod
    def _get_stats(ids: List[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out: List[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, texts: Sequence[str], vocab_size: int) -> None:
        """Learn BPE merges from raw text until `vocab_size` tokens exist."""
        if vocab_size <= len(self.vocab):
            # Initial vocabulary (256 bytes + special tokens) already covers the
            # requested size; nothing to merge.
            return

        # Represent every document as a list of byte token ids. Special tokens
        # are prepended as single tokens so they are never merged/split.
        special_ids = {
            tok: self._id_for_bytes(tok.encode("utf-8")) for tok in self.special_tokens
        }
        sequences: List[List[int]] = []
        for text in texts:
            ids: List[int] = []
            # Naive but correct: treat the whole document as one stream and mark
            # document boundaries with end-of-text.
            ids.extend(text.encode("utf-8"))
            ids.append(special_ids["<|endoftext|>"])
            sequences.append([b for b in ids])

        target = vocab_size
        while len(self.vocab) < target:
            stats: Counter = Counter()
            for seq in sequences:
                stats.update(self._get_stats(seq))
            if not stats:
                break
            # Most frequent pair wins; ties broken by lower id1 then id2 for determinism.
            pair, _ = max(stats.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            id1, id2 = pair
            new_id = self._next_id
            self._next_id += 1
            self.vocab[new_id] = _bytes_of(id1, self.vocab) + _bytes_of(id2, self.vocab)
            self.merges.append((id1, id2, new_id))
            self.pair_to_new_id[pair] = new_id
            self.rank[pair] = len(self.merges) - 1
            sequences = [self._merge(seq, pair, new_id) for seq in sequences]

    def _id_for_bytes(self, b: bytes) -> int:
        for tid, tb in self.vocab.items():
            if tb == b:
                return tid
        # Should not happen for byte values / special tokens after _init_default_vocab.
        raise KeyError(f"bytes {b!r} not in vocabulary")

    # ---------------------------------------------------------------- encode/decode
    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids: List[int] = list(text.encode("utf-8"))
        # Iteratively merge the lowest-ranked adjacent pair until none remain.
        while len(ids) >= 2:
            best_rank = None
            best_i = -1
            best_pair: Optional[Tuple[int, int]] = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                r = self.rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
                    best_pair = pair
            if best_i < 0 or best_pair is None:
                break
            new_id = self.pair_to_new_id[best_pair]
            ids = ids[:best_i] + [new_id] + ids[best_i + 2 :]
        if add_eos:
            ids.append(self._id_for_bytes("<|endoftext|>".encode("utf-8")))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        parts = [self.vocab.get(int(i), b"") for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- save/load
    def save(self, vocab_file: str, merges_file: str) -> None:
        os.makedirs(os.path.dirname(vocab_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(merges_file) or ".", exist_ok=True)
        vocab_serial = {str(tid): tb.hex() for tid, tb in self.vocab.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "special_tokens": self.special_tokens,
                    "vocab": vocab_serial,
                },
                f,
            )
        merges_serial = [[i1, i2, nid] for (i1, i2, nid) in self.merges]
        with open(merges_file, "w", encoding="utf-8") as f:
            json.dump(merges_serial, f)

    @classmethod
    def load(cls, vocab_file: str, merges_file: str) -> "ByteLevelBPETokenizer":
        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {int(tid): bytes.fromhex(h) for tid, h in data["vocab"].items()}
        special = data.get("special_tokens", SPECIAL_TOKENS)
        merges: List[Tuple[int, int, int]] = []
        with open(merges_file, "r", encoding="utf-8") as f:
            for i1, i2, nid in json.load(f):
                merges.append((int(i1), int(i2), int(nid)))
        return cls(vocab=vocab, merges=merges, special_tokens=special)

    def __len__(self) -> int:
        return len(self.vocab)
