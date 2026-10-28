"""Exact and lexical near-duplicate detection without embedding dependencies."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from helix.quality import TOKEN_RE


@dataclass
class DedupeReport:
    accepted: int
    exact_duplicates: int
    near_duplicates: int

    def to_json(self) -> dict[str, int]:
        return {'accepted': self.accepted, 'exact_duplicates': self.exact_duplicates,
                'near_duplicates': self.near_duplicates}


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def shingle_hashes(text: str, size: int = 5, count: int = 300) -> set[str]:
    words = [word.lower() for word in TOKEN_RE.findall(text)][:count]
    if len(words) < size:
        return {hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()}
    return {hashlib.blake2b(' '.join(words[i:i+size]).encode('utf-8'),
                            digest_size=8).hexdigest() for i in range(len(words) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def deduplicate(documents: list[tuple[str, str]],
                threshold: float = 0.80) -> tuple[list[tuple[str, str]], DedupeReport]:
    accepted: list[tuple[str, str]] = []
    seen_digests: set[str] = set()
    near_duplicates = 0
    seen_shingles: list[set[str]] = []
    for doc_id, text in documents:
        digest = document_hash(text)
        if digest in seen_digests:
            continue
        shingles = shingle_hashes(text)
        if any(jaccard(shingles, previous) >= threshold for previous in seen_shingles):
            near_duplicates += 1
            continue
        seen_digests.add(digest)
        seen_shingles.append(shingles)
        accepted.append((doc_id, text))
    report = DedupeReport(len(accepted), len(documents) - len(accepted) - near_duplicates,
                          near_duplicates)
    return accepted, report
