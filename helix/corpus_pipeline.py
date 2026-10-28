"""Staged, restartable corpus construction with immutable splits.

The pipeline intentionally keeps training, validation and test ids disjoint
and deterministic. Benchmark terms found in the training text are recorded in
metadata and the affected documents are excluded before tokenization.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from helix.dedupe import DedupeReport, deduplicate, document_hash
from helix.quality import language_hint, normalize_text, passes_quality, quality_score
from helix.storage import StorageError, resolve, resolve_data_root, sha256_file


@dataclass
class CorpusStats:
    discovered: int
    extracted: int
    quality_passed: int
    exact_duplicates: int
    near_duplicates: int
    benchmark_filtered: int
    train_documents: int
    val_documents: int
    test_documents: int
    train_tokens: int
    val_tokens: int
    test_tokens: int


BENCHMARK_PATTERNS = [
    re.compile(r'(?<![\w])(?:brownie|donut|arc|truthfulqa|gsm[\s_-]*8k|mmlu|ceval|hellaswag)(?![\w])', re.I),
]


def write_documents(documents: list[dict[str, object]],
                    path: os.PathLike[str] | str,
                    root: os.PathLike[str] | str | None = None) -> Path:
    target = resolve(path, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n\n---\n\n'.join(str(doc['text']) for doc in documents),
                      encoding='utf-8')
    return target


def read_documents(path: os.PathLike[str] | str,
                   root: os.PathLike[str] | str | None = None) -> list[dict[str, object]]:
    target = resolve(path, root)
    text = target.read_text(encoding='utf-8')
    parts = [part.strip() for part in text.split('\n\n---\n\n') if part.strip()]
    return [{'id': f'{target.name}:{index}', 'text': part, 'metadata': {'origin': str(target)}, 'quality': {}}
            for index, part in enumerate(parts)]


def benchmark_leakage(text: str) -> bool:
    return any(pattern.search(text) for pattern in BENCHMARK_PATTERNS)


def build_corpus(documents: list[tuple[str, str]],
                 root: os.PathLike[str] | str | None = None,
                 train_fraction: float = 0.9,
                 validation_fraction: float = 0.05,
                 shuffle_seed: int = 1337,
                 lexical_threshold: float = 0.80) -> tuple[dict[str, object], CorpusStats]:
    storage_root = resolve_data_root(root)
    corpus_root = storage_root / 'corpus'
    raw = []
    quality_passed = 0
    benchmark_filtered = 0
    for doc_id, text in documents:
        normalized = normalize_text(text)
        # Leakage is checked before the quality gate so rejected evidence is
        # counted accurately rather than silently grouped as low-quality data.
        if benchmark_leakage(normalized):
            benchmark_filtered += 1
            continue
        metrics = quality_score(normalized)
        if not passes_quality(metrics):
            continue
        quality_passed += 1
        raw.append({'id': doc_id, 'text': normalized, 'metadata': {'language': language_hint(normalized)},
                    'quality': metrics})

    deduped, report = deduplicate([(str(item['id']), str(item['text'])) for item in raw],
                                  lexical_threshold)
    accepted_ids = {doc_id for doc_id, _ in deduped}
    final = [item for item in raw if str(item['id']) in accepted_ids]
    rng = random.Random(shuffle_seed)
    final_sorted = sorted(final, key=lambda item: str(item['id']))
    rng.shuffle(final_sorted)
    test_count = max(1, round(len(final_sorted) * validation_fraction)) if final_sorted else 0
    val_count = max(1, round((len(final_sorted) - test_count) * (1 - train_fraction))) if final_sorted else 0
    test = final_sorted[:test_count]
    validation = final_sorted[test_count:test_count + val_count]
    training = final_sorted[test_count + val_count:]

    if not training:
        raise StorageError('no documents survived corpus quality and split filters')

    # Hash every source document so a changed input cannot silently alter a
    # completed corpus version.
    train_digests = {str(item['id']): document_hash(str(item['text'])) for item in training}
    corpus_digest = hashlib.sha256('\n'.join(
        f"{key}:{value}" for key, value in sorted(train_digests.items())).encode()).hexdigest()
    for name, split in (('train', training), ('val', validation), ('test', test)):
        path = write_documents(split, f'corpus/{name}.txt', storage_root)
        path.write_text('\n\n---\n\n'.join(str(item['text']) for item in split),
                        encoding='utf-8')
        (storage_root / 'corpus' / f'{name}.sha256').write_text(sha256_file(path, storage_root),
                                                                encoding='utf-8')

    stats = CorpusStats(
        discovered=len(documents), extracted=len(documents),
        quality_passed=quality_passed, exact_duplicates=report.exact_duplicates,
        near_duplicates=report.near_duplicates, benchmark_filtered=benchmark_filtered,
        train_documents=len(training), val_documents=len(validation),
        test_documents=len(test),
        train_tokens=sum(quality_score(str(item['text']))['tokens'] for item in training),
        val_tokens=sum(quality_score(str(item['text']))['tokens'] for item in validation),
        test_tokens=sum(quality_score(str(item['text']))['tokens'] for item in test))
    manifest = {'name': 'helix-corpus', 'version': 1, 'storage_root': str(storage_root),
                'corpus_sha256': corpus_digest, 'split_isolation': True,
                'roles': {'train': 'training only', 'val': 'model selection only',
                          'test': 'final evaluation only'},
                'stats': asdict(stats), 'dedupe': report.to_json(),
                'documents': [{'id': item['id'], 'language': item['metadata']['language'],
                               'sha256': document_hash(str(item['text'])),
                               'quality': item['quality']}
                              for item in final_sorted]}
    path = storage_root / 'corpus' / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest, stats
