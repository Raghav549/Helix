"""One entry point for restartable source-to-token corpus construction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from helix.corpus_pipeline import build_corpus, read_documents
from helix.extract import extract_source
from helix.ingest import ingest_source
from helix.sources import Catalog, SourceRecord
from helix.storage import REPOSITORY_ROOT, StorageError, resolve, resolve_data_root, sha256_file
from helix.tokenizer import ByteLevelBPETokenizer
from helix.data import build_from_corpus


def _seed_documents(source: str, text: str) -> list[tuple[str, str]]:
    """Split authored seed topics into separate, logically partitioned documents."""
    parts = [part.strip() for part in text.split("\n\n== ") if part.strip()]
    if len(parts) < 2:
        return [(source, text)]
    documents: list[tuple[str, str]] = []
    for index, part in enumerate(parts):
        # The leading topic stays with its body; later parts retain the topic
        # prefix after the split marker.
        body = part[len("== "):].strip() if index == 0 else part.split(" ==\n", 1)[-1].strip()
        if body:
            documents.append((f"{source}:{index:02d}", body))
    return documents


def _load_json(path: os.PathLike[str] | str,
               root: os.PathLike[str] | str | None = None) -> object:
    target = resolve(path, root)
    return json.loads(target.read_text(encoding='utf-8'))


def run(root: os.PathLike[str] | str | None = None,
        sources_path: os.PathLike[str] | str = 'manifests/sources.json',
        extract_max_docs: int = 10_000,
        lexical_threshold: float = 1.0) -> dict[str, object]:
    storage_root = resolve_data_root(root)
    catalog_path = resolve(sources_path, storage_root)
    if not catalog_path.is_file():
        raise StorageError(f'source catalog not found: {catalog_path}')

    catalog = Catalog.load(catalog_path, storage_root)
    catalog.validate()
    documents = []
    for record in catalog.eligible():
        # HELIX-authored source records may point to local content; HTTP
        # records use the bounded downloader and hash the staged file.
        local_path = REPOSITORY_ROOT / 'data/seed_corpus.txt'
        if record.name == 'helix-seed-corpus':
            if not local_path.is_file():
                raise StorageError(f'HELIX seed corpus is missing from {local_path}')
            ingest = ingest_source(record, storage_root, local_path=local_path)
            source_path = storage_root / ingest.path
            extracted = extract_source(source_path, ingest.source,
                                       {'ingest_sha256': ingest.sha256,
                                        'content_type': ingest.content_type})
            documents.extend(_seed_documents(record.name, extracted.text))
            continue
        ingest = ingest_source(record, storage_root)
        source_path = storage_root / ingest.path
        extracted = extract_source(source_path, ingest.source,
                                   {'ingest_sha256': ingest.sha256,
                                    'content_type': ingest.content_type})
        documents.append((extracted.doc_id, extracted.text))
    manifest, stats = build_corpus(documents, storage_root,
                                   lexical_threshold=lexical_threshold)

    corpus_path = storage_root / 'corpus' / 'train.txt'
    vocab_path = storage_root / 'artifacts' / 'data' / 'vocab.json'
    merges_path = storage_root / 'artifacts' / 'data' / 'merges.json'
    train_path = storage_root / 'artifacts' / 'data' / 'train.npy'
    val_path = storage_root / 'artifacts' / 'data' / 'val.npy'
    tokenizer, train_ids, val_ids = build_from_corpus(
        str(corpus_path), str(vocab_path), str(merges_path),
        vocab_size=512, train_frac=0.9,
        train_out=str(train_path), val_out=str(val_path))
    for path, arr in ((train_path, train_ids), (val_path, val_ids)):
        with path.open('wb') as stream:
            import numpy as np
            np.save(stream, arr)
    metadata = {'corpus_sha256': sha256_file(corpus_path, storage_root),
                'tokenizer_vocab_size': len(tokenizer),
                'train_tokens': int(len(train_ids)), 'val_tokens': int(len(val_ids))}
    (storage_root / 'artifacts' / 'data' / 'train.npy.meta.json').write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return {'corpus_manifest': str(storage_root / 'corpus' / 'manifest.json'),
            'corpus_stats': asdict(stats), 'tokenizer_metadata': metadata}
