#!/usr/bin/env python3
"""Measure retrieval quality on a small labelled question set.

The retriever is lexical (BM25) plus a hashed bag-of-words cosine signal, so this
script reports exactly that: recall@1, recall@k, MRR and citation validity for a
hand-labelled set of questions over the repository's own documentation. It is not
a semantic-search benchmark and the report says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helix.retrieval import RetrievalIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate HELIX local retrieval')
    parser.add_argument('--cases', default='data/retrieval_eval.json')
    parser.add_argument('--root', default=None, help='override the corpus root')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--exclude', action='append', default=None,
                        help='relative directory prefix inside the corpus root to skip (repeatable)')
    parser.add_argument('--out', default=None, help='write the JSON report here')
    args = parser.parse_args()

    spec = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    root = Path(args.root or spec.get('corpus_root', 'docs'))
    if not root.is_dir():
        print(f'corpus root not found: {root}', file=sys.stderr)
        return 2

    exclude = args.exclude if args.exclude is not None else list(spec.get('exclude', []))
    index = RetrievalIndex.ingest(str(root), exclude=exclude)
    cases = spec['cases']
    results: List[dict] = []
    hits_at_1 = hits_at_k = 0
    reciprocal_ranks: List[float] = []
    invalid_citations = 0

    for case in cases:
        accepted = case['expected']
        accepted = [accepted] if isinstance(accepted, str) else list(accepted)
        found = index.search(case['query'], k=args.k)
        names = [Path(chunk.source).name for chunk in found]
        ranks = [names.index(name) + 1 for name in accepted if name in names]
        rank = min(ranks) if ranks else None
        if rank == 1:
            hits_at_1 += 1
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        for chunk in found:
            source = Path(root) / chunk.source
            if not source.is_file() or chunk.end < chunk.start:
                invalid_citations += 1
        results.append({
            'query': case['query'],
            'expected': accepted,
            'rank': rank,
            'top': [
                {'source': Path(chunk.source).name, 'score': round(chunk.score, 4),
                 'citation': chunk.citation}
                for chunk in found
            ],
        })

    total = len(cases)
    report = {
        'retriever': 'bm25+hashed-bow-cosine',
        'corpus_root': str(root),
        'exclude': exclude,
        'documents': index.document_count,
        'chunks': index.chunk_count,
        'index_checksum': index.checksum,
        'k': args.k,
        'total': total,
        'recall_at_1': round(hits_at_1 / total, 4) if total else 0.0,
        f'recall_at_{args.k}': round(hits_at_k / total, 4) if total else 0.0,
        'mrr': round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
        'invalid_citations': invalid_citations,
        'caveat': ('Small hand-labelled set over the project documentation with a lexical '
                   'retriever; scores are indicative, not a general retrieval benchmark, and '
                   'questions were written after the documents were, so they share vocabulary.'),
        'results': results,
    }

    text = json.dumps(report, indent=2) + '\n'
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
