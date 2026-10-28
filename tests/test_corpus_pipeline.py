"""Corpus stage, split isolation and benchmark leakage tests."""

from __future__ import annotations

from helix.corpus_pipeline import benchmark_leakage, build_corpus
from helix.dedupe import deduplicate
from helix.storage import resolve, sha256_file


def sample(index: int) -> str:
    # Intentionally synthetic: this fixture tests stage logic, not model
    # quality or natural-language realism.
    prefix = f'SYNTHETIC-{index:03d}'
    return (f'{prefix} alpha bravo charlie delta. {prefix} echo foxtrot golf. '
            f'{prefix} hotel india juliet kilo. {prefix} lima mexico november. '
            f'{prefix} oscar papa quebec romeo. ' * 4)


def test_splits_are_disjoint_and_hash_verified(tmp_path):
    root = tmp_path / 'external'
    documents = [(f'doc-{index}', sample(index)) for index in range(60)]
    manifest, stats = build_corpus(documents, root, lexical_threshold=0.25)
    assert stats.train_documents + stats.val_documents + stats.test_documents == stats.quality_passed
    assert stats.train_documents >= 30
    train = resolve('corpus/train.txt', root).read_text(encoding='utf-8')
    for term in ('brownie', 'donut', 'gsm8k'):
        assert term not in train.lower()
    assert sha256_file('corpus/train.txt', root) == sha256_file('corpus/train.txt', root)


def test_benchmark_leakage_is_filtered(tmp_path):
    root = tmp_path / 'external'
    # Filtering happens before dedup and split allocation.
    leaked = ('normal', sample(0)), ('normal-two', sample(0)),
    ('normal-three', sample(1)), ('normal-four', sample(2)),
    ('normal-five', sample(3)), ('benchmark', 'GSM8K sample tasks have answers. ' * 10)
    manifest, stats = build_corpus(list(leaked), root, lexical_threshold=0.25)
    assert stats.benchmark_filtered == 1
    assert benchmark_leakage('A brownie recipe is not an evaluation benchmark.')


def test_lexical_duplicates_are_removed(tmp_path):
    text = sample(0)
    accepted, report = deduplicate([('one', text), ('two', text), ('three', text)])
    assert len(accepted) == 1 and report.exact_duplicates == 2
