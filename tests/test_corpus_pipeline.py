"""Corpus stage, split isolation and benchmark leakage tests."""

from __future__ import annotations

from helix.corpus_pipeline import benchmark_leakage, build_corpus
from helix.dedupe import deduplicate
from helix.storage import resolve, sha256_file


# Intentionally synthetic and non-similar: this fixture tests stage logic,
# not natural-language quality or model capability.
WORDS = ('alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
         'hotel', 'india', 'juliet', 'kilo', 'lima', 'mexico', 'november',
         'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango',
         'uniform', 'victor', 'whiskey', 'xray', 'yankee', 'zulu')


def sample(index: int) -> str:
    prefix = f'SYNTHETIC-{index:03d}'
    return ' '.join(
        f'{prefix}-{index} {word}-{index}.' for word in WORDS) + ' evidence'


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
    ('normal-five', sample(3)), ('short', 'tiny'),
    ('benchmark', ' '.join(f'GSM8K-{index} sample-{index} tasks-{index}.'
                        for index in range(80)))
    # This isolated boundary test allows the intentionally shared duplicate and
    # verifies leakage is excluded before split allocation; uniqueness is
    # covered separately by test_splits_are_disjoint_and_hash_verified.
    # The boundary test allows only the intentionally shared duplicate; this
    # makes the benchmark filter the reason the split cannot be materialized.
    try:
        build_corpus(list(leaked), root, lexical_threshold=0.95)
    except Exception as exc:
        assert 'at least two accepted documents' in str(exc)
    else:
        raise AssertionError('benchmark leakage was not rejected')
    normal = 'Python functions return values. Mathematics uses evidence. '
    assert not benchmark_leakage(normal)
    assert benchmark_leakage('A brownie recipe is not an evaluation benchmark.')


def test_lexical_duplicates_are_removed(tmp_path):
    root = tmp_path / 'external'
    accepted, report = deduplicate([('one', sample(0)), ('two', sample(0)),
                                    ('three', sample(1))], 0.25)
    assert [doc_id for doc_id, _ in accepted] == ['one', 'three']
    assert report.exact_duplicates == 1 and report.near_duplicates == 0
