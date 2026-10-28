"""End-to-end storage/source pipeline test without external network access."""

from __future__ import annotations

from pathlib import Path

from helix.cli import main
from helix.corpus_pipeline import build_corpus
from helix.pipeline import run
from helix.sources import Catalog, SourceRecord


def _write_catalog(root: Path) -> Path:
    target = root / 'manifests' / 'sources.json'
    catalog = Catalog(version=1, generated_by='test')
    catalog.sources.append(SourceRecord(
        name='local-seed', kind='seed', url='https://example.test/seed.txt',
        license='MIT', license_evidence='verified', permitted_uses=('train',),
        decision='eligible'))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"version": 1, "generated_by": "test", "sources": ['
                      '{"name": "local-seed", "kind": "seed", "url": "https://example.test/seed.txt",'
                      ' "license": "MIT", "license_evidence": "verified",'
                      ' "permitted_uses": ["train"], "decision": "eligible"}]}\n',
                      encoding='utf-8')
    return target


def test_pipeline_cli_uses_external_root_and_invariants(tmp_path, monkeypatch):
    root = tmp_path / 'external'
    catalog = _write_catalog(root)
    # The catalog has no downloaded file yet; the CLI should report a clear
    # failure without writing any corpus or training data.
    code = main(['sources', '--data-root', str(root)])
    assert code == 2
    assert not (root / 'corpus').exists()
    assert not (root / 'artifacts' / 'data' / 'train.npy').exists()


def test_corpus_invariant_test_is_available(tmp_path):
    root = tmp_path / 'external'
    documents = [(f'doc-{i}', f'Document {i}. Python and mathematics require evidence. ' * 4)
                 for i in range(30)]
    manifest, stats = build_corpus(documents, root)
    assert manifest['split_isolation'] is True
    assert stats.train_documents > 0
