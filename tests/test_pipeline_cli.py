"""End-to-end storage/source pipeline test without external network access."""

from __future__ import annotations

import json
import http.server
import threading
import numpy as np
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

from helix.sources import add_sources
from helix.tokenizer import ByteLevelBPETokenizer

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


def _local_source_catalog(root: Path, base_url: str) -> Path:
    target = root / 'manifests' / 'sources.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    source = {'version': 1, 'generated_by': 'test', 'sources': [
        {'name': 'document-0', 'kind': 'text', 'url': f'{base_url}/document-0.txt',
         'license': 'MIT', 'license_evidence': 'verified',
         'permitted_uses': ['train'], 'decision': 'eligible'},
        {'name': 'document-1', 'kind': 'text', 'url': f'{base_url}/document-1.txt',
         'license': 'MIT', 'license_evidence': 'verified',
         'permitted_uses': ['train'], 'decision': 'eligible'}]}
    target.write_text(json.dumps(source), encoding='utf-8')
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
    words = ('alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
         'hotel', 'india', 'juliet', 'kilo', 'lima', 'mexico', 'november',
         'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango',
         'uniform', 'victor', 'whiskey', 'xray', 'yankee', 'zulu')
    documents = [(f'doc-{i:02d}', ' '.join(
        f'SYNTHETIC-{i:02d}-{i} {word}-{i}.' for word in words) + ' evidence')
        for i in range(30)]
    manifest, stats = build_corpus(documents, root)
    assert manifest['split_isolation'] is True
    assert stats.train_documents > 0


def test_source_pipeline_generates_external_tokenizer(tmp_path):
    root = tmp_path / 'external'
    root.mkdir(parents=True)
    text_one = (REPOSITORY_ROOT / 'data/seed_corpus.txt').read_text(encoding='utf-8')
    text_two = '\n\n'.join(
        f'HELIX-RESEARCH-DOCUMENT-TWO SECTION {index}. ' + sentence + '.'
        for index, sentence in enumerate([
            'Proofs use explicit assumptions', 'Code must be executable',
            'Measurements include uncertainty', 'Reproducible research records versions',
            'Validation compares expected and observed values',
            'Implementation checks maintainable behavior',
            'Statistical reporting distinguishes models from evidence',
            'Interpretation depends on stable measurements',
            'Algorithms require deterministic boundary conditions',
            'Documentation supports later verification']))
    body = {'/document-0.txt': text_one.encode('utf-8'),
            '/document-1.txt': text_two.encode('utf-8')}
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            content = body[self.path]
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(content)
        def log_message(self, *args): pass
    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        catalog = _local_source_catalog(root, f'http://127.0.0.1:{server.server_address[1]}')
        from helix.pipeline import run
        result = run(root=root, sources_path=catalog, lexical_threshold=1.0)
    finally:
        server.shutdown()
    assert result['corpus_stats']['benchmark_filtered'] == 0
    assert result['tokenizer_metadata']['train_tokens'] > 0
    assert result['tokenizer_metadata']['val_tokens'] > 0
    train = np.load(root / 'artifacts' / 'data' / 'train.npy', allow_pickle=False)
    val = np.load(root / 'artifacts' / 'data' / 'val.npy', allow_pickle=False)
    tok = ByteLevelBPETokenizer.load(root / 'artifacts' / 'data' / 'vocab.json',
                                     root / 'artifacts' / 'data' / 'merges.json')
    assert tok.decode([int(train[0])])
    assert tok.decode([int(val[0])])


def test_source_pipeline_rejects_leakage_before_quality(tmp_path):
    root = tmp_path / 'external'
    root.mkdir(parents=True)
    target = root / 'manifests' / 'sources.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'version': 1, 'generated_by': 'test', 'sources': [
        {'name': 'benchmark-fixture', 'kind': 'text', 'url': 'https://example.test/leak.txt',
         'license': 'MIT', 'license_evidence': 'verified',
         'permitted_uses': ['train'], 'decision': 'eligible'}]}), encoding='utf-8')
    # Call the corpus boundary directly to avoid network ingest.
    from helix.corpus_pipeline import build_corpus
    try:
        build_corpus([('benchmark', 'GSM8K tasks have answers. ' * 20)], root)
    except Exception as exc:
        assert 'no documents survived' in str(exc)
    else:
        raise AssertionError('benchmark leakage was not rejected')
    assert not (root / 'artifacts' / 'data' / 'train.npy').exists()
