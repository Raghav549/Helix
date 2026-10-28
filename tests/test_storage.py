"""Storage separation, manifest and migration integrity tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from helix.manifests import Manifest, ManifestRecord
from helix.storage import (REPOSITORY_ROOT as REPO_ROOT, StorageError, ensure_space,
                           migrate_verified, repository_root, resolve_data_root, resolve,
                           sha256_file, validate_storage_layout)


def test_external_root_must_remain_outside_repository(tmp_path):
    with pytest.raises(StorageError):
        resolve_data_root(repository_root() / 'artifacts')


def test_external_root_may_use_helix_data_directory_name(tmp_path):
    root = resolve_data_root(tmp_path / 'HELIX' / 'data')
    assert root.name == 'data' and not root.is_relative_to(repository_root())


def test_relative_paths_resolve_to_external_root(tmp_path):
    root = resolve_data_root(tmp_path / 'external')
    target = resolve('artifacts/llm/ckpt-final.pt', root)
    assert target.parent == root / 'artifacts/llm'
    assert not target.is_relative_to(repository_root())


def test_manifest_report_is_small_source_metadata(tmp_path):
    root = resolve_data_root(tmp_path / 'external')
    report = Manifest('artifacts', 1, 'helix-storage', str(root))
    report.records.append(ManifestRecord('llm', 'checkpoint', 'artifacts/llm/ckpt-final.pt',
                                         hashlib.sha256(b'metadata').hexdigest(), 7, 'repository'))
    target = report.save('artifacts/manifest.json', root)
    assert target.stat().st_size < 10_000
    assert target.parent == root / 'artifacts'


def test_migration_verifies_source_and_copy(tmp_path):
    root = resolve_data_root(tmp_path / 'external')
    original = repository_root() / 'artifacts/llm/ckpt-final.pt'
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    migrated = migrate_verified({'llm/ckpt-final.pt': original},
                                {'llm/ckpt-final.pt': digest},
                                'artifacts', root)
    target = migrated['llm/ckpt-final.pt']
    assert target.is_file() and sha256_file(target, root) == digest
    # Running twice is safe and retains the verified file.
    second = migrate_verified({'llm/ckpt-final.pt': original},
                              {'llm/ckpt-final.pt': digest},
                              'artifacts', root)
    assert second['llm/ckpt-final.pt'].read_bytes() == original.read_bytes()


def test_migration_rejects_corruption_without_deleting_original(tmp_path):
    root = resolve_data_root(tmp_path / 'external')
    source = tmp_path / 'bad.pt'
    source.write_bytes(b'not the expected content')
    with pytest.raises(StorageError, match='source hash mismatch'):
        migrate_verified({'bad.pt': source}, {'bad.pt': '0' * 64}, 'artifacts', root)
    assert source.read_bytes() == b'not the expected content'


def test_disk_space_guard(tmp_path):
    root = resolve_data_root(tmp_path / 'external')
    with pytest.raises(StorageError, match='free'):
        ensure_space('artifacts/out.bin', 10**30, root)


def test_layout_validation(tmp_path):
    root = resolve_data_root(tmp_path / 'external')
    validate_storage_layout(root)
    for name in ('caches', 'outputs', 'artifacts'):
        assert (root / name).is_dir()


def test_direct_dataclass_does_not_resolve_external(tmp_path):
    # Regression: direct constructors must not redirect checked-in files.
    from helix.config import DataConfig, HelixConfig, ModelConfig, TrainConfig
    cfg = HelixConfig(model=ModelConfig(vocab_size=512),
                      train=TrainConfig(out_dir=str(tmp_path / 'outputs')),
                      data=DataConfig(train_file='data/train.npy',
                                      val_file='data/val.npy',
                                      vocab_file='data/vocab.json',
                                      merges_file='data/merges.json',
                                      corpus_file='data/seed_corpus.txt'))
    assert cfg.data.corpus_file == 'data/seed_corpus.txt'
    assert cfg.train.out_dir.startswith(str(tmp_path))


def test_cli_source_run_builds_external_corpus(tmp_path):
    # This regression ensures an ordinary CLI source run does not write into
    # the repository and that the initial authored seed is accepted.
    root = tmp_path / 'external'
    root.mkdir(parents=True, exist_ok=True)
    # A private catalog avoids the network-only authored-source special case.
    (root / 'seed_corpus.txt').write_text(
        'HELIX-A: Python functions return values. Mathematics uses evidence. '
        'A first document must have enough tokens for training.\n\n'
        'HELIX-B: Code must be executable. Measurements include uncertainty. '
        'A second document must be held out from tokenizer training.\n', encoding='utf-8')
    import http.server
    import json
    import threading
    bodies = {'/document-one.txt': (
                  '\n\n'.join(
                      f'HELIX TRAINING DOCUMENT ONE SECTION {index}. ' +
                      ' '.join(
                          f'{word}{word}'
                          for word in (
                              'pyre', 'latus', 'solve', 'works', 'proof',
                              'system', 'energy', 'maths', 'method', 'science',
                              'source', 'logic', 'seven', 'forward', 'factor',
                              'counter', 'qualm', 'vertex', 'theorem', 'value',
                              'design', 'cipher', 'number', 'motion', 'result',
                              'memory', 'record', 'linear', 'gradient',
                              'pencil', 'island', 'signal', 'context',
                              'maple', 'vector', 'object', 'forty', 'phase',
                              'units', 'curve', 'basic', 'native'
                          )[: index % 15 + index // 15]
                      ) for index in range(10))).encode(),
              '/document-two.txt': (
                  '\n\n'.join(
                      f'HELIX VALIDATION DOCUMENT TWO SECTION {index}. ' +
                      ' '.join(
                          f'{word}{word}'
                          for word in (
                              'ample', 'bronze', 'fabric', 'gather', 'harbor',
                              'knot', 'lunar', 'marine', 'needle', 'ocean',
                              'prairie', 'quartz', 'rabbit', 'solstice', 'timber',
                              'urban', 'velvet', 'willow', 'yellow', 'zenith',
                              'azure', 'boulder', 'cedar', 'delta', 'ember',
                              'fjord', 'granite', 'harvest', 'ivory', 'jade',
                              'kestrel', 'lavender', 'meadow', 'nectar', 'opal',
                              'pine', 'quail', 'ridge', 'silver', 'tundra'
                          )[: index % 15 + index // 15]
                      ) for index in range(10))).encode()}
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            content = bodies[self.path]
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
    base_url = f'http://127.0.0.1:{server.server_address[1]}'
    (root / 'manifests').mkdir()
    (root / 'manifests' / 'sources.json').write_text(json.dumps(
        {"version": 1, "generated_by": "test", "sources": [
            {"name": "document-one", "kind": "text",
             "url": base_url + "/document-one.txt",
             "license": "MIT", "license_evidence": "verified",
             "permitted_uses": ["train"], "decision": "eligible"},
            {"name": "document-two", "kind": "text",
             "url": base_url + "/document-two.txt",
             "license": "MIT", "license_evidence": "verified",
             "permitted_uses": ["train"], "decision": "eligible"}]}), encoding='utf-8')
    try:
        from helix.cli import main
        assert main(['sources', '--data-root', str(root)]) == 0
    finally:
        server.shutdown()
    assert (root / 'corpus' / 'train.txt').is_file()
    assert not (Path(repository_root()) / 'corpus').exists()
    assert not (Path(repository_root()) / 'outputs').exists()


def test_tokenizer_round_trip_in_external_storage(tmp_path):
    from helix.tokenizer import ByteLevelBPETokenizer
    tok = ByteLevelBPETokenizer()
    tok.train(['Python returns values. HELIX trains from random initialization. ' * 4], 280)
    vocab = tmp_path / 'vocab.json'
    merges = tmp_path / 'merges.json'
    tok.save(str(vocab), str(merges))
    text = 'Python returns values.'
    assert ByteLevelBPETokenizer.load(str(vocab), str(merges)).decode(
        tok.encode(text, add_eos=True)).startswith(text)
