"""Storage separation, manifest and migration integrity tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from helix.manifests import Manifest, ManifestRecord
from helix.storage import (StorageError, ensure_space, migrate_verified,
                           repository_root, resolve_data_root, resolve,
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
    assert cfg.data.corpus_file.startswith(str(repository_root()))
    assert cfg.train.out_dir.startswith(str(tmp_path))


def test_cli_source_run_builds_external_corpus(tmp_path):
    # This regression ensures an ordinary CLI source run does not write into
    # the repository and that the initial authored seed is accepted.
    root = tmp_path / 'external'
    root.mkdir(parents=True, exist_ok=True)
    Path(root / 'seed_corpus.txt').write_text(
        'Python functions return values. Mathematics uses evidence. '
        'History examines dates and consequences. ' * 8, encoding='utf-8')
    from helix.cli import main
    assert main(['sources', '--data-root', str(root)]) == 0
    assert (root / 'corpus' / 'train.txt').is_file()
    assert not (Path(repository_root()) / 'corpus').exists()
    assert not (Path(repository_root()) / 'outputs').exists()
