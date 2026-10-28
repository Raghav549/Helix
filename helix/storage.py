"""External HELIX storage with strict repository isolation.

The source repository contains only code, configuration, tests, small metadata
and source provenance. Heavy model artifacts, tokenized data, caches, outputs
and deployment copies live under ``HELIX_DATA_ROOT`` (default ``/var/lib/helix``
when writable, otherwise ``~/.helix/data``). Relative paths intentionally
resolve against that root rather than the repository.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# Roots that must never be part of an external storage location. This prevents
# a misconfigured environment variable from moving HELIX data inside the repo.
REPOSITORY_DIR_NAMES = frozenset(
    ('helix', 'artifacts', 'configs', 'docs', 'tests', 'scripts')
)


class StorageError(RuntimeError):
    """A storage operation failed a safety or integrity check."""


def default_data_root() -> Path:
    candidate = Path('/var/lib/helix')
    if candidate.exists() or os.access('/var/lib', os.W_OK):
        return candidate
    return Path.home() / '.helix' / 'data'


def repository_root() -> Path:
    return REPOSITORY_ROOT


def _normalize_root(value: os.PathLike[str] | str) -> Path:
    path = Path(os.path.expanduser(str(value))).resolve()
    if path == REPOSITORY_ROOT or path.is_relative_to(REPOSITORY_ROOT):
        raise StorageError('HELIX data root must resolve outside the repository')
    near_repo = {part.lower() for part in path.parts[:-1]} & REPOSITORY_DIR_NAMES
    if near_repo:
        raise StorageError(f'external data root cannot be under a HELIX repository directory: {sorted(near_repo)}')
    return path


def resolve_data_root(value: os.PathLike[str] | str | None = None) -> Path:
    """Resolve and create the external root, refusing repository-relative paths."""
    root = _normalize_root(value or os.environ.get('HELIX_DATA_ROOT') or default_data_root())
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / '.helix-data').mkdir(exist_ok=True)
        (root / '.helix-data' / 'README.txt').write_text(
            'HELIX external storage. Keep checkpoints, tokenized corpora, outputs and caches here.\n',
            encoding='utf-8')
    except OSError as exc:
        raise StorageError(f'cannot initialize HELIX data root {root}: {exc}') from exc
    return root


def resolve(path: os.PathLike[str] | str,
            data_root: os.PathLike[str] | str | None = None) -> Path:
    """Resolve a project-relative path against the external data root."""
    root = resolve_data_root(data_root)
    candidate = Path(os.path.expanduser(str(path)))
    return candidate if candidate.is_absolute() else (root / candidate)


def _chunks(path: Path, size: int = 1024 * 1024) -> Iterable[bytes]:
    with path.open('rb') as stream:
        while True:
            block = stream.read(size)
            if not block:
                return
            yield block


def sha256_file(path: os.PathLike[str] | str,
                data_root: os.PathLike[str] | str | None = None) -> str:
    digest = hashlib.sha256()
    for block in _chunks(resolve(path, data_root)):
        digest.update(block)
    return digest.hexdigest()


def verify_files(paths: Iterable[os.PathLike[str] | str],
                 expected_hashes: dict[str, str],
                 data_root: os.PathLike[str] | str | None = None) -> None:
    """Fail closed when a manifest hash is missing or does not match."""
    root = resolve_data_root(data_root)
    for path in paths:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = root / resolved
        expected = expected_hashes.get(str(path), expected_hashes.get(resolved.name, ''))
        if not expected or sha256_file(resolved) != expected:
            raise StorageError(f'hash verification failed for {path}')


def ensure_space(target: os.PathLike[str] | str,
                 required_bytes: int,
                 data_root: os.PathLike[str] | str | None = None) -> None:
    root = resolve_data_root(data_root)
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = root / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(target_path.parent if target_path.parent.exists() else root))
    if usage.free < required_bytes:
        raise StorageError(f'need {required_bytes} bytes but {usage.free} bytes are free on {root}')


def migrate_verified(sources: dict[str, os.PathLike[str] | str],
                     hashes: dict[str, str],
                     destination: os.PathLike[str] | str | None = None,
                     data_root: os.PathLike[str] | str | None = None) -> dict[str, Path]:
    """Copy files only after their expected hashes match; retain originals.

    A destination file that already has the correct hash is left unchanged,
    which makes repeated migrations safe. Any mismatch removes only that newly
    written destination file and leaves the verified destination alone.
    """
    root = resolve_data_root(data_root)
    dest_root = resolve(destination or 'migrated', root)
    dest_root.mkdir(parents=True, exist_ok=True)
    migrated: dict[str, Path] = {}
    for logical_name, source in sources.items():
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = REPOSITORY_ROOT / source_path
        expected = hashes.get(logical_name)
        if not expected:
            raise StorageError(f'missing hash for migration source {logical_name}')
        actual = sha256_file(source_path)
        if actual != expected:
            raise StorageError(f'source hash mismatch for {logical_name}')
        target = dest_root / logical_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256_file(target) == expected:
            migrated[logical_name] = target
            continue
        shutil.copy2(source_path, target)
        if sha256_file(target) != expected:
            target.unlink(missing_ok=True)
            raise StorageError(f'copied hash mismatch for {logical_name}')
        migrated[logical_name] = target
    return migrated


def validate_storage_layout(data_root: os.PathLike[str] | str | None = None) -> None:
    root = resolve_data_root(data_root)
    for required in ('caches', 'outputs', 'artifacts'):
        (root / required).mkdir(exist_ok=True)
