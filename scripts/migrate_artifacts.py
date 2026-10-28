#!/usr/bin/env python3
"""Migrate verified HELIX artifacts from Git to HELIX_DATA_ROOT.

The script is intentionally migration-only. It never deletes repository files
and exits before copying if a SHA-256 manifest value does not match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from helix.storage import (StorageError, migrate_verified, resolve,
                           resolve_data_root, sha256_file, validate_storage_layout)


def sha256sums_path() -> Path:
    return Path(__file__).resolve().parent.parent / 'artifacts' / 'SHA256SUMS'


def parse_manifest(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return logical names and Git-tracked SHA-256 values from artifacts/SHA256SUMS."""
    hashes: dict[str, str] = {}
    paths: dict[str, str] = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        if not raw or raw.startswith('#'):
            continue
        parts = raw.split()
        if len(parts) != 2:
            raise StorageError(f'malformed checksum line: {raw!r}')
        digest, relative = parts
        if len(digest) != 64:
            raise StorageError(f'unsupported checksum line: {raw!r}')
        logical = str(Path(relative).relative_to('artifacts')) if relative.startswith('artifacts/') else relative
        if logical in hashes:
            raise StorageError(f'duplicate checksum for {logical}')
        hashes[logical] = digest
        paths[logical] = relative
    return paths, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--destination', default='artifacts')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    try:
        paths, hashes = parse_manifest(sha256sums_path())
        root = resolve_data_root(args.data_root)
        destination = resolve(args.destination, root)
        if args.check:
            verified = verified_files(paths, hashes, destination)
            print(f'External HELIX artifacts verified: {verified} of {len(paths)} manifest entries, root={root}')
            return 0

        source_paths = {logical: (Path(__file__).resolve().parent.parent / relative)
                        for logical, relative in paths.items()}
        migrated = migrate_verified(source_paths, hashes, destination, root)
        validate_storage_layout(root)
        print(json.dumps({'root': str(root), 'migrated': len(migrated),
                          'bytes': sum(path.stat().st_size for path in migrated.values())},
                         indent=2))
        return 0
    except StorageError as exc:
        print(f'Migration rejected: {exc}', file=sys.stderr)
        return 2


def verified_files(paths: dict[str, str], hashes: dict[str, str],
                   destination: Path) -> int:
    verified = 0
    for logical in paths:
        source = destination / logical
        if source.is_file() and sha256_file(source) == hashes[logical]:
            verified += 1
    return verified


if __name__ == '__main__':
    raise SystemExit(main())
