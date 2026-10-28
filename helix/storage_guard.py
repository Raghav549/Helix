"""Pre-commit repository guard against accidentally committing HELIX data."""

from __future__ import annotations

import os
import subprocess
import sys

from helix.storage import REPOSITORY_ROOT, StorageError, resolve_data_root


def forbidden_paths(files: list[str]) -> list[str]:
    findings: list[str] = []
    for item in files:
        parts = item.split('/')
        # Repository-local heavy data and its generated metadata are forbidden.
        if len(parts) >= 2 and parts[0] in {'artifacts', 'out', 'data'}:
            if item.endswith(('.pt', '.safetensors', '.bin', '.npy', '.gz', '.zip')):
                findings.append(item)
            elif parts[-1] in {'vocab.json', 'merges.json', 'train.npy',
                               'val.npy', 'seed_corpus.txt', 'sft_pairs.json',
                               'manifest.json'}:
                findings.append(item)
        # Checkpoint and shard formats are never appropriate in the source tree.
        if item.endswith(('.pt', '.safetensors', '.bin', '.npy')):
            findings.append(item)
    return sorted(set(findings))


def main() -> int:
    commit = os.environ.get('HELIX_STORAGE_CHECK_COMMIT', '')
    if not commit:
        print('No commit ID provided; storage guard skipped by configuration', file=sys.stderr)
        return 0
    command = ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit]
    try:
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, text=True,
                                capture_output=True, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f'Cannot inspect commit: {exc}', file=sys.stderr)
        return 2
    findings = forbidden_paths(result.stdout.splitlines())
    if findings:
        print('Refusing commit: HELIX storage artifacts are not allowed in Git', file=sys.stderr)
        for item in findings:
            print(f'  {item}', file=sys.stderr)
        return 1
    try:
        resolve_data_root(os.environ.get('HELIX_DATA_ROOT'))
    except StorageError as exc:
        print(f'Invalid HELIX_DATA_ROOT: {exc}', file=sys.stderr)
        return 1
    print('HELIX storage guard passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
