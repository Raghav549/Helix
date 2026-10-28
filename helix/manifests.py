"""Machine-readable source-of-truth manifests for HELIX external data."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from helix.storage import StorageError, resolve_data_root


@dataclass
class ManifestRecord:
    name: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    origin: str


@dataclass
class SourceRecord:
    name: str
    kind: str
    url: str
    license: str
    license_evidence: str
    permitted_uses: tuple[str, ...]
    language: str
    format: str
    retrieval_date: str
    quarantined: bool = False
    notes: str = ''
    children: list['SourceRecord'] = field(default_factory=list)


@dataclass
class Manifest:
    name: str
    version: int
    generated_by: str
    storage_root: str
    records: list[ManifestRecord] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def save(self, path: os.PathLike[str] | str,
             root: os.PathLike[str] | str | None = None) -> Path:
        from helix.storage import resolve
        target = resolve(path, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + '\n',
                          encoding='utf-8')
        return target


def _load_sources(path: Path) -> list[SourceRecord]:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        records: list[SourceRecord] = []
        for item in raw.get('sources', []):
            item = dict(item)
            item['permitted_uses'] = tuple(item.get('permitted_uses', []))
            item['children'] = [dict(child) for child in item.get('children', [])]
            records.append(SourceRecord(**item))
        return records
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise StorageError(f'cannot load source manifest {path}: {exc}') from exc


def source_manifest(root: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    resolved_root = resolve_data_root(root)
    path = resolved_root / 'manifests' / 'sources.json'
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise StorageError(f'cannot load source manifest {path}: {exc}') from exc
    return {'version': 1, 'sources': []}


def save_source_manifest(payload: dict[str, Any],
                         root: os.PathLike[str] | str | None = None) -> Path:
    from helix.storage import resolve
    target = resolve('manifests/sources.json', root)
    if not isinstance(payload, dict) or not isinstance(payload.get('sources'), list):
        raise StorageError('source manifest must contain a sources list')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return target


def ensure_dir(path: os.PathLike[str] | str,
               root: os.PathLike[str] | str | None = None) -> Path:
    from helix.storage import resolve
    target = resolve(path, root)
    target.mkdir(parents=True, exist_ok=True)
    return target


def storage_inventory(root: os.PathLike[str] | str | None = None) -> dict[str, int]:
    resolved_root = resolve_data_root(root)
    inventory = {'files': 0, 'bytes': 0}
    for path in resolved_root.rglob('*'):
        if path.is_file() and not path.is_symlink():
            inventory['files'] += 1
            inventory['bytes'] += path.stat().st_size
    return inventory
