"""Universal source discovery, catalogs, and rights-aware staging.

Each connector records access metadata only; it never treats availability as
permission to train. Consumers must evaluate ``rights`` before moving records
from ``discovered`` into ``eligible`` or ``quarantined``.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from helix.storage import StorageError, resolve, resolve_data_root

RIGHTS_DECISIONS = {'eligible', 'needs_review', 'quarantined'}


@dataclass
class SourceRecord:
    name: str
    kind: str
    url: str
    license: str
    license_evidence: str
    permitted_uses: tuple[str, ...] = ('store',)
    source_url: str = ''
    language: str = ''
    format: str = ''
    retrieval_date: str = ''
    decision: str = 'needs_review'
    quarantined: bool = True
    notes: str = ''
    versions: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> 'SourceRecord':
        item = dict(item)
        # The existing ``url`` field is the retrieval URL; ``source_url`` is the
        # exact reviewed content location when it differs from the landing page.
        item.setdefault('source_url', item.get('url', ''))
        item['permitted_uses'] = tuple(item.get('permitted_uses', []))
        item['versions'] = list(item.get('versions', []))
        unknown = set(item) - set(cls.__dataclass_fields__)
        if unknown:
            raise StorageError(f'unknown source fields: {sorted(unknown)}')
        record = cls(**item)
        if record.decision not in RIGHTS_DECISIONS:
            raise StorageError(f'invalid rights decision: {record.decision}')
        record.quarantined = record.decision != 'eligible'
        return record

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['permitted_uses'] = list(self.permitted_uses)
        return payload


@dataclass
class Catalog:
    version: int
    generated_by: str
    sources: list[SourceRecord] = field(default_factory=list)

    @classmethod
    def load(cls, path: os.PathLike[str] | str | None = None,
             root: os.PathLike[str] | str | None = None) -> 'Catalog':
        target = resolve(path or 'manifests/sources.json', root)
        try:
            payload = json.loads(target.read_text(encoding='utf-8'))
            return cls(version=int(payload.get('version', 1)),
                       generated_by=str(payload.get('generated_by', 'helix')),
                       sources=[SourceRecord.from_json(item) for item in payload.get('sources', [])])
        except (OSError, TypeError, ValueError, KeyError, StorageError) as exc:
            raise StorageError(f'cannot load source catalog {target}: {exc}') from exc

    def save(self, path: os.PathLike[str] | str | None = None,
             root: os.PathLike[str] | str | None = None) -> Path:
        target = resolve(path or 'manifests/sources.json', root)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': self.version, 'generated_by': self.generated_by,
                   'sources': [record.to_json() for record in self.sources]}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n',
                          encoding='utf-8')
        return target

    def validate(self) -> None:
        names = {record.name for record in self.sources}
        if len(names) != len(self.sources):
            raise StorageError('source catalog contains duplicate names')
        for record in self.sources:
            if not record.name or not record.url:
                raise StorageError('source name and URL are required')
            if record.license_evidence != 'verified' and record.decision == 'eligible':
                raise StorageError(f'{record.name} is eligible without verified license evidence')
            if 'train' not in record.permitted_uses and record.decision == 'eligible':
                raise StorageError(f'{record.name} does not permit training')

    def eligible(self) -> list[SourceRecord]:
        return [record for record in self.sources if record.decision == 'eligible']


def load_sources(path: os.PathLike[str] | str | None = None,
                 root: os.PathLike[str] | str | None = None) -> list[SourceRecord]:
    return Catalog.load(path, root).sources


def add_sources(records: Iterable[SourceRecord],
                path: os.PathLike[str] | str | None = None,
                root: os.PathLike[str] | str | None = None) -> Catalog:
    target = resolve(path or 'manifests/sources.json', root)
    existing = Catalog(version=1, generated_by='helix')
    if target.is_file():
        existing = Catalog.load(target, root)
    names = {record.name for record in existing.sources}
    for record in records:
        if record.name in names:
            raise StorageError(f'duplicate source: {record.name}')
        existing.sources.append(record)
        names.add(record.name)
    existing.validate()
    existing.save(target, root)
    return Catalog.load(target, root)


def ingest_rows(rows: Iterable[dict[str, str]],
                path: os.PathLike[str] | str | None = None,
                root: os.PathLike[str] | str | None = None) -> int:
    """Ingest CSV/TSV discovery rows without silently allowing training."""
    records = []
    for row in rows:
        required = {'name', 'kind', 'url', 'license', 'license_evidence'}
        missing = required - set(row)
        if missing:
            raise StorageError(f'discovery row missing {sorted(missing)}: {row}')
        decision = (row.get('decision') or 'needs_review').lower()
        if decision not in RIGHTS_DECISIONS:
            raise StorageError(f'invalid discovery decision: {decision}')
        permitted = tuple(part.strip().lower() for part in
                          (row.get('permitted_uses') or '').split(',') if part.strip())
        records.append(SourceRecord(
            name=row['name'], kind=row['kind'], url=row['url'],
            license=row['license'], license_evidence=row['license_evidence'],
            permitted_uses=permitted, language=row.get('language', ''),
            format=row.get('format', ''), retrieval_date=row.get('retrieval_date', ''),
            decision=decision, notes=row.get('notes', ''),
            versions=[v for v in (row.get('versions') or '').split(',') if v]))
    catalog = add_sources(records, path, root)
    return len(catalog.sources)


def read_delimited_rows(text: str, delimiter: str = ',') -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]
