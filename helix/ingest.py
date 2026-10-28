"""Bounded, resumable HTTP staging for rights-aware source material."""

from __future__ import annotations

import email
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from helix.storage import StorageError, resolve, resolve_data_root
from helix.sources import Catalog, SourceRecord

DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
USER_AGENT = 'HELIX-source-ingest/0.13 (+https://github.com/Raghav549/Helix)'


class TimeoutHandler(urllib.request.HTTPHandler):
    """Enforce a wall-clock limit on redirects as well as the first response."""

    def __init__(self, timeout: int = 30) -> None:
        super().__init__()
        self.timeout = timeout

    def http_open(self, request):
        return self.do_open(urllib.request.HTTPConnection, request, timeout=self.timeout)

    https_open = http_open


@dataclass
class IngestResult:
    source: str
    path: str
    size_bytes: int
    sha256: str
    content_type: str
    etag: str
    resumed: bool

    def to_json(self) -> dict[str, object]:
        return self.__dict__.copy()


def _validate_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {'https', 'http'} or not parsed.netloc:
        raise StorageError(f'only HTTP(S) source URLs are allowed: {url}')
    if parsed.username or parsed.password:
        raise StorageError('source URLs must not contain credentials')
    return url


def _safe_name(name: str, url: str) -> str:
    label = Path(urllib.parse.urlparse(url).path).name or 'source'
    stem = Path(label).stem.replace('/', '_').replace('\\', '_')
    if len(stem) > 96:
        stem = stem[:96]
    return f'{name}--{stem}'


def ingest_source(record: SourceRecord,
                  root: os.PathLike[str] | str | None = None,
                  max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
                  max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES) -> IngestResult:
    """Download one source to ``raw/`` with size, total-byte and hash guards."""
    storage_root = resolve_data_root(root)
    if record.decision != 'eligible':
        raise StorageError(f'source {record.name} is not eligible for ingest')
    url = _validate_url(record.url)
    destination = resolve(f'raw/{_safe_name(record.name, url)}', storage_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = sum(path.stat().st_size for path in (storage_root / 'raw').glob('**/*')
                if path.is_file()) if (storage_root / 'raw').exists() else 0
    if total >= max_total_bytes:
        raise StorageError(f'external raw storage quota exhausted: {total} bytes')

    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    resumed = False
    if destination.is_file():
        existing = hashlib.sha256(destination.read_bytes()).hexdigest()
        request.add_header('Range', f'bytes={destination.stat().st_size}-')
        resumed = True
    else:
        destination.touch()
    opener = urllib.request.build_opener(TimeoutHandler(timeout=30))
    try:
        with opener.open(request) as response:
            content_type = response.headers.get_content_type()
            etag = response.headers.get('ETag', '')
            expected = int(response.headers.get('Content-Length', '0')) + destination.stat().st_size
            if expected and expected > max_file_bytes:
                raise StorageError(f'source exceeds {max_file_bytes} bytes')
            with destination.open('ab') as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if destination.stat().st_size > max_file_bytes:
                        raise StorageError(f'source exceeds {max_file_bytes} bytes')
                    if total + destination.stat().st_size > max_total_bytes:
                        raise StorageError('external raw storage quota exhausted')
                    stream.write(block)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, StorageError) as exc:
        destination.unlink(missing_ok=True)
        raise StorageError(f'failed to ingest {record.name}: {exc}') from exc

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return IngestResult(record.name, destination.relative_to(storage_root).as_posix(),
                        destination.stat().st_size, digest, content_type, etag, resumed)


def ingest_catalog(path: os.PathLike[str] | str | None = None,
                   root: os.PathLike[str] | str | None = None,
                   max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
                   max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES) -> list[IngestResult]:
    catalog = Catalog.load(path, root)
    catalog.validate()
    return [ingest_source(record, root, max_file_bytes, max_total_bytes)
            for record in catalog.eligible()]


def write_ingest_manifest(results: list[IngestResult],
                          path: os.PathLike[str] | str | None = None,
                          root: os.PathLike[str] | str | None = None) -> Path:
    target = resolve(path or 'manifests/ingest.json', root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(
        {'version': 1, 'retrieved_at_epoch': time.time(),
         'records': [result.to_json() for result in results]},
        indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return target
