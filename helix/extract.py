"""Deterministic text extraction from downloaded source archives."""

from __future__ import annotations

import gzip
import html
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from helix.storage import StorageError

ALLOWED_SUFFIXES = {'.txt', '.md', '.rst', '.html', '.htm', '.xml', '.json', '.csv',
                    '.tsv', '.epub', '.pdf', '.tar', '.gz', '.zip'}


@dataclass
class ExtractedDocument:
    doc_id: str
    text: str
    metadata: dict[str, str]


def _decode(data: bytes, encoding: str = 'utf-8') -> str:
    for candidate in (encoding, 'utf-8', 'cp1252', 'latin-1'):
        try:
            return data.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    raise StorageError('unsupported source encoding')


def _normalize_newlines(text: str) -> str:
    return text.replace('\r\n', '\n').replace('\r', '\n')


def _strip_html(text: str) -> str:
    text = re.sub(r'(?is)<(script|style|noscript|svg|head)\b.*?>.*?</\1>', ' ', text)
    text = re.sub(r'(?is)<br\s*/?>', '\n', text)
    text = re.sub(r'(?is)</(p|div|li|section|article|table|tr)>', '\n', text)
    text = re.sub(r'(?is)<[^>]+>', ' ', text)
    return html.unescape(re.sub(r'[ \t\f\v]+', ' ', text))


def _extract_epub_without_dependencies(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [name for name in archive.namelist() if name.endswith('.xhtml')]
            names.sort(key=lambda name: (0 if 'nav' in name.lower() else 1, name))
            return '\n\n'.join(_normalize_newlines(_strip_html(_decode(archive.read(name))))
                               for name in names)
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise StorageError(f'cannot extract EPUB without local dependency: {exc}') from exc


def _extract_pdf_minimal(data: bytes) -> str:
    # The baseline does not include an external PDF library. This extractor
    # intentionally leaves scanned or encoded content for the quarantine path
    # instead of pretending a meaningful OCR result.
    latin1 = data.decode('latin-1')
    chunks = [chunk for chunk in re.findall(r'\((?:[^()\\]|\\.)*\)', latin1)
              if len(chunk) > 12]
    return _normalize_newlines(' '.join(chunks))


def extract_source(source_path: os.PathLike[str] | str,
                   doc_id: str,
                   metadata: dict[str, str]) -> ExtractedDocument:
    path = Path(source_path)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise StorageError(f'source format is not supported: {path.name}')
    if path.stat().st_size > 256 * 1024 * 1024:
        raise StorageError(f'source exceeds extraction limit: {path.name}')

    if path.suffix.lower() == '.gz':
        with gzip.open(path, 'rb') as stream:
            data = stream.read()
    elif path.suffix.lower() == '.tar':
        with tarfile.open(path, 'r:*') as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            data = b''.join(archive.extractfile(m).read() for m in members[:50])
    elif path.suffix.lower() in {'.zip', '.epub'}:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith('/')]
            if path.suffix.lower() == '.epub':
                text = _extract_epub_without_dependencies(path.read_bytes())
            else:
                allowed = [name for name in names if Path(name).suffix.lower()
                           in {'.txt', '.md', '.rst', '.html', '.htm', '.xml'}]
                text = '\n\n'.join(
                    _normalize_newlines(_strip_html(_decode(archive.read(name))))
                    if Path(name).suffix.lower() in {'.html', '.htm', '.xml'}
                    else _normalize_newlines(_decode(archive.read(name)))
                    for name in allowed[:100])
            return ExtractedDocument(doc_id, text.strip(), {**metadata, 'archive_entries': str(len(names))})
        data = b''
    elif path.suffix.lower() == '.pdf':
        text = _extract_pdf_minimal(path.read_bytes())
        return ExtractedDocument(doc_id, text.strip(), {**metadata, 'extraction': 'pdf-minimal'})
    else:
        data = path.read_bytes()

    suffix = path.suffix.lower()
    text = _normalize_newlines(_decode(data))
    if suffix in {'.html', '.htm', '.xml'}:
        text = _strip_html(text)
    return ExtractedDocument(doc_id, text.strip(), metadata)
