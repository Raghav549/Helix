"""Persistent, provenance-preserving ingestion for the HELIX Knowledge Mill.

Ingestion updates retrieval immediately; it never modifies model weights.  A
separate, reviewed training export is deliberately required for training jobs.
"""
from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import socket
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from helix.retrieval import RetrievalIndex

MAX_BYTES = 4 * 1024 * 1024
TEXT_TYPES = {'.txt', '.md', '.markdown', '.rst', '.html', '.htm', '.json', '.csv', '.pdf'}
IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.webp'}


class KnowledgeError(ValueError):
    """An uploaded or remote source is unsafe or unsupported."""


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _safe_name(value: str, default: str = 'document.txt') -> str:
    name = Path(value or default).name.replace('\x00', '').strip()
    if not name:
        name = default
    stem = ''.join(ch if ch.isalnum() or ch in '._-' else '-' for ch in name)
    return stem[:120] or default


def _plain_html(raw: bytes) -> str:
    text = raw.decode('utf-8', errors='replace')
    import re
    text = re.sub(r'(?is)<(script|style|noscript)[^>]*>.*?</\1>', ' ', text)
    text = re.sub(r'(?is)<br\s*/?>|</(?:p|div|li|h[1-6]|tr)>', '\n', text)
    return html.unescape(re.sub(r'(?is)<[^>]+>', ' ', text))


def _private_host(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise KnowledgeError('URL host could not be resolved') from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            return True
    return False


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # pragma: no cover - urllib callback
        return None


class KnowledgeMill:
    """Files, provenance and a deterministic retrieval index outside the repo."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.documents_dir = self.root / 'documents'
        self.index_dir = self.root / 'index'
        self.database = self.root / 'knowledge.sqlite3'
        self._lock = threading.RLock()
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL, source_kind TEXT NOT NULL, source_url TEXT,
                content_type TEXT NOT NULL, bytes INTEGER NOT NULL, license_note TEXT,
                submitted_by TEXT NOT NULL, created_at TEXT NOT NULL, indexable INTEGER NOT NULL,
                state TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS documents_created ON documents(created_at DESC)')

    @staticmethod
    def fetch_url(url: str) -> tuple[bytes, str, str]:
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise KnowledgeError('Only public http(s) URLs without credentials are allowed')
        if _private_host(parsed.hostname):
            raise KnowledgeError('Private, loopback and local-network URL targets are not allowed')
        request = Request(url, headers={'User-Agent': 'HELIX-KnowledgeMill/1.0', 'Accept': 'text/plain,text/html,application/pdf,image/*'})
        try:
            with build_opener(_NoRedirect).open(request, timeout=12) as response:
                content_type = response.headers.get_content_type() or 'application/octet-stream'
                length = int(response.headers.get('Content-Length') or 0)
                if length > MAX_BYTES:
                    raise KnowledgeError('Remote document exceeds the 4 MiB ingestion limit')
                body = response.read(MAX_BYTES + 1)
        except HTTPError as exc:
            if 300 <= exc.code < 400:
                raise KnowledgeError('Redirecting URLs are not accepted; submit the final public URL') from exc
            raise KnowledgeError(f'Remote URL returned HTTP {exc.code}') from exc
        except URLError as exc:
            raise KnowledgeError('Could not fetch the remote URL') from exc
        if len(body) > MAX_BYTES:
            raise KnowledgeError('Remote document exceeds the 4 MiB ingestion limit')
        name = _safe_name(Path(parsed.path).name or 'download')
        return body, content_type, name

    @staticmethod
    def _normalise(raw: bytes, filename: str, content_type: str) -> tuple[bytes, str, bool, str]:
        suffix = Path(filename).suffix.lower()
        media = (content_type or '').lower().split(';', 1)[0]
        if suffix in IMAGE_TYPES or media.startswith('image/'):
            if media not in ('image/png', 'image/jpeg', 'image/webp'):
                raise KnowledgeError('Only PNG, JPEG and WebP images are accepted')
            return raw, media, False, 'stored_for_image_analysis'
        if suffix == '.pdf' or media == 'application/pdf':
            if suffix != '.pdf': filename += '.pdf'
            return raw, 'application/pdf', True, 'queued_for_indexing'
        if suffix not in TEXT_TYPES and media not in ('text/plain', 'text/html', 'application/json', 'text/csv'):
            raise KnowledgeError('Accepted sources are text, HTML, JSON, CSV, PDF, PNG, JPEG and WebP')
        text = _plain_html(raw) if media == 'text/html' or suffix in ('.html', '.htm') else raw.decode('utf-8', errors='replace')
        text = '\n'.join(line.rstrip() for line in text.replace('\r\n', '\n').split('\n')).strip()
        if len(text) < 32:
            raise KnowledgeError('Text sources must contain at least 32 characters after safe parsing')
        return text.encode('utf-8'), 'text/plain; charset=utf-8', True, 'queued_for_indexing'

    def ingest(self, raw: bytes, filename: str, content_type: str, *, source_kind: str,
               source_url: str | None, submitted_by: str, license_note: str = '') -> dict[str, Any]:
        if not raw or len(raw) > MAX_BYTES:
            raise KnowledgeError('Sources must be between 1 byte and 4 MiB')
        filename = _safe_name(filename)
        normalised, normalised_type, indexable, state = self._normalise(raw, filename, content_type)
        digest = hashlib.sha256(normalised).hexdigest()
        document_id = digest[:24]
        extension = '.txt' if indexable and Path(filename).suffix.lower() not in ('.pdf',) else Path(filename).suffix.lower()
        stored_name = f'{document_id}{extension or ".bin"}'
        with self._lock:
            with self._connect() as db:
                existing = db.execute('SELECT * FROM documents WHERE sha256 = ?', (digest,)).fetchone()
                if existing:
                    return {**dict(existing), 'deduplicated': True, 'indexed': existing['state'] == 'indexed'}
                target = self.documents_dir / stored_name
                temporary = target.with_suffix(target.suffix + '.tmp')
                temporary.write_bytes(normalised)
                os.replace(temporary, target)
                record = (document_id, digest, filename, stored_name, source_kind, source_url,
                          normalised_type, len(normalised), license_note.strip()[:500], submitted_by[:120],
                          _now(), int(indexable), state)
                db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', record)
            if indexable:
                self.rebuild_index()
                with self._connect() as db:
                    db.execute("UPDATE documents SET state = 'indexed' WHERE id = ?", (document_id,))
                    state = 'indexed'
        return {'id': document_id, 'sha256': digest, 'original_name': filename, 'source_kind': source_kind,
                'source_url': source_url, 'content_type': normalised_type, 'bytes': len(normalised),
                'deduplicated': False, 'indexed': bool(indexable), 'state': state}

    def rebuild_index(self) -> dict[str, Any]:
        manifest = {'version': 'knowledge-mill-v1', 'documents': {}}
        with self._connect() as db:
            for row in db.execute('SELECT stored_name, original_name, sha256, license_note, source_url FROM documents WHERE indexable = 1'):
                manifest['documents'][row['stored_name']] = {'title': row['original_name'], 'sha256': row['sha256'],
                                                             'license': row['license_note'], 'source_url': row['source_url']}
        (self.documents_dir / 'helix-retrieval.json').write_text(json.dumps(manifest, sort_keys=True), encoding='utf-8')
        index = RetrievalIndex.ingest(self.documents_dir)
        index.save(self.index_dir)
        return {'documents': index.document_count, 'chunks': index.chunk_count, 'checksum': index.checksum}

    def status(self) -> dict[str, Any]:
        with self._connect() as db:
            total = db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            indexed = db.execute("SELECT COUNT(*) FROM documents WHERE state = 'indexed'").fetchone()[0]
        try:
            index = RetrievalIndex.load(self.index_dir)
            return {'configured': True, 'documents': index.document_count, 'chunks': index.chunk_count,
                    'checksum': index.checksum, 'stored_documents': total, 'indexed_documents': indexed}
        except (FileNotFoundError, ValueError):
            return {'configured': False, 'documents': 0, 'chunks': 0, 'stored_documents': total,
                    'indexed_documents': indexed}

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [dict(row) for row in db.execute('SELECT id, original_name, source_kind, source_url, content_type, bytes, license_note, submitted_by, created_at, state FROM documents ORDER BY created_at DESC LIMIT ?', (max(1, min(limit, 100)),))]
