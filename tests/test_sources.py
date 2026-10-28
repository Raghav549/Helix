"""Source catalog, ingest, extraction, quality, and pipeline integrity tests."""

from __future__ import annotations

import gzip
import io
import zipfile

import pytest

from helix.discovery import connectors
from helix.extract import extract_source
from helix.ingest import ingest_source
from helix.quality import language_hint, passes_quality, quality_score
from helix.sources import Catalog, SourceRecord, add_sources


def test_connectors_do_not_claim_training_permission_without_evidence():
    for item in connectors():
        assert item.base_url.startswith('https://')
        assert 'store' in item.permitted_uses


def test_catalog_requires_training_permission_and_evidence(tmp_path):
    record = SourceRecord('example', 'book', 'https://example.com/a.txt', 'PD',
                          'verified', ('train',), decision='eligible')
    catalog = add_sources([record], root=tmp_path / 'external')
    assert catalog.eligible()[0].name == 'example'
    ambiguous = SourceRecord('ambiguous', 'book', 'https://example.com/b.txt', 'unknown',
                             'not-found', ('train',), decision='eligible')
    with pytest.raises(Exception, match='license evidence'):
        add_sources([ambiguous], root=tmp_path / 'external')


def test_catalog_rejects_nontraining_source(tmp_path):
    record = SourceRecord('no-train', 'book', 'https://example.com/a.txt', 'CC-BY-NC',
                          'verified', ('store',), decision='eligible')
    with pytest.raises(Exception, match='does not permit training'):
        add_sources([record], root=tmp_path / 'external')


def test_quality_rejects_boilerplate_and_accepts_normal_text():
    assert not passes_quality(quality_score('word ' * 40))
    assert passes_quality(quality_score(
        'The quick brown fox jumps over the lazy dog. ' * 20))
    assert language_hint('你好世界 ' * 20) == 'zh'


def test_extract_txt_html_and_archive(tmp_path):
    text_source = tmp_path / 'source.txt'
    text_source.write_text('hello\r\nworld', encoding='utf-8')
    doc = extract_source(text_source, 'text', {})
    assert doc.text == 'hello\nworld'
    html_source = tmp_path / 'source.html'
    html_source.write_text('<html><head><style>x{}</style></head><p>hello</p></html>',
                           encoding='utf-8')
    assert 'hello' in extract_source(html_source, 'html', {}).text
    with zipfile.ZipFile(tmp_path / 'source.zip', 'w') as archive:
        archive.writestr('a.txt', 'archive text')
    assert 'archive text' in extract_source(tmp_path / 'source.zip', 'zip', {}).text


def test_ingest_rejects_non_http_source(tmp_path):
    record = SourceRecord('local', 'book', 'file:///etc/passwd', 'PD', 'verified', ('train',), decision='eligible')
    with pytest.raises(Exception, match='HTTP'):
        ingest_source(record, root=tmp_path / 'external')


def test_ingest_writes_hashed_file_with_local_server(tmp_path, monkeypatch):
    from pathlib import Path
    import http.server
    import threading
    root = tmp_path / 'external'
    body = b'Eligible public-domain source text for HELIX. ' * 40
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args): pass
    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        record = SourceRecord('local-server', 'book', f'http://127.0.0.1:{server.server_address[1]}/a.txt',
                              'PD', 'verified', ('train',), decision='eligible')
        result = ingest_source(record, root=root, max_total_bytes=len(body) + 1)
        assert result.size_bytes == len(body)
        assert Path(root, result.path).read_bytes() == body
        assert len(result.sha256) == 64
    finally:
        server.shutdown()
