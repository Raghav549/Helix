"""Knowledge Mill ingestion: provenance, dedupe and retrieval boundaries."""
from __future__ import annotations

import pytest

from helix.knowledge import KnowledgeError, KnowledgeMill


def test_text_is_persisted_deduplicated_and_retrievable(tmp_path):
    mill = KnowledgeMill(tmp_path / 'knowledge')
    first = mill.ingest(b'Wheat is milled into flour after cleaning and grinding.', 'mill.txt',
                        'text/plain', source_kind='text', source_url=None,
                        submitted_by='agent-1', license_note='user-submitted')
    again = mill.ingest(b'Wheat is milled into flour after cleaning and grinding.', 'copy.txt',
                        'text/plain', source_kind='text', source_url=None, submitted_by='agent-1')
    assert first['indexed'] is True
    assert again['deduplicated'] is True
    status = mill.status()
    assert status['configured'] is True and status['documents'] == 1
    assert mill.list_documents()[0]['license_note'] == 'user-submitted'


def test_images_are_recorded_but_not_misrepresented_as_text_retrieval(tmp_path):
    mill = KnowledgeMill(tmp_path / 'knowledge')
    record = mill.ingest(b'not-decoded-here', 'scene.png', 'image/png', source_kind='upload',
                         source_url=None, submitted_by='agent-1')
    assert record['indexed'] is False
    assert record['state'] == 'stored_for_image_analysis'
    assert mill.status()['documents'] == 0


def test_local_network_urls_are_refused_before_fetch(monkeypatch):
    monkeypatch.setattr('helix.knowledge._private_host', lambda host: True)
    with pytest.raises(KnowledgeError, match='Private'):
        KnowledgeMill.fetch_url('http://127.0.0.1/private.txt')
