"""Retrieval ingestion, ranking and citation tests."""
import json
from pathlib import Path

import pytest

from helix.retrieval import RetrievalIndex, _split, _tokens

FIXTURE_ROOT = Path(__file__).parent / 'fixtures' / 'retrieval'


@pytest.fixture(scope='module')
def index() -> RetrievalIndex:
    return RetrievalIndex.ingest(FIXTURE_ROOT, chunk_chars=240, overlap=40)


def test_ingest_reports_documents_chunks_and_stable_checksum(index):
    assert index.document_count == 3
    assert index.chunk_count >= 3
    again = RetrievalIndex.ingest(FIXTURE_ROOT, chunk_chars=240, overlap=40)
    assert again.checksum == index.checksum, 'ingestion must be deterministic'
    sources = {document['source'] for document in index.documents}
    assert sources == {'transformer-notes.md', 'evaluation-policy.html', 'forecasting-baseline.txt'}
    for document in index.documents:
        assert document['sha256'] and document['bytes'] > 0


def test_html_ingestion_strips_markup_and_scripts(index):
    html_docs = [document for document in index.documents if document['source'].endswith('.html')]
    assert html_docs
    text = ' '.join(chunk.text for chunk in index.chunks if chunk.source.endswith('.html'))
    assert 'Held-out evaluation data' in text
    assert '<p>' not in text and '<script' not in text
    assert 'console.log' not in text


def test_search_ranks_the_relevant_document_first(index):
    results = index.search('causal attention mask softmax', k=3)
    assert results
    assert results[0].source == 'transformer-notes.md'
    assert 'mask' in results[0].text.lower()
    assert results[0].score > 0
    assert results[0].citation.startswith('transformer-notes.md#chars=')
    assert results[0].to_dict()['citation'] == results[0].citation


def test_search_cites_page_and_offsets_and_returns_empty_for_noise(index):
    results = index.search('forecast holdout error interval', k=2)
    assert results and results[0].source == 'forecasting-baseline.txt'
    payload = results[0].to_dict()
    assert payload['start'] < payload['end']
    assert payload['snippet']
    assert 'linear trend' in ' '.join(chunk.text.lower() for chunk in index.chunks)
    assert index.search('', k=3) == []
    assert index.search('zzzzqqqwwww', k=3) == []


def test_results_are_capped_by_k_and_ordered_by_score(index):
    results = index.search('mask position token model evaluation', k=2)
    assert len(results) <= 2
    scores = [chunk.score for chunk in results]
    assert scores == sorted(scores, reverse=True)


def test_save_load_round_trip_preserves_ranking(index, tmp_path):
    path = index.save(tmp_path)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['checksum'] == index.checksum
    assert payload['embedding']['kind'] == 'hashed-bag-of-words'
    assert 'embedding' not in payload['chunks'][0]
    loaded = RetrievalIndex.load(tmp_path)
    assert loaded.checksum == index.checksum
    original = [chunk.doc_id + str(chunk.start) for chunk in index.search('attention mask', k=3)]
    restored = [chunk.doc_id + str(chunk.start) for chunk in loaded.search('attention mask', k=3)]
    assert original == restored


def test_evidence_block_is_numbered_capped_and_attributes_sources(index):
    results = index.search('attention mask evaluation policy', k=3)
    evidence = index.build_evidence(results, max_chars=500)
    assert evidence.startswith('[1] ')
    assert 'transformer-notes.md#chars=' in evidence or 'evaluation-policy.html#chars=' in evidence
    assert len(evidence) <= 500
    assert evidence.count('[') >= 1


def test_ingest_rejects_invalid_options(tmp_path):
    with pytest.raises(ValueError):
        RetrievalIndex.ingest(tmp_path, chunk_chars=10)
    with pytest.raises(ValueError):
        RetrievalIndex.ingest(tmp_path, chunk_chars=200, overlap=150)
    with pytest.raises(ValueError):
        RetrievalIndex.ingest(tmp_path / 'missing')


def test_ingest_ignores_unsupported_and_hidden_files(tmp_path):
    (tmp_path / 'keep.md').write_text('Durable content about retrieval ranking and citations. ' * 6, encoding='utf-8')
    (tmp_path / 'binary.bin').write_bytes(b'\x00\x01\x02' * 100)
    (tmp_path / '.hidden.md').write_text('secret ' * 50, encoding='utf-8')
    index = RetrievalIndex.ingest(tmp_path)
    assert index.document_count == 1
    assert index.documents[0]['source'] == 'keep.md'


def test_load_rejects_missing_or_foreign_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        RetrievalIndex.load(tmp_path)
    (tmp_path / 'index.json').write_text('{"format": 99, "chunks": []}', encoding='utf-8')
    with pytest.raises(ValueError):
        RetrievalIndex.load(tmp_path)


def test_chunk_split_and_tokenizer_helpers():
    text = 'Sentence one. ' * 80
    spans = _split(text, 200, 20)
    assert spans[0][0] == 0
    assert all(end > start for start, end in spans)
    assert spans[-1][1] == len(text)
    assert _tokens('The Attention MASK, and the model!') == ['attention', 'mask', 'model']


def test_ingest_can_exclude_relative_directories(tmp_path):
    from helix.retrieval import RetrievalIndex

    (tmp_path / 'keep.md').write_text('# Kept\n\nhelium balloon pressure valve calibration notes.\n', encoding='utf-8')
    reports = tmp_path / 'reports'
    reports.mkdir()
    (reports / 'ignored.md').write_text('# Ignored\n\nhelium balloon pressure valve calibration notes.\n', encoding='utf-8')

    everything = RetrievalIndex.ingest(tmp_path)
    filtered = RetrievalIndex.ingest(tmp_path, exclude=['reports'])
    assert everything.document_count == 2
    assert filtered.document_count == 1
    assert all(not chunk.source.startswith('reports/') for chunk in filtered.chunks)
    # Exclusion must not disturb the kept document's checksum stability.
    assert filtered.search('helium balloon calibration')
