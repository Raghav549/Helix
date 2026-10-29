"""Regression tests for the production failure: HTML relayed as JSON.

These exercise the browser-facing API contract of the Flask backend:
JSON only, request ids everywhere, honest degraded responses, rate limiting and
correct versioned routing.
"""
import io
import json
from types import SimpleNamespace

import pytest

import helix.infer as infer
import helix.web as web
from helix.web import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'PROJECT_ROOT', str(tmp_path))
    monkeypatch.setattr(web, 'resolve_data_root', lambda *args, **kwargs: tmp_path)
    application = create_app(
        config_path=str(tmp_path / 'missing-config.json'),
        llm_checkpoint=str(tmp_path / 'missing-checkpoint.pt'),
    )
    application.testing = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def _json(response):
    assert response.mimetype == 'application/json', response.mimetype
    return response.get_json()


def test_unknown_api_route_is_json_not_html(client):
    response = client.get('/api/definitely-not-a-route')
    assert response.status_code == 404
    body = _json(response)
    assert body['code'] == 'route_not_found'
    assert body['layer'] == 'helix-backend'
    assert body['request_id']
    assert b'<html' not in response.data.lower()


def test_method_not_allowed_is_json(client):
    response = client.get('/api/chat/stream')
    assert response.status_code == 405
    body = _json(response)
    assert body['code'] == 'method_not_allowed'


def test_every_response_carries_a_request_id_and_is_echoed(client):
    response = client.get('/api/status', headers={'X-Request-ID': 'client-req-1234'})
    assert response.headers['X-Request-ID'] == 'client-req-1234'
    assert _json(response)['request_id'] == 'client-req-1234'
    # An unsafe id must not be reflected into headers or logs.
    spoofed = client.get('/api/status', headers={'X-Request-ID': 'spaces make this invalid'})
    assert spoofed.headers['X-Request-ID'] != 'spaces make this invalid'
    assert len(spoofed.headers['X-Request-ID']) >= 8


def test_versioned_alias_routes_to_the_same_handler(client):
    current = _json(client.get('/api/status'))
    versioned = _json(client.get('/api/v1/status'))
    assert versioned['build'] == current['build']
    assert versioned['capabilities']['chat'] is True
    assert client.get('/api/v1/status').headers['X-API-Version'] == '1'


def test_health_reports_degraded_when_model_artifacts_are_missing(client):
    body = _json(client.get('/api/health'))
    assert body['model_ready'] is False
    assert body['degraded'] is True
    assert body['status'] == 'degraded'
    assert body['degraded_reason']
    assert '/tmp' not in json.dumps(body), 'absolute paths must not leak'
    # Liveness stays available so the platform keeps the service diagnosable.
    assert client.get('/healthz').status_code == 200


def test_chat_without_artifacts_returns_actionable_json_503(client):
    response = client.post('/api/chat', json={'message': 'hello'})
    assert response.status_code == 503
    body = _json(response)
    assert body['code'] == 'model_unavailable'
    assert body['retryable'] is False
    assert 'diagnostics' in body['detail']
    stream = client.post('/api/chat/stream', json={'message': 'hello'})
    assert stream.status_code == 503 and stream.mimetype == 'application/json'


def test_oversized_json_body_is_rejected_as_json(client):
    payload = json.dumps({'message': 'x' * (7 * 1024 * 1024)})
    response = client.post('/api/chat', data=payload, content_type='application/json')
    assert response.status_code == 413
    assert _json(response)['code'] == 'request_too_large'


def test_rate_limit_returns_json_429_with_retry_after(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'resolve_data_root', lambda *args, **kwargs: tmp_path)
    monkeypatch.setenv('HELIX_RATE_LIMIT_PER_MINUTE', '60')
    monkeypatch.setenv('HELIX_RATE_LIMIT_BURST', '2')
    application = create_app(config_path=str(tmp_path / 'missing.json'),
                             llm_checkpoint=str(tmp_path / 'missing.pt'))
    application.testing = True
    client = application.test_client()
    statuses = [client.post('/api/code/analyze', json={'code': 'x = 1'}).status_code for _ in range(4)]
    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429
    limited = client.post('/api/code/analyze', json={'code': 'x = 1'})
    assert limited.mimetype == 'application/json'
    assert _json(limited)['code'] == 'rate_limited'
    assert int(limited.headers['Retry-After']) >= 1


def test_diagnostics_reports_presence_flags_only(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'resolve_data_root', lambda *args, **kwargs: tmp_path)
    monkeypatch.setenv('HELIX_AUTH_PASSWORD', 'do-not-leak-me')
    monkeypatch.setenv('HELIX_MODEL_BUNDLE_URL', 'https://github.com/example/asset.zip')
    application = create_app(config_path=str(tmp_path / 'missing.json'),
                             llm_checkpoint=str(tmp_path / 'missing.pt'))
    application.testing = True
    response = application.test_client().get('/api/diagnostics')
    body = _json(response)
    assert body['environment_present']['HELIX_AUTH_PASSWORD'] is True
    assert 'do-not-leak-me' not in response.get_data(as_text=True)
    assert 'github.com' not in response.get_data(as_text=True)
    assert body['rate_limits']['general_per_minute'] >= 1
    assert 'startup_error' in body and 'storage_writable' in body


def _ready_app(tmp_path, monkeypatch):
    """An app whose artifact files exist, with the model stubbed out."""
    config = tmp_path / 'config.json'
    checkpoint = tmp_path / 'checkpoint.pt'
    config.write_text('{}', encoding='utf-8')
    checkpoint.write_bytes(b'not-a-real-checkpoint')
    monkeypatch.setattr(web, 'resolve_data_root', lambda *args, **kwargs: tmp_path)
    application = create_app(config_path=str(config), llm_checkpoint=str(checkpoint))
    application.testing = True
    return application


def test_streaming_chat_emits_meta_deltas_and_done(tmp_path, monkeypatch):
    class FakeModel:
        def __call__(self, *args, **kwargs):  # pragma: no cover - unused
            raise AssertionError('fake model must not be called directly')

    def fake_llm(self, config, checkpoint):
        return FakeModel(), object(), SimpleNamespace(train=SimpleNamespace(device='cpu'))

    def fake_stream(model, tok, prompt, limit, **kwargs):
        for piece in ('Recursion ', 'is ', 'a function calling itself.'):
            yield piece

    monkeypatch.setattr(web.ModelCache, 'llm', fake_llm)
    monkeypatch.setattr(infer, 'stream_generate', fake_stream)
    client = _ready_app(tmp_path, monkeypatch).test_client()

    response = client.post('/api/chat/stream', json={'message': 'What is recursion?'}, buffered=True)
    assert response.status_code == 200
    assert response.mimetype == 'text/event-stream'
    assert response.headers['Cache-Control'] == 'no-cache, no-transform'
    assert response.headers['X-Accel-Buffering'] == 'no'
    text = response.get_data(as_text=True)
    assert '"meta"' in text and '"done": true' in text
    assert 'Recursion ' in text
    assert '<html' not in text.lower()
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]
    assert events[0]['meta']['model'].startswith('helix')
    assert events[0]['meta']['grounded_in_local_documents'] is False
    assert events[-1]['done'] is True


def test_streaming_generation_failure_is_a_json_error_event(tmp_path, monkeypatch):
    def fake_llm(self, config, checkpoint):
        return object(), object(), SimpleNamespace(train=SimpleNamespace(device='cpu'))

    def failing_stream(*args, **kwargs):
        yield 'partial'
        raise RuntimeError('boom at /srv/helix/secret/path.pt')

    monkeypatch.setattr(web.ModelCache, 'llm', fake_llm)
    monkeypatch.setattr(infer, 'stream_generate', failing_stream)
    client = _ready_app(tmp_path, monkeypatch).test_client()
    response = client.post('/api/chat/stream', json={'message': 'hi'}, buffered=True)
    text = response.get_data(as_text=True)
    assert '"code": "generation_failed"' in text
    assert '/srv/helix' not in text, 'internal paths must not reach the browser'


def test_image_urls_use_the_proxied_api_prefix(client, tmp_path):
    from helix.physics import render_scene, scene_from_prompt
    assert web.PUBLIC_INDEX.endswith('public/index.html')
    image, _ = render_scene(scene_from_prompt('a blue sphere on the left', 96))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    upload = (io.BytesIO(buffer.getvalue()), 'scene.png')
    result = client.post('/api/image/edit', data={'image': upload, 'prompt': 'a red cube'},
                         content_type='multipart/form-data')
    # Model artifacts are absent in this fixture, so either a real edit or a
    # clean 500 JSON is acceptable - but never HTML.
    assert result.mimetype == 'application/json'
    image_url = result.get_json().get('image_url')
    if image_url:
        assert image_url.startswith('/api/out/web_images/')
        assert client.get(image_url).mimetype == 'image/png'
        legacy = image_url.replace('/api', '')
        assert client.get(legacy).mimetype == 'image/png'
    assert client.get('/api/out/web_images/not-a-hash.png').status_code == 404


def test_search_endpoint_is_json_when_unconfigured(client):
    response = client.post('/api/research/search', json={'query': 'attention'})
    assert response.status_code == 503
    body = _json(response)
    assert body['code'] == 'retrieval_not_configured'
    sources = _json(client.get('/api/research/sources'))
    assert sources['configured'] is False


def test_search_endpoint_queries_a_configured_index(tmp_path, monkeypatch):
    from helix.retrieval import RetrievalIndex
    fixtures = web.Path(__file__).parent / 'fixtures' / 'retrieval'
    index = RetrievalIndex.ingest(fixtures, chunk_chars=240, overlap=40)
    index.save(tmp_path / 'retrieval')
    monkeypatch.setattr(web, 'resolve_data_root', lambda *args, **kwargs: tmp_path)
    application = create_app(config_path=str(tmp_path / 'missing.json'),
                             llm_checkpoint=str(tmp_path / 'missing.pt'),
                             retrieval_root=str(tmp_path / 'retrieval'))
    application.testing = True
    client = application.test_client()
    response = client.post('/api/research/search', json={'query': 'causal attention mask', 'k': 3})
    body = _json(response)
    assert body['results'][0]['source'] == 'transformer-notes.md'
    assert body['results'][0]['citation'].startswith('transformer-notes.md#')
    assert body['index']['documents'] == 3
    assert 'not a semantic model' in body['notice']
    sources = _json(client.get('/api/research/sources'))
    assert sources['configured'] is True and len(sources['documents']) == 3


def test_status_never_claims_unimplemented_capabilities(client):
    caps = _json(client.get('/api/status'))['capabilities']
    assert caps['web_search'] is False
    assert caps['training_jobs'] is False
    assert isinstance(caps['retrieval'], bool)


def test_client_disconnect_releases_the_compute_lock(tmp_path, monkeypatch):
    """A browser that stops mid-stream must not wedge the single compute slot."""

    def fake_llm(self, config, checkpoint):
        return object(), object(), SimpleNamespace(train=SimpleNamespace(device='cpu'))

    def endless_stream(*args, **kwargs):
        for index in range(200):
            yield f'token{index} '

    monkeypatch.setattr(web.ModelCache, 'llm', fake_llm)
    monkeypatch.setattr(infer, 'stream_generate', endless_stream)
    client = _ready_app(tmp_path, monkeypatch).test_client()

    response = client.post('/api/chat/stream', json={'message': 'long answer please'}, buffered=False)
    assert response.status_code == 200
    iterator = iter(response.response)
    first = next(iterator)
    assert b'meta' in first or b'delta' in first
    iterator.close()          # the client disconnects here
    response.close()

    # The lock must be free: a new request is served instead of a 429.
    second = client.post('/api/chat/stream', json={'message': 'again'}, buffered=True)
    assert second.status_code == 200, second.get_data(as_text=True)
    assert '"done": true' in second.get_data(as_text=True)
