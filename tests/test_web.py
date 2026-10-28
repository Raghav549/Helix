"""Integrated API, origin, consent, and real-checkpoint smoke tests."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
import helix.web as web

REPO = Path(web.PROJECT_ROOT)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'PROJECT_ROOT', str(tmp_path))
    app = web.create_app(
        config_path=str(REPO / 'configs/helix-research.json'),
        llm_checkpoint=str(REPO / 'artifacts/sft/sckpt-best.pt'),
        image_gen_config=str(REPO / 'artifacts/image-gen/config.json'),
        image_gen_checkpoint=str(REPO / 'artifacts/image-gen/ckpt-final.pt'),
        vision_checkpoint=str(REPO / 'artifacts/vision/ckpt-final.pt'),
        caption_dir=str(REPO / 'artifacts/caption'),
    )
    app.testing = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_status_static_and_security_headers(client):
    response = client.get('/')
    assert response.status_code == 200
    for text in (b'HELIX', b'Chat', b'Image Generation', b'Coding Agent', b'Research Lab'):
        assert text in response.data
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert "script-src 'self'" in response.headers['Content-Security-Policy']
    assert client.get('/static/app.js').status_code == 200
    status = client.get('/api/status').get_json()
    assert status['ok'] and status['device'] == 'cpu'
    assert set(status['models']) == {'llm', 'image_gen', 'vision', 'caption'}


def test_origin_restrictions_and_proxy_scheme(client):
    data = {'text': 'never saved', 'consent': True}
    assert client.post('/api/memory', json=data, headers={'Origin': 'https://attacker.example'}).status_code == 403
    assert client.post('/api/memory', json=data, headers={'Origin': 'null'}).status_code == 403
    assert client.post('/api/memory', json=data, headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
    assert client.post('/api/memory', json=data, headers={'Origin': 'https://localhost'}).status_code == 403
    # Proxy-terminated TLS: forwarded scheme and public host must agree.
    r = client.post('/api/memory', json=data, headers={'Host': 'preview.example',
        'Origin': 'https://preview.example', 'X-Forwarded-Proto': 'https'})
    assert r.status_code == 201


def test_memory_consent_edit_delete_and_permissions(client, tmp_path):
    assert client.post('/api/memory', json={'text': 'hello'}).status_code == 400
    assert client.get('/api/memory').get_json()['memories'] == []
    r = client.post('/api/memory', json={'text': 'hello', 'consent': True})
    assert r.status_code == 201
    mid = r.get_json()['id']
    assert client.put(f'/api/memory/{mid}', json={'text': 'changed'}).status_code == 400
    assert client.put(f'/api/memory/{mid}', json={'text': 'changed', 'consent': True}).get_json()['ok']
    assert client.get('/api/memory').get_json()['memories'][0]['text'] == 'changed'
    assert (tmp_path / 'out/memory.sqlite3').stat().st_mode & 0o777 == 0o600
    assert client.delete(f'/api/memory/{mid}').get_json()['ok']
    assert client.get('/api/memory').get_json()['memories'] == []


def test_chat_memory_only_when_opted_in(client, monkeypatch):
    import helix.infer
    prompts = []
    def fake_stream(model, tok, prompt, *args, **kwargs):
        prompts.append(prompt)
        yield 'response'
    monkeypatch.setattr(helix.infer, 'stream_generate', fake_stream)
    monkeypatch.setattr(web.ModelCache, 'llm', lambda self, config, checkpoint: (None, None, SimpleNamespace(train=SimpleNamespace(device='cpu'))))
    client.post('/api/memory', json={'text': 'private-locally-saved-note', 'consent': True})
    assert client.post('/api/chat', json={'message': 'hello'}).status_code == 200
    assert 'private-locally-saved-note' not in prompts[-1]
    assert client.post('/api/chat', json={'message': 'hello', 'use_memory': True}).status_code == 200
    assert 'private-locally-saved-note' in prompts[-1]


def test_approved_execution_and_network_isolation(client):
    assert client.post('/api/code/execute', json={'code': 'result = 42'}).status_code == 403
    response = client.post('/api/code/execute', json={'code': 'result = 7 * 6', 'approved': True})
    assert response.status_code == 200
    result = response.get_json()
    if not result['ok'] and 'isolation unavailable' in (result['error'] or ''):
        pytest.skip('host isolation facilities unavailable; executor failed closed')
    assert result['ok'] and '42' in result['output']
    attack = client.post('/api/code/execute', json={'code': 'import socket\nsocket.socket()', 'approved': True}).get_json()
    assert not attack['ok']
    assert 'PermissionError' in attack['error'] or 'Operation not permitted' in attack['error']


def test_code_analysis_patch_and_private_audit(client, tmp_path):
    secret = 'unique-private-code-39157'
    response = client.post('/api/code/analyze', json={'code': f'def foo():\n    return "{secret}"\n'})
    assert response.status_code == 200
    patch = client.post('/api/code/patch', json={'original': 'a\n', 'modified': 'b\n'}).get_json()
    assert '-a' in patch['diff'] and '+b' in patch['diff']
    audit = (tmp_path / 'out/audit.jsonl').read_text()
    assert secret not in audit and 'code' in audit


def test_math_csv_experiments_and_invalid_inputs(client):
    response = client.post('/api/research/math', json={'operation': 'differentiate',
        'expression': 'x**3 + 2*x', 'variable': 'x', 'save_experiment': True})
    data = response.get_json()
    assert response.status_code == 200 and data['result'] == '3*x**2 + 2' and data['experiment_id']
    assert client.post('/api/research/math', json={'expression': "__import__('os').system('id')"}).status_code == 400
    result = client.post('/api/research/csv', json={'csv': 'a,b\n1,2\n3,4\n'}).get_json()
    assert result['rows'] == 2 and result['column_stats']['a']['mean'] == 2
    experiments = client.get('/api/research/experiments').get_json()['experiments']
    assert any(e['name'] == 'Symbolic calculation' for e in experiments)
    assert client.post('/api/chat', json={'message': '', 'max_new_tokens': 2}).status_code == 400


def test_physics_render_and_file_access(client):
    r = client.post('/api/image/generate', json={'prompt': 'a blue sphere on the left', 'mode': 'physics'})
    assert r.status_code == 200
    result = r.get_json()
    assert result['metadata']['scene']['color'] == 'blue'
    assert client.get(result['image_url']).mimetype == 'image/png'
    assert client.post('/api/image/generate', json={'prompt': 'a forest', 'mode': 'physics'}).status_code == 400
    assert client.get('/out/web_images/../../memory.sqlite3').status_code == 404


@pytest.mark.skipif(not (REPO / 'artifacts/caption/ckpt-best.pt').exists(), reason='trained artifacts not present')
def test_real_models_stream_image_understanding_and_edit(client):
    # Uses committed/released checkpoints; no fake model output or remote service.
    stream = client.post('/api/chat/stream', json={'message': 'What is recursion?', 'max_new_tokens': 6}, buffered=True)
    assert stream.status_code == 200 and b'"done":true' in stream.data
    source = io.BytesIO((REPO / 'artifacts/image-gen/physics-reference.png').read_bytes())
    r = client.post('/api/image/analyze', data={'image': (source, 'reference.png')}, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data['width'] == 32 and 'synthetic_domain_caption' in data and 'not' in data['caption_warning'].lower()
    guided = client.post('/api/image/generate', json={'prompt': 'blue cube on the right', 'mode': 'physics_guided', 'seed': 42})
    assert guided.status_code == 200 and guided.get_json()['metadata']['reference_url']
    source = io.BytesIO((REPO / 'artifacts/image-gen/physics-reference.png').read_bytes())
    edited = client.post('/api/image/edit', data={'image': (source, 'reference.png'), 'prompt': 'a green sphere', 'strength': '.3', 'seed': '4'}, content_type='multipart/form-data')
    assert edited.status_code == 200, edited.get_json()
    assert client.get(edited.get_json()['image_url']).mimetype == 'image/png'


def test_csv_plot_pairs_rows_and_returns_image(client):
    data = {'csv': 'x,y\n1,2\nbad,50\n3,6\n', 'x': 'x', 'y': 'y'}
    result = client.post('/api/research/plot', json=data)
    assert result.status_code == 200 and result.get_json()['points'] == 2
    assert client.get(result.get_json()['image_url']).mimetype == 'image/png'
    assert client.post('/api/research/plot', json={**data, 'x': 'missing'}).status_code == 400


def test_code_test_skeleton_is_not_execution(client):
    result = client.post('/api/code/test-skeleton', json={'code': 'def add(a,b):\n    return a+b\n'})
    assert result.status_code == 200
    assert 'test_add_is_callable' in result.get_json()['tests']
    assert 'not executed' in result.get_json()['notice'].lower()
    assert client.post('/api/code/test-skeleton', json={'code': 'def broken(:'}).status_code == 400


def test_malformed_json_strings_numbers_and_origin_fail_safely(client):
    assert client.post('/api/memory', data=b'{"text":"\\ud800", "consent":true}',
                       content_type='application/json').status_code == 400
    assert client.post('/api/image/generate', data=b'{"prompt":"sphere", "seed":NaN}',
                       content_type='application/json').status_code == 400
    assert client.post('/api/memory', json={'text':'x', 'consent':True},
                       headers={'Origin':'http://['}).status_code == 403
    assert client.get('/api/memory').get_json()['memories'] == []


def test_long_multibyte_image_prompts_are_rejected(client):
    data = {'prompt': 'sphere ' + 'é'*600, 'mode': 'learned'}
    assert client.post('/api/image/generate', json=data).status_code == 400


def test_explicit_external_origin_allows_tls_offload_but_rejects_other_origins(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'PROJECT_ROOT', str(tmp_path))
    public = 'https://8766-example.e2b.app'
    app = web.create_app(external_origin=public)
    client = app.test_client()
    headers = {'Host': '8766-example.e2b.app', 'Origin': public, 'Sec-Fetch-Site': 'same-origin'}
    # Simulates the real preview proxy: internal HTTP and no forwarded scheme.
    assert client.post('/api/code/analyze', json={'code': 'x = 1'}, headers=headers).status_code == 200
    for origin in ('https://attacker.example', 'http://8766-example.e2b.app',
                   'https://8766-example.e2b.app:444', 'https://8766-example.e2b.app/path', 'null'):
        assert client.post('/api/code/analyze', json={'code': 'x = 1'},
                           headers={**headers, 'Origin': origin}).status_code == 403
    assert client.post('/api/code/analyze', json={'code': 'x = 1'},
                       headers={**headers, 'Sec-Fetch-Site': 'cross-site'}).status_code == 403


@pytest.mark.parametrize('origin', ['https://host/path', 'https://host?x=1', 'https://host#fragment',
                                    'ftp://host', 'https://user:password@host', 'https://host:invalid',
                                    'https://[', ''])
def test_invalid_external_origin_is_rejected(origin):
    with pytest.raises(ValueError, match='external_origin'):
        web.create_app(external_origin=origin)
