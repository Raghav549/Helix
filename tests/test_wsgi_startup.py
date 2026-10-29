"""Startup behaviour of the production WSGI entrypoint.

The production incident started with the backend refusing to start at all
(import-time bundle verification), which made the platform serve HTML. These
tests pin the degraded-but-reachable behaviour and the integrity checks.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(code: str, env: dict, cwd: Path = REPO) -> subprocess.CompletedProcess:
    full_env = {**os.environ, 'PYTHONPATH': str(REPO), **env}
    full_env.pop('HELIX_REQUIRE_AUTH', None)
    return subprocess.run([sys.executable, '-c', code], cwd=str(cwd), env=full_env,
                          capture_output=True, text=True, timeout=180)


def test_missing_bundle_starts_degraded_and_stays_diagnosable(tmp_path):
    env = {'HELIX_DATA_ROOT': str(tmp_path / 'data')}
    result = _run(
        "import json\n"
        "from helix.wsgi import app\n"
        "c = app.test_client()\n"
        "h = c.get('/healthz')\n"
        "chat = c.post('/api/chat', json={'message': 'hello'})\n"
        "stream = c.post('/api/chat/stream', json={'message': 'hello'})\n"
        "print(json.dumps({\n"
        "  'health_status': h.status_code, 'health': h.get_json(),\n"
        "  'chat_status': chat.status_code, 'chat': chat.get_json(),\n"
        "  'chat_type': chat.mimetype,\n"
        "  'stream_status': stream.status_code, 'stream_type': stream.mimetype,\n"
        "}))\n", env)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload['health_status'] == 200
    assert payload['health']['degraded'] is True
    assert payload['health']['model_ready'] is False
    assert 'manifest' in payload['health']['degraded_reason'].lower()
    assert payload['chat_status'] == 503
    assert payload['chat_type'] == 'application/json'
    assert payload['chat']['code'] == 'model_unavailable'
    assert payload['stream_type'] == 'application/json'
    assert payload['stream_status'] == 503
    assert str(tmp_path) not in json.dumps(payload), 'absolute paths must not leak'


def _write_bundle(root: Path, corrupt: bool = False) -> None:
    files = {
        'artifacts/sft/sckpt-best.pt': b'weights-bytes',
        'artifacts/data/vocab.json': b'{"vocab": 1}',
        'artifacts/data/merges.json': b'[]',
        'configs/helix-deploy.json': b'{"model": {}}',
    }
    records = {}
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    if corrupt:
        records['artifacts/data/merges.json']['sha256'] = '0' * 64
    manifest = root / 'artifacts/model-bundle-manifest.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'format': 1, 'files': records}), encoding='utf-8')


def _stage(repo: Path, data: Path) -> tuple:
    from helix import wsgi

    original_repo, original_root = wsgi.REPOSITORY_ROOT, wsgi.resolve_data_root
    try:
        wsgi.REPOSITORY_ROOT = str(repo)
        wsgi.resolve_data_root = lambda *args, **kwargs: data
        staged = wsgi._stage_verified_model_artifacts()
        return staged, None
    except Exception as exc:  # noqa: BLE001 - the caller asserts on the message
        return None, str(exc)
    finally:
        wsgi.REPOSITORY_ROOT, wsgi.resolve_data_root = original_repo, original_root


def test_verified_bundle_is_staged_into_runtime_storage(tmp_path):
    repo = tmp_path / 'repo'
    _write_bundle(repo)
    data = tmp_path / 'data'
    data.mkdir()
    staged, error = _stage(repo, data)
    assert error is None, error
    assert (staged / 'artifacts/sft/sckpt-best.pt').read_bytes() == b'weights-bytes'
    assert (staged / 'configs/helix-deploy.json').is_file()


def test_corrupt_bundle_is_rejected(tmp_path):
    repo = tmp_path / 'repo'
    _write_bundle(repo, corrupt=True)
    data = tmp_path / 'data'
    data.mkdir()
    staged, error = _stage(repo, data)
    assert staged is None
    assert 'integrity check failed' in error
    assert str(tmp_path) not in error, 'the error message must be path-free'


def test_wsgi_source_contains_no_hardcoded_credentials():
    source = (REPO / 'helix/wsgi.py').read_text(encoding='utf-8')
    assert 'password' not in source.lower() or 'redacted' in source.lower()
    assert 'sync: false' not in source
