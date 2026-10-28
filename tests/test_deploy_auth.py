"""Hosted deployment authentication tests."""
import base64

import pytest

from helix.web import create_app


def _basic(user, password):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_required_auth_fails_closed_when_credentials_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HELIX_REQUIRE_AUTH", "1")
    monkeypatch.delenv("HELIX_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("HELIX_AUTH_PASSWORD", raising=False)
    monkeypatch.setattr("helix.web.resolve_data_root", lambda *args, **kwargs: tmp_path)
    with pytest.raises(RuntimeError, match="credentials are missing"):
        create_app(config_path="/tmp/config.json", llm_checkpoint="/tmp/checkpoint.pt")


def test_basic_auth_protects_ui_and_api_but_not_health(monkeypatch, tmp_path):
    monkeypatch.setenv("HELIX_REQUIRE_AUTH", "true")
    monkeypatch.setenv("HELIX_AUTH_USERNAME", "helix-test")
    monkeypatch.setenv("HELIX_AUTH_PASSWORD", "not-a-production-secret")
    monkeypatch.setattr("helix.web.resolve_data_root", lambda *args, **kwargs: tmp_path)
    app = create_app(config_path="/tmp/config.json", llm_checkpoint="/tmp/checkpoint.pt")
    app.testing = True
    client = app.test_client()

    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 401
    denied = client.get("/api/status")
    assert denied.status_code == 401
    assert denied.headers["WWW-Authenticate"].startswith("Basic realm=\"HELIX\"")
    assert client.get("/", headers=_basic("helix-test", "wrong")).status_code == 401
    allowed = client.get("/api/status", headers=_basic("helix-test", "not-a-production-secret"))
    assert allowed.status_code == 200
    assert client.get("/", headers=_basic("helix-test", "not-a-production-secret")).status_code == 200
