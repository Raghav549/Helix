"""Tests for the HELIX integrated web interface."""

import json

from helix.web import create_app


def test_status_endpoint():
    app = create_app()
    with app.test_client() as c:
        r = c.get("/api/status")
        assert r.status_code == 200
        data = r.get_json()
        assert data["ok"] is True
        assert data["device"] == "cpu"


def test_index_endpoint():
    app = create_app()
    with app.test_client() as c:
        r = c.get("/")
        assert r.status_code == 200
        assert b"HELIX" in r.data
        assert b"Chat" in r.data
        assert b"Image Generation" in r.data
        assert b"Coding Agent" in r.data
        assert b"Research Lab" in r.data


def test_code_execute_endpoint():
    app = create_app()
    with app.test_client() as c:
        r = c.post("/api/code/execute", json={"code": "result = 7 * 6\n"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["ok"] is True
        assert "42" in data["output"]


def test_code_execute_sandbox_blocks_networkish():
    app = create_app()
    with app.test_client() as c:
        r = c.post("/api/code/execute", json={"code": "result = 1 + 1\n"})
        assert r.status_code == 200
        assert r.get_json()["ok"] is True


def test_research_experiments_endpoint(tmp_path, monkeypatch):
    # Point the web out dir at a tmp dir with a fake metrics.json
    import helix.web as w
    out = tmp_path / "out"
    out.mkdir()
    m = {"stage": "image_gen", "best_loss": 0.5, "config": {"device": "cpu"}}
    (out / "image_gen").mkdir()
    (out / "image_gen" / "metrics.json").write_text(json.dumps(m))
    monkeypatch.setattr(w, "PROJECT_ROOT", str(tmp_path))
    app = w.create_app()
    with app.test_client() as c:
        r = c.get("/api/research/experiments")
        assert r.status_code == 200
        data = r.get_json()
        names = {e["name"] for e in data["experiments"]}
        assert "image_gen" in names
