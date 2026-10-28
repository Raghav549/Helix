"""Tests for the HELIX coding agent."""

import os
import tempfile

import pytest

from helix.coding import (RepoInspector, StaticAnalyzer, PatchGenerator,
                          TestGenerator, CodingAgent, demo_code_fix)


def test_static_analysis_python():
    src = '''import os
import sys as s

def add(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    """A calculator."""

    def mul(self, a, b):
        if a > 0:
            return a * b
        return 0

# TODO: finish this
x = 1
'''
    a = StaticAnalyzer(src, "example.py").analyze()
    assert a.file == "example.py"
    kinds = {s.kind for s in a.symbols}
    assert "function" in kinds and "class" in kinds and "import" in kinds
    funcs = [s for s in a.symbols if s.kind == "function"]
    assert any(s.name == "add" for s in funcs)
    assert a.todo_count >= 1
    assert a.complexity >= 2


def test_static_analysis_syntax_error():
    a = StaticAnalyzer("def foo(\n", "bad.py").analyze()
    assert a.has_syntax_error()
    assert a.syntax_error is not None


def test_patch_generator():
    pg = PatchGenerator()
    orig = "line one\nline two\nline three\n"
    mod = "line one\nline two EDITED\nline three\n"
    p = pg.make_patch("f.py", orig, mod)
    assert "f.py" in p.diff
    assert "EDITED" in p.diff


def test_test_generator():
    tg = TestGenerator()
    src = "def add(a, b):\n    return a + b\n"
    out = tg.generate_tests("example.py", src)
    assert "import unittest" in out
    assert "add" in out


def test_repo_inspector(tmp_path):
    (tmp_path / "a.py").write_text("print('hi')\n")
    (tmp_path / "b.md").write_text("# readme\n")
    (tmp_path / "hidden.txt").write_text("x\n")
    ri = RepoInspector(str(tmp_path))
    files = ri.list_files()
    assert "a.py" in files and "b.md" in files
    info = ri.file_info("a.py")
    assert info is not None and info.language == "python"
    content = ri.read_file("a.py")
    assert content == "print('hi')\n"


def test_coding_agent_execute():
    agent = CodingAgent(root=tempfile.gettempdir())
    res = agent.execute_code("result = 21 * 2\n", approved=True)
    if not res.ok and res.error and "isolation unavailable" in res.error:
        pytest.skip("host does not permit the fail-closed Linux sandbox")
    assert res.ok is True and "42" in res.output


def test_demo_code_fix():
    out = demo_code_fix("result = 10 / 2\n", approved=True)
    if out.get("execution", {}).get("error") and "isolation unavailable" in out["execution"]["error"]:
        pytest.skip("host does not permit the fail-closed Linux sandbox")
    assert "analysis" in out and "execution" in out
    assert out["execution"]["ok"] is True
    assert "5.0" in out["execution"]["output"]
