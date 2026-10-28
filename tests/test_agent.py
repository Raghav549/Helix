"""Tests for the HELIX agent system (deterministic tool + loop tests)."""

import os
import tempfile
import time

import torch

from helix.agent import (
    Agent, ToolRegistry, CalculatorTool, PythonTool, DocumentTool,
    ToolResult,
)
from helix.config import ModelConfig
from helix.model import HelixTransformer
from helix.tokenizer import ByteLevelBPETokenizer


def test_calculator():
    calc = CalculatorTool()
    r = calc.execute(expression="2 + 3 * 4")
    assert r.ok and r.output == "14"
    r = calc.execute(expression="2 ** 10")
    assert r.ok and r.output == "1024"
    r = calc.execute(expression="__import__('os')")
    assert not r.ok  # disallowed node


def test_python_sandbox_computes_result():
    py = PythonTool(timeout_s=10.0, memory_mb=128)
    r = py.execute(code="result = 40 + 2\n")
    assert r.ok and "42" in r.output, r.output
    assert r.duration_s >= 0


def test_python_sandbox_timeout():
    py = PythonTool(timeout_s=1.0, memory_mb=128)
    r = py.execute(code="import time; time.sleep(10)\n")
    assert not r.ok and r.error == "timeout"


def test_document_tool():
    docs = {"a": "The helix is a protein.", "b": "Python is a programming language."}
    doc = DocumentTool(docs)
    r = doc.execute(query="helix")
    assert r.ok and "helix" in r.output.lower() and "[a]" in r.output
    r = doc.execute(query="nothing")
    assert r.ok and "no matching" in r.output


def test_tool_registry():
    reg = ToolRegistry.default_registry()
    assert "calculator" in reg.names() and "python" in reg.names()
    assert reg.get("calculator") is not None
    assert reg.get("nonexistent") is None
    desc = reg.descriptions()
    assert "calculator" in desc and "python" in desc


def _tiny_agent_model_and_tok():
    tok = ByteLevelBPETokenizer()
    tok.train(["hello world"], vocab_size=200)
    model = HelixTransformer(ModelConfig(
        vocab_size=len(tok), block_size=32, n_layer=1, n_head=2, n_embd=32,
        n_kv_head=1, dropout=0.0,
    )).eval()
    return model, tok


def test_agent_loop_executes_preapproved_tool(monkeypatch):
    model, tok = _tiny_agent_model_and_tok()
    reg = ToolRegistry.default_registry()
    agent = Agent(model, tok, registry=reg, device="cpu", max_steps=4,
                  max_time_s=30.0, require_approval=True)
    # Pre-approve the python tool (consequential) so the gate passes.
    agent.set_tool_approved("python", True, "result = 40 + 2")

    # Monkeypatch _plan to force a python call then an answer.
    calls = {"n": 0}

    def fake_plan(goal, history):
        calls["n"] += 1
        if calls["n"] == 1:
            return "PLAN: compute 40+2\nCALL: python result = 40 + 2\n"
        return "ANSWER: 42\n"

    monkeypatch.setattr(agent, "_plan", fake_plan)
    run = agent.run("compute 40+2")
    assert run.stopped_reason == "completed"
    assert "42" in run.final_answer
    assert len(run.steps) == 2
    assert run.steps[0].tool == "python" and run.steps[0].result.ok
    assert run.steps[0].approved is True
    # Audit log recorded the step.
    assert len(agent.audit_log) == 2
    assert agent.audit_log[0]["status"] == "ok"


def test_agent_rejects_unapproved_consequential_tool(monkeypatch):
    model, tok = _tiny_agent_model_and_tok()
    reg = ToolRegistry.default_registry()
    agent = Agent(model, tok, registry=reg, device="cpu", max_steps=4,
                  max_time_s=30.0, require_approval=True)
    # Do NOT pre-approve python -> consequential tool rejected.

    def fake_plan(goal, history):
        return "CALL: python result = 1 + 1\n"

    monkeypatch.setattr(agent, "_plan", fake_plan)
    run = agent.run("compute")
    # The step was rejected; model keeps being asked until max_steps, then stops.
    assert run.stopped_reason in ("max_steps", "no_tool", "completed")
    rejected = [s for s in run.steps if s.tool == "python"]
    assert rejected and not rejected[0].approved
    assert rejected[0].result is not None and rejected[0].result.error == "rejected by approval gate"


def test_agent_unknown_tool(monkeypatch):
    model, tok = _tiny_agent_model_and_tok()
    reg = ToolRegistry.default_registry()
    agent = Agent(model, tok, registry=reg, device="cpu", max_steps=3,
                  max_time_s=30.0, require_approval=True)

    def fake_plan(goal, history):
        return "CALL: nosuchtool do something\n"

    monkeypatch.setattr(agent, "_plan", fake_plan)
    run = agent.run("x")
    assert run.steps and run.steps[0].result is not None
    assert "unknown tool" in run.steps[0].result.error
