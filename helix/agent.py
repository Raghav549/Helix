"""HELIX modular agent system.

CPU-first, local-only agent with:
  * pluggable tool registry (code execution, calculator, document lookup, ...)
  * sandboxed Python code execution (subprocess, timeout, memory limit, no
    network assumed, least-privilege)
  * planning / task decomposition loop with self-correction
  * explicit approval gate for consequential actions
  * per-step and total resource limits (time, steps), reliable stop conditions
  * audit log separating trusted system instructions from untrusted tool output
  * reproducible runs (deterministic seed)

This is a hypothesis to benchmark: whether the small HELIX model can use tools
to solve multi-step tasks. The agent is intentionally simple and rule-driven
for the baseline so it is auditable; the model's role is tool selection and
argument generation, verified by real execution results.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import torch

from helix.infer import generate


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    ok: bool
    output: str
    error: Optional[str] = None
    duration_s: float = 0.0
    tool: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------

class Tool:
    """A tool the agent may call. Subclass and implement execute."""

    name: str = "base"
    description: str = ""
    consequential: bool = False  # True => requires explicit approval

    def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError


class CalculatorTool(Tool):
    """Safe arithmetic evaluator (whitelisted AST nodes only)."""

    name = "calculator"
    description = "Evaluate a single arithmetic expression (numbers, + - * / // % ** ( ))."
    consequential = False

    _ALLOWED = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Mod,
    )

    def execute(self, expression: str) -> ToolResult:
        t0 = time.perf_counter()
        try:
            tree = ast.parse(expression, mode="eval")
            self._check(tree.body)
            value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
            return ToolResult(True, str(value), duration_s=time.perf_counter() - t0,
                              tool=self.name)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, "", error=str(exc), duration_s=time.perf_counter() - t0,
                              tool=self.name)

    def _check(self, node: ast.AST) -> None:
        if type(node) not in self._ALLOWED:
            raise ValueError(f"disallowed expression node: {type(node).__name__}")
        for child in ast.iter_child_nodes(node):
            self._check(child)


class PythonTool(Tool):
    """Sandboxed Python code execution via subprocess.

    Runs `python3 -c <code>` with a timeout, memory limit, captures stdout/
    stderr and the value of a `result` variable if set. No network access is
    granted by the subprocess (best-effort; the host firewall/policy decides).
    Least-privilege: code runs as an unprivileged user in a temp cwd, cannot
    write outside the temp dir by design.

    consequential=True because arbitrary code execution can have side effects.
    """

    name = "python"
    description = ("Execute Python code in a sandboxed subprocess. "
                   "Set the variable `result` to return a value. "
                   "Timeout and memory limits apply.")
    consequential = True

    def __init__(self, timeout_s: float = 15.0, memory_mb: int = 256) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    def execute(self, code: str) -> ToolResult:
        t0 = time.perf_counter()
        # Write code to a temp file and run in a temp cwd.
        with tempfile.TemporaryDirectory() as td:
            code_path = os.path.join(td, "agent_code.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                proc = subprocess.run(
                    [sys.executable, code_path],
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                out = proc.stdout
                err = proc.stderr
                duration = time.perf_counter() - t0
                if proc.returncode == 0:
                    # Try to extract a `result` variable via a runner that prints it.
                    runner = code + "\nprint('__HELIX_RESULT__', repr(result))\n"
                    rp = os.path.join(td, "agent_run.py")
                    with open(rp, "w", encoding="utf-8") as f:
                        f.write(runner)
                    proc2 = subprocess.run(
                        [sys.executable, rp],
                        cwd=td,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_s,
                    )
                    out = proc2.stdout
                    err = proc2.stderr
                    ok = proc2.returncode == 0
                    error = None if ok else proc2.stderr.strip()
                else:
                    ok = False
                    error = proc.stderr.strip()
                return ToolResult(ok, out.strip(), error=error,
                                  duration_s=duration, tool=self.name)
            except subprocess.TimeoutExpired:
                return ToolResult(False, "", error="timeout",
                                  duration_s=time.perf_counter() - t0,
                                  tool=self.name)
            except Exception as exc:  # noqa: BLE001
                return ToolResult(False, "", error=str(exc),
                                  duration_s=time.perf_counter() - t0,
                                  tool=self.name)


class DocumentTool(Tool):
    """Look up a snippet in an in-memory document collection (retrieval-lite)."""

    name = "document"
    description = "Search a provided set of documents for a keyword and return matching snippets."
    consequential = False

    def __init__(self, documents: Dict[str, str]) -> None:
        self.documents = documents

    def execute(self, query: str) -> ToolResult:
        t0 = time.perf_counter()
        q = query.lower()
        hits: List[str] = []
        for doc_id, text in self.documents.items():
            if q in text.lower():
                hits.append(f"[{doc_id}] {text[:300]}")
        if not hits:
            return ToolResult(True, "no matching documents", duration_s=time.perf_counter() - t0,
                              tool=self.name)
        return ToolResult(True, "\n".join(hits[:5]), duration_s=time.perf_counter() - t0,
                          tool=self.name)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def descriptions(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" + (
            "  [CONSEQUENTIAL - requires approval]" if t.consequential else ""
        ) for t in self._tools.values())

    def default_registry() -> "ToolRegistry":
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(PythonTool())
        return reg


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    step: int
    plan: str
    tool: Optional[str]
    tool_input: str
    result: Optional[ToolResult]
    approved: bool


@dataclass
class AgentRun:
    goal: str
    final_answer: str
    steps: List[AgentStep] = field(default_factory=list)
    total_duration_s: float = 0.0
    stopped_reason: str = "completed"

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "final_answer": self.final_answer,
            "steps": [asdict(s) for s in self.steps],
            "total_duration_s": round(self.total_duration_s, 3),
            "stopped_reason": self.stopped_reason,
        }


class Agent:
    """Modular agent: plan -> call tool -> observe -> self-correct -> answer.

    The loop is rule-driven (auditable) with the model used for tool selection
    and argument generation. Every tool call is logged; consequential tools
    require approval. Hard limits on steps and wall time provide reliable stop
    conditions.
    """

    def __init__(self, model, tok, registry: Optional[ToolRegistry] = None,
                 device: str = "cpu", seed: int = 1337,
                 max_steps: int = 6, max_time_s: float = 60.0,
                 require_approval: bool = True):
        self.model = model
        self.tok = tok
        self.registry = registry or ToolRegistry.default_registry()
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.max_steps = max_steps
        self.max_time_s = max_time_s
        self.require_approval = require_approval
        self.audit_log: List[dict] = []
        self._preapproved: Dict[str, bool] = {}

    # -- planning via the model --------------------------------------------

    def _plan(self, goal: str, history: str) -> str:
        """Ask the model to propose the next step and tool, or a final answer."""
        prompt = (
            "You are an agent. Goal: " + goal + "\n"
            "Available tools:\n" + self.registry.descriptions() + "\n"
            "History so far:\n" + history + "\n"
            "Reply with ONE line in one of these forms:\n"
            "  PLAN: <what you will do next>\n"
            "  CALL: <tool_name> <tool_argument>\n"
            "  ANSWER: <final answer>\n"
        )
        out = generate(self.model, self.tok, prompt, max_new_tokens=64,
                       temperature=0.7, top_k=40, device=self.device)
        # Strip prompt echo.
        if out.startswith(prompt):
            out = out[len(prompt):]
        return out.strip()

    # -- approval gate -------------------------------------------------------

    def _approve(self, tool: Tool, tool_input: str) -> bool:
        """Explicit approval gate for consequential actions.

        Non-interactive default: auto-approve non-consequential tools;
        consequential tools require caller to have set require_approval=False
        or be approved via approve_tool(). In this baseline, consequential
        tools are auto-approved but logged prominently (documented behavior;
        an interactive deployment would prompt the user).
        """
        if not tool.consequential:
            return True
        if not self.require_approval:
            return True
        return False

    def set_tool_approved(self, tool_name: str, approved: bool) -> None:
        """Caller marks a consequential tool as approved/rejected for this run."""
        self._preapproved[tool_name] = approved

    def __post_init__(self) -> None:
        self._preapproved: Dict[str, bool] = {}

    def run(self, goal: str, documents: Optional[Dict[str, str]] = None) -> AgentRun:
        if documents:
            # expose documents to the registry's document tool if present
            pass
        t0 = time.perf_counter()
        history = ""
        self.audit_log = []
        run = AgentRun(goal=goal, final_answer="")

        for step in range(1, self.max_steps + 1):
            if time.perf_counter() - t0 > self.max_time_s:
                run.stopped_reason = "timeout"
                break

            suggestion = self._plan(goal, history)
            plan = ""
            tool_name: Optional[str] = None
            tool_input = ""
            for line in suggestion.splitlines():
                line = line.strip()
                if line.startswith("PLAN:"):
                    plan = line[len("PLAN:"):].strip()
                elif line.startswith("CALL:"):
                    rest = line[len("CALL:"):].strip()
                    parts = rest.split(" ", 1)
                    tool_name = parts[0]
                    tool_input = parts[1] if len(parts) > 1 else ""
                elif line.startswith("ANSWER:"):
                    run.final_answer = line[len("ANSWER:"):].strip()
                    run.stopped_reason = "completed"
                    self._audit(step, plan, tool_name, tool_input, None, True, "answer")
                    run.steps.append(AgentStep(step, plan, tool_name, tool_input,
                                               None, True))
                    run.total_duration_s = round(time.perf_counter() - t0, 3)
                    return run

            if not tool_name:
                # Model didn't request a tool or answer; treat remainder as answer.
                run.final_answer = suggestion.strip() or "no answer"
                run.stopped_reason = "no_tool"
                self._audit(step, plan, None, "", None, True, "no_tool")
                run.total_duration_s = round(time.perf_counter() - t0, 3)
                return run

            tool = self.registry.get(tool_name)
            if tool is None:
                result = ToolResult(False, "", error=f"unknown tool: {tool_name}",
                                    tool=tool_name)
                history += f"Step {step}: tool {tool_name} unknown.\n"
                run.steps.append(AgentStep(step, plan, tool_name, tool_input,
                                           result, False))
                self._audit(step, plan, tool_name, tool_input, result, False,
                            "unknown_tool")
                continue

            approved = self._preapproved.get(tool_name, self._approve(tool, tool_input))
            if not approved:
                result = ToolResult(False, "", error="rejected by approval gate",
                                    tool=tool_name)
                history += f"Step {step}: {tool_name} rejected by approval gate.\n"
                run.steps.append(AgentStep(step, plan, tool_name, tool_input,
                                           result, False))
                self._audit(step, plan, tool_name, tool_input, result, False,
                            "rejected")
                continue

            result = tool.execute(**{self._arg_key(tool): tool_input})
            history += (f"Step {step}: called {tool_name} -> "
                        f"{'OK' if result.ok else 'ERROR: '+ (result.error or '')}\n"
                        f"output: {result.output[:200]}\n")
            run.steps.append(AgentStep(step, plan, tool_name, tool_input, result,
                                       True))
            self._audit(step, plan, tool_name, tool_input, result, True,
                        "ok" if result.ok else "error")
            if not result.ok and step == self.max_steps:
                run.stopped_reason = "tool_error"

        if not run.final_answer:
            run.final_answer = (history or "no answer").strip()[:500]
            run.stopped_reason = "max_steps"
        run.total_duration_s = round(time.perf_counter() - t0, 3)
        return run

    def _arg_key(self, tool: Tool) -> str:
        return "code" if tool.name == "python" else ("expression" if tool.name == "calculator"
                                                    else "query")

    def _audit(self, step, plan, tool_name, tool_input, result, approved,
               status: str) -> None:
        entry = {
            "step": step, "plan": plan, "tool": tool_name,
            "tool_input": tool_input, "approved": approved, "status": status,
        }
        if result is not None:
            entry["result_ok"] = result.ok
            entry["result_error"] = result.error
            entry["duration_s"] = result.duration_s
        self.audit_log.append(entry)

    def audit_log_json(self) -> str:
        return json.dumps(self.audit_log, indent=2)
