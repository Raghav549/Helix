"""HELIX modular agent system.

CPU-first, local-only agent with:
  * pluggable tool registry (code execution, calculator, document lookup, ...)
  * sandboxed Python code execution (Linux namespaces, seccomp and limits)
  * planning / task decomposition loop with self-correction
  * explicit approval gate for consequential actions
  * step limits and between-call time checks; sandbox time/resource limits
  * per-step audit log (may contain sensitive inputs; protect it)
  * reproducible runs (deterministic seed)

This is a hypothesis to benchmark: whether the small HELIX model can use tools
to solve multi-step tasks. The agent is intentionally simple and rule-driven
for the baseline so it is auditable; the model's role is tool selection and
argument generation, verified by real execution results.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

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
        import math
        import operator
        t0 = time.perf_counter()
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
               ast.Mod: operator.mod, ast.Pow: operator.pow}
        def calc(node, depth=0):
            if depth > 32:
                raise ValueError("expression too deep")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = node.value
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                value = calc(node.operand, depth + 1)
                value = -value if isinstance(node.op, ast.USub) else value
            elif isinstance(node, ast.BinOp) and type(node.op) in ops:
                left, right = calc(node.left, depth + 1), calc(node.right, depth + 1)
                if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or
                        (abs(left) > 1 and abs(right) * math.log2(abs(left)) > 4096)):
                    raise ValueError("exponent too large")
                value = ops[type(node.op)](left, right)
            else:
                raise ValueError(f"disallowed expression node: {type(node).__name__}")
            if isinstance(value, complex) or (isinstance(value, int) and value.bit_length() > 4096) or \
               (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError("result outside numeric limits")
            return value
        try:
            if len(expression) > 2048:
                raise ValueError("expression too long")
            value = calc(ast.parse(expression, mode="eval").body)
            return ToolResult(True, str(value), duration_s=time.perf_counter() - t0, tool=self.name)
        except Exception as exc:
            return ToolResult(False, "", error=str(exc), duration_s=time.perf_counter() - t0, tool=self.name)


class PythonTool(Tool):
    """Execute Python once in an isolated, bounded Linux sandbox.

    Requires user, mount, PID and network namespaces plus libseccomp. Execution
    fails closed on platforms lacking these facilities. Treat the boundary as
    defense in depth; for hostile workloads use a separately provisioned VM.
    """

    name = "python"
    description = "Execute Python in a restricted, network-free Linux sandbox."
    consequential = True

    def __init__(self, timeout_s: float = 15.0, memory_mb: int = 256) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb

    def execute(self, code: str) -> ToolResult:
        from helix.sandbox import run_python
        result = run_python(code, timeout_s=self.timeout_s, memory_mb=self.memory_mb)
        return ToolResult(result.ok, result.output, result.error, result.duration_s,
                          self.name)


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
    require exact-payload approval. Step and wall-time limits are checked
    between model/tool calls; they do not preempt a running model forward pass.
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
        if require_approval is not True:
            raise ValueError("consequential tools always require exact-payload approval")
        self.audit_log: List[dict] = []
        self._preapproved: set[tuple[str, str]] = set()

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
        """Consume an explicit, exact-payload, single-use grant for consequential tools."""
        if not tool.consequential:
            return True
        import hashlib
        digest = hashlib.sha256(tool_input.encode('utf-8')).hexdigest()
        grant = (tool.name, digest)
        if grant in self._preapproved:
            self._preapproved.remove(grant)
            return True
        return False

    def set_tool_approved(self, tool_name: str, approved: bool, tool_input: str | None = None) -> None:
        """Approve/revoke one exact payload; never grant arbitrary model-selected code."""
        if approved is not True and approved is not False:
            raise ValueError("approval must be an explicit boolean")
        if not isinstance(tool_input, str):
            raise ValueError("approval must include the exact tool input")
        import hashlib
        grant = (tool_name, hashlib.sha256(tool_input.encode('utf-8')).hexdigest())
        if approved:
            self._preapproved.add(grant)
        else:
            self._preapproved.discard(grant)

    def run(self, goal: str, documents: Optional[Dict[str, str]] = None) -> AgentRun:
        if documents:
            # Documents remain untrusted observations, never permission grants.
            self.registry.register(DocumentTool(documents))
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

            approved = self._approve(tool, tool_input)
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
            if run.stopped_reason == "completed":
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
