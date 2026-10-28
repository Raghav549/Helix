"""HELIX evaluation framework.

CPU-runnable, small original evaluation probes with explicit
categories: memorisation, reasoning, factual recall, code execution,
mathematical proofs, multilingual, long-context, retrieval, tool safety,
calibration.

Every task is a hypothesis test: it reports a metric and a pass/fail against a
configurable threshold. Results are written to a JSON report. Tasks are
generated at eval time where possible, but contamination and generalization
are not guaranteed by this small probe suite.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch

from helix.infer import load_model_and_tokenizer, generate


@dataclass
class EvalResult:
    task: str
    category: str
    metric: str
    value: float
    threshold: Optional[float]
    pass_: bool
    details: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    model: str
    checkpoint: str
    device: str
    results: List[EvalResult] = field(default_factory=list)

    def add(self, result: EvalResult) -> None:
        self.results.append(result)

    def summary(self) -> dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.pass_)
        return {
            "model": self.model,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / max(total, 1), 4),
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)


class EvalTask:
    """Base class for an evaluation task."""

    name: str = "base"
    category: str = "uncategorized"

    def run(self, model, tok, device: str) -> EvalResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Small original diagnostic tasks
# ---------------------------------------------------------------------------

class ArithmeticTask(EvalTask):
    """Elementary arithmetic addition. Numbers drawn at eval time."""

    name = "arithmetic_addition"
    category = "reasoning"

    def run(self, model, tok, device: str, n: int = 20, seed: int = 0) -> EvalResult:
        import random
        rng = random.Random(seed)
        correct = 0
        examples = []
        for _ in range(n):
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)
            prompt = f"{a} + {b} ="
            completion = generate(model, tok, prompt, max_new_tokens=8,
                                  temperature=0.0, top_k=1, device=device)
            tail = completion[len(prompt):].strip()
            import re
            match = re.match(r"\s*(\d+)(?!\d)", tail)
            expected = str(a + b)
            ok = bool(match) and match.group(1) == expected
            correct += int(ok)
            examples.append({"a": a, "b": b, "expected": expected,
                             "completion": tail, "ok": ok})
        value = correct / max(n, 1)
        return EvalResult(self.name, self.category, "accuracy", value, 0.5,
                          value >= 0.5,
                          {"correct": correct, "total": n, "examples": examples[:5]})


class CodeCompletionTask(EvalTask):
    """Complete a short Python function body (heuristic pass)."""

    name = "code_completion_is_even"
    category = "code_execution"

    def run(self, model, tok, device: str) -> EvalResult:
        prompt = ("def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n    return ")
        completion = generate(model, tok, prompt, max_new_tokens=12,
                              temperature=0.2, top_k=5, device=device)
        tail = completion[len(prompt):].strip().split("\n", 1)[0]
        # Execute generated code against independent examples, never just look
        # for a textual substring. Fail closed if isolation is unavailable.
        from helix.sandbox import run_python
        code = ("def is_even(n):\n    return " + tail + "\n" +
                "for n in range(-12, 13):\n    assert is_even(n) == (n % 2 == 0)\n")
        execution = run_python(code, timeout_s=3)
        ok = execution.ok
        value = 1.0 if ok else 0.0
        return EvalResult(self.name, self.category, "isolated_test_pass", value, 1.0,
                          ok, {"completion": tail, "execution_error": execution.error})


class SentenceCompletionTask(EvalTask):
    """Cloze-style factual completion."""

    name = "sentence_completion_capital"
    category = "memorisation"

    def run(self, model, tok, device: str) -> EvalResult:
        prompt = "The capital of France is"
        completion = generate(model, tok, prompt, max_new_tokens=6,
                              temperature=0.0, top_k=1, device=device)
        tail = completion[len(prompt):].strip().lower()
        ok = "paris" in tail
        value = 1.0 if ok else 0.0
        return EvalResult(self.name, self.category, "accuracy", value, 1.0,
                          ok, {"completion": tail})


class LongContextRecallTask(EvalTask):
    """Needle-in-haystack recall inside a long context."""

    name = "long_context_recall"
    category = "long_context"

    def run(self, model, tok, device: str,
            needle: str = "The secret code is ALPHA-9921.",
            filler_lines: int = 40) -> EvalResult:
        haystack = "\n".join(f"filler line {i}" for i in range(filler_lines))
        context = (f"{haystack}\n{needle}\n{haystack}\n"
                   f"Question: What is the secret code?")
        needle_ids = tok.encode(needle)
        context_ids = tok.encode(context)
        # A missing needle outside the supported window is not a successful
        # long-context evaluation. Never count the echoed prompt as an answer.
        within_window = len(context_ids) <= model.config.block_size
        completion = generate(model, tok, context, max_new_tokens=12,
                              temperature=0.0, top_k=1, device=device)
        tail = completion[len(context):] if completion.startswith(context) else completion
        ok = within_window and "ALPHA-9921" in tail
        value = 1.0 if ok else 0.0
        return EvalResult(self.name, self.category, "accuracy", value, 1.0,
                          ok, {"completion": tail, "context_tokens": len(context_ids),
                               "context_limit": model.config.block_size,
                               "needle_within_window": within_window})


class CalibrationTask(EvalTask):
    """A small true-vs-false likelihood probe, not statistical calibration."""

    name = "calibration_true_vs_false"
    category = "calibration"

    def run(self, model, tok, device: str) -> EvalResult:
        true_stmt = "Water boils at 100 degrees Celsius at sea level."
        false_stmt = "Water boils at 500 degrees Celsius at sea level."

        def loss_for(text: str) -> float:
            ids = torch.tensor([tok.encode(text)], device=device)
            with torch.no_grad():
                logits, loss = model(ids[:, :-1], ids[:, 1:])
            return float(loss.item())

        lt = loss_for(true_stmt)
        lf = loss_for(false_stmt)
        ok = lt < lf
        value = 1.0 if ok else 0.0
        return EvalResult(self.name, self.category,
                          "true_loss_less_than_false_loss", value, 1.0, ok,
                          {"true_loss": round(lt, 4), "false_loss": round(lf, 4)})


# ---------------------------------------------------------------------------
# Task registry (extensible)
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, EvalTask] = {}
for _task in (ArithmeticTask(), CodeCompletionTask(), SentenceCompletionTask(),
              LongContextRecallTask(), CalibrationTask()):
    TASK_REGISTRY[_task.name] = _task


def register_task(task: EvalTask) -> None:
    TASK_REGISTRY[task.name] = task


def run_eval(config_path: str, checkpoint_path: str, out_path: str,
             tasks: Optional[List[str]] = None, device: str = "cpu") -> EvalReport:
    """Run selected eval tasks against a checkpoint and write a JSON report."""
    model, tok, cfg = load_model_and_tokenizer(config_path, checkpoint_path)
    model.to(device)
    report = EvalReport(model=config_path, checkpoint=checkpoint_path, device=device)
    selected = tasks or list(TASK_REGISTRY)
    for name in selected:
        if name not in TASK_REGISTRY:
            raise ValueError(f"unknown eval task: {name}")
        task = TASK_REGISTRY[name]
        t0 = time.time()
        result = task.run(model, tok, device)
        result.details = dict(result.details)
        result.details["elapsed_s"] = round(time.time() - t0, 3)
        report.add(result)
        print(f"[eval] {name}: {result.metric}={result.value:.4f} pass={result.pass_}")
    report.save(out_path)
    return report
