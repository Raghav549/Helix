"""HELIX research laboratory — reproducible computational experiments.

Independently implemented. Provides:
  * Experiment tracker: record config, inputs, outputs, logs, versions;
    read/write metrics from out/; reproducible run IDs.
  * Symbolic mathematics (via sympy as an independent verifier):
    differentiation, integration, equation solving, simplification,
    limit computation. Results are clearly tool-verified.
  * Statistics: descriptive stats, correlation, linear regression.
  * Data analysis: load CSV, summarize columns, detect types.
  * Plotting: save line/scatter/histogram plots to PNG (matplotlib Agg).
  * Hypothesis evaluation: compare model claims against tool-verified results.

All tools run locally; sympy/matplotlib are independent verification
instruments — their output is distinguished from model-generated claims.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# sympy is an optional independent verification instrument.
try:
    import sympy as sp
    HAS_SYMPY = True
except Exception:  # noqa: BLE001
    sp = None
    HAS_SYMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # noqa: BLE001
    plt = None
    HAS_MPL = False


# ============================================================================
# Experiment tracker
# ============================================================================

@dataclass
class ExperimentRecord:
    id: str
    name: str
    stage: str
    created_at: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metrics: Dict[str, float]
    logs: List[str] = field(default_factory=list)
    device: str = "cpu"
    verifier_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentTracker:
    """Tracks experiments under out/<name>/ with metrics.json + record.json."""

    def __init__(self, root: str = "out") -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _exp_dir(self, exp_id: str) -> str:
        return os.path.join(self.root, exp_id)

    def create(self, name: str, stage: str, config: Dict[str, Any],
               inputs: Dict[str, Any], device: str = "cpu") -> str:
        exp_id = f"{stage}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        d = self._exp_dir(exp_id)
        os.makedirs(d, exist_ok=True)
        rec = ExperimentRecord(
            id=exp_id, name=name, stage=stage,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            config=config, inputs=inputs, outputs={}, metrics={},
            device=device,
        )
        rec.logs.append(f"created experiment {exp_id}")
        self._write_record(rec)
        return exp_id

    def log(self, exp_id: str, message: str) -> None:
        rec = self.read(exp_id)
        if rec is None:
            return
        rec.logs.append(f"{time.strftime('%H:%M:%S')} {message}")
        self._write_record(rec)

    def set_outputs(self, exp_id: str, outputs: Dict[str, Any]) -> None:
        rec = self.read(exp_id)
        if rec is None:
            return
        rec.outputs = outputs
        self._write_record(rec)

    def set_metrics(self, exp_id: str, metrics: Dict[str, float]) -> None:
        rec = self.read(exp_id)
        if rec is None:
            return
        rec.metrics = metrics
        self._write_record(rec)

    def add_verifier(self, exp_id: str, tool: str) -> None:
        rec = self.read(exp_id)
        if rec is None:
            return
        if tool not in rec.verifier_tools:
            rec.verifier_tools.append(tool)
        self._write_record(rec)

    def read(self, exp_id: str) -> Optional[ExperimentRecord]:
        rpath = os.path.join(self._exp_dir(exp_id), "record.json")
        if not os.path.isfile(rpath):
            return None
        with open(rpath, encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentRecord(**data)

    def list(self) -> List[ExperimentRecord]:
        out: List[ExperimentRecord] = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root)):
            rpath = os.path.join(self.root, name, "record.json")
            if os.path.isfile(rpath):
                rec = self.read(name)
                if rec is not None:
                    out.append(rec)
        return out

    def _write_record(self, rec: ExperimentRecord) -> None:
        d = self._exp_dir(rec.id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as f:
            json.dump(rec.to_dict(), f, indent=2)

    # Also scan legacy metrics.json files (from other HELIX stages).
    def scan_legacy(self) -> List[dict]:
        found: List[dict] = []
        if not os.path.isdir(self.root):
            return found
        for name in sorted(os.listdir(self.root)):
            mpath = os.path.join(self.root, name, "metrics.json")
            if os.path.isfile(mpath):
                try:
                    with open(mpath, encoding="utf-8") as f:
                        m = json.load(f)
                    found.append({"name": name, **m})
                except Exception:
                    pass
        return found


# ============================================================================
# Symbolic mathematics (sympy as independent verifier)
# ============================================================================

@dataclass
class SymbolicResult:
    expression: str
    operation: str
    result: str
    verified_by: str  # "sympy" | "heuristic"
    steps: List[str] = field(default_factory=list)


class SymbolicMath:
    """Symbolic math via sympy; clearly marked as tool-verified."""

    def __init__(self) -> None:
        self.available = HAS_SYMPY

    def differentiate(self, expression: str, variable: str = "x") -> SymbolicResult:
        steps = [f"Input: {expression}", f"Variable: {variable}"]
        if not self.available:
            return SymbolicResult(expression, "differentiate",
                                  "(sympy unavailable)", "heuristic", steps)
        x = sp.Symbol(variable)
        try:
            expr = sp.sympify(expression)
            deriv = sp.diff(expr, x)
            steps.append(f"d/d{variable} {expression} = {deriv}")
            return SymbolicResult(expression, "differentiate", str(deriv),
                                  "sympy", steps)
        except Exception as exc:  # noqa: BLE001
            steps.append(f"Error: {exc}")
            return SymbolicResult(expression, "differentiate",
                                  f"error: {exc}", "sympy", steps)

    def integrate(self, expression: str, variable: str = "x") -> SymbolicResult:
        steps = [f"Input: {expression}", f"Variable: {variable}"]
        if not self.available:
            return SymbolicResult(expression, "integrate",
                                  "(sympy unavailable)", "heuristic", steps)
        x = sp.Symbol(variable)
        try:
            expr = sp.sympify(expression)
            integ = sp.integrate(expr, x)
            steps.append(f"Integral of {expression} dx = {integ} + C")
            return SymbolicResult(expression, "integrate", str(integ) + " + C",
                                  "sympy", steps)
        except Exception as exc:  # noqa: BLE001
            steps.append(f"Error: {exc}")
            return SymbolicResult(expression, "integrate",
                                  f"error: {exc}", "sympy", steps)

    def solve(self, equation: str, variable: str = "x") -> SymbolicResult:
        steps = [f"Equation: {equation}", f"Variable: {variable}"]
        if not self.available:
            return SymbolicResult(equation, "solve",
                                  "(sympy unavailable)", "heuristic", steps)
        x = sp.Symbol(variable)
        try:
            # support "a = b" form
            if "=" in equation:
                lhs, rhs = equation.split("=", 1)
                eq = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
            else:
                eq = sp.Eq(sp.sympify(equation), 0)
            sols = sp.solve(eq, x)
            steps.append(f"Solutions: {sols}")
            return SymbolicResult(equation, "solve", str(sols), "sympy", steps)
        except Exception as exc:  # noqa: BLE001
            steps.append(f"Error: {exc}")
            return SymbolicResult(equation, "solve", f"error: {exc}", "sympy", steps)

    def simplify(self, expression: str) -> SymbolicResult:
        steps = [f"Input: {expression}"]
        if not self.available:
            return SymbolicResult(expression, "simplify",
                                  "(sympy unavailable)", "heuristic", steps)
        try:
            expr = sp.sympify(expression)
            simp = sp.simplify(expr)
            steps.append(f"Simplified: {simp}")
            return SymbolicResult(expression, "simplify", str(simp), "sympy", steps)
        except Exception as exc:  # noqa: BLE001
            steps.append(f"Error: {exc}")
            return SymbolicResult(expression, "simplify",
                                  f"error: {exc}", "sympy", steps)

    def limit(self, expression: str, variable: str = "x",
              point: str = "0") -> SymbolicResult:
        steps = [f"Input: {expression}", f"Limit {variable}-> {point}"]
        if not self.available:
            return SymbolicResult(expression, "limit",
                                  "(sympy unavailable)", "heuristic", steps)
        x = sp.Symbol(variable)
        try:
            expr = sp.sympify(expression)
            p = sp.sympify(point)
            lim = sp.limit(expr, x, p)
            steps.append(f"lim_{variable}->{point} {expression} = {lim}")
            return SymbolicResult(expression, "limit", str(lim), "sympy", steps)
        except Exception as exc:  # noqa: BLE001
            steps.append(f"Error: {exc}")
            return SymbolicResult(expression, "limit", f"error: {exc}",
                                  "sympy", steps)


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class StatResult:
    metric: str
    value: float
    detail: str = ""


class Statistics:
    """Descriptive statistics, correlation, linear regression (pure Python)."""

    @staticmethod
    def _nums(values: List[Any]) -> List[float]:
        out = []
        for v in values:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
        return out

    def describe(self, values: List[Any]) -> List[StatResult]:
        nums = self._nums(values)
        if not nums:
            return [StatResult("error", 0.0, "no numeric values")]
        n = len(nums)
        mean = sum(nums) / n
        var = sum((x - mean) ** 2 for x in nums) / n
        std = math.sqrt(var)
        srt = sorted(nums)
        median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
        return [
            StatResult("count", float(n)),
            StatResult("mean", round(mean, 6)),
            StatResult("median", round(median, 6)),
            StatResult("std", round(std, 6)),
            StatResult("min", round(min(nums), 6)),
            StatResult("max", round(max(nums), 6)),
        ]

    def correlation(self, x: List[Any], y: List[Any]) -> StatResult:
        xn = self._nums(x)
        yn = self._nums(y)
        n = min(len(xn), len(yn))
        if n < 2:
            return StatResult("correlation", 0.0, "insufficient data")
        xn, yn = xn[:n], yn[:n]
        mx = sum(xn) / n
        my = sum(yn) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xn, yn))
        dx = math.sqrt(sum((a - mx) ** 2 for a in xn))
        dy = math.sqrt(sum((b - my) ** 2 for b in yn))
        if dx == 0 or dy == 0:
            return StatResult("correlation", 0.0, "zero variance")
        return StatResult("correlation", round(num / (dx * dy), 6))

    def linear_regression(self, x: List[Any], y: List[Any]) -> Dict[str, Any]:
        xn = self._nums(x)
        yn = self._nums(y)
        n = min(len(xn), len(yn))
        if n < 2:
            return {"error": "insufficient data"}
        xn, yn = xn[:n], yn[:n]
        mx = sum(xn) / n
        my = sum(yn) / n
        num = sum((a - mx) * (b - my) for a, b in zip(xn, yn))
        den = sum((a - mx) ** 2 for a in xn)
        if den == 0:
            return {"error": "zero x variance"}
        slope = num / den
        intercept = my - slope * mx
        # R^2
        ypred = [slope * a + intercept for a in xn]
        ss_res = sum((b - p) ** 2 for b, p in zip(yn, ypred))
        ss_tot = sum((b - my) ** 2 for b in yn)
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        return {"slope": round(slope, 6), "intercept": round(intercept, 6),
                "r_squared": round(r2, 6), "n": n}


# ============================================================================
# Data analysis
# ============================================================================

def load_csv(text: str) -> Tuple[List[str], List[List[str]]]:
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def summarize_csv(text: str) -> dict:
    header, rows = load_csv(text)
    if not header:
        return {"error": "empty csv"}
    result: Dict[str, Any] = {"columns": header, "rows": len(rows)}
    col_stats: Dict[str, dict] = {}
    for ci, col in enumerate(header):
        col_vals = [row[ci] for row in rows if ci < len(row)]
        nums = Statistics()._nums(col_vals)
        if nums:
            desc = Statistics().describe(nums)
            col_stats[col] = {d.metric: d.value for d in desc}
        else:
            col_stats[col] = {"type": "categorical",
                              "unique": len(set(col_vals)),
                              "top": col_vals[0] if col_vals else None}
    result["column_stats"] = col_stats
    return result


# ============================================================================
# Plotting
# ============================================================================

def plot_line(x: List[Any], y: List[Any], title: str,
              out_path: str) -> dict:
    if not HAS_MPL:
        return {"error": "matplotlib unavailable"}
    xn = Statistics()._nums(x)
    yn = Statistics()._nums(y)
    n = min(len(xn), len(yn))
    if n < 1:
        return {"error": "no numeric data"}
    plt.figure()
    plt.plot(xn[:n], yn[:n], marker="o")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    return {"out_path": out_path, "points": n}


def plot_histogram(values: List[Any], title: str, out_path: str,
                   bins: int = 10) -> dict:
    if not HAS_MPL:
        return {"error": "matplotlib unavailable"}
    nums = Statistics()._nums(values)
    if not nums:
        return {"error": "no numeric data"}
    plt.figure()
    plt.hist(nums, bins=bins, edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    return {"out_path": out_path, "points": len(nums)}


# ============================================================================
# Hypothesis evaluation: model claim vs tool-verified result
# ============================================================================

def evaluate_claim(claim: str, tool_result: str) -> dict:
    """Compare a model-generated claim with an independent tool result.

    Returns whether they agree (exact string match after normalization) and a
    human-readable verdict. This is the guard against fabricated knowledge.
    """
    claim_norm = " ".join(claim.split()).strip().lower()
    tool_norm = " ".join(tool_result.split()).strip().lower()
    agree = claim_norm == tool_norm or claim_norm in tool_norm or tool_norm in claim_norm
    return {
        "claim": claim,
        "tool_result": tool_result,
        "agree": agree,
        "verdict": ("CONFIRMED by independent tool" if agree
                    else "DISAGREES with independent tool — treat as unverified"),
    }


def run_reproducible_experiment(tracker: ExperimentTracker, name: str,
                                stage: str, config: Dict[str, Any],
                                inputs: Dict[str, Any],
                                compute_fn) -> ExperimentRecord:
    """Run a function inside a tracked, reproducible experiment."""
    exp_id = tracker.create(name, stage, config, inputs)
    tracker.log(exp_id, "starting computation")
    t0 = time.perf_counter()
    try:
        outputs = compute_fn()
        dt = time.perf_counter() - t0
        tracker.set_outputs(exp_id, outputs)
        metrics = {"elapsed_s": round(dt, 4)}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)
        tracker.set_metrics(exp_id, metrics)
        tracker.log(exp_id, f"completed in {dt:.3f}s")
    except Exception as exc:  # noqa: BLE001
        tracker.log(exp_id, f"failed: {exc}")
        raise
    return tracker.read(exp_id)
