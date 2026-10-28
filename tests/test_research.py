"""Tests for the HELIX research laboratory."""

import os
import tempfile

from helix.research import (ExperimentTracker, SymbolicMath, Statistics,
                            load_csv, summarize_csv, plot_line, plot_histogram,
                            evaluate_claim, run_reproducible_experiment)


def test_symbolic_differentiate():
    sm = SymbolicMath()
    r = sm.differentiate("x**2 + 3*x + 2", "x")
    assert r.verified_by == "sympy"
    assert "2*x + 3" in r.result or r.result == "2*x + 3"


def test_symbolic_integrate():
    sm = SymbolicMath()
    r = sm.integrate("2*x", "x")
    assert r.verified_by == "sympy"


def test_symbolic_solve():
    sm = SymbolicMath()
    r = sm.solve("x**2 - 4", "x")
    assert r.verified_by == "sympy"
    assert "-2" in r.result and "2" in r.result


def test_symbolic_simplify():
    sm = SymbolicMath()
    r = sm.simplify("(x + 1)**2 - x**2")
    assert r.verified_by == "sympy"


def test_symbolic_limit():
    sm = SymbolicMath()
    r = sm.limit("sin(x)/x", "x", "0")
    assert r.verified_by == "sympy"
    assert r.result == "1"


def test_statistics():
    st = Statistics()
    d = st.describe([1, 2, 3, 4, 5])
    vals = {s.metric: s.value for s in d}
    assert vals["mean"] == 3.0
    assert vals["min"] == 1.0
    assert vals["max"] == 5.0


def test_correlation():
    st = Statistics()
    c = st.correlation([1, 2, 3, 4], [2, 4, 6, 8])
    assert abs(c.value - 1.0) < 1e-6


def test_linear_regression():
    st = Statistics()
    lr = st.linear_regression([1, 2, 3, 4], [2, 4, 6, 8])
    assert abs(lr["slope"] - 2.0) < 1e-6
    assert abs(lr["r_squared"] - 1.0) < 1e-6


def test_csv_summarize():
    text = "a,b\n1,2\n3,4\n5,6\n"
    h, rows = load_csv(text)
    assert h == ["a", "b"] and len(rows) == 3
    s = summarize_csv(text)
    assert s["rows"] == 3
    assert s["column_stats"]["a"]["mean"] == 3.0


def test_experiment_tracker(tmp_path):
    tr = ExperimentTracker(str(tmp_path / "exp"))
    eid = tr.create("test-exp", "demo", {"lr": 0.1}, {"n": 10})
    tr.log(eid, "step one")
    tr.set_outputs(eid, {"loss": 0.5})
    tr.set_metrics(eid, {"loss": 0.5, "acc": 0.9})
    tr.add_verifier(eid, "sympy")
    rec = tr.read(eid)
    assert rec is not None and rec.name == "test-exp"
    assert rec.metrics["loss"] == 0.5
    assert rec.metrics["acc"] == 0.9
    assert "sympy" in rec.verifier_tools
    assert len(tr.list()) == 1


def test_run_reproducible_experiment(tmp_path):
    tr = ExperimentTracker(str(tmp_path / "exp2"))
    def compute():
        return {"loss": 0.25, "iters": 10}
    rec = run_reproducible_experiment(tr, "repro", "demo", {"x": 1}, {"y": 2}, compute)
    assert rec.metrics["loss"] == 0.25
    assert rec.metrics["iters"] == 10
    assert rec.metrics["elapsed_s"] >= 0


def test_evaluate_claim():
    ev = evaluate_claim("2*x + 3", "2*x + 3")
    assert ev["agree"] is True
    assert ev["verdict"].startswith("CONFIRMED")
    ev2 = evaluate_claim("x**2", "2*x + 3")
    assert ev2["agree"] is False
    assert ev2["verdict"].startswith("DISAGREES")


def test_plots(tmp_path):
    if not __import__("helix.research", fromlist=["HAS_MPL"]).HAS_MPL:
        import pytest
        pytest.skip("matplotlib unavailable")
    p1 = str(tmp_path / "line.png")
    r = plot_line([1, 2, 3], [1, 4, 9], "sq", p1)
    assert r["points"] == 3 and os.path.isfile(p1)
    p2 = str(tmp_path / "hist.png")
    r = plot_histogram([1, 1, 2, 3, 5], "hist", p2)
    assert r["points"] == 5 and os.path.isfile(p2)


def test_plot_preserves_row_pairing_and_csv_is_bounded(tmp_path, monkeypatch):
    import pytest
    import helix.research as research
    if not research.HAS_MPL: pytest.skip('matplotlib unavailable')
    observed = []
    original = research.plt.plot
    def capture(x, y, *args, **kwargs):
        observed.append((list(x), list(y)))
        return original(x, y, *args, **kwargs)
    monkeypatch.setattr(research.plt, 'plot', capture)
    result = research.plot_line(['1', 'bad', '3'], ['10', '20', '30'], 'pairs', str(tmp_path/'pair.png'))
    assert result['points'] == 2 and observed == [([1.0, 3.0], [10.0, 30.0])]
    with pytest.raises(ValueError): research.load_csv('a,a\n1,2')
    with pytest.raises(ValueError): research.load_csv(','.join(['x'] * 129))
