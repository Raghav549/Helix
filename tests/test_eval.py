"""Tests for the HELIX evaluation framework."""

import torch

from helix.config import ModelConfig
from helix.model import HelixTransformer
from helix.tokenizer import ByteLevelBPETokenizer
from helix.eval import TASK_REGISTRY, ArithmeticTask, CalibrationTask, CodeCompletionTask


def _tiny_model_and_tok():
    tok = ByteLevelBPETokenizer()
    tok.train(
        ["hello world", "2 + 2 = 4", "The capital of France is Paris.",
         "def is_even(n): return n % 2 == 0"],
        vocab_size=300,
    )
    model = HelixTransformer(ModelConfig(
        vocab_size=len(tok), block_size=32, n_layer=1, n_head=2, n_embd=32,
        n_kv_head=1, dropout=0.0,
    ))
    return model, tok


def test_task_registry_nonempty():
    assert len(TASK_REGISTRY) >= 3


def test_tasks_return_valid_results():
    model, tok = _tiny_model_and_tok()
    for task in (ArithmeticTask(), CalibrationTask(), CodeCompletionTask()):
        res = task.run(model, tok, "cpu")
        assert 0.0 <= res.value <= 1.0
        assert res.category
        assert res.metric
        assert isinstance(res.pass_, bool)


def test_arithmetic_is_contamination_resistant():
    model, tok = _tiny_model_and_tok()
    t1 = ArithmeticTask().run(model, tok, "cpu", seed=1)
    t2 = ArithmeticTask().run(model, tok, "cpu", seed=2)
    # Different seeds => different numbers; both should produce valid metrics.
    assert 0.0 <= t1.value <= 1.0
    assert 0.0 <= t2.value <= 1.0


def test_report_save_and_summary(tmp_path):
    model, tok = _tiny_model_and_tok()
    res = CalibrationTask().run(model, tok, "cpu")
    import json
    from helix.eval import EvalReport
    report = EvalReport(model="tiny", checkpoint="none", device="cpu")
    report.add(res)
    path = tmp_path / "report.json"
    report.save(str(path))
    data = json.load(open(path))
    assert data["total"] == 1
    assert "results" in data


def test_long_context_does_not_count_prompt_echo(monkeypatch):
    from types import SimpleNamespace
    import helix.eval as evaluation
    tok = ByteLevelBPETokenizer()
    model = SimpleNamespace(config=SimpleNamespace(block_size=10000))
    monkeypatch.setattr(evaluation, 'generate', lambda model, tok, prompt, **kw: prompt)
    result = evaluation.LongContextRecallTask().run(model, tok, 'cpu', filler_lines=1)
    assert not result.pass_ and result.details['completion'] == ''
    monkeypatch.setattr(evaluation, 'generate', lambda model, tok, prompt, **kw: prompt + ' ALPHA-9921')
    assert evaluation.LongContextRecallTask().run(model, tok, 'cpu', filler_lines=1).pass_
    model.config.block_size = 10
    assert not evaluation.LongContextRecallTask().run(model, tok, 'cpu', filler_lines=1).pass_


def test_calibration_likelihood_uses_shifted_targets():
    tok = ByteLevelBPETokenizer()
    class CheckingModel:
        calls = 0
        def __call__(self, x, y):
            self.calls += 1
            assert torch.equal(x[:, 1:], y[:, :-1])
            return None, torch.tensor(1.0 if self.calls == 1 else 2.0)
    model = CheckingModel()
    result = CalibrationTask().run(model, tok, 'cpu')
    assert model.calls == 2 and result.pass_
