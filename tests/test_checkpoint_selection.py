import pytest

from scripts.select_chat_checkpoint import choose_checkpoint


def report(passed, loss, names=("a", "b", "c")):
    return {
        "passed": passed,
        "total": len(names),
        "validation": {"loss": loss},
        "results": [{"task": name, "pass_": i < passed} for i, name in enumerate(names)],
    }


def test_base_wins_when_task_scores_tie_and_sft_language_loss_regresses():
    selected, manifest = choose_checkpoint(report(1, 3.4), report(1, 4.5))
    assert selected == "base"
    assert "base preserves" in manifest["reason"]


def test_sft_wins_when_it_passes_more_capability_probes():
    selected, manifest = choose_checkpoint(report(1, 3.4), report(2, 4.8))
    assert selected == "sft"
    assert "more held-out capability probes" in manifest["reason"]


def test_sft_can_win_a_tie_if_language_loss_is_better():
    selected, _ = choose_checkpoint(report(1, 3.4), report(1, 3.2))
    assert selected == "sft"


def test_mismatched_task_suites_are_rejected():
    with pytest.raises(ValueError):
        choose_checkpoint(report(1, 3.4, ("a", "b")), report(1, 3.2, ("b", "a")))


def test_chat_sft_wins_tie_when_instruction_loss_improves_and_lm_regression_is_bounded():
    metrics = {"initial_heldout_loss": 3.0, "best_sft_val_loss": 1.7}
    selected, manifest = choose_checkpoint(report(1, 3.4), report(1, 4.5), metrics)
    assert selected == "sft"
    assert manifest["sft_instruction_loss_improvement_fraction"] > 0.2


def test_base_wins_when_instruction_loss_does_not_improve_enough():
    metrics = {"initial_heldout_loss": 3.0, "best_sft_val_loss": 2.7}
    selected, _ = choose_checkpoint(report(1, 3.4), report(1, 4.5), metrics)
    assert selected == "base"
