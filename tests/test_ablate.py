"""Smoke test for the architecture ablation runner."""

import json
import os
import tempfile

from helix.config import HelixConfig, ModelConfig, TrainConfig, DataConfig, save_config
from helix.ablate import run_ablation


def test_ablation_runs_and_reports(tmp_path):
    cfg = HelixConfig(
        model=ModelConfig(vocab_size=300, block_size=32, n_layer=1, n_head=2,
                          n_embd=32, n_kv_head=1, dropout=0.0,
                          block_type="transformer"),
        train=TrainConfig(batch_size=2, block_size=32, max_iters=8,
                          eval_interval=4, eval_iters=4, learning_rate=1e-3,
                          warmup_iters=2, lr_decay_iters=8, log_interval=4,
                          save_interval=8, out_dir=str(tmp_path / "out")),
        data=DataConfig(train_file=str(tmp_path / "train.npy"),
                        val_file=str(tmp_path / "val.npy"),
                        vocab_file=str(tmp_path / "vocab.json"),
                        merges_file=str(tmp_path / "merges.json"),
                        corpus_file="data/seed_corpus.txt", train_frac=0.9),
    )
    cfg_path = tmp_path / "config.json"
    save_config(cfg, str(cfg_path))

    report = run_ablation(str(cfg_path), out_path=str(tmp_path / "report.json"),
                          block_types=["transformer", "linear_attn"])
    assert os.path.exists(str(tmp_path / "report.json"))
    assert "variants" in report
    assert len(report["variants"]) == 2
    btypes = {v["block_type"] for v in report["variants"]}
    assert btypes == {"transformer", "linear_attn"}
    for v in report["variants"]:
        assert "final_val_loss" in v
        assert "tokens_per_sec" in v
        assert "n_params" in v
        assert v["checkpoint"].endswith(".pt")
    # best_by_val_loss should be one of the variants.
    assert report["best_by_val_loss"] in btypes

    data = json.load(open(str(tmp_path / "report.json")))
    assert data["best_by_val_loss"] in btypes
