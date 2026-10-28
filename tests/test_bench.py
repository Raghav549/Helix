"""Tests for the HELIX benchmark harness."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import os
import tempfile

from helix.config import HelixConfig, ModelConfig, TrainConfig, DataConfig, save_config
from helix.bench import detect_hardware, peak_rss_mb, recommend, HardwareSpec


def test_detect_hardware_returns_sane_values():
    h = detect_hardware()
    assert h.cpu_count >= 1
    assert h.ram_gb >= 0
    assert h.torch_threads >= 1
    assert h.device == "cpu"
    assert isinstance(h.simd_hints, list)
    assert isinstance(h.notes, str)


def test_peak_rss_nonnegative():
    v = peak_rss_mb()
    assert v >= 0.0


def test_recommendations():
    h = HardwareSpec(cpu_count=2, cpu_count_logical=2, ram_gb=1.5,
                     torch_threads=2, torch_version="x", simd_hints=[], device="cpu")
    from helix.bench import BenchReport
    report = BenchReport(hardware=h, config={})
    recs = recommend(h, report)
    assert any("2 GB" in r for r in recs)
    assert any("CPUs" in r for r in recs)
    assert any("AVX2/NEON" in r for r in recs)


def test_bench_runs_end_to_end(tmp_path):
    cfg = HelixConfig(
        model=ModelConfig(vocab_size=200, block_size=64, n_layer=1, n_head=2,
                          n_embd=32, n_kv_head=1, dropout=0.0),
        train=TrainConfig(batch_size=2, block_size=64, max_iters=4, eval_interval=2,
                          eval_iters=4, learning_rate=1e-3, warmup_iters=1,
                          lr_decay_iters=4, log_interval=2, save_interval=4,
                          out_dir=str(tmp_path / "out")),
        data=DataConfig(train_file=str(tmp_path / "train.npy"),
                        val_file=str(tmp_path / "val.npy"),
                        vocab_file=str(tmp_path / "vocab.json"),
                        merges_file=str(tmp_path / "merges.json"),
                        corpus_file=str(REPO / "data/seed_corpus.txt"), train_frac=0.9),
    )
    cfg_path = tmp_path / "config.json"
    save_config(cfg, str(cfg_path))

    from helix.bench import run_bench
    report = run_bench(str(cfg_path), out_path=str(tmp_path / "bench.json"),
                       block_sizes=[32, 64])
    assert len(report.results) > 0
    names = {r.name for r in report.results}
    assert "forward_backward_toks_per_sec" in names
    assert report.hardware.cpu_count >= 1
    out = tmp_path / "bench.json"
    assert out.exists()
