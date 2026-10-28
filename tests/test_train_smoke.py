"""Smoke test: real end-to-end training from random initialization (tiny)."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import json
import os
import tempfile

from helix.config import HelixConfig, ModelConfig, TrainConfig, DataConfig, save_config
from helix.train import train


def _write_cfg(tmp):
    cfg = HelixConfig(
        model=ModelConfig(
            vocab_size=160, block_size=32, n_layer=1, n_head=2, n_embd=32,
            n_kv_head=1, dropout=0.0,
        ),
        train=TrainConfig(
            batch_size=4, block_size=32, max_iters=6, eval_interval=3,
            eval_iters=4, learning_rate=1e-3, warmup_iters=2,
            lr_decay_iters=6, log_interval=3, save_interval=6,
            out_dir=os.path.join(tmp, "out"), resume_from=None,
        ),
        data=DataConfig(
            train_file=os.path.join(tmp, "train.npy"),
            val_file=os.path.join(tmp, "val.npy"),
            vocab_file=os.path.join(tmp, "vocab.json"),
            merges_file=os.path.join(tmp, "merges.json"),
            corpus_file=str(REPO / "data/seed_corpus.txt"), train_frac=0.9,
        ),
    )
    path = os.path.join(tmp, "config.json")
    save_config(cfg, path)
    return path


def test_train_from_random_init_and_resume(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    train(cfg_path)
    out_dir = os.path.join(tmp_path, "out")
    assert os.path.exists(os.path.join(out_dir, "ckpt-final.pt"))
    assert os.path.exists(os.path.join(out_dir, "log.jsonl"))

    # resume check
    cfg = HelixConfig(
        model=ModelConfig(vocab_size=160, block_size=32, n_layer=1, n_head=2,
                          n_embd=32, n_kv_head=1, dropout=0.0),
        train=TrainConfig(
            batch_size=4, block_size=32, max_iters=10, eval_interval=3,
            eval_iters=4, learning_rate=1e-3, warmup_iters=2,
            lr_decay_iters=10, log_interval=3, save_interval=10,
            out_dir=os.path.join(tmp_path, "out2"),
            resume_from=os.path.join(out_dir, "ckpt-final.pt"),
        ),
        data=DataConfig(
            train_file=os.path.join(tmp_path, "train.npy"),
            val_file=os.path.join(tmp_path, "val.npy"),
            vocab_file=os.path.join(tmp_path, "vocab.json"),
            merges_file=os.path.join(tmp_path, "merges.json"),
            corpus_file=str(REPO / "data/seed_corpus.txt"), train_frac=0.9,
        ),
    )
    cfg_path2 = os.path.join(tmp_path, "config2.json")
    save_config(cfg, cfg_path2)
    train(cfg_path2)
    assert os.path.exists(os.path.join(tmp_path, "out2", "ckpt-final.pt"))
