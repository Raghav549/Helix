"""CPU-first training loop: gradient accumulation, checkpoint save/resume,
deterministic seeds, JSON logging, and validation-driven best-checkpointing.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import numpy as np
import torch

from helix.config import HelixConfig, load_config
from helix.data import build_from_corpus, get_batch, load_ids
from helix.model import HelixTransformer


def get_lr(it: int, train_cfg) -> float:
    """Cosine learning-rate decay with linear warmup."""
    if it < train_cfg.warmup_iters:
        return train_cfg.learning_rate * it / max(train_cfg.warmup_iters, 1)
    if it > train_cfg.lr_decay_iters:
        return train_cfg.min_lr
    decay_ratio = (it - train_cfg.warmup_iters) / max(
        train_cfg.lr_decay_iters - train_cfg.warmup_iters, 1
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_cfg.min_lr + coeff * (train_cfg.learning_rate - train_cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model: HelixTransformer,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    train_cfg,
) -> dict:
    model.eval()
    out: dict = {}
    for split in ("train", "val"):
        losses = torch.zeros(train_cfg.eval_iters)
        for k in range(train_cfg.eval_iters):
            X, Y = get_batch(
                split,
                train_ids,
                val_ids,
                train_cfg.batch_size,
                train_cfg.block_size,
                train_cfg.device,
            )
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = round(float(losses.mean()), 6)
    model.train()
    return out


def save_checkpoint(
    model: HelixTransformer,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val: float,
    out_dir: str,
    tag: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val": best_val,
    }
    path = os.path.join(out_dir, f"ckpt-{tag}.pt")
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: HelixTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = cfg.train.device
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    print(f"[HELIX] device={device} threads={torch.get_num_threads()}")

    # ---- data ----
    data_ok = os.path.exists(cfg.data.train_file) and os.path.exists(cfg.data.val_file)
    if data_ok:
        train_ids = load_ids(cfg.data.train_file)
        val_ids = load_ids(cfg.data.val_file)
        print(
            f"[HELIX] data train={len(train_ids)} val={len(val_ids)} tokens "
            f"(vocab target {cfg.model.vocab_size})"
        )
    else:
        print(f"[HELIX] building tokenizer and data from {cfg.data.corpus_file}")
        tok, train_ids, val_ids = build_from_corpus(
            cfg.data.corpus_file,
            cfg.data.vocab_file,
            cfg.data.merges_file,
            cfg.model.vocab_size,
            cfg.data.train_frac,
            cfg.data.train_file,
            cfg.data.val_file,
        )
        # Align configured vocab size to the actual learned vocabulary.
        cfg.model.vocab_size = len(tok)
        print(f"[HELIX] learned vocab size = {len(tok)}")

    # ---- align model vocab to the actual learned vocabulary (covers the
    # ---- data-already-exists branch, keeping checkpoints consistent) ----
    import json as _json
    with open(cfg.data.vocab_file, encoding="utf-8") as _f:
        cfg.model.vocab_size = len(_json.load(_f)["vocab"])
    print(f"[HELIX] model vocab size set to {cfg.model.vocab_size}")

    # ---- model ----
    model = HelixTransformer(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[HELIX] model parameters: {n_params:,}")
    optimizer = model.configure_optimizers(cfg.train)

    iter_num = 0
    best_val = float("inf")
    start_time = time.time()

    # ---- resume ----
    if cfg.train.resume_from and os.path.exists(cfg.train.resume_from):
        ckpt = load_checkpoint(cfg.train.resume_from, model, optimizer)
        iter_num = int(ckpt.get("iter_num", 0))
        best_val = float(ckpt.get("best_val", best_val))
        print(f"[HELIX] resumed from {cfg.train.resume_from} at iter {iter_num}")

    log_path = os.path.join(cfg.train.out_dir, "log.jsonl")
    os.makedirs(cfg.train.out_dir, exist_ok=True)

    model.train()
    t0 = time.time()
    running_loss = 0.0

    while iter_num < cfg.train.max_iters:
        lr = get_lr(iter_num, cfg.train)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # gradient accumulation
        for micro in range(max(1, cfg.train.gradient_accumulation_steps)):
            X, Y = get_batch(
                "train",
                train_ids,
                val_ids,
                cfg.train.batch_size,
                cfg.train.block_size,
                device,
            )
            _, loss = model(X, Y)
            loss = loss / max(1, cfg.train.gradient_accumulation_steps)
            loss.backward()
            running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1

        if iter_num % cfg.train.log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"iter {iter_num}: loss {running_loss/max(1,cfg.train.log_interval):.4f} "
                f"lr {lr:.2e} ({elapsed*1000/cfg.train.log_interval:.1f} ms/iter)"
            )
            running_loss = 0.0
            t0 = time.time()

        if iter_num % cfg.train.eval_interval == 0 or iter_num == cfg.train.max_iters:
            vals = estimate_loss(model, train_ids, val_ids, cfg.train)
            print(
                f"step {iter_num}: train_loss {vals['train']:.4f} val_loss {vals['val']:.4f}"
            )
            log_entry = {
                "iter": iter_num,
                "train_loss": vals["train"],
                "val_loss": vals["val"],
                "lr": lr,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            if vals["val"] < best_val:
                best_val = vals["val"]
                save_checkpoint(model, optimizer, iter_num, best_val, cfg.train.out_dir, "best")

        if cfg.train.save_interval and iter_num % cfg.train.save_interval == 0:
            save_checkpoint(model, optimizer, iter_num, best_val, cfg.train.out_dir, f"iter{iter_num}")

    # final checkpoint
    save_checkpoint(model, optimizer, iter_num, best_val, cfg.train.out_dir, "final")
    total = time.time() - start_time
    print(f"[HELIX] training done in {total:.1f}s. best_val={best_val:.4f}")
    print(f"[HELIX] artifacts in {cfg.train.out_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HELIX training")
    parser.add_argument("--config", required=True, help="path to config JSON")
    args = parser.parse_args()
    train(args.config)
