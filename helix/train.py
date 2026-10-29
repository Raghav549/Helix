"""CPU-first training loop: gradient accumulation, checkpoint save/resume,
deterministic seeds, JSON logging, and validation-driven best-checkpointing.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional
from dataclasses import asdict
from contextlib import nullcontext

import numpy as np
import torch

from helix.config import HelixConfig, load_config, save_config
from helix.data import build_from_corpus, get_batch, load_ids
from helix.model import HelixTransformer


def get_lr(it: int, train_cfg) -> float:
    """Cosine learning-rate decay with linear warmup."""
    if it < train_cfg.warmup_iters:
        return train_cfg.learning_rate * (it + 1) / max(train_cfg.warmup_iters, 1)
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
    config=None,
    scaler=None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val": best_val,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": asdict(config) if config is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else {},
    }
    path = os.path.join(out_dir, f"ckpt-{tag}.pt")
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(
    path: str,
    model: HelixTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def _peak_rss_mb() -> float:
    """Maximum resident set size of this process, in MiB (Linux/macOS)."""
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:  # pragma: no cover - platform dependent
        return 0.0


def _environment(device: str) -> dict:
    import platform
    import torch

    from helix import __version__ as helix_version

    try:
        import numpy as _np
        numpy_version = _np.__version__
    except Exception:  # pragma: no cover
        numpy_version = "unavailable"
    return {"helix_version": helix_version, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": numpy_version,
            "device": str(device), "threads": torch.get_num_threads(),
            "accelerator": "cuda" if torch.cuda.is_available() else
                           ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"),
            "platform": platform.platform()}


def _provenance(cfg) -> dict:
    """Record exactly which data, tokenizer and code produced this run."""
    import hashlib
    import subprocess

    from helix import __version__ as helix_version

    def digest(path: str) -> dict:
        if not path or not os.path.isfile(path):
            return {"path": os.path.basename(path or ""), "sha256": None, "bytes": None}
        hasher = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return {"path": os.path.basename(path), "sha256": hasher.hexdigest(),
                "bytes": os.path.getsize(path)}

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        git_commit = commit.stdout.strip() if commit.returncode == 0 else "unknown"
    except Exception:  # pragma: no cover - git may be unavailable
        git_commit = "unknown"
    meta = {}
    meta_path = cfg.data.train_file + ".meta.json"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            meta = {}
    return {"helix_version": helix_version, "git_commit": git_commit,
            "corpus": digest(cfg.data.corpus_file),
            "train_ids": digest(cfg.data.train_file),
            "val_ids": digest(cfg.data.val_file),
            "vocab": digest(cfg.data.vocab_file),
            "merges": digest(cfg.data.merges_file),
            "data_meta": meta}


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    save_config(cfg, config_path)
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = cfg.train.device
    from helix.runtime import configure_threads
    configure_threads()

    print(f"[HELIX] device={device} threads={torch.get_num_threads()}")

    # ---- data ----
    if cfg.train.block_size > cfg.model.block_size:
        raise ValueError("training block_size exceeds model context")
    if cfg.train.dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError("dtype must be float32, bfloat16 or float16")
    if cfg.train.dtype == "float16" and not str(device).startswith("cuda"):
        raise ValueError("float16 training requires CUDA; use float32 or bfloat16 on CPU")
    if min(cfg.train.batch_size, cfg.train.max_iters, cfg.train.eval_interval,
           cfg.train.eval_iters, cfg.train.log_interval, cfg.train.gradient_accumulation_steps) < 1:
        raise ValueError("training counts and intervals must be positive")
    data_ok = all(os.path.exists(p) for p in (cfg.data.train_file, cfg.data.val_file,
                                                cfg.data.vocab_file, cfg.data.merges_file))
    if data_ok:
        # Never silently train on prepared tokens belonging to a different corpus,
        # split, tokenizer or vocabulary target when provenance metadata exists.
        meta_path = cfg.data.train_file + ".meta.json"
        if os.path.isfile(meta_path):
            import hashlib
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if os.path.isfile(cfg.data.corpus_file):
                with open(cfg.data.corpus_file, 'rb') as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                if digest != meta.get('corpus_sha256'):
                    raise ValueError('prepared data corpus hash mismatch; rebuild the data and tokenizer')
            if meta.get('vocab_target', cfg.model.vocab_size) != cfg.model.vocab_size or meta.get('train_fraction', cfg.data.train_frac) != cfg.data.train_frac:
                raise ValueError('prepared data split or vocabulary target mismatch; rebuild the data and tokenizer')
            for name, path in (('train', cfg.data.train_file), ('val', cfg.data.val_file),
                               ('vocab', cfg.data.vocab_file), ('merges', cfg.data.merges_file)):
                expected = meta.get('file_sha256', {}).get(name)
                if expected:
                    with open(path, 'rb') as f:
                        if hashlib.sha256(f.read()).hexdigest() != expected:
                            raise ValueError(f'prepared {name} hash mismatch; rebuild the data and tokenizer')
        else:
            print('[HELIX] WARNING: prepared data have no provenance metadata; verify their origin independently')
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
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(cfg.train.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda" and cfg.train.dtype == "float16"))
    allow_nonfinite_skip = os.getenv("HELIX_ALLOW_NONFINITE_SKIP", "").strip().lower() in ("1", "true", "yes", "on")

    iter_num = 0
    best_val = float("inf")
    start_time = time.time()

    # ---- resume ----
    if cfg.train.resume_from:
        ckpt = load_checkpoint(cfg.train.resume_from, model, optimizer)
        iter_num = int(ckpt.get("iter_num", 0))
        best_val = float(ckpt.get("best_val", best_val))
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if device_type == "cuda" and ckpt.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[HELIX] resumed from {cfg.train.resume_from} at iter {iter_num}")

    log_path = os.path.join(cfg.train.out_dir, "log.jsonl")
    os.makedirs(cfg.train.out_dir, exist_ok=True)

    model.train()
    t0 = time.time()
    running_loss = 0.0
    skipped_steps = 0
    nonfinite_events = 0
    last_grad_norm = 0.0
    loss_was_nonfinite = False

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
            with (torch.autocast(device_type=device_type, dtype=autocast_dtype)
                  if autocast_dtype is not None else nullcontext()):
                _, loss = model(X, Y)
                loss = loss / cfg.train.gradient_accumulation_steps
            if not torch.isfinite(loss):
                nonfinite_events += 1
                optimizer.zero_grad(set_to_none=True)
                if not allow_nonfinite_skip:
                    raise RuntimeError(
                        f"non-finite loss at iteration {iter_num}; lower the learning rate or "
                        "set HELIX_ALLOW_NONFINITE_SKIP=1 to skip such steps and record them")
                skipped_steps += 1
                loss_was_nonfinite = True
                break
            scaler.scale(loss).backward()
            running_loss += loss.item()

        if loss_was_nonfinite:
            loss_was_nonfinite = False
            iter_num += 1
            continue
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            nonfinite_events += 1
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            if not allow_nonfinite_skip:
                raise RuntimeError(
                    f"non-finite gradient norm at iteration {iter_num}; training aborted so a "
                    "corrupt checkpoint is never written")
            iter_num += 1
            continue
        last_grad_norm = float(grad_norm)
        scaler.step(optimizer)
        scaler.update()
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
                save_checkpoint(model, optimizer, iter_num, best_val, cfg.train.out_dir, "best", cfg, scaler)

        if cfg.train.save_interval and iter_num % cfg.train.save_interval == 0:
            save_checkpoint(model, optimizer, iter_num, best_val, cfg.train.out_dir, f"iter{iter_num}", cfg, scaler)

    # final checkpoint
    save_checkpoint(model, optimizer, iter_num, best_val, cfg.train.out_dir, "final", cfg, scaler)
    total = time.time() - start_time
    print(f"[HELIX] training done in {total:.1f}s. best_val={best_val:.4f}")
    print(f"[HELIX] artifacts in {cfg.train.out_dir}/")
    tokens_processed = iter_num * cfg.train.batch_size * cfg.train.block_size * cfg.train.gradient_accumulation_steps
    metrics = {"stage": "pretraining", "initialization": "random" if not cfg.train.resume_from else "resume",
               "config": asdict(cfg), "parameters": n_params, "iterations": iter_num,
               "train_tokens_processed": tokens_processed,
               "best_validation_loss": best_val, "elapsed_s": round(total, 4),
               "tokens_per_second": round(tokens_processed / total, 2) if total > 0 else None,
               "skipped_steps": skipped_steps, "nonfinite_events": nonfinite_events,
               "last_grad_norm": round(last_grad_norm, 6),
               "peak_rss_mb": _peak_rss_mb(),
               "environment": _environment(cfg.train.device),
               "provenance": _provenance(cfg)}
    with open(os.path.join(cfg.train.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HELIX training")
    parser.add_argument("--config", required=True, help="path to config JSON")
    args = parser.parse_args()
    train(args.config)
