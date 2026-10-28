"""HELIX architecture ablation runner.

Trains several small models from random initialization on the SAME corpus
with the SAME token budget, differing in block type (compute differs)
(transformer / linear_attn / recurrent / hybrid). Reports validation loss,
training throughput, and quality per second for each variant so the next
architecture investment is based on measured evidence, not assumptions.

Every variant trains from scratch (random init) — no pretrained weights.
CPU-first; tiny models and few iterations so a full ablation completes on an
ordinary CPU in minutes.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List

import torch

from helix.config import HelixConfig, ModelConfig, TrainConfig, DataConfig, load_config, save_config
from helix.data import build_from_corpus, get_batch, load_ids
from helix.model import HelixTransformer


def get_lr(it: int, train_cfg) -> float:
    if it < train_cfg.warmup_iters:
        return train_cfg.learning_rate * it / max(train_cfg.warmup_iters, 1)
    if it > train_cfg.lr_decay_iters:
        return train_cfg.min_lr
    decay_ratio = (it - train_cfg.warmup_iters) / max(
        train_cfg.lr_decay_iters - train_cfg.warmup_iters, 1
    )
    coeff = 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * decay_ratio))
    return train_cfg.min_lr + coeff * (train_cfg.learning_rate - train_cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, train_cfg) -> Dict[str, float]:
    model.eval()
    out: Dict[str, float] = {}
    for split in ("train", "val"):
        # Fixed, architecture-independent batches at every evaluation.
        generator = torch.Generator().manual_seed(train_cfg.seed + (17 if split == "train" else 29))
        losses = torch.zeros(train_cfg.eval_iters)
        for k in range(train_cfg.eval_iters):
            X, Y = get_batch(split, train_ids, val_ids, train_cfg.batch_size,
                             train_cfg.block_size, train_cfg.device, generator=generator)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = round(float(losses.mean()), 6)
    model.train()
    return out


def train_variant(cfg: HelixConfig, train_ids, val_ids, out_dir: str) -> Dict:
    """Train one variant from random init; return metrics."""
    torch.manual_seed(cfg.train.seed)
    import numpy as np
    np.random.seed(cfg.train.seed)
    device = cfg.train.device
    from helix.runtime import configure_threads
    configure_threads()

    os.makedirs(out_dir, exist_ok=True)
    model = HelixTransformer(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = model.configure_optimizers(cfg.train)

    initial_val = estimate_loss(model, train_ids, val_ids, cfg.train)["val"]
    batch_generator = torch.Generator().manual_seed(cfg.train.seed + 47)
    start = time.perf_counter()
    train_losses: List[float] = []
    val_losses: List[float] = []
    total_tokens = 0

    model.train()
    for it in range(cfg.train.max_iters):
        lr = get_lr(it, cfg.train)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        X, Y = get_batch("train", train_ids, val_ids, cfg.train.batch_size,
                         cfg.train.block_size, device, generator=batch_generator)
        total_tokens += X.numel()
        _, loss = model(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(float(loss.item()))

        if it % cfg.train.eval_interval == 0 or it == cfg.train.max_iters - 1:
            v = estimate_loss(model, train_ids, val_ids, cfg.train)
            val_losses.append(v["val"])

    dt = time.perf_counter() - start
    final_val = estimate_loss(model, train_ids, val_ids, cfg.train)["val"]
    ckpt_path = os.path.join(out_dir, f"ckpt-{cfg.model.block_type.replace(',', '_')}.pt")
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, ckpt_path)

    return {
        "block_type": cfg.model.block_type,
        "n_params": n_params,
        "max_iters": cfg.train.max_iters,
        "train_tokens": total_tokens,
        "elapsed_s": round(dt, 3),
        "tokens_per_sec": round(total_tokens / max(dt, 1e-9), 2),
        "final_train_loss": round(train_losses[-1], 6) if train_losses else None,
        "final_val_loss": round(final_val, 6),
        "val_loss_curve": [round(v, 6) for v in val_losses],
        "checkpoint": ckpt_path,
        "initial_val_loss": initial_val,
        "heldout_loss_reduction_per_second": round((initial_val - final_val) / max(dt, 1e-9), 8),
    }


def run_ablation(config_path: str, out_path: str = "out/ablation-report.json",
                 block_types: List[str] | None = None) -> Dict:
    """Train one variant per block_type and compare."""
    base = load_config(config_path)
    if block_types is None:
        block_types = ["transformer", "linear_attn", "recurrent"]

    # Build/load data once so every variant sees identical tokens.
    from helix.data import build_from_corpus
    import os as _os
    if not (_os.path.exists(base.data.train_file) and _os.path.exists(base.data.val_file)):
        print(f"[ablate] building data from {base.data.corpus_file}")
        tok, train_ids, val_ids = build_from_corpus(
            base.data.corpus_file, base.data.vocab_file, base.data.merges_file,
            base.model.vocab_size, base.data.train_frac,
            base.data.train_file, base.data.val_file,
        )
        base.model.vocab_size = len(tok)
    else:
        train_ids = load_ids(base.data.train_file)
        val_ids = load_ids(base.data.val_file)

    # Align vocab to learned tokenizer.
    import json as _json
    if _os.path.exists(base.data.vocab_file):
        with open(base.data.vocab_file, encoding="utf-8") as f:
            base.model.vocab_size = len(_json.load(f)["vocab"])

    device = base.train.device
    print(f"[ablate] device={device} vocab={base.model.vocab_size} "
          f"train={len(train_ids)} val={len(val_ids)} tokens")

    results: List[Dict] = []
    for bt in block_types:
        cfg = copy.deepcopy(base)
        cfg.model.block_type = bt
        # Keep param count comparable across variants (same n_embd/n_layer/heads).
        tag = bt.replace(",", "_")
        out_dir = os.path.join(base.train.out_dir, f"ablate-{tag}")
        print(f"\n[ablate] variant: {bt}  (seed={cfg.train.seed}, iters={cfg.train.max_iters})")
        res = train_variant(cfg, train_ids, val_ids, out_dir)
        results.append(res)
        print(f"[ablate] {bt}: val_loss={res['final_val_loss']:.4f} "
              f"tok/s={res['tokens_per_sec']:.0f} "
              f"elapsed={res['elapsed_s']:.1f}s  params={res['n_params']:,}")

    # Ranking only applies to this small controlled experiment.
    ranked = sorted(results, key=lambda r: r["final_val_loss"])
    report = {
        "hardware_device": device,
        "variants": results,
        "best_by_val_loss": ranked[0]["block_type"] if ranked else None,
        "best_val_loss": ranked[0]["final_val_loss"] if ranked else None,
        "note": "Lower val loss = better. All variants trained from random init "
                "on identical batches with the same token budget; parameter count and wall compute differ.",
    }
    _os.makedirs(_os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ablate] best by val loss: {report['best_by_val_loss']} "
          f"({report['best_val_loss']:.4f})")
    print(f"[ablate] report saved to {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HELIX architecture ablation")
    parser.add_argument("--config", default="configs/helix-tiny.json")
    parser.add_argument("--out-path", default="out/ablation-report.json")
    parser.add_argument("--block-types", default="transformer,linear_attn,recurrent",
                        help="comma-separated block types to compare")
    args = parser.parse_args()
    types = [t.strip() for t in args.block_types.split(",") if t.strip()]
    run_ablation(args.config, args.out_path, types)
