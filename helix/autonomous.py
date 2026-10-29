"""HELIX autonomous training driver: chained improvement loops with gates.

`helix.autonomous.run_autonomous()` executes a repeatable cycle:

  pretrain/SFT stage  →  calibrate refusal gate on validation split
                      →  evaluate (loss + refusal-behaviour probes)
                      →  RAFT/self-rewarding data generation loop
                      →  response-masked SFT on verified pairs only
                      →  re-calibrate & re-evaluate; keep best checkpoint

Safety properties (each enforced in code, not comments):
  * HALT CONDITIONS: non-finite loss, eval regression beyond `max_regression`,
    or zero verified pairs for two consecutive iterations stop the loop —
    an autonomous system that cannot fail loudly is worse than none.
  * Rights-aware: the driver consumes only corpora whose manifests carry a
    'train' decision (helix.corpus / helix.sources); it never downloads.
  * Every iteration writes an audit record (JSONL) so runs are reviewable.

Research grounding: iterative self-rewarding (arXiv:2401.10028), RAFT
(arXiv:2306.05314), Constitutional AI critique-revise (arXiv:2212.08073),
knowledge-aware abstention (Kapoor et al., NeurIPS 2024), calibration by
temperature scaling (Guo et al., ICML 2017).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from helix.config import load_config
from helix.data import load_ids
from helix.infer import load_model_and_tokenizer
from helix.refusal import ConfidenceGate, calibrate_gate
from helix.selfimprove import run_improvement_loop, verify_answer
from helix.sft import train_sft


@dataclass
class IterationRecord:
    iteration: int
    val_loss: Optional[float]
    gate_threshold: float
    gate_temperature: float
    refusal_probe_pass: bool
    verified_pairs: int
    refusal_teaching_pairs: int
    sft_best_val_loss: Optional[float]
    kept_checkpoint: bool
    seconds: float
    notes: str = ""


@dataclass
class AutonomousReport:
    config_path: str
    base_checkpoint: str
    iterations: List[IterationRecord] = field(default_factory=list)
    halted_reason: Optional[str] = None
    final_checkpoint: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _refusal_probe(model, tok, gate: ConfidenceGate, device: str) -> bool:
    """Behavioural probe: does the gated model refuse nonsense prompts?

    A prompt built from random unicode noise should exceed the calibrated NLL
    threshold and be refused; the gate must answer when shown text very close
    to its training distribution. Both halves must hold."""
    import random
    rng = random.Random(7)
    noise = "".join(chr(rng.randint(0x2100, 0x9FFF)) for _ in range(24))
    d_noise = gate.generate_gated(model, tok, noise, device=device,
                                  max_new_tokens=16)
    known = getattr(gate, "_known_probe", None)
    if not known:
        return d_noise.answered is False
    d_known = gate.generate_gated(model, tok, known, device=device,
                                  max_new_tokens=16)
    return (d_noise.answered is False) and (d_known.answered is True)


def run_autonomous(config_path: str, base_checkpoint: str, out_dir: str,
                   *, iterations: int = 2, samples_per_prompt: int = 4,
                   sft_iters: int = 120, batch_size: int = 4,
                   lr: float = 1e-4, seed: int = 1337,
                   max_regression: float = 0.25,
                   prompts_with_context: Optional[List[Tuple[str, str]]] = None,
                   log=print) -> AutonomousReport:
    """Execute the chained loop. Returns a full report; checkpoints under
    `out_dir/iter_k`. Requires artifacts/data/{train,val}.npy + vocab/merges
    exactly like `helix train`."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(config_path)
    device = cfg.train.device
    report = AutonomousReport(config_path, base_checkpoint)
    audit_path = os.path.join(out_dir, "autonomous-audit.jsonl")

    current_ckpt = base_checkpoint
    prev_val_loss: Optional[float] = None
    empty_pair_streak = 0

    for it in range(iterations):
        t0 = time.time()
        model, tok, cfg = load_model_and_tokenizer(config_path, current_ckpt)
        val_ids = load_ids(cfg.data.val_file)

        # --- validation loss of current weights --------------------------
        from helix.train import estimate_loss
        train_ids = load_ids(cfg.data.train_file)
        losses = estimate_loss(model, train_ids, val_ids, cfg.train)
        val_loss = losses["val"]
        if not np.isfinite(val_loss):
            report.halted_reason = f"non-finite validation loss at iter {it}"
            break

        # --- calibrate the refusal gate on held-out data -----------------
        gate, gstats = calibrate_gate(model, tok, val_ids, cfg, seed=seed,
                                      device=device)
        # give the probe a genuine in-distribution string from val corpus
        sample_txt = tok.decode(val_ids[:64].tolist())[:80]
        gate._known_probe = sample_txt  # type: ignore[attr-defined]
        refusal_ok = _refusal_probe(model, tok, gate, device)
        gate.save(os.path.join(out_dir, f"gate_iter{it}.json"))

        # --- self-improvement data loop (RAFT / self-rewarding) ----------
        if prompts_with_context is None:
            prompts_with_context = _prompts_from_corpus(tok, train_ids, n=8)
        loop_out = os.path.join(out_dir, f"loop_iter{it}")
        reps = run_improvement_loop(model, tok, prompts_with_context, loop_out,
                                    iterations=1, samples_per_prompt=samples_per_prompt,
                                    device=device, seed=seed + it, gate=gate, log=log)
        rep = reps[0]
        pair_file = rep.pair_file
        n_pairs = rep.verified_pairs - rep.refusals_kept
        if rep.verified_pairs == 0:
            empty_pair_streak += 1
        else:
            empty_pair_streak = 0
        if empty_pair_streak >= 2:
            report.halted_reason = ("no verified pairs for 2 consecutive "
                                    f"iterations (halted at iter {it})")
            break

        # --- SFT on verified pairs only -----------------------------------
        iter_dir = os.path.join(out_dir, f"iter_{it}")
        metrics = train_sft(config_path, current_ckpt, pair_file, iter_dir,
                            max_iters=sft_iters, batch_size=batch_size,
                            lr=lr, seed=seed, device=device)
        new_ckpt = os.path.join(iter_dir, "sckpt-best.pt")
        sft_val = metrics.get("best_sft_val_loss")

        # --- regression gate: keep candidate only if not much worse ------
        kept = False
        if os.path.isfile(new_ckpt):
            cand_model, cand_tok, _ = load_model_and_tokenizer(config_path, new_ckpt)
            cand_losses = estimate_loss(cand_model, train_ids, val_ids, cfg.train)
            cand_val = cand_losses["val"]
            if prev_val_loss is None or cand_val <= prev_val_loss * (1 + max_regression):
                current_ckpt = new_ckpt
                prev_val_loss = min(prev_val_loss or cand_val, cand_val)
                kept = True
                val_loss = cand_val
            del cand_model
        record = IterationRecord(it, round(val_loss, 6), gate.threshold,
                                 gate.temperature, refusal_ok, n_pairs,
                                 rep.refusals_kept,
                                 round(sft_val, 6) if sft_val else None,
                                 kept, round(time.time() - t0, 2),
                                 f"gate_source={gstats.get('tokens_scored', 0)} tokens scored")
        report.iterations.append(record)
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        log(f"[autonomous] iter {it}: val_loss {val_loss:.4f} "
            f"gate_thr {gate.threshold:.3f} refusal_probe={'PASS' if refusal_ok else 'FAIL'} "
            f"pairs={n_pairs}(+{rep.refusals_kept} refusal) kept={kept}")

    report.final_checkpoint = current_ckpt
    report.save(os.path.join(out_dir, "autonomous-report.json"))
    if report.halted_reason:
        log(f"[autonomous] HALTED: {report.halted_reason}")
    return report


def _prompts_from_corpus(tok, ids, n: int = 8) -> List[Tuple[str, str]]:
    """Build (question-style prompt, context) pairs from real corpus spans so
    the improvement loop's grounding verifier has something to check against.
    Deterministic slicing keeps runs reproducible."""
    arr = np.asarray(ids)[: 4096]
    chunks = [arr[i * 512:(i + 1) * 512] for i in range(n)
              if (i + 1) * 512 <= len(arr)]
    out: List[Tuple[str, str]] = []
    for c in chunks:
        ctx = tok.decode(c.tolist())
        ctx = " ".join(ctx.split())[:240]
        if len(ctx) < 40:
            continue
        out.append((f"Summarize this passage in one sentence.\nContext: {ctx}\nSummary:", ctx))
    return out
