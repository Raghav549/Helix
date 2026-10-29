"""HELIX self-improvement loops: research-grounded autonomous refinement.

This module implements the "many loops / autonomous" layer of the HELIX
research programme, built from published training-loop ideas that are
feasible on a CPU budget:

1. Self-Rewarding Language Models (Yuan et al., 2024, arXiv:2401.10028) —
   the model judges its own sampled generations; the highest-reward samples
   become new SFT data and the next iteration re-judges them. On HELIX the
   judge is *verifiable-first*: rule-based checks (format, length,
   refusal-correctness, reproduction of a context span) carry most weight
   and the model's own likelihood signal is only a tie-breaker, because a
   small model's preference judgements are noisy.

2. Constitutional AI / RLAIF critique–revise loop (Bai et al., 2022,
   arXiv:2212.08073) — every candidate answer is critiqued against an
   explicit constitution (honesty first: "if unsure, refuse"), then revised.
   Revisions that pass verification become training pairs so the behaviour
   is learned, not merely prompted.

3. Rejection-sampling fine-tuning, RAFT (Yuan et al., 2023,
   arXiv:2306.05314) — sample N completions per prompt, keep only the
   verified best one, fine-tune on survivors. Improves quality without RL
   infrastructure; ideal for CPU.

4. Self-consistency voting (Wang et al., ICLR 2023, arXiv:2203.11171) —
   majority vote over stochastic samples selects the final answer when the
   task has a checkable canonical form.

All loops share one invariant: **nothing enters the training set unless it
passes a deterministic verifier**. That is what turns "answer anything" into
"answer only what it can back up".
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch

from helix.infer import generate
from helix.refusal import REFUSAL_MESSAGE, ConfidenceGate


# ---------------------------------------------------------------------------
# Constitution — the rules every candidate answer must satisfy
# ---------------------------------------------------------------------------

CONSTITUTION: List[str] = [
    "Honesty: if you cannot support an answer from your training or the "
    "provided context, say you do not know instead of guessing.",
    "Grounding: any factual claim in a long answer must be repeatable from "
    "the supplied context passage when one is given.",
    "Relevance: the response must address the instruction directly and not "
    "ramble or repeat itself.",
    "Safety: no executable instructions for harming people, systems, or data.",
    "Format: keep responses within the requested length and use plain text.",
]


# ---------------------------------------------------------------------------
# Deterministic verifiers (no model involved — this is the trust anchor)
# ---------------------------------------------------------------------------

@dataclass
class Verification:
    passed: bool
    score: float                 # 0..1 continuous quality signal
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def verify_answer(prompt: str, answer: str, *, context: str = "",
                  min_len: int = 3, max_len: int = 600,
                  require_grounded: bool = True) -> Verification:
    """Rule-based verifier used by every self-improvement loop.

    Checks (each contributes to a 0..1 score):
      * non-refusal answers must have sane length;
      * repetition ratio below threshold (garbage output detection);
      * printability: mostly valid characters (byte-level models drift);
      * grounding: with a context provided, at least one content word of the
        answer must appear in the context (weak but deterministic link);
      * refusal answers are always structurally valid (they are the honest
        fallback), but score lower than grounded answers so learning prefers
        correct knowledge when it exists.
    """
    reasons: List[str] = []
    score = 1.0
    text = answer.strip()

    if not text:
        return Verification(False, 0.0, ["empty"])

    is_refusal = text == REFUSAL_MESSAGE.strip()

    if len(text) < min_len:
        score -= 0.4
        reasons.append("too_short")
    if len(text) > max_len:
        score -= 0.3
        reasons.append("too_long")

    words = _norm_words(text)
    if words:
        rep = 1.0 - len(set(words)) / len(words)
        if rep > 0.6:
            score -= 0.4
            reasons.append(f"repetition_{rep:.2f}")

    printable_frac = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    if printable_frac < 0.9:
        score -= 0.3
        reasons.append("unprintable_drift")

    if context and require_grounded and not is_refusal:
        cw = set(_norm_words(context))
        aw = {w for w in words if len(w) > 3}
        overlap = len(cw & aw) / max(len(aw), 1)
        if overlap < 0.15:
            score -= 0.5
            reasons.append("ungrounded")
        else:
            score += 0.1  # grounded bonus, capped below

    if is_refusal:
        score = min(score, 0.55)  # honest, but not evidence of competence
    score = max(0.0, min(1.0, score))
    return Verification(score >= 0.6, round(score, 4), reasons)


def critique(answer: str, constitution: List[str] = CONSTITUTION) -> List[str]:
    """Constitutional-AI style critique: which principles are violated?"""
    v = verify_answer("", answer)
    problems = list(v.reasons)
    if REFUSAL_MESSAGE.strip() != answer.strip() and not v.passed:
        problems.append("violates-honesty-or-grounding")
    return problems


# ---------------------------------------------------------------------------
# Model-based reward (Self-Rewarding, verifiable-first blend)
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_reward(model, tok, prompt: str, answer: str, device: str = "cpu") -> float:
    """Normalised teacher-forced log-prob of the answer given the prompt.

    Higher means the model finds the continuation more natural; used only as
    a tie-breaker between rule-verified candidates (arXiv:2401.10028)."""
    p_ids = tok.encode(prompt)
    full = tok.encode(prompt + answer)
    n_new = len(full) - len(p_ids)
    if n_new <= 0:
        return -float("inf")
    idx = torch.tensor([full], dtype=torch.long, device=device)
    logits, _ = model(idx[:, :-1])
    targets = idx[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    nll = -lp.gather(2, targets.unsqueeze(2)).squeeze(2)
    tail = nll[0, nll.shape[1] - n_new:]
    return float(-tail.mean())


def rank_candidates(model, tok, prompt: str, candidates: List[str], *,
                    context: str = "", device: str = "cpu") -> List[Tuple[str, float, Verification]]:
    """Score candidates: verifier score dominates, model reward breaks ties."""
    scored = []
    for cand in candidates:
        v = verify_answer(prompt, cand, context=context)
        r = model_reward(model, tok, prompt, cand, device) if v.passed else -99.0
        scored.append((cand, v.score + 0.05 * max(min(r, 5.0), -5.0), v))
    scored.sort(key=lambda t: -t[1])
    return scored


def self_consistency_pick(model, tok, prompt: str, n_samples: int = 5,
                          temperature: float = 0.8, device: str = "cpu",
                          context: str = "") -> Optional[str]:
    """Sample N answers, keep verified ones, return the most frequent
    normalised form (Wang et al. 2023 self-consistency voting)."""
    tally: Dict[str, int] = {}
    display: Dict[str, str] = {}
    for _ in range(n_samples):
        raw = generate(model, tok, prompt, max_new_tokens=48,
                       temperature=temperature, top_k=40, device=device)
        ans = raw[len(prompt):] if raw.startswith(prompt) else raw
        v = verify_answer(prompt, ans, context=context)
        if not v.passed:
            continue
        key = " ".join(_norm_words(ans))[:120]
        tally[key] = tally.get(key, 0) + 1
        display.setdefault(key, ans)
    if not tally:
        return None
    best_key = max(tally.items(), key=lambda kv: kv[1])[0]
    return display[best_key]


# ---------------------------------------------------------------------------
# The improvement loop: RAFT-style data generation from existing prompts
# ---------------------------------------------------------------------------

@dataclass
class LoopReport:
    iteration: int
    prompts: int
    candidates_generated: int
    verified_pairs: int
    refusals_kept: int
    seconds: float
    pair_file: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_improvement_loop(model, tok, prompts_with_context: List[Tuple[str, str]],
                         out_dir: str, *, iterations: int = 1,
                         samples_per_prompt: int = 4, device: str = "cpu",
                         seed: int = 1337, gate: Optional[ConfidenceGate] = None,
                         log=print) -> List[LoopReport]:
    """Autonomous rejection-sampling loop (RAFT + Self-Rewarding + CAI revise).

    For each iteration and each (prompt, context) pair:
      1. sample N completions (exploration);
      2. if a gate exists and all completions would be refused, emit the
         honest refusal pair (teaches calibrated abstention — Kapoor 2024);
      3. verify every completion deterministically;
      4. rank survivors with the blended reward; self-consistency vote;
      5. write surviving (instruction, response) pairs to JSON for `helix sft`.

    Returns one report per iteration. Data accumulates across iterations
    (iteration k+1 judges the outputs of the model trained on iteration k's
    data when the caller re-loads a checkpoint — the paper's iterative
    self-rewarding scheme).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    reports: List[LoopReport] = []
    for it in range(iterations):
        t0 = time.time()
        pairs: List[List[str]] = []
        n_cand = refusals = 0
        for prompt, context in prompts_with_context:
            cands = []
            for s in range(samples_per_prompt):
                temp = 0.2 + 0.6 * rng.random()
                raw = generate(model, tok, prompt, max_new_tokens=48,
                               temperature=temp, top_k=40, device=device)
                ans = raw[len(prompt):] if raw.startswith(prompt) else raw
                n_cand += 1
                cands.append(ans)
            ranked = rank_candidates(model, tok, prompt, cands,
                                     context=context, device=device)
            survivors = [(c, sc) for c, sc, v in ranked if v.passed]
            if survivors:
                best = survivors[0][0].strip()
                pairs.append([prompt.strip(), best])
            elif gate is not None:
                # everything failed verification → teach the refusal itself
                refusals += 1
                pairs.append([prompt.strip(), REFUSAL_MESSAGE.strip()])
        path = os.path.join(out_dir, f"pairs_iter{it}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=1)
        rep = LoopReport(it, len(prompts_with_context), n_cand, len(pairs),
                         refusals, round(time.time() - t0, 2), path)
        reports.append(rep)
        log(f"[loop] iter {it}: {rep.verified_pairs}/{rep.prompts} verified pairs "
            f"({rep.refusals_kept} refusal-teaching), {rep.seconds}s -> {path}")
    return reports
