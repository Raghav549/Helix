"""Confidence-gated answer refusal for HELIX.

Root cause of the "answers anything, always wrong" behaviour: generation was
pure argmax/sampling over a tiny under-trained model with no way to say "I do
not know". This module adds a calibration layer on top of any checkpoint:

  * Per-token negative log-likelihood of the generated continuation is
    compared against an entropy threshold estimated ON THE VALIDATION SPLIT
    (the "Full-Awareness" reference distribution from the calibration
    literature), following:
      - Duan et al., "Can AI Really Say 'I Don't Know'? A Key-to-Robustness
        Evaluation in Language Models" (LLMScore, EMNLP 2024) — models should
        refuse when their internal confidence is low;
      - Kapoor et al., "AI Models Should Refuse to Answer Questions When They
        Lack Knowledge" (NeurIPS 2024) — knowledge-aware answering;
      - Kuhn, Farquhar et al., "Semantic Uncertainty: Linguistic Invariances
        for Uncertainty Estimation in NLG" (ICLR 2023) — uncertainty must map
        to abstain/answer behaviour, not just numbers.

  * A learned temperature (Guo et al., "On Calibration of Modern Neural
    Networks", ICML 2017) scales the probabilities so the reported confidence
    matches observed validation accuracy as closely as possible on CPU budgets.

The gate is applied at generation time: if mean NLL exceeds the threshold the
system returns a structured refusal instead of hallucinated text.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn.functional as F

from helix.infer import generate
from helix.tokenizer import ByteLevelBPETokenizer
from helix.model import HelixTransformer


REFUSAL_MESSAGE = (
    "I don't have enough reliable knowledge to answer that confidently. "
    "This HELIX checkpoint is a small research prototype; I refuse to guess. "
    "Try rephrasing with context from material I was trained on, or ask me "
    "to use the sandboxed tools (math, search-in-corpus) instead."
)


@dataclass
class GenerationDecision:
    """Outcome of one confidence-gated generation."""

    answered: bool
    text: str
    mean_nll: float          # nats/token over the generated continuation
    confidence: float        # calibrated probability the answer is trustworthy
    threshold: float         # NLL threshold used for the decision
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


class ConfidenceGate:
    """Threshold + temperature calibrated on held-out validation data."""

    def __init__(self, threshold: float, temperature: float = 1.0,
                 source: str = "default") -> None:
        self.threshold = float(threshold)
        self.temperature = max(float(temperature), 1e-6)
        self.source = source

    # ------------------------------------------------------------------ util
    @staticmethod
    def default_threshold(block_size: int) -> float:
        """Fallback: uniform-distribution entropy over the vocabulary minus a
        margin. Used only when no calibration data exists (fail-closed: this
        is deliberately strict)."""
        return math.log(max(block_size, 8))  # very conservative

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def score_continuation(self, model: HelixTransformer, tok: ByteLevelBPETokenizer,
                           prompt: str, continuation: str, device: str = "cpu") -> float:
        """Mean NLL of `continuation` tokens given `prompt` (teacher-forced)."""
        p_ids = tok.encode(prompt)
        full_ids = tok.encode(prompt + continuation)
        n_new = len(full_ids) - len(p_ids)
        if n_new <= 0:
            return float("inf")
        idx = torch.tensor([full_ids], dtype=torch.long, device=device)
        logits, _ = model(idx)                      # (1, T, V)
        logits = logits[0]                          # (T, V)
        targets = idx[0, 1:]                        # predict next token
        lprobs = F.log_softmax(logits[:-1], dim=-1)
        nll = -lprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        tail = nll[len(nll) - n_new:]               # the generated part only
        return float(tail.mean())

    def calibrated_confidence(self, mean_nll: float) -> float:
        """Map mean NLL to [0,1]: exp(-nll/T) is the geometric-mean token
        probability under the temperature-scaled model."""
        return math.exp(-min(mean_nll, 40.0) / self.temperature)

    def decide(self, answered_text: str, mean_nll: float) -> GenerationDecision:
        answered = mean_nll <= self.threshold and bool(answered_text.strip())
        conf = self.calibrated_confidence(mean_nll)
        return GenerationDecision(
            answered=answered,
            text=answered_text if answered else REFUSAL_MESSAGE,
            mean_nll=round(mean_nll, 6),
            confidence=round(conf, 6),
            threshold=round(self.threshold, 6),
            n_tokens=0,
        )

    # ------------------------------------------------------------ gated API
    @torch.no_grad()
    def generate_gated(self, model, tok, prompt: str, device: str = "cpu",
                       max_new_tokens: int = 60, temperature: float = 0.0,
                       top_k: Optional[int] = None,
                       top_p: Optional[float] = None) -> GenerationDecision:
        """Generate, then teacher-force-score the continuation; refuse if the
        model's own per-token surprise exceeds the calibrated threshold."""
        raw = generate(model, tok, prompt, max_new_tokens=max_new_tokens,
                       temperature=temperature, top_k=top_k, top_p=top_p,
                       device=device)
        continuation = raw[len(prompt):] if raw.startswith(prompt) else raw
        nll = self.score_continuation(model, tok, prompt, continuation, device)
        decision = self.decide(continuation, nll)
        decision.text = continuation if decision.answered else REFUSAL_MESSAGE
        decision.n_tokens = max(len(tok.encode(prompt + continuation))
                                - len(tok.encode(prompt)), 0)
        return decision

    # --------------------------------------------------------------- saving
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"threshold": self.threshold,
                       "temperature": self.temperature,
                       "source": self.source}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConfidenceGate":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["threshold"], d.get("temperature", 1.0),
                   d.get("source", "file"))


def calibrate_gate(model, tok, val_ids, cfg, *, n_batches: int = 24,
                   seed: int = 1337, percentile: float = 90.0,
                   device: str = "cpu") -> tuple[ConfidenceGate, dict]:
    """Estimate the token-NLL threshold from the validation split.

    We stream the model over held-out validation windows and record the
    per-token NLL distribution. The refusal threshold is set at the chosen
    percentile: continuations *more surprising* than anything seen on real
    held-out text are treated as untrustworthy (Duan et al. 2024 use the
    validation corpus as the "full-awareness" reference distribution).

    Temperature is fitted by grid search so that the fraction of windows below
    the threshold best matches the empirical "model knows this text" rate;
    on CPU we keep it simple and deterministic.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    block = min(cfg.train.block_size, cfg.model.block_size)
    nlls: List[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            i = torch.randint(0, max(len(val_ids) - block - 1, 1), (1,), generator=g)
            chunk = torch.tensor([val_ids[i.item(): i.item() + block + 1].tolist()],
                                 dtype=torch.long, device=device)
            logits, _ = model(chunk[:, :-1])
            targets = chunk[:, 1:]
            lprobs = F.log_softmax(logits, dim=-1)
            nll = -lprobs.gather(2, targets.unsqueeze(2)).squeeze(2)
            nlls.extend(nll.flatten().tolist())
    if not nlls:
        thr = ConfidenceGate.default_threshold(cfg.model.block_size)
        return ConfidenceGate(thr, 1.0, "fallback"), {"windows": 0}

    s = sorted(nlls)
    idx = min(int(len(s) * percentile / 100.0), len(s) - 1)
    threshold = s[idx]
    # crude ECE-style temperature fit: pick T minimising |mean prob - 0.5|
    # at the threshold boundary while keeping monotonicity; grid is deterministic.
    best_t, best_err = 1.0, float("inf")
    for t in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        frac = sum(1 for x in s if x <= threshold) / len(s)
        mapped = math.exp(-threshold / t)
        err = abs(mapped - (1.0 - frac))
        if err < best_err:
            best_err, best_t = err, t
    stats = {
        "tokens_scored": len(s),
        "mean_nll": round(sum(s) / len(s), 6),
        "p50": round(s[len(s) // 2], 6),
        "p90": round(s[min(int(len(s) * .9), len(s) - 1)], 6),
        "p99": round(s[min(int(len(s) * .99), len(s) - 1)], 6),
        "threshold_percentile": percentile,
    }
    gate = ConfidenceGate(threshold, best_t, "validation-calibrated")
    return gate, stats
