"""HELIX curriculum planner: staged, rights-aware training over subject domains.

The full ambition — archive-scale books, 120+ languages, and distillation from
the five biggest open-source LLMs (Llama, Mistral, Falcon, OLMo, GPT-NeoX
families) — is planned here as an ordered curriculum with per-stage token
budgets, eval gates and rights requirements. The planner produces a machine-
readable plan (`plan.json`) that the autonomous driver executes *only* against
corpora whose manifests already carry a `train` rights decision, reusing the
existing rights-aware staging in helix/sources.py and helix/corpus_pipeline.py.

Curriculum ordering follows the classic "basic → research level" progression
per subject, interleaved by difficulty so gradient updates stay stable
(curriculum learning, Bengio et al., 2009; and the stage-gated data mixing of
LLaMA-3 / OLMo technical reports). Each stage ends with an eval gate; failing
a gate halts the loop instead of compounding garbage (this is the anti-
"answers anything" safeguard at the data level).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Subject stages: (domain, tier, target token share). Tiers:
# basic -> intermediate -> advanced -> research. Language coverage tiers are
# explicit because a 512-token byte BPE can only *partially* cover 120+
# scripts; full coverage requires a larger Unicode-aware vocabulary, which the
# plan flags as a prerequisite rather than pretending it is free.
SUBJECT_STAGES: List[Dict] = [
    # --- foundational literacy first (tokenizer + general web prose) ---
    {"domain": "general", "tier": "basic", "share": 0.10},
    # --- mathematics: arithmetic -> proof/research ---
    {"domain": "mathematics", "tier": "basic", "share": 0.06},
    {"domain": "mathematics", "tier": "intermediate", "share": 0.04},
    {"domain": "mathematics", "tier": "advanced", "share": 0.03},
    {"domain": "mathematics", "tier": "research", "share": 0.02},
    # --- physics ---
    {"domain": "physics", "tier": "basic", "share": 0.04},
    {"domain": "physics", "tier": "intermediate", "share": 0.03},
    {"domain": "physics", "tier": "advanced", "share": 0.02},
    {"domain": "physics", "tier": "research", "share": 0.02},
    # --- chemistry / biology / medicine ---
    {"domain": "chemistry", "tier": "basic", "share": 0.03},
    {"domain": "chemistry", "tier": "advanced", "share": 0.02},
    {"domain": "biology", "tier": "basic", "share": 0.03},
    {"domain": "biology", "tier": "research", "share": 0.02},
    {"domain": "medicine", "tier": "intermediate", "share": 0.02},
    # --- philosophy & history ---
    {"domain": "philosophy", "tier": "basic", "share": 0.03},
    {"domain": "philosophy", "tier": "research", "share": 0.02},
    {"domain": "history", "tier": "basic", "share": 0.04},
    {"domain": "history", "tier": "advanced", "share": 0.03},
    # --- languages: high-, mid- and low-resource tiers (120+ script goal) ---
    {"domain": "language", "tier": "high_resource", "share": 0.12},
    {"domain": "language", "tier": "mid_resource", "share": 0.08},
    {"domain": "language", "tier": "low_resource", "share": 0.05},
    {"domain": "conversation", "tier": "fluent", "share": 0.05},
    # --- code (reasoning carrier; HumanEval-class verified tasks) ---
    {"domain": "code", "tier": "basic", "share": 0.04},
    {"domain": "code", "tier": "advanced", "share": 0.03},
    # --- distillation corpora from top open-source LLM families ---
    # These are *generated-instruction* datasets released under permissive
    # licenses (e.g. OpenAssistant, Alpaca-style sets derived from Llama/Mistral/
    # Falcon/OLMo/Pythra-family models); rights must be confirmed per dataset.
    {"domain": "distill_open_llms", "tier": "instructions", "share": 0.06},
]

TIER_ORDER = ["basic", "high_resource", "mid_resource", "instructions",
              "intermediate", "conversation", "fluent", "advanced",
              "low_resource", "research"]


@dataclass
class StagePlan:
    index: int
    domain: str
    tier: str
    target_tokens: int
    eval_gate: str            # metric name that must improve or loop halts
    min_gate_improvement: float
    rights_required: List[str] = field(default_factory=lambda: ["train"])
    status: str = "planned"   # planned | ready | blocked_no_rights | done


@dataclass
class CurriculumPlan:
    total_token_budget: int
    vocab_size: int
    block_size: int
    stages: List[StagePlan]
    notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def build_plan(total_token_budget: int, vocab_size: int = 4096,
               block_size: int = 256,
               available_domains: Optional[Dict[str, int]] = None) -> CurriculumPlan:
    """Turn the subject table into concrete stages with token budgets.

    `available_domains` maps "domain/tier" -> trainable tokens found in
    rights-cleared local manifests. Stages without cleared data are marked
    `blocked_no_rights`; the driver must halt on blocked stages rather than
    silently substituting scraped text.
    """
    shares = {tuple((s["domain"], s["tier"])): s["share"] for s in SUBJECT_STAGES}
    total_share = sum(shares.values())
    stages: List[StagePlan] = []
    for i, s in enumerate(SUBJECT_STAGES):
        budget = int(total_token_budget * s["share"] / total_share)
        key = f"{s['domain']}/{s['tier']}"
        avail = (available_domains or {}).get(key, 0)
        status = "ready" if avail >= budget // 4 else "blocked_no_rights"
        stages.append(StagePlan(i, s["domain"], s["tier"], budget,
                                f"{s['domain']}_{s['tier']}_val_loss",
                                0.02, ["train"], status))
    notes = [
        "A 512-token byte-level BPE cannot faithfully represent 120+ languages; "
        "stages assume a rebuilt Unicode-aware vocabulary of >=4k tokens before "
        "any language stage is marked ready.",
        "Distillation stages require per-dataset license review; model outputs "
        "of third-party LLMs are training data only where their terms allow.",
        "Eval gates use held-out loss plus refusal-calibration accuracy; a "
        "stage that fails its gate twice halts the curriculum (no garbage "
        "compounding).",
    ]
    return CurriculumPlan(total_token_budget, vocab_size, block_size, stages, notes)


def next_ready_stage(plan: CurriculumPlan) -> Optional[StagePlan]:
    for st in plan.stages:
        if st.status == "ready":
            return st
    return None


def mark_done(plan: CurriculumPlan, index: int) -> None:
    plan.stages[index].status = "done"
