#!/usr/bin/env python3
"""Select the chat checkpoint using held-out task performance and LM validation."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report.get("results"), list) or not report["results"]:
        raise ValueError("evaluation report must contain a non-empty results list")
    task_names = [item.get("task") for item in report["results"]]
    if any(not isinstance(name, str) or not name for name in task_names):
        raise ValueError("evaluation report contains an invalid task name")
    validation = report.get("validation") or {}
    loss = validation.get("loss", math.inf)
    try:
        loss = float(loss)
    except (TypeError, ValueError):
        loss = math.inf
    if not math.isfinite(loss):
        loss = math.inf
    total = int(report.get("total", len(report["results"])))
    passed = int(report.get("passed", sum(bool(item.get("pass_")) for item in report["results"])))
    if total < 1 or passed < 0 or passed > total:
        raise ValueError("evaluation report has invalid pass counts")
    return {"passed": passed, "total": total, "pass_rate": passed / total,
            "validation_loss": loss, "tasks": task_names}


def choose_checkpoint(base_report: dict[str, Any], sft_report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    base, sft = _metrics(base_report), _metrics(sft_report)
    if base["tasks"] != sft["tasks"]:
        raise ValueError("base and SFT evaluations must use the same ordered task set")
    if sft["passed"] > base["passed"]:
        selected = "sft"
        reason = "SFT passes more held-out capability probes"
    elif sft["passed"] < base["passed"]:
        selected = "base"
        reason = "base checkpoint passes more held-out capability probes"
    elif sft["validation_loss"] < base["validation_loss"]:
        selected = "sft"
        reason = "task scores tie and SFT has lower held-out language-model loss"
    else:
        selected = "base"
        reason = "task scores tie and base preserves equal or better held-out language-model loss"
    return selected, {"rule_version": 1, "selected_stage": selected, "reason": reason,
                      "base": base, "sft": sft,
                      "note": "Small diagnostic suite only; selection is not proof of broad intelligence or generalization."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--sft-eval", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-eval", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    base_report = json.loads(Path(args.base_eval).read_text(encoding="utf-8"))
    sft_report = json.loads(Path(args.sft_eval).read_text(encoding="utf-8"))
    selected, manifest = choose_checkpoint(base_report, sft_report)
    checkpoint = Path(args.base_checkpoint if selected == "base" else args.sft_checkpoint)
    evaluation = Path(args.base_eval if selected == "base" else args.sft_eval)
    for required in (checkpoint, evaluation):
        if not required.is_file():
            raise SystemExit(f"required selection input is missing: {required}")
    output_checkpoint, output_eval, manifest_path = map(Path, (args.output_checkpoint, args.output_eval, args.manifest))
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_eval.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, output_checkpoint)
    shutil.copyfile(evaluation, output_eval)
    manifest["selected_checkpoint"] = str(checkpoint.name)
    manifest["selected_evaluation"] = str(evaluation.name)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected_stage": selected, "reason": manifest["reason"],
                      "output_checkpoint": str(output_checkpoint),
                      "output_evaluation": str(output_eval),
                      "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
