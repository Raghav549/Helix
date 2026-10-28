#!/usr/bin/env python3
"""Package a trained HELIX checkpoint and its tokenizer into a small deploy bundle."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--config", default="configs/helix-research.json")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--sft-metrics", default=None, help="Optional SFT metrics JSON embedded in the bundle manifest")
    parser.add_argument("--sft-data-manifest", default=None, help="Optional generated SFT data provenance manifest")
    parser.add_argument("--output", default="out/helix-chat-bundle.zip")
    args = parser.parse_args()
    inputs = {
        "artifacts/sft/sckpt-best.pt": Path(args.checkpoint),
        "artifacts/data/vocab.json": Path(args.vocab),
        "artifacts/data/merges.json": Path(args.merges),
        "configs/helix-deploy.json": Path(args.config),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"required bundle input missing: {path} ({name})")
    evaluation = Path(args.evaluation)
    if not evaluation.is_file():
        raise SystemExit(f"evaluation report missing: {evaluation}")
    sft_metrics = None
    if args.sft_metrics:
        metrics_path = Path(args.sft_metrics)
        if not metrics_path.is_file():
            raise SystemExit(f"SFT metrics report missing: {metrics_path}")
        sft_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    sft_data_manifest = None
    if args.sft_data_manifest:
        data_manifest_path = Path(args.sft_data_manifest)
        if not data_manifest_path.is_file():
            raise SystemExit(f"SFT data manifest missing: {data_manifest_path}")
        sft_data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "format": 1,
        "model": "HELIX random-initialized experimental language model",
        "warning": "Small research model; not a general-purpose or production-grade assistant.",
        "files": {name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for name, path in inputs.items()},
        "evaluation": json.loads(evaluation.read_text(encoding="utf-8")),
        "sft_training": sft_metrics,
        "sft_data_manifest": sft_data_manifest,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in inputs.items():
            archive.write(path, name)
        archive.writestr("artifacts/model-bundle-manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"bundle": str(output), "sha256": sha256(output),
                      "bytes": output.stat().st_size, "files": list(inputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
