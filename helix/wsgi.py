"""Production WSGI entrypoint with verified runtime model-artifact staging."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from helix.storage import REPOSITORY_ROOT, resolve_data_root
from helix.web import create_app


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_verified_model_artifacts() -> Path:
    """Copy immutable build-bundled weights/tokenizer to runtime data storage."""
    root = Path(REPOSITORY_ROOT)
    manifest_path = root / "artifacts/model-bundle-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("HELIX model bundle is missing; run scripts/fetch_model_bundle.py during build")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "artifacts/sft/sckpt-best.pt",
        "artifacts/data/vocab.json",
        "artifacts/data/merges.json",
    }
    records = manifest.get("files", {})
    if manifest.get("format") != 1 or not expected.issubset(records):
        raise RuntimeError("HELIX model bundle manifest is invalid")
    data_root = resolve_data_root()
    for name in sorted(expected):
        source = root / name
        record = records[name]
        if not source.is_file() or source.stat().st_size != record.get("bytes") or _sha256(source) != record.get("sha256"):
            raise RuntimeError(f"HELIX model bundle integrity check failed: {name}")
        target = data_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == source.stat().st_size and _sha256(target) == record["sha256"]:
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(source.read_bytes())
        if _sha256(temporary) != record["sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"HELIX runtime copy integrity check failed: {name}")
        os.replace(temporary, target)
    return data_root


_DATA_ROOT = _stage_verified_model_artifacts()
app = create_app(
    config_path=os.getenv("HELIX_CONFIG_PATH", "configs/helix-deploy.json"),
    llm_checkpoint=os.getenv("HELIX_CHECKPOINT_PATH", str(_DATA_ROOT / "artifacts/sft/sckpt-best.pt")),
    image_gen_config=os.getenv("HELIX_IMAGE_CONFIG_PATH", "artifacts/image-gen/config.json"),
    image_gen_checkpoint=os.getenv("HELIX_IMAGE_CHECKPOINT_PATH", "artifacts/image-gen/ckpt-final.pt"),
    vision_checkpoint=os.getenv("HELIX_VISION_CHECKPOINT_PATH", "artifacts/vision/ckpt-final.pt"),
    caption_dir=os.getenv("HELIX_CAPTION_DIR", "artifacts/caption"),
    external_origin=os.getenv("HELIX_EXTERNAL_ORIGIN") or None,
)
