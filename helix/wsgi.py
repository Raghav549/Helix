"""Production WSGI entrypoint with verified runtime model-artifact staging.

Startup policy: verification is strict, but a failure must not make the service
unreachable. If the model bundle is missing or corrupt, the process still starts,
`/healthz` and `/api/health` report `degraded`, and chat returns a precise JSON
503 instead of the platform serving an HTML error page (which is what produced
the production "is not valid JSON" incident).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from helix.storage import REPOSITORY_ROOT, resolve_data_root
from helix.web import create_app

EXPECTED_BUNDLE_FILES = (
    "artifacts/sft/sckpt-best.pt",
    "artifacts/data/vocab.json",
    "artifacts/data/merges.json",
    "configs/helix-deploy.json",
)
MANIFEST_NAME = "artifacts/model-bundle-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(**fields) -> None:
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "service": "helix-backend", "event": "startup"}
    record.update(fields)
    print(json.dumps(record, sort_keys=True), flush=True)


def _stage_verified_model_artifacts() -> Path:
    """Copy build-time-verified weights/tokenizer into writable runtime storage."""
    root = Path(REPOSITORY_ROOT)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError("model bundle manifest is missing; run scripts/fetch_model_bundle.py during build")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files", {})
    if manifest.get("format") != 1 or not set(EXPECTED_BUNDLE_FILES).issubset(records):
        raise RuntimeError("model bundle manifest is invalid")
    data_root = resolve_data_root()
    for name in sorted(EXPECTED_BUNDLE_FILES):
        source = root / name
        record = records[name]
        if not source.is_file() or source.stat().st_size != record.get("bytes") \
                or _sha256(source) != record.get("sha256"):
            raise RuntimeError(f"model bundle integrity check failed: {name}")
        target = data_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == source.stat().st_size \
                and _sha256(target) == record["sha256"]:
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(source.read_bytes())
        if _sha256(temporary) != record["sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"runtime copy integrity check failed: {name}")
        os.replace(temporary, target)
    return data_root


def _sanitize(message: str) -> str:
    import re
    message = re.sub(r"(?<!\w)/(?:[\w.\-]+/)+[\w.\-]+", "<path>", message)
    message = re.sub(r"(?i)(password|token|secret|authorization)\s*[=:]\s*\S+", r"\1=<redacted>", message)
    return message[:300]


_data_root: Optional[Path] = None
_startup_error: Optional[str] = None
try:
    _data_root = _stage_verified_model_artifacts()
    _log(stage="ok", data_root=str(_data_root))
except Exception as exc:  # noqa: BLE001 - startup must degrade, not crash
    _startup_error = _sanitize(f"{type(exc).__name__}: {exc}")
    _log(stage="degraded", error=_startup_error)

try:
    _fallback_root = resolve_data_root()
except Exception:  # pragma: no cover - storage unavailable
    _fallback_root = Path("/tmp")

_artifact_root = _data_root or _fallback_root
# Both the checkpoint and the deploy configuration are bundle members, so the
# staged copy under HELIX_DATA_ROOT is the authoritative default.
_config_default = str(_artifact_root / "configs/helix-deploy.json")
_checkpoint_default = str(_artifact_root / "artifacts/sft/sckpt-best.pt")

app = create_app(
    config_path=os.getenv("HELIX_CONFIG_PATH", _config_default),
    llm_checkpoint=os.getenv("HELIX_CHECKPOINT_PATH", _checkpoint_default),
    image_gen_config=os.getenv("HELIX_IMAGE_CONFIG_PATH", "artifacts/image-gen/config.json"),
    image_gen_checkpoint=os.getenv("HELIX_IMAGE_CHECKPOINT_PATH", "artifacts/image-gen/ckpt-final.pt"),
    vision_checkpoint=os.getenv("HELIX_VISION_CHECKPOINT_PATH", "artifacts/vision/ckpt-final.pt"),
    caption_dir=os.getenv("HELIX_CAPTION_DIR", "artifacts/caption"),
    external_origin=os.getenv("HELIX_EXTERNAL_ORIGIN") or None,
    startup_error=_startup_error,
    build_sha=os.getenv("HELIX_BUILD_SHA") or os.getenv("RENDER_GIT_COMMIT"),
    retrieval_root=os.getenv("HELIX_RETRIEVAL_ROOT") or None,
)
