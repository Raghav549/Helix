#!/usr/bin/env python3
"""Download and verify a HELIX model bundle from a GitHub Release."""
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BUNDLE_BYTES = 100 * 1024 * 1024
ALLOWED_HOSTS = {"github.com", "release-assets.githubusercontent.com"}
EXPECTED = {
    "artifacts/sft/sckpt-best.pt": 80 * 1024 * 1024,
    "artifacts/data/vocab.json": 8 * 1024 * 1024,
    "artifacts/data/merges.json": 8 * 1024 * 1024,
    "configs/helix-deploy.json": 1024 * 1024,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    url = os.environ.get("HELIX_MODEL_BUNDLE_URL", "").strip()
    expected_sha = os.environ.get("HELIX_MODEL_BUNDLE_SHA256", "").strip().lower()
    if not url or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise SystemExit("Set HELIX_MODEL_BUNDLE_URL and a 64-character HELIX_MODEL_BUNDLE_SHA256 before deploying.")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or "/releases/download/" not in parsed.path:
        raise SystemExit("Model bundle URL must be an HTTPS GitHub Release asset URL.")
    request = urllib.request.Request(url, headers={"User-Agent": "HELIX-model-bundle-installer/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname not in ALLOWED_HOSTS:
            raise SystemExit("Release asset redirected to an unapproved host.")
        chunks, total = [], 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BUNDLE_BYTES:
                raise SystemExit("Model bundle exceeds 100 MiB limit.")
            chunks.append(chunk)
    raw = b"".join(chunks)
    actual_sha = sha256_bytes(raw)
    if actual_sha != expected_sha:
        raise SystemExit(f"Model bundle SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    with tempfile.NamedTemporaryFile(suffix=".zip") as temp:
        temp.write(raw)
        temp.flush()
        with zipfile.ZipFile(temp.name) as archive:
            names = set(archive.namelist())
            manifest_name = "artifacts/model-bundle-manifest.json"
            if manifest_name not in names:
                raise SystemExit("Bundle manifest is missing.")
            manifest = json.loads(archive.read(manifest_name))
            if manifest.get("format") != 1 or set(manifest.get("files", {})) != set(EXPECTED):
                raise SystemExit("Bundle manifest format or file list is invalid.")
            if names != set(EXPECTED) | {manifest_name}:
                raise SystemExit("Bundle contains unexpected files.")
            extracted = {}
            for name, max_bytes in EXPECTED.items():
                info = archive.getinfo(name)
                if info.file_size > max_bytes or info.file_size < 1:
                    raise SystemExit(f"Bundle member size is invalid: {name}")
                data = archive.read(name)
                record = manifest["files"][name]
                if len(data) != record.get("bytes") or sha256_bytes(data) != record.get("sha256"):
                    raise SystemExit(f"Bundle member integrity check failed: {name}")
                extracted[name] = data
    for name, data in extracted.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    print(json.dumps({"ok": True, "bundle_sha256": actual_sha,
                      "installed": sorted(extracted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
