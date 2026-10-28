"""HELIX corpus discovery and ingestion pipeline.

Extensible, provenance-tracked, rights-aware. Discovers eligible sources and
records per-document provenance so material whose legal status cannot be
established is quarantined and excluded from training.

Zero-budget-first: the local-workflow path scans a user-provided directory
(no network assumed). Network ingestion (Internet Archive, Wikipedia dumps,
Open Library, etc.) is recorded as candidate sources in the manifest and can
be enabled separately when connectivity and budget allow.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Rights categories
RIGHTS_DOWNLOAD = "download"      # may be downloaded
RIGHTS_TRAIN = "train"            # may be used for training
RIGHTS_REDISTRIBUTE = "redistribute"  # may be redistributed

TEXT_EXTENSIONS = {".txt", ".md", ".text", ".rst", ".nfo", ".csv", ".tsv", ".jsonl"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".c", ".cpp", ".h", ".hpp", ".rs", ".go",
                   ".java", ".rb", ".sh", ".pl", ".lua", ".hs", ".ml", ".scala",
                   ".swift", ".kt", ".r", ".m", ".mm", ".asm", ".s", ".f", ".f90",
                   ".cob", ".cbl", ".bas", ".pas", ".adoc", ".tex"}

ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


@dataclass
class CorpusSource:
    source_id: str
    name: str
    url: str
    fmt: str = "text"                       # text | html | pdf | epub | code | notebook | image
    language: str = "und"
    license: str = "unknown"
    rights: List[str] = field(default_factory=list)   # subset of download/train/redistribute
    retrieved_at: Optional[str] = None
    content_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    attribution: Optional[str] = None
    notes: Optional[str] = None
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    quality_score: Optional[float] = None       # 0..1 heuristic
    extraction_confidence: Optional[float] = None # 0..1


@dataclass
class CorpusManifest:
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    sources: List[CorpusSource] = field(default_factory=list)

    def add(self, source: CorpusSource) -> None:
        for i, s in enumerate(self.sources):
            if s.source_id == source.source_id:
                self.sources[i] = source
                return
        self.sources.append(source)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": self.version,
            "created_at": self.created_at,
            "sources": [asdict(s) for s in self.sources],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CorpusManifest":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        m = cls(version=payload.get("version", "1.0"),
                created_at=payload.get("created_at", ""))
        for s in payload.get("sources", []):
            m.sources.append(CorpusSource(**s))
        return m

    def trainable(self) -> List[CorpusSource]:
        return [s for s in self.sources
                if RIGHTS_TRAIN in s.rights and not s.quarantined]

    def quarantined(self) -> List[CorpusSource]:
        return [s for s in self.sources if s.quarantined]

    def summary(self) -> dict:
        return {
            "total": len(self.sources),
            "trainable": len(self.trainable()),
            "quarantined": len(self.quarantined()),
            "by_fmt": _count_by(self.sources, "fmt"),
            "by_lang": _count_by(self.sources, "language"),
        }


def _count_by(items: List, attr: str) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for it in items:
        k = getattr(it, attr, "unknown")
        d[k] = d.get(k, 0) + 1
    return d


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: str) -> Optional[str]:
    """Read a file trying several encodings; return None if undecodable."""
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def _simple_language(text: str) -> str:
    """Cheap language hint from character ranges (fallback only)."""
    if not text:
        return "und"
    cjk = sum(1 for ch in text
              if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = max(1, len(text))
    if cjk / total > 0.05:
        return "zh/ja/ko"
    if cyr / total > 0.05:
        return "ru/cyrillic"
    if lat / total > 0.5:
        return "latin-script"
    return "und"


def _quality_score(text: str) -> float:
    """0..1 heuristic: penalizes short, control-char-heavy text."""
    if not text or len(text) < 10:
        return 0.0
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r")
    ratio = control / max(1, len(text))
    length_score = min(1.0, len(text) / 1000.0)
    return round(max(0.0, 1.0 - ratio * 10) * length_score, 3)


def quarantine(source: CorpusSource, reason: str) -> CorpusSource:
    source.quarantined = True
    source.quarantine_reason = reason
    source.rights = [r for r in source.rights if r != RIGHTS_TRAIN]
    return source


# ---------------------------------------------------------------------------
# Local (zero-network) ingestion
# ---------------------------------------------------------------------------

def scan_directory(directory: str, manifest: Optional[CorpusManifest] = None,
                   rights: Optional[List[str]] = None,
                   default_license: str = "user-provided") -> CorpusManifest:
    """Scan a local directory for ingestable text/code files.

    Rights default to train-only for user-provided local files (download /
    redistribute not assumed). Each file becomes a CorpusSource with content
    hash, language hint, quality score, and retrieval timestamp.
    """
    if manifest is None:
        manifest = CorpusManifest()
    if rights is None:
        rights = [RIGHTS_TRAIN]

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXTENSIONS and ext not in CODE_EXTENSIONS:
                continue
            path = os.path.join(root, fname)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size == 0:
                continue
            text = read_text(path)
            if text is None:
                q = CorpusSource(
                    source_id=hashlib.sha256(path.encode()).hexdigest()[:16],
                    name=fname, url=path, fmt=ext.lstrip("."),
                    rights=[], retrieved_at=now, size_bytes=size,
                    quarantined=True, quarantine_reason="undecodable",
                )
                manifest.add(q)
                continue
            lang = _simple_language(text)
            qscore = _quality_score(text)
            if qscore < 0.05:
                q = CorpusSource(
                    source_id=sha256_file(path)[:16], name=fname, url=path,
                    fmt=ext.lstrip("."), language=lang, license=default_license,
                    rights=[], retrieved_at=now,
                    content_hash=sha256_file(path), size_bytes=size,
                    quality_score=qscore, extraction_confidence=1.0,
                    quarantined=True, quarantine_reason="low_quality",
                )
                manifest.add(q)
                continue
            fmt = "code" if ext in CODE_EXTENSIONS else "text"
            src = CorpusSource(
                source_id=sha256_file(path)[:16], name=fname, url=path,
                fmt=fmt, language=lang, license=default_license,
                rights=list(rights), retrieved_at=now,
                content_hash=sha256_file(path), size_bytes=size,
                quality_score=qscore, extraction_confidence=1.0,
                notes=f"discovered by scan_directory({directory})",
            )
            manifest.add(src)
    return manifest


# ---------------------------------------------------------------------------
# Candidate network sources (extensible, not exhaustive / unverified)
# ---------------------------------------------------------------------------

KNOWN_SOURCE_TEMPLATES: List[CorpusSource] = [
    CorpusSource(
        source_id="wikipedia_dump", name="Wikipedia content dump",
        url="https://dumps.wikimedia.org/", fmt="xml", language="multilingual",
        license="CC-BY-SA 3.0 / GFDL",
        rights=[RIGHTS_DOWNLOAD, RIGHTS_TRAIN],
        attribution="Wikimedia Foundation",
        notes="Training permitted under CC-BY-SA with attribution; verify per-dump.",
    ),
    CorpusSource(
        source_id="wikisource", name="Wikisource public-domain texts",
        url="https://wikisource.org/", fmt="html", language="multilingual",
        license="public domain / CC-BY-SA",
        rights=[RIGHTS_DOWNLOAD, RIGHTS_TRAIN, RIGHTS_REDISTRIBUTE],
        attribution="Wikimedia Foundation",
        notes="Public-domain and freely licensed texts.",
    ),
    CorpusSource(
        source_id="gutenberg", name="Project Gutenberg public-domain ebooks",
        url="https://www.gutenberg.org/", fmt="text", language="multilingual",
        license="public domain (US)",
        rights=[RIGHTS_DOWNLOAD, RIGHTS_TRAIN, RIGHTS_REDISTRIBUTE],
        attribution="Project Gutenberg",
        notes="Confirm each item's license page; non-US public domain may differ.",
    ),
    CorpusSource(
        source_id="internet_archive", name="Internet Archive texts",
        url="https://archive.org/", fmt="text", language="multilingual",
        license="varies", rights=[RIGHTS_DOWNLOAD],
        notes="Many items are lending-restricted; verify per-item rights before training.",
    ),
]


def discover() -> CorpusManifest:
    """Build a manifest of candidate sources (preliminary, unverified).

    Extensible: callers may append sources. Rights listed are preliminary
    claims and are NOT sufficient for training until verified at ingest time.
    Anything whose rights cannot be established should be quarantined.
    """
    manifest = CorpusManifest()
    for t in KNOWN_SOURCE_TEMPLATES:
        manifest.add(t)
    return manifest
