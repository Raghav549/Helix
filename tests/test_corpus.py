"""Tests for the corpus discovery and ingestion pipeline."""

import os
import tempfile

from helix.corpus import (
    CorpusSource, CorpusManifest, scan_directory, discover, quarantine,
    RIGHTS_TRAIN, RIGHTS_DOWNLOAD, RIGHTS_REDISTRIBUTE, sha256_file,
)


def test_scan_directory_creates_manifest(tmp_path):
    (tmp_path / "a.txt").write_text(
        "hello world this is a test document about science and reasoning"
    )
    (tmp_path / "b.py").write_text("def add(a, b):\n    return a + b\n")
    m = scan_directory(str(tmp_path))
    assert m.summary()["total"] >= 2
    trainable = m.trainable()
    assert len(trainable) >= 1
    for s in trainable:
        assert RIGHTS_TRAIN in s.rights
        assert s.content_hash is not None
        assert s.size_bytes is not None


def test_scan_undecodable_quarantined(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe\x00\x01nottext")
    m = scan_directory(str(tmp_path))
    q = m.quarantined()
    assert len(q) >= 1
    # latin-1 always decodes, so garbage bytes surface as low_quality rather
    # than undecodable; either way the file is quarantined and excluded.
    assert q[0].quarantined
    assert q[0].quarantine_reason in ("undecodable", "low_quality")
    assert RIGHTS_TRAIN not in q[0].rights


def test_quarantine_removes_train_right():
    s = CorpusSource(source_id="x", name="x", url="x",
                     rights=[RIGHTS_TRAIN, RIGHTS_DOWNLOAD])
    s = quarantine(s, "bad license")
    assert s.quarantined
    assert RIGHTS_TRAIN not in s.rights
    assert s.quarantine_reason == "bad license"


def test_discover_has_candidates():
    m = discover()
    assert m.summary()["total"] >= 1
    ids = {s.source_id for s in m.sources}
    assert {"wikipedia_dump", "gutenberg", "internet_archive"} <= ids


def test_manifest_roundtrip(tmp_path):
    m = CorpusManifest()
    s = CorpusSource(source_id="s1", name="n", url="u",
                     rights=[RIGHTS_TRAIN, RIGHTS_REDISTRIBUTE],
                     quality_score=0.9)
    m.add(s)
    path = tmp_path / "manifest.json"
    m.save(str(path))
    m2 = CorpusManifest.load(str(path))
    assert len(m2.sources) == 1
    assert m2.sources[0].source_id == "s1"
    assert m2.trainable()[0].rights == [RIGHTS_TRAIN, RIGHTS_REDISTRIBUTE]


def test_sha256_deterministic(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("deterministic content")
    h1 = sha256_file(str(p))
    h2 = sha256_file(str(p))
    assert h1 == h2
    assert len(h1) == 64
