# Data Pipeline (planned)

The v0.1 baseline tokenizes `data/seed_corpus.txt` into train/val arrays
(`helix/data.build_from_corpus`) and trains the byte-level BPE tokenizer from
scratch.

Future stages (documented, not yet implemented):
- Extensible corpus discovery/ingestion: Internet Archive, Open Library,
  Wikipedia/Wikisource/Wikibooks dumps, research repositories, code repos.
- Per-document provenance: source URL/ID, retrieval timestamp, license, rights,
  attribution, version, language, extraction confidence, dataset hash.
- Legal filtering: distinguish download/train/redistribute-permitted vs
  quarantine; respect robots.txt, access controls, repository licenses.
- Format support: HTML, PDF, EPUB, plain text, scanned books (OCR), code,
  notebooks, tables, equations; preserve structure (definitions, proofs,
  code blocks, citations, cross-references).
- Quality: OCR confidence scoring, language detection, encoding normalization,
  spam/malware filtering, exact + semantic dedup, near-dup detection,
  contamination checks, benchmark-leakage prevention, quality scoring,
  source reliability scoring, contradiction tracking.
- Separate processing paths: natural language, source code, math/scientific,
  structured data, dialogue, multimodal.
- Learning-signal transforms: next-token, code completion, math derivations,
  proof construction, QA, grounded summarization, translation, debugging,
  unit-test generation, tool-use trajectories.
