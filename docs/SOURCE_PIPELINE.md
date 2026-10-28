# HELIX source and data pipeline

The source repository contains code, configuration, tests, documentation,
dataset manifest definitions and small metadata. Heavy model outputs,
tokenized arrays, raw downloads, caches and generated media live outside Git
under `HELIX_DATA_ROOT` (default `/var/lib/helix`, otherwise `~/.helix/data`).

## External root

```sh
export HELIX_DATA_ROOT=/media/helix/data
python -m helix.storage.guard    # pre-commit equivalent
python scripts/migrate_artifacts.py --check
```

`HELIX_DATA_ROOT` must resolve outside the repository. The migration script
copies every file listed in the source release manifest only after verifying
its SHA-256, then verifies the copied file again. Originals are retained.

## Discovery and rights

`helix/discovery.py` defines connector metadata for Internet Archive advanced
search, Open Library, Wikimedia dumps, Common Crawl, OpenAlex, Hugging Face
datasets and GitHub topic searches. Connectors catalog access metadata only;
availability is never treated as training permission.

`helix/sources.py` stores each source as a machine-readable record with URL,
kind, license, license evidence, permitted uses, language, format, retrieval
date, decision and notes. A record is eligible only when evidence is verified
and training is explicitly permitted; ambiguous records are quarantined by
default.

## Stages

1. `helix/ingest.py` downloads only eligible HTTP(S) URLs to `raw/` with
   per-file, total-byte and timeout limits.
2. `helix/extract.py` extracts text, HTML, archives and EPUB deterministically.
   Scanned PDFs and unsupported media are quarantined rather than faked.
3. `helix/quality.py` normalizes text and computes conservative quality
   signals without rejecting a language by itself.
4. `helix/dedupe.py` removes exact and lexical near-duplicates.
5. `helix/corpus_pipeline.py` filters benchmark names, assigns deterministic
   train/validation/test splits, hashes every accepted document, writes split
   hashes and a corpus manifest.
6. `helix/pipeline.py` runs all stages from a source catalog.
7. `helix/data.py` trains BPE on the training split only and writes token
   arrays and hash metadata.

## Current status

The initial source is the HELIX-authored MIT-licensed seed corpus already in
the repository. It is intentionally small and is enough to validate the
pipeline and train a tiny baseline. The connector catalog is implemented, but
bulk legal downloads and OCR have not been run against external collections.
No unverified source is automatically promoted to training.
