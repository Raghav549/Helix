#!/usr/bin/env python3
"""Discover broad source metadata without downloading content or granting training rights."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from helix.sources import Catalog, SourceRecord
from helix.storage import resolve_data_root

USER_AGENT = "HELIX-Research-Discovery/0.1 (metadata-only; rights review required)"
ALLOWED_HOSTS = {
    "archive.org", "openlibrary.org", "api.openalex.org", "api.github.com",
    "dumps.wikimedia.org",
}
TOPICS = (
    "mathematics", "logic philosophy epistemology", "programming languages compilers",
    "software engineering operating systems databases", "physics astronomy",
    "chemistry biology", "medicine clinical science", "psychology cognitive science",
    "history geography", "linguistics multilingual communication", "economics law",
    "artificial intelligence machine learning", "engineering electronics",
    "statistics probability optimization", "computer science cybersecurity",
)


def utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def request_json(url: str, timeout: int = 20) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"refusing non-HTTPS or unapproved metadata endpoint: {url}")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if parsed.hostname == "api.github.com" and github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    request = urllib.request.Request(url, headers=headers)
    retryable = {429, 500, 502, 503, 504}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                final = urllib.parse.urlparse(response.geturl())
                if final.scheme != "https" or final.hostname not in ALLOWED_HOSTS:
                    raise ValueError(f"metadata endpoint redirected to unapproved host: {response.geturl()}")
                raw = response.read(12 * 1024 * 1024 + 1)
                if len(raw) > 12 * 1024 * 1024:
                    raise ValueError("metadata response exceeds 12 MiB safety limit")
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("metadata response must be a JSON object")
                return payload
        except HTTPError as exc:
            if exc.code not in retryable or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            try:
                delay = min(30, max(1, int(retry_after)))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(delay)
        except URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("metadata request exhausted retry attempts")


def _record(name: str, kind: str, url: str, license_name: str, source: str,
            language: str = "", fmt: str = "metadata", notes: str = "") -> SourceRecord:
    return SourceRecord(
        name=name, kind=kind, url=url, source_url=url,
        license=license_name or "unknown",
        license_evidence=f"{source} metadata only; exact item terms not verified",
        permitted_uses=("store",), language=language, format=fmt,
        retrieval_date=utc_date(), decision="needs_review", quarantined=True,
        notes=notes or "Metadata candidate only. Verify item-level rights, full-text access, privacy, attribution and redistribution terms before training.",
    )


def parse_internet_archive(payload: dict[str, Any], query: str) -> list[SourceRecord]:
    records = []
    for item in payload.get("response", {}).get("docs", []):
        identifier = item.get("identifier")
        if not identifier:
            continue
        license_name = str(item.get("licenseurl") or item.get("rights") or "unknown")
        url = f"https://archive.org/details/{urllib.parse.quote(str(identifier), safe='')}"
        records.append(_record(
            f"internet-archive:{identifier}", "book/historical/document", url, license_name,
            "Internet Archive Advanced Search", language=",".join(item.get("language", []) if isinstance(item.get("language"), list) else [str(item.get("language", ""))]).strip(","),
            fmt="metadata", notes=f"Discovered from query {query!r}. A catalog rights field is not a substitute for reviewing the item page and file-level terms.",
        ))
    return records


def parse_open_library(payload: dict[str, Any], query: str) -> list[SourceRecord]:
    records = []
    for item in payload.get("docs", []):
        key = item.get("key")
        if not key:
            continue
        url = "https://openlibrary.org" + str(key)
        records.append(_record(
            f"open-library:{key}", "book/metadata", url, "catalog-specific/unknown",
            "Open Library Search API", language=",".join(item.get("language") or []),
            notes=f"Catalog metadata for query {query!r}; Open Library metadata is not full text and does not itself grant book training rights.",
        ))
    return records


def parse_openalex(payload: dict[str, Any], query: str) -> list[SourceRecord]:
    records = []
    for item in payload.get("results", []):
        wid = item.get("id")
        if not wid:
            continue
        location = item.get("primary_location") or {}
        landing = location.get("landing_page_url") or item.get("doi") or wid
        oa = item.get("open_access") or {}
        license_name = location.get("license") or oa.get("oa_status") or "open-access status only"
        records.append(_record(
            f"openalex:{str(wid).rsplit('/', 1)[-1]}", "research/paper", str(landing), str(license_name),
            "OpenAlex Works API", fmt="metadata",
            notes=f"Discovered from query {query!r}; OA status is not equivalent to a verified training licence. Check the publisher/repository full text and its licence.",
        ))
    return records


def parse_github(payload: dict[str, Any], query: str) -> list[SourceRecord]:
    records = []
    for item in payload.get("items", []):
        full_name = item.get("full_name")
        url = item.get("html_url")
        if not full_name or not url:
            continue
        license_name = (item.get("license") or {}).get("spdx_id") or "no machine-readable licence"
        records.append(_record(
            f"github:{full_name}", "software/source-code/documentation", str(url), str(license_name),
            "GitHub repository search API", language=str(item.get("language") or ""), fmt="repository metadata",
            notes=f"Discovered from query {query!r}; repository-level licence metadata does not establish rights for every file, submodule, issue, commit or vendored dependency.",
        ))
    return records


def parse_wikimedia_index() -> list[SourceRecord]:
    return [_record(
        "wikimedia:dumps-index", "encyclopedia/dump-catalog",
        "https://dumps.wikimedia.org/backup-index.html",
        "CC BY-SA/GFDL (project-level; dump/version terms must be checked)",
        "Wikimedia dump index", fmt="catalog index",
        notes="Index only. Choose a specific project, language, dump version and licence/attribution policy before downloading any dump.",
    )]


def discover(source: str, query: str, limit: int) -> list[SourceRecord]:
    limit = max(1, min(int(limit), 50))
    if source == "internet_archive":
        params = {
            "q": f'mediatype:texts AND (title:({query}) OR description:({query}) OR subject:({query}))',
            "fl[]": ["identifier", "title", "creator", "date", "licenseurl", "rights", "language", "format"],
            "rows": limit, "page": 1, "output": "json",
        }
        payload = request_json("https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params, doseq=True))
        return parse_internet_archive(payload, query)
    if source == "open_library":
        params = {"q": query, "limit": limit, "fields": "key,title,author_name,first_publish_year,language,has_fulltext,ia"}
        return parse_open_library(request_json("https://openlibrary.org/search.json?" + urllib.parse.urlencode(params)), query)
    if source == "openalex":
        params = {"search": query, "filter": "is_oa:true", "per-page": limit}
        return parse_openalex(request_json("https://api.openalex.org/works?" + urllib.parse.urlencode(params)), query)
    if source == "github":
        params = {"q": f'{query} license:mit', "sort": "stars", "order": "desc", "per_page": limit}
        return parse_github(request_json("https://api.github.com/search/repositories?" + urllib.parse.urlencode(params)), query)
    if source == "wikimedia":
        # Metadata-only pointer; no bulk dump is downloaded by this command.
        return parse_wikimedia_index()
    raise ValueError(f"unknown source connector: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("all", "internet_archive", "open_library", "openalex", "github", "wikimedia"), default="all")
    parser.add_argument("--query", action="append", help="Broad topic query; repeat for additional domains. Defaults to a broad multi-domain set.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum metadata results per query/source (1-50).")
    parser.add_argument("--root", help="External HELIX_DATA_ROOT override.")
    parser.add_argument("--output", default="discovery/candidates.json", help="Output path relative to external data root.")
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        parser.error("--limit must be between 1 and 50")
    root = resolve_data_root(args.root or os.environ.get("HELIX_DATA_ROOT"))
    queries = args.query or list(TOPICS)
    sources = ("internet_archive", "open_library", "openalex", "github", "wikimedia") if args.source == "all" else (args.source,)
    records: dict[str, SourceRecord] = {}
    errors = []
    for source in sources:
        source_queries = [queries[0]] if source == "wikimedia" else queries
        for query in source_queries:
            try:
                for record in discover(source, query, args.limit):
                    records.setdefault(record.name, record)
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                errors.append({"source": source, "query": query, "error": f"{type(exc).__name__}: {exc}"})
    catalog = Catalog(version=1, generated_by="scripts/discover_sources.py", sources=list(records.values()))
    catalog.validate()
    target = catalog.save(args.output, root=root)
    report = {
        "output": str(target), "sources_queried": list(sources), "queries": queries,
        "candidate_count": len(catalog.sources), "eligible_for_training": len(catalog.eligible()),
        "needs_review": sum(item.decision == "needs_review" for item in catalog.sources),
        "errors": errors,
        "notice": "Metadata discovery only. No content was downloaded and no candidate was approved for training.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
