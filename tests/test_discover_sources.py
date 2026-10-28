"""Metadata discovery is deliberately quarantined until rights review."""
from scripts.discover_sources import (
    parse_github, parse_internet_archive, parse_open_library, parse_openalex,
    parse_wikimedia_index,
)


def test_archive_candidates_never_become_training_eligible_from_catalog_rights():
    records = parse_internet_archive({"response": {"docs": [
        {"identifier": "example-book", "title": "Mathematics", "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/", "language": ["eng"]}
    ]}}, "mathematics")
    assert len(records) == 1
    assert records[0].url.endswith("/example-book")
    assert records[0].decision == "needs_review"
    assert records[0].quarantined is True
    assert "train" not in records[0].permitted_uses


def test_open_library_is_catalog_metadata_not_full_text_permission():
    records = parse_open_library({"docs": [{"key": "/works/OL123W", "title": "A Book", "language": ["eng"]}]}, "books")
    assert records[0].name == "open-library:/works/OL123W"
    assert records[0].format == "metadata"
    assert records[0].decision == "needs_review"


def test_openalex_open_access_status_is_not_a_verified_training_license():
    records = parse_openalex({"results": [{
        "id": "https://openalex.org/W123", "title": "Research",
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "primary_location": {"landing_page_url": "https://example.org/paper", "license": None},
    }]}, "science")
    assert records[0].name == "openalex:W123"
    assert records[0].decision == "needs_review"
    assert "verified" in records[0].license_evidence


def test_github_license_metadata_still_requires_file_level_review():
    records = parse_github({"items": [{
        "full_name": "org/project", "html_url": "https://github.com/org/project",
        "license": {"spdx_id": "MIT"}, "language": "Python",
    }]}, "python")
    assert records[0].license == "MIT"
    assert records[0].decision == "needs_review"
    assert "every file" in records[0].notes


def test_wikimedia_discovery_only_points_to_dump_index():
    record = parse_wikimedia_index()[0]
    assert record.url == "https://dumps.wikimedia.org/backup-index.html"
    assert record.decision == "needs_review"
    assert "train" not in record.permitted_uses
