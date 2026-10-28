"""Evidence-based source discovery without hard-coding book titles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Connector:
    name: str
    kind: str
    base_url: str
    license_evidence: str
    permitted_uses: tuple[str, ...]


CONNECTORS = (
    Connector('archive_org_advanced_search', 'books/historical',
              'https://archive.org/advancedsearch.php', 'catalog-specific',
              ('store',)),
    Connector('open_library_search', 'books/metadata',
              'https://openlibrary.org/search.json', 'catalog-specific',
              ('store',)),
    Connector('open_library_subjects', 'books/metadata',
              'https://openlibrary.org/subjects', 'catalog-specific',
              ('store',)),
    Connector('wikimedia_dumps', 'encyclopedic/historical',
              'https://dumps.wikimedia.org', 'CC BY-SA/GFDL',
              ('store', 'train')),
    Connector('common_crawl_index', 'web/open',
              'https://index.commoncrawl.org', 'catalog-specific',
              ('store',)),
    Connector('openalex', 'research/metadata',
              'https://api.openalex.org', 'catalog-specific',
              ('store',)),
    Connector('huggingface_datasets', 'datasets',
              'https://huggingface.co/api/datasets', 'dataset-specific',
              ('store',)),
    Connector('github_topics', 'software/documentation',
              'https://api.github.com/search/repositories', 'repository-specific',
              ('store', 'train')),
)


def connectors() -> list[Connector]:
    return list(CONNECTORS)


def connector(name: str) -> Connector:
    try:
        return next(item for item in CONNECTORS if item.name == name)
    except StopIteration:
        raise ValueError(f'unknown connector: {name}')
