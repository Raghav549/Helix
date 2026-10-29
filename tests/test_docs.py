"""Release documentation guards.

These keep the README honest and internally consistent: the rejected sentence
stays gone, every required section exists, every relative link resolves, and no
document claims an artifact that has not been produced.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / 'README.md').read_text(encoding='utf-8')
LOWER = README.lower()

REQUIRED_SECTIONS = [
    '## Architecture',
    '## Quick start',
    '## API',
    '## Model and tokenizer',
    '## Training',
    '## Data provenance and licensing',
    '## Evaluation',
    '## Security and privacy',
    '## Deployment',
    '## Troubleshooting',
    '## Testing',
    '## Documentation',
]


def test_rejected_sentence_is_gone():
    assert 'This is a small research prototype, not a general-purpose assistant' not in README
    assert 'not a general-purpose assistant' not in LOWER


def test_readme_has_the_required_sections():
    for section in REQUIRED_SECTIONS:
        assert section in README, f'missing section: {section}'


def test_readme_states_measured_limits_instead_of_claims():
    assert '803,712' in README
    assert '4.97' in README
    assert '1 of 5' in README or '1/5' in README
    for overclaim in ('state-of-the-art', 'best-in-class', 'matches GPT', 'human-level'):
        assert overclaim.lower() not in LOWER


def test_every_relative_link_resolves():
    links = re.findall(r'\]\((?!https?://|#)([^)]+)\)', README)
    assert links, 'the README should link to its documentation'
    for link in links:
        target = link.split('#')[0]
        assert (ROOT / target).exists(), f'broken link in README: {link}'


def test_no_document_claims_a_downloadable_apk():
    for path in (ROOT / 'README.md', ROOT / 'mobile' / 'README.md', ROOT / 'docs' / 'STATUS.md'):
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8').lower()
        for phrase in ('download the apk', 'apk is available', 'install the released apk'):
            assert phrase not in text, f'{path.name} claims an artifact that may not exist'


def test_changelog_exists_and_is_dated_by_section():
    changelog = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    assert '## Unreleased' in changelog
    assert 'docs/MEASUREMENTS.md' in changelog


def test_status_document_records_blockers():
    status = (ROOT / 'docs' / 'STATUS.md').read_text(encoding='utf-8')
    assert '## Blockers' in status
    assert 'Render' in status
