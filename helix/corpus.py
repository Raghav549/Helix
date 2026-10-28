"""Rights-reviewed corpus discovery, local extraction, filtering and manifests.

Discovery does not grant rights. Local ingestion requires an explicit license,
license evidence and a user assertion of training rights. Unknown material is
quarantined; no legal conclusions are inferred from file visibility or an API.
"""
from __future__ import annotations
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

RIGHTS_DOWNLOAD = 'download'
RIGHTS_TRAIN = 'train'
RIGHTS_REDISTRIBUTE = 'redistribute'
TEXT_EXTENSIONS = {'.txt', '.md', '.text', '.rst', '.nfo', '.csv', '.tsv', '.jsonl', '.tex'}
CODE_EXTENSIONS = {'.py', '.js', '.ts', '.c', '.cpp', '.h', '.hpp', '.rs', '.go',
                   '.java', '.rb', '.sh', '.pl', '.lua', '.hs', '.ml', '.scala',
                   '.swift', '.kt', '.r', '.m', '.asm', '.s', '.f', '.f90',
                   '.cob', '.cbl', '.bas', '.pas', '.adoc'}
EXTRA_EXTENSIONS = {'.html', '.htm', '.pdf', '.epub', '.ipynb', '.png', '.jpg', '.jpeg'}
MAX_FILE_BYTES = 16 * 1024 * 1024


@dataclass
class CorpusSource:
    source_id: str
    name: str
    url: str
    fmt: str = 'text'
    language: str = 'und'
    license: str = 'unknown'
    rights: List[str] = field(default_factory=list)
    retrieved_at: Optional[str] = None
    content_hash: Optional[str] = None
    size_bytes: Optional[int] = None
    attribution: Optional[str] = None
    notes: Optional[str] = None
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    quality_score: Optional[float] = None
    extraction_confidence: Optional[float] = None
    license_evidence: Optional[str] = None
    source_version: Optional[str] = None
    rights_confirmed: bool = False


@dataclass
class CorpusManifest:
    version: str = '2.0'
    created_at: str = field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    sources: List[CorpusSource] = field(default_factory=list)

    def add(self, source: CorpusSource) -> None:
        for i, old in enumerate(self.sources):
            if old.source_id == source.source_id:
                self.sources[i] = source
                return
        self.sources.append(source)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'version': self.version, 'created_at': self.created_at,
                       'sources': [asdict(s) for s in self.sources]}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'CorpusManifest':
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return cls(data.get('version', '1.0'), data.get('created_at', ''),
                   [CorpusSource(**s) for s in data.get('sources', [])])

    def trainable(self) -> List[CorpusSource]:
        return [s for s in self.sources if RIGHTS_TRAIN in s.rights and not s.quarantined
                and s.rights_confirmed and isinstance(s.license, str)
                and s.license.strip().lower() not in ('unknown', 'varies', '', 'none')
                and isinstance(s.license_evidence, str) and bool(s.license_evidence.strip())
                and bool(s.content_hash)]

    def quarantined(self) -> List[CorpusSource]:
        return [s for s in self.sources if s.quarantined]

    def summary(self) -> dict:
        return {'total': len(self.sources), 'trainable': len(self.trainable()),
                'quarantined': len(self.quarantined()),
                'by_fmt': _count_by(self.sources, 'fmt'), 'by_lang': _count_by(self.sources, 'language')}


def _count_by(items, attr):
    result = {}
    for item in items:
        value = getattr(item, attr, 'unknown')
        result[value] = result.get(value, 0) + 1
    return result


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: str) -> Optional[str]:
    try:
        raw = Path(path).read_bytes()
        if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
            return raw.decode('utf-16')
        return raw.decode('utf-8-sig')
    except (UnicodeError, OSError):
        return None


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self.hidden += 1
        elif tag in ('p', 'div', 'li', 'h1', 'h2', 'h3', 'pre', 'tr', 'br'):
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript') and self.hidden:
            self.hidden -= 1
        elif tag in ('p', 'div', 'li', 'pre', 'tr'):
            self.parts.append('\n')
    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _html_text(text):
    parser = _HTMLText()
    parser.feed(text)
    return ''.join(parser.parts).strip()


def extract_text(path: str) -> tuple[str, str]:
    """Extract supported formats. Reject unsupported image OCR explicitly."""
    ext = Path(path).suffix.lower()
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError('file exceeds 16 MiB extraction limit')
    if ext in TEXT_EXTENSIONS or ext in CODE_EXTENSIONS or ext in ('.html', '.htm'):
        text = read_text(path)
        if text is None:
            raise ValueError('undecodable')
        if ext in ('.html', '.htm'):
            return _html_text(text), 'html'
        return text, 'code' if ext in CODE_EXTENSIONS else 'text'
    if ext == '.ipynb':
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        cells = []
        for cell in data.get('cells', []):
            kind = cell.get('cell_type')
            if kind not in ('markdown', 'code'):
                continue
            source = cell.get('source', '')
            cells.append(f'[{kind}]\n' + (''.join(source) if isinstance(source, list) else str(source)))
        return '\n\n'.join(cells), 'notebook'
    if ext == '.epub':
        with zipfile.ZipFile(path) as z:
            entries = [x for x in z.infolist() if x.filename.lower().endswith(('.html', '.xhtml', '.htm'))]
            if sum(x.file_size for x in entries) > MAX_FILE_BYTES:
                raise ValueError('epub uncompressed size exceeds limit')
            chapters = [_html_text(z.read(x).decode('utf-8', 'strict')) for x in sorted(entries, key=lambda x: x.filename)]
            return '\n\n'.join(chapters), 'epub'
    if ext == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError('PDF requires optional pypdf; scanned PDFs require OCR') from exc
        reader = PdfReader(path)
        if reader.is_encrypted or len(reader.pages) > 500:
            raise ValueError('encrypted PDF or page limit exceeded')
        text = '\n\n'.join(page.extract_text() or '' for page in reader.pages)
        if len(text.strip()) < 20:
            raise ValueError('PDF text extraction insufficient; OCR required')
        return text, 'pdf'
    raise ValueError('format requires OCR or is not supported')


def _simple_language(text: str) -> str:
    if not text: return 'und'
    ranges = [('devanagari', '\u0900', '\u097f'), ('cyrillic', '\u0400', '\u04ff'),
              ('cjk', '\u3040', '\u9fff')]
    for name, lo, hi in ranges:
        if sum(lo <= c <= hi for c in text) / len(text) > .05: return name
    if sum(c.isascii() and c.isalpha() for c in text) / len(text) > .4: return 'latin-script'
    return 'und'


def _quality_score(text: str) -> float:
    if len(text.strip()) < 10: return 0.0
    control = sum(ord(c) < 32 and c not in '\t\n\r' for c in text)
    replacement = text.count('\ufffd')
    return round(max(0., 1 - 10 * (control + replacement) / len(text)) * min(1., len(text) / 1000), 3)


def quarantine(source: CorpusSource, reason: str) -> CorpusSource:
    source.quarantined = True
    source.quarantine_reason = reason
    source.rights = [r for r in source.rights if r != RIGHTS_TRAIN]
    return source


def _shingles(text: str) -> set:
    words = re.findall(r'\w+', text.casefold())
    return {tuple(words[i:i+5]) for i in range(max(0, len(words) - 4))}


def scan_directory(directory: str, manifest: Optional[CorpusManifest] = None,
                   rights: Optional[List[str]] = None, default_license: str = 'unknown',
                   license_evidence: Optional[str] = None, source_version: Optional[str] = None,
                   attribution: Optional[str] = None, contamination_strings: tuple = ()) -> CorpusManifest:
    """Scan bounded local files. Explicit rights + evidence are an attestation,
    not automated legal review. Quarantine duplicates, suspected leaks and unknowns.
    """
    manifest = manifest if manifest is not None else CorpusManifest()
    confirmed = bool(rights and RIGHTS_TRAIN in rights and isinstance(default_license, str)
                     and default_license.strip().lower() not in ('', 'unknown', 'varies', 'none')
                     and isinstance(license_evidence, str) and license_evidence.strip())
    seen = {s.content_hash for s in manifest.sources if s.content_hash and not s.quarantined}
    prior_shingles = []
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    directory = os.path.realpath(directory)
    for root, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', 'node_modules', '__pycache__', 'out'})
        for filename in sorted(files):
            path = os.path.join(root, filename)
            ext = Path(filename).suffix.lower()
            if ext not in TEXT_EXTENSIONS | CODE_EXTENSIONS | EXTRA_EXTENSIONS or os.path.islink(path):
                continue
            try:
                size = os.path.getsize(path)
                if size > MAX_FILE_BYTES:
                    raise ValueError('file exceeds 16 MiB extraction limit')
                digest = sha256_file(path)
                local = os.path.relpath(path, os.getcwd())
                source_url = path if local == '..' or local.startswith('..' + os.sep) else local
                source = CorpusSource(hashlib.sha256(path.encode()).hexdigest()[:20], filename, source_url,
                    fmt=ext.lstrip('.'), license=default_license, rights=list(rights or []),
                    retrieved_at=now, content_hash=digest, size_bytes=size, attribution=attribution,
                    license_evidence=license_evidence, source_version=source_version, rights_confirmed=confirmed)
                text, source.fmt = extract_text(path)
                source.language, source.quality_score = _simple_language(text), _quality_score(text)
                source.extraction_confidence = 1.0 if source.fmt in ('text', 'code') else None
                shingles = _shingles(text)
                if not confirmed:
                    quarantine(source, 'training rights and licence evidence not confirmed')
                elif source.quality_score < 0.01:
                    quarantine(source, 'low_quality')
                elif digest in seen:
                    quarantine(source, 'exact duplicate')
                elif any(q and q.casefold() in text.casefold() for q in contamination_strings if len(q) >= 8):
                    quarantine(source, 'held-out benchmark overlap')
                elif len(shingles) > 10 and any(len(shingles & old) / len(shingles | old) >= .9 for old in prior_shingles):
                    quarantine(source, 'near duplicate (5-word Jaccard >= 0.9)')
                else:
                    seen.add(digest)
                    prior_shingles.append(shingles)
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
                local = os.path.relpath(path, os.getcwd())
                source_url = path if local == '..' or local.startswith('..' + os.sep) else local
                source = CorpusSource(hashlib.sha256(path.encode()).hexdigest()[:20], filename, source_url,
                    fmt=ext.lstrip('.'), license=default_license, retrieved_at=now,
                    quarantined=True, quarantine_reason=str(exc))
            manifest.add(source)
    return manifest


def build_training_corpus(manifest: CorpusManifest, output_path: str) -> dict:
    """Materialize only reviewed, hash-matching local records; fail on mutation."""
    sources = manifest.trainable()
    if not sources:
        raise ValueError('no reviewed trainable sources')
    docs = []
    for source in sources:
        if not os.path.isfile(source.url) or os.path.islink(source.url):
            raise ValueError(f'not a local regular source: {source.name}')
        if sha256_file(source.url) != source.content_hash:
            raise ValueError(f'source changed since review: {source.name}')
        text, _ = extract_text(source.url)
        docs.append(text.strip())
    content = '\n\n'.join(docs) + '\n'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Path(output_path).write_text(content, encoding='utf-8')
    return {'sources': len(sources), 'sha256': sha256_file(output_path), 'bytes': len(content.encode())}


def discover_github(query: str, limit: int = 10) -> CorpusManifest:
    """Search public repository metadata. Every hit remains quarantined until
    a human reviews per-file rights and ingests a versioned local copy.
    """
    if not query.strip() or len(query) > 200 or not 1 <= limit <= 30:
        raise ValueError('invalid query or limit')
    url = 'https://api.github.com/search/repositories?' + urllib.parse.urlencode({'q': query, 'per_page': limit})
    req = urllib.request.Request(url, headers={'User-Agent': 'HELIX-research', 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.load(response)
    manifest = CorpusManifest()
    for item in payload.get('items', []):
        lic = item.get('license') or {}
        manifest.add(CorpusSource('github-' + str(item['id']), item['full_name'], item['html_url'],
            fmt='repository', license=lic.get('spdx_id') or 'unknown',
            license_evidence=lic.get('url'), source_version=item.get('default_branch'),
            retrieved_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            notes='Discovery metadata only; review individual files and pin a commit.',
            quarantined=True, quarantine_reason='rights not reviewed'))
    return manifest


def discover() -> CorpusManifest:
    """Return example discovery entry points, never pre-approved datasets."""
    result = CorpusManifest()
    for ident, name, url in [
        ('wikipedia_dump', 'Wikipedia dumps', 'https://dumps.wikimedia.org/'),
        ('wikisource', 'Wikisource', 'https://wikisource.org/'),
        ('gutenberg', 'Project Gutenberg', 'https://www.gutenberg.org/'),
        ('internet_archive', 'Internet Archive', 'https://archive.org/'),
    ]:
        result.add(CorpusSource(ident, name, url, license='varies',
            quarantined=True, quarantine_reason='individual item rights not reviewed'))
    return result
