"""Local document retrieval with source attribution.

Design constraints, deliberately modest:

* No hosted embedding service and no network access. Embeddings are hashed
  bag-of-words vectors, so retrieval is lexical plus a smoothed similarity
  signal. It is a real, testable ranking function, not a semantic model, and
  the docs say so.
* Every returned chunk carries its document id, title, relative source path,
  character offsets and (for PDFs) a page number so an answer can cite exactly
  where evidence came from.
* Retrieved text is untrusted input: it is delimited, length-capped and never
  interpreted as instructions by the caller.
* Ingestion is deterministic: the same files with the same options produce the
  same index bytes, so an index can be checksummed and reviewed.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EMBED_DIM = 512
SUPPORTED_SUFFIXES = {'.md', '.markdown', '.txt', '.html', '.htm', '.rst', '.json', '.csv', '.pdf'}
DEFAULT_CHUNK_CHARS = 900
DEFAULT_OVERLAP_CHARS = 120
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024

_TOKEN_RE = re.compile(r'[a-z0-9]+')
_STOPWORDS = frozenset('''a an and are as at be been by for from has have how i if in into is it
its of on or that the their there these this to was were what when where which who why will with
you your'''.split())


def _tokens(text: str) -> List[str]:
    return [token for token in _TOKEN_RE.findall(text.lower())
            if token not in _STOPWORDS and len(token) > 1]


def _embed(tokens: Iterable[str]) -> List[float]:
    vector = [0.0] * EMBED_DIM
    for token in tokens:
        digest = hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], 'big') % EMBED_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


@dataclass
class Chunk:
    doc_id: str
    title: str
    source: str
    text: str
    start: int
    end: int
    page: Optional[int] = None
    score: float = 0.0
    citation: str = ''
    embedding: List[float] = field(default_factory=list, repr=False)
    tokens: List[str] = field(default_factory=list, repr=False)

    def to_dict(self, snippet_chars: int = 400) -> dict:
        snippet = self.text.strip()
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rstrip() + '…'
        return {'doc_id': self.doc_id, 'title': self.title, 'source': self.source,
                'page': self.page, 'start': self.start, 'end': self.end,
                'citation': self.citation or self.make_citation(),
                'score': round(self.score, 4), 'snippet': snippet}

    def make_citation(self) -> str:
        if self.page:
            return f'{self.source}#page={self.page}'
        return f'{self.source}#chars={self.start}-{self.end}'


def _read_document(path: Path) -> Tuple[str, Optional[int]]:
    """Return plain text plus an optional page number for PDF-backed docs."""
    suffix = path.suffix.lower()
    if suffix == '.html' or suffix == '.htm':
        raw = path.read_text(encoding='utf-8', errors='replace')
        raw = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', raw)
        raw = re.sub(r'(?is)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>', '\n', raw)
        return html.unescape(re.sub(r'<[^>]+>', ' ', raw)), None
    return path.read_text(encoding='utf-8', errors='replace'), None


def _pdf_pages(path: Path) -> List[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('PDF ingestion needs the optional pypdf dependency (pip install -e ".[pdf]")') from exc
    reader = PdfReader(str(path))
    return [(page.extract_text() or '') for page in reader.pages]


def _split(text: str, chunk_chars: int, overlap: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    length = len(text)
    start = 0
    while start < length:
        end = min(length, start + chunk_chars)
        if end < length:
            # Prefer a paragraph or sentence boundary near the target size.
            window = text[start:end]
            for marker in ('\n\n', '. ', '\n'):
                cut = window.rfind(marker)
                if cut > chunk_chars * 0.5:
                    end = start + cut + len(marker)
                    break
        spans.append((start, end))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return spans


class RetrievalIndex:
    """A deterministic, checksummable index over local documents."""

    FORMAT = 1

    def __init__(self, chunks: List[Chunk], documents: List[dict], root: str = '', version: str = ''):
        self.chunks = chunks
        self.documents = documents
        self.root = root
        self.version = version
        self._df: Dict[str, int] = {}
        self._avg_len = 0.0
        self._recompute_statistics()

    # -- statistics --------------------------------------------------------
    def _recompute_statistics(self) -> None:
        df: Dict[str, int] = {}
        total = 0
        for chunk in self.chunks:
            total += len(chunk.tokens)
            for token in set(chunk.tokens):
                df[token] = df.get(token, 0) + 1
        self._df = df
        self._avg_len = (total / len(self.chunks)) if self.chunks else 0.0

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def checksum(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(f'{chunk.doc_id}|{chunk.start}|{chunk.end}|{chunk.text}'.encode('utf-8'))
        return digest.hexdigest()

    # -- construction ------------------------------------------------------
    @classmethod
    def ingest(cls, root: str | os.PathLike[str], chunk_chars: int = DEFAULT_CHUNK_CHARS,
               overlap: int = DEFAULT_OVERLAP_CHARS, max_chunks_per_document: int = 400,
               exclude: Sequence[str] = ()) -> 'RetrievalIndex':
        """Index a directory. `exclude` holds relative path prefixes to skip."""
        if chunk_chars < 120 or chunk_chars > 4000:
            raise ValueError('chunk_chars must be between 120 and 4000')
        if not 0 <= overlap < chunk_chars // 2:
            raise ValueError('overlap must be smaller than half of chunk_chars')
        root_path = Path(root)
        if not root_path.is_dir():
            raise ValueError('retrieval root must be an existing directory')
        skip_prefixes = tuple(prefix.strip('/') + '/' for prefix in exclude if prefix and prefix.strip('/'))
        manifest = {}
        manifest_path = root_path / 'helix-retrieval.json'
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                manifest = {}

        chunks: List[Chunk] = []
        documents: List[dict] = []
        for path in sorted(root_path.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name == 'helix-retrieval.json':
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if any(part.startswith('.') for part in path.relative_to(root_path).parts):
                continue
            try:
                if path.stat().st_size > MAX_DOCUMENT_BYTES:
                    continue
            except OSError:
                continue
            relative = path.relative_to(root_path).as_posix()
            if skip_prefixes and relative.startswith(skip_prefixes):
                continue
            doc_id = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:16]
            source_metadata = (manifest.get('documents', {}).get(relative, {}) or {})
            title = str(source_metadata.get('title') or path.stem.replace('_', ' ').replace('-', ' ').strip() or relative)
            pages: Optional[List[str]] = None
            if path.suffix.lower() == '.pdf':  # pragma: no cover - optional
                pages = _pdf_pages(path)
                text = '\n\n'.join(pages)
            else:
                text, _ = _read_document(path)
            text = text.replace('\r\n', '\n').strip()
            if not text:
                continue
            spans = _split(text, chunk_chars, overlap)[:max_chunks_per_document]
            for index, (start, end) in enumerate(spans):
                body = text[start:end].strip()
                if len(body) < 32:
                    continue
                page = None
                if pages:
                    consumed = len(text[:start].split('\n\n'))
                    page = min(len(pages), max(1, consumed))
                chunk = Chunk(doc_id=doc_id, title=title, source=relative, text=body,
                              start=start, end=end, page=page)
                chunk.tokens = _tokens(body)
                chunk.embedding = _embed(chunk.tokens)
                chunks.append(chunk)
            documents.append({'doc_id': doc_id, 'source': relative, 'title': title,
                              'bytes': len(text.encode('utf-8')), 'chunks': len(spans),
                              'license': source_metadata.get('license'),
                              'source_url': source_metadata.get('source_url'),
                              'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
        return cls(chunks, documents, root=str(root_path),
                   version=str(manifest.get('version', 'local')))

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | os.PathLike[str]) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            'format': self.FORMAT, 'version': self.version, 'root': self.root,
            'embedding': {'kind': 'hashed-bag-of-words', 'dim': EMBED_DIM},
            'documents': self.documents,
            'checksum': self.checksum,
            'chunks': [{**{key: value for key, value in asdict(chunk).items()
                          if key not in ('embedding', 'tokens')}} for chunk in self.chunks],
        }
        path = target / 'index.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding='utf-8')
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, directory: str | os.PathLike[str]) -> 'RetrievalIndex':
        path = Path(directory) / 'index.json'
        if not path.is_file():
            raise FileNotFoundError('no retrieval index in this directory')
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('retrieval index is implausibly large')
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('format') != cls.FORMAT:
            raise ValueError('unsupported retrieval index format')
        chunks = []
        for record in payload.get('chunks', []):
            if not isinstance(record, dict) or not isinstance(record.get('text'), str):
                continue
            chunk = Chunk(doc_id=str(record.get('doc_id', '')), title=str(record.get('title', '')),
                          source=str(record.get('source', '')), text=record['text'],
                          start=int(record.get('start', 0)), end=int(record.get('end', 0)),
                          page=record.get('page'))
            chunk.tokens = _tokens(chunk.text)
            chunk.embedding = _embed(chunk.tokens)
            chunk.citation = chunk.make_citation()
            chunks.append(chunk)
        documents = [record for record in payload.get('documents', []) if isinstance(record, dict)]
        return cls(chunks, documents, root=str(payload.get('root', '')), version=str(payload.get('version', '')))

    # -- search ------------------------------------------------------------
    def search(self, query: str, k: int = 4) -> List[Chunk]:
        query_tokens = _tokens(query or '')
        if not query_tokens or not self.chunks:
            return []
        k = max(1, min(int(k), 20))
        query_embedding = _embed(query_tokens)
        unique = set(query_tokens)
        total_docs = len(self.chunks)
        k1, b = 1.2, 0.75
        raw: List[Tuple[float, float, Chunk]] = []
        for chunk in self.chunks:
            frequencies: Dict[str, int] = {}
            for token in chunk.tokens:
                if token in unique:
                    frequencies[token] = frequencies.get(token, 0) + 1
            if not frequencies:
                continue
            length = max(1, len(chunk.tokens))
            bm25 = 0.0
            for token, frequency in frequencies.items():
                df = self._df.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                bm25 += idf * (frequency * (k1 + 1)) / (
                    frequency + k1 * (1 - b + b * length / max(self._avg_len, 1e-6)))
            cosine = max(0.0, _cosine(query_embedding, chunk.embedding))
            raw.append((bm25, cosine, chunk))
        if not raw:
            return []
        max_bm25 = max(entry[0] for entry in raw) or 1.0
        ranked = []
        for bm25, cosine, chunk in raw:
            score = 0.65 * (bm25 / max_bm25) + 0.35 * cosine
            if score <= 0:
                continue
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].doc_id, item[1].start))
        results = []
        for score, chunk in ranked[:k]:
            chunk.score = score
            chunk.citation = chunk.make_citation()
            results.append(chunk)
        return results

    def build_evidence(self, sources: Sequence[dict] | Sequence[Chunk], max_chars: int = 2400) -> str:
        """Render retrieved chunks as numbered, length-capped evidence."""
        blocks: List[str] = []
        used = 0
        for number, source in enumerate(sources, start=1):
            text = source.text if isinstance(source, Chunk) else str(source.get('snippet', ''))
            citation = source.make_citation() if isinstance(source, Chunk) else str(source.get('citation', ''))
            header = f'[{number}] {citation}'
            body = ' '.join(text.split())
            if len(body) > 600:
                body = body[:600].rstrip() + '…'
            block = f'{header}\n{body}'
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return '\n\n'.join(blocks)


def main() -> int:  # pragma: no cover - CLI helper
    import argparse
    parser = argparse.ArgumentParser(description='Build a HELIX retrieval index from local documents')
    parser.add_argument('--root', required=True, help='directory containing .md/.txt/.html documents')
    parser.add_argument('--output', required=True, help='directory for index.json')
    parser.add_argument('--chunk-chars', type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP_CHARS)
    args = parser.parse_args()
    index = RetrievalIndex.ingest(args.root, args.chunk_chars, args.overlap)
    path = index.save(args.output)
    print(json.dumps({'index': str(path), 'documents': index.document_count,
                      'chunks': index.chunk_count, 'checksum': index.checksum}, indent=2))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
