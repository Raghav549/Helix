"""Normalize extracted text and compute conservative quality signals."""

from __future__ import annotations

import re
import unicodedata

MAX_DOCUMENT_CHARS = 500_000

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u0080-\U0010ffff]")
LATIN_RE = re.compile(r'[\u0000-\u007f]')
CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()[:MAX_DOCUMENT_CHARS]


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def language_hint(text: str) -> str:
    sample = text[:50_000]
    latin = len(LATIN_RE.findall(sample))
    cjk = len(CJK_RE.findall(sample))
    letters = max(1, latin + cjk)
    if cjk / letters > 0.25:
        return 'zh'
    if latin / letters > 0.75:
        return 'latin'
    return 'mixed'


def repetition_ratio(text: str) -> float:
    words = [word.lower() for word in TOKEN_RE.findall(text[:100_000])]
    if len(words) < 4:
        return 0.0
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return sum(value * value for value in counts.values()) / (len(words) * len(words))


def quality_score(text: str) -> dict[str, object]:
    tokens = token_count(text)
    letters = sum(1 for char in text if char.isalpha())
    letters_per_token = letters / max(1, tokens)
    unique_words = len({word.lower() for word in TOKEN_RE.findall(text[:100_000])})
    vocabulary_density = unique_words / max(1, tokens)
    repetition = repetition_ratio(text)
    # Conservative thresholds do not reject solely on language; short or
    # repetitive content is excluded while evidence is recorded.
    score = 0.0
    if tokens >= 100: score += 0.35
    if letters_per_token >= 3.0: score += 0.20
    if vocabulary_density >= 0.12: score += 0.20
    if 0.02 <= repetition <= 0.75: score += 0.25
    return {'tokens': tokens, 'letters_per_token': round(letters_per_token, 4),
            'vocabulary_density': round(vocabulary_density, 4),
            'repetition_ratio': round(repetition, 4), 'score': round(score, 4)}


def passes_quality(metrics: dict[str, object], minimum_score: float = 0.6) -> bool:
    tokens = int(metrics['tokens'])
    return (tokens >= 100 and metrics['score'] >= minimum_score and
            float(metrics['repetition_ratio']) < 0.85)
