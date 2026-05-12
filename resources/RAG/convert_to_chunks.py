"""Turn PFAF section dicts into RAG chunks: sentence splits; long spans use a sliding window."""
from __future__ import annotations

import re
from typing import NamedTuple

from extract_normalize import PfafStructured


class ChunkStats(NamedTuple):
    """Chunk list plus how many pieces were produced by sliding-window splits (oversized sentences)."""

    chunks: list[str]
    chunks_from_window: int

# Above this length, a single "sentence" is windowed instead of kept whole.
_DEFAULT_MAX_SENTENCE_CHARS = 400
# Window size and overlap (chars) for oversized sentences.
_DEFAULT_WINDOW_CHARS = 300
_DEFAULT_OVERLAP_CHARS = 80


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _window_slide(text: str, size: int, overlap: int) -> list[str]:
    """Slide a window over ``text`` with ``overlap`` between consecutive windows."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        i += step
    return out


def _format_chunk(section_title: str, body: str) -> str:
    """Section title as label (e.g. ``Medicinal Uses:``) then body."""
    title = section_title.strip()
    if not title.endswith(":"):
        title = f"{title}:"
    return f"{title}\n\n{body.strip()}"


def chunks_from_sections(
    sections: dict[str, str],
    *,
    max_sentence_chars: int = _DEFAULT_MAX_SENTENCE_CHARS,
    window_chars: int = _DEFAULT_WINDOW_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> ChunkStats:
    """
    Build text chunks from a ``sections`` mapping (heading → body).

    - Splits each body on sentence boundaries (``.?!`` + whitespace).
    - Each chunk starts with the section key as a label (``Medicinal Uses:``).
    - Sentences longer than ``max_sentence_chars`` are split with a sliding window
      of ``window_chars`` and ``overlap_chars``.

    ``chunks_from_window`` counts chunk texts produced from that window path (one
    oversized sentence can yield multiple window chunks).
    """
    result: list[str] = []
    chunks_from_window = 0
    for section_title, body in sections.items():
        body = (body or "").strip()
        if not body:
            continue
        sentences = _split_sentences(body)
        if not sentences:
            continue
        for sent in sentences:
            if len(sent) <= max_sentence_chars:
                result.append(_format_chunk(section_title, sent))
            else:
                for w in _window_slide(sent, window_chars, overlap_chars):
                    chunks_from_window += 1
                    result.append(_format_chunk(section_title, w))
    return ChunkStats(result, chunks_from_window)


def chunks_from_structured(
    structured: PfafStructured,
    *,
    max_sentence_chars: int = _DEFAULT_MAX_SENTENCE_CHARS,
    window_chars: int = _DEFAULT_WINDOW_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> ChunkStats:
    """Convenience: chunk ``structured.sections`` only."""
    return chunks_from_sections(
        structured.sections,
        max_sentence_chars=max_sentence_chars,
        window_chars=window_chars,
        overlap_chars=overlap_chars,
    )
