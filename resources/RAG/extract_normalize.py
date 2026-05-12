"""Parse PFAF ``Plant.aspx`` HTML into latin name, metadata (fact table), and cleaned sections for RAG."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from bs4 import BeautifulSoup, Tag

# Headings to skip (ads, footer, boilerplate, trailing page chrome)
_SKIP_SECTION_TITLES = frozenset(
    {
        "The Bookshop: Edible Plant Books",
        "Plant Uses",
        "Content",
        "QR Code",
        "Author",
        "Botanical References",
        "Links / References",
        "Readers comment",
        "Other Names",
        "Native Range",
        "Conservation Status",
        "Weed Potential",
    }
)

# Book promo blocks use long h3 titles like "Now available: … [Paperback and eBook]".
_SKIP_SECTION_TITLE_PREFIXES = ("Now available:",)

# PFAF HTML sometimes uses non-void <br>…</br> so BeautifulSoup nests the following page inside <br>.
_VOID_HTML_TAGS = frozenset({"br", "hr", "img", "input", "meta", "link"})

# Drop Summary when empty or shorter than this (after cleaning).
_SUMMARY_MIN_LEN = 80

# Citation-style brackets: [1, 2, 183], [K], [1-2]
_CITATION_BRACKETS = re.compile(
    r"\[\d+(?:\s*,\s*\d+)*\]|" r"\[K\]|" r"\[\d+\s*[-–]\s*\d+\]"
)

# Medicinal disclaimer paragraph
_MEDICINAL_DISCLAIMER = re.compile(
    r"Plants For A Future can not take any responsibility for any adverse effects from the use of plants\. "
    r"Always seek advice from a professional before using a plant medicinally\.\s*",
    re.IGNORECASE,
)

# Bookshop promo block (often embedded in a section body)
_BOOKSHOP_BLOCK = re.compile(
    r"The Bookshop: Edible Plant Books.*?Shop Now",
    re.DOTALL | re.IGNORECASE,
)

# Tail junk after cultivation / propagation (Carbon widget, converter, bookshop)
_TAIL_JUNK = re.compile(
    r"(?:^|\n)\s*References\s*\n\s*Carbon Farming Information.*?Shop Now",
    re.DOTALL | re.IGNORECASE,
)
_TEMP_CONVERTER_BLOCK = re.compile(
    r"Temperature Converter.*?Fahrenheit:\s*",
    re.DOTALL | re.IGNORECASE,
)
_PFAF_BOOKSHOP_TAIL = re.compile(
    r"The PFAF Bookshop.*?Shop Now",
    re.DOTALL | re.IGNORECASE,
)

# Lines that are only navigation / filler (exact match after strip, case-insensitive)
_DROP_STANDALONE_LINES = frozenset(
    {
        "references",
        "more on edible uses",
        "more on medicinal uses",
        "more on other uses",
        "uk hardiness map",
        "us hardiness map",
    }
)


def _slug_key(label: str) -> str:
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "field"


def _is_in_footer(tag: Tag) -> bool:
    for p in tag.parents:
        if not isinstance(p, Tag):
            continue
        if p.name == "footer":
            return True
        cid = (p.get("id") or "").lower()
        if "footer" in cid:
            return True
        classes = " ".join(p.get("class") or []).lower()
        if "footer" in classes and p.name in ("div", "footer", "section"):
            return True
    return False


def _is_ads(tag: Tag) -> bool:
    if tag.name == "ins" and "adsbygoogle" in (tag.get("class") or []):
        return True
    if tag.get("id") == "google_translate_element":
        return True
    return False


def _table_facts(soup: BeautifulSoup) -> dict[str, str]:
    """Key fields from the striped fact table (Common Name, Family, …)."""
    out: dict[str, str] = {}
    table = soup.find("table", class_=lambda c: c and "table-hover" in c and "table-striped" in c)
    if not table:
        return out

    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        raw_key = cells[0].get_text(separator=" ", strip=True)
        raw_key = re.sub(r"\s*\(info\)\s*$", "", raw_key, flags=re.I).strip()
        if not raw_key:
            continue
        key = _slug_key(raw_key)

        val_cell = cells[1]
        text = val_cell.get_text(separator=" ", strip=True)
        if not text:
            imgs = val_cell.find_all("img")
            if imgs:
                text = " | ".join(img.get("alt") or img.get("title") or "" for img in imgs).strip()
        if text:
            out[key] = text
    return out


def _content_after_heading(heading: Tag) -> str:
    """Text from siblings after ``heading`` until the next h1/h2/h3."""
    parts: list[str] = []
    cur = heading.next_sibling
    while cur is not None:
        if isinstance(cur, Tag):
            if cur.name in ("h1", "h2", "h3"):
                break
            if _is_ads(cur):
                cur = cur.next_sibling
                continue
            if cur.name == "script":
                cur = cur.next_sibling
                continue
            if cur.name in _VOID_HTML_TAGS:
                cur = cur.next_sibling
                continue
            t = cur.get_text(separator="\n", strip=True)
            if t:
                parts.append(t)
        cur = cur.next_sibling
    return "\n\n".join(parts).strip()


def _sections_from_headings(soup: BeautifulSoup) -> dict[str, str]:
    """h2 / h3 keys → full paragraph text until the next heading of same or higher level."""
    body = soup.find("body")
    if not body:
        return {}

    sections: dict[str, str] = {}
    for tag in body.find_all(["h2", "h3"]):
        if _is_in_footer(tag):
            continue
        if _is_ads(tag):
            continue
        title = tag.get_text(separator=" ", strip=True)
        if not title:
            continue
        if title in _SKIP_SECTION_TITLES:
            continue
        if any(title.startswith(p) for p in _SKIP_SECTION_TITLE_PREFIXES):
            continue
        if tag.name == "h2" and "text-white" in (tag.get("class") or []):
            continue

        block = _content_after_heading(tag)
        if block:
            sections[title] = block

    return sections


def _collapse_whitespace(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _remove_citation_brackets(text: str) -> str:
    return _CITATION_BRACKETS.sub("", text)


def _filter_drop_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.split("\n"):
        if line.strip().lower() in _DROP_STANDALONE_LINES:
            continue
        lines.append(line)
    return "\n".join(lines)


def _strip_boilerplate_sections(text: str) -> str:
    t = text
    t = _MEDICINAL_DISCLAIMER.sub("", t)
    t = _BOOKSHOP_BLOCK.sub("", t)
    t = _TAIL_JUNK.sub("", t)
    t = _TEMP_CONVERTER_BLOCK.sub("", t)
    t = _PFAF_BOOKSHOP_TAIL.sub("", t)
    return t


def _clean_text(text: str) -> str:
    t = _strip_boilerplate_sections(text)
    t = _remove_citation_brackets(t)
    t = _filter_drop_lines(t)
    t = _collapse_whitespace(t)
    return t


_KNOWN_HAZARDS_SECTION_TITLE = "Known hazards"


def _normalize_metadata(table: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in table.items():
        if k in ("weed_potential", "known_hazards"):
            continue
        out[k] = _clean_text(v)
    return out


def _strip_latin_author_suffix(latin: str | None) -> str | None:
    """Remove trailing author abbreviation, e.g. ``Morus rubra - L.`` → ``Morus rubra``."""
    if not latin:
        return None
    t = latin.strip()
    t = re.sub(r"\s*[-–]\s*L\.?\s*$", "", t, flags=re.I).strip()
    return t or None


def _normalize_sections(sections: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for title, body in sections.items():
        cleaned = _clean_text(body)
        if not cleaned:
            continue
        if title == "Summary" and len(cleaned) < _SUMMARY_MIN_LEN:
            continue
        out[title] = cleaned
    return out


@dataclass
class PfafStructured:
    """PFAF plant page: ``latin_name``; ``metadata`` is the fact table minus fields promoted to ``sections``."""

    latin_name: str | None
    metadata: dict[str, str]
    sections: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_pfaf_html(html: str) -> PfafStructured:
    """
    Parse raw HTML from a PFAF ``Plant.aspx`` response body.

    Returns ``latin_name`` as the main plant name, ``metadata`` from the striped table (no
    ``weed_potential`` or ``known_hazards``), and ``sections`` with boilerplate removed, citations
    stripped, and whitespace normalized. ``known_hazards`` from the fact table is added as the
    section ``Known hazards``. Short or empty ``Summary`` sections are dropped.
    """
    soup = BeautifulSoup(html, "html.parser")

    latin = None
    h1 = soup.find("h1")
    if h1:
        latin = h1.get_text(separator=" ", strip=True)
    if not latin:
        span = soup.find("span", id=re.compile(r"lbldisplatinname", re.I))
        if span:
            latin = span.get_text(separator=" ", strip=True)

    latin = _strip_latin_author_suffix(latin)

    table = _table_facts(soup)
    raw_sections = _sections_from_headings(soup)

    metadata = _normalize_metadata(table)
    sections = _normalize_sections(raw_sections)

    kh = table.get("known_hazards")
    if kh:
        cleaned = _clean_text(kh)
        if cleaned:
            sections[_KNOWN_HAZARDS_SECTION_TITLE] = cleaned

    return PfafStructured(
        latin_name=latin,
        metadata=metadata,
        sections=sections,
    )
