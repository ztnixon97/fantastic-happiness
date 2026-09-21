"""HTML to evidence text.

A small, dependency-free extractor. It pulls the readable body, the metadata
publishers actually provide (canonical link, Open Graph, JSON-LD) and nothing
executable. Scripts are discarded rather than interpreted, with one exception:
``application/ld+json`` blocks are parsed as *data*, since that is where
publication dates and bylines usually live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

from research.normalize.text import clean_text, normalize_whitespace

#: Elements whose text is chrome, not content.
_SKIP_CONTENT = frozenset(
    {
        "script", "style", "noscript", "svg", "canvas", "form", "nav", "aside",
        "footer", "header", "button", "select", "option", "iframe", "template",
        "figure", "figcaption", "label", "input", "textarea",
    }
)

#: Elements that end a line of prose.
_BLOCK = frozenset(
    {
        "p", "div", "section", "article", "br", "li", "tr", "h1", "h2", "h3",
        "h4", "h5", "h6", "blockquote", "pre", "ul", "ol", "table", "dd", "dt",
    }
)

_DATE_KEYS = (
    "article:published_time",
    "datePublished",
    "og:published_time",
    "publishdate",
    "publish-date",
    "date",
    "dc.date",
    "dc.date.issued",
    "citation_publication_date",
    "parsely-pub-date",
)

_AUTHOR_KEYS = ("author", "article:author", "citation_author", "parsely-author")


@dataclass(slots=True)
class ExtractedPage:
    title: str | None = None
    text: str = ""
    canonical_url: str | None = None
    description: str | None = None
    site_name: str | None = None
    language: str | None = None
    published_raw: str | None = None
    authors: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    jsonld: list[Any] = field(default_factory=list)
    #: ``(href, anchor text)`` pairs, capped; used for primary-source chasing.
    links: list[tuple[str, str]] = field(default_factory=list)

    @property
    def published_at(self) -> datetime | None:
        return parse_date(self.published_raw)


def parse_date(value: str | None) -> datetime | None:
    """Parse the date formats publishers actually emit, or return ``None``.

    Guessing a date is worse than having none: a fabricated timestamp
    silently corrupts every timeline built on it.
    """
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    candidates = [raw]
    if raw.endswith("Z"):
        candidates.append(raw[:-1] + "+00:00")
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%dT%H:%M:%S%z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class _PageParser(HTMLParser):
    def __init__(self, *, max_links: int = 200) -> None:
        super().__init__(convert_charrefs=True)
        self.page = ExtractedPage()
        self._skip_depth = 0
        self._in_title = False
        self._in_ldjson = False
        self._article_depth = 0
        self._chunks: list[str] = []
        self._article_chunks: list[str] = []
        self._title_parts: list[str] = []
        self._ldjson_parts: list[str] = []
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._max_links = max_links

    # -- tags ----------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): (value or "") for key, value in attrs}
        if tag == "script" and attributes.get("type", "").lower() == "application/ld+json":
            self._in_ldjson = True
            self._ldjson_parts = []
            return
        if tag in _SKIP_CONTENT:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag == "article":
            self._article_depth += 1
        elif tag == "html":
            self.page.language = attributes.get("lang") or self.page.language
        elif tag == "meta":
            self._handle_meta(attributes)
        elif tag == "link" and "canonical" in attributes.get("rel", "").lower():
            self.page.canonical_url = attributes.get("href") or self.page.canonical_url
        elif tag == "a" and attributes.get("href"):
            self._link_href = attributes["href"]
            self._link_text = []
        if tag in _BLOCK:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_ldjson:
            self._in_ldjson = False
            self._store_ldjson("".join(self._ldjson_parts))
            return
        if tag in _SKIP_CONTENT:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        elif tag == "article":
            self._article_depth = max(0, self._article_depth - 1)
        elif tag == "a" and self._link_href is not None:
            if len(self.page.links) < self._max_links:
                anchor = normalize_whitespace("".join(self._link_text))[:200]
                self.page.links.append((self._link_href, anchor))
            self._link_href = None
            self._link_text = []
        if tag in _BLOCK:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self._in_ldjson:
            self._ldjson_parts.append(data)
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._skip_depth:
            return
        if self._link_href is not None:
            self._link_text.append(data)
        self._emit(data)

    # -- helpers -------------------------------------------------------
    def _emit(self, text: str) -> None:
        if self._skip_depth or self._in_ldjson:
            return
        self._chunks.append(text)
        if self._article_depth:
            self._article_chunks.append(text)

    def _handle_meta(self, attributes: dict[str, str]) -> None:
        key = (
            attributes.get("property")
            or attributes.get("name")
            or attributes.get("itemprop")
            or ""
        ).lower()
        content = attributes.get("content", "").strip()
        if not key or not content:
            return
        self.page.meta.setdefault(key, content)
        if key in ("og:title", "twitter:title") and not self.page.title:
            self.page.title = content
        elif key in ("og:description", "description"):
            self.page.description = self.page.description or content
        elif key == "og:site_name":
            self.page.site_name = self.page.site_name or content
        elif key == "og:url" and not self.page.canonical_url:
            self.page.canonical_url = content
        elif key in _DATE_KEYS and not self.page.published_raw:
            self.page.published_raw = content
        elif key in _AUTHOR_KEYS:
            for name in _split_authors(content):
                if name not in self.page.authors:
                    self.page.authors.append(name)

    def _store_ldjson(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except ValueError:
            return
        self.page.jsonld.append(data)
        for node in _iter_ldjson_nodes(data):
            if not isinstance(node, dict):
                continue
            if not self.page.published_raw:
                for key in ("datePublished", "dateCreated", "uploadDate"):
                    if node.get(key):
                        self.page.published_raw = str(node[key])
                        break
            author = node.get("author")
            for name in _ldjson_names(author):
                if name not in self.page.authors:
                    self.page.authors.append(name)
            publisher = node.get("publisher")
            if not self.page.site_name:
                names = _ldjson_names(publisher)
                if names:
                    self.page.site_name = names[0]
            if not self.page.title and isinstance(node.get("headline"), str):
                self.page.title = node["headline"]

    def finish(self) -> ExtractedPage:
        if self._title_parts and not self.page.title:
            self.page.title = normalize_whitespace("".join(self._title_parts)) or None
        body = "".join(self._article_chunks or self._chunks)
        self.page.text = clean_text(body)
        if self.page.title:
            self.page.title = _strip_site_suffix(
                normalize_whitespace(self.page.title), self.page.site_name
            )
        return self.page


def _strip_site_suffix(title: str, site_name: str | None) -> str:
    """Drop the outlet name publishers append to <title>.

    ``NRC approves design | Example News`` is the same headline as
    ``NRC approves design``; keeping the suffix would make two copies of one
    story look like two different headlines.
    """
    if not site_name:
        return title
    for separator in (" | ", " - ", " \u2013 ", " \u2014 ", " :: ", " \u00b7 "):
        suffix = f"{separator}{site_name}"
        if title.lower().endswith(suffix.lower()) and len(title) > len(suffix) + 5:
            return title[: -len(suffix)].strip()
    return title


def _iter_ldjson_nodes(data: Any, depth: int = 0) -> list[Any]:
    if depth > 4:
        return []
    if isinstance(data, list):
        nodes: list[Any] = []
        for item in data:
            nodes.extend(_iter_ldjson_nodes(item, depth + 1))
        return nodes
    if isinstance(data, dict):
        nodes = [data]
        for key in ("@graph", "mainEntity", "mainEntityOfPage"):
            if key in data:
                nodes.extend(_iter_ldjson_nodes(data[key], depth + 1))
        return nodes
    return []


def _ldjson_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return _split_authors(value)
    if isinstance(value, dict):
        name = value.get("name")
        return _split_authors(name) if isinstance(name, str) else []
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            names.extend(_ldjson_names(item))
        return names
    return []


def _split_authors(value: str | None) -> list[str]:
    if not value:
        return []
    text = normalize_whitespace(value)
    if not text or len(text) > 300:
        return []
    for separator in (";", " and ", "|", ","):
        if separator in text:
            parts = [part.strip(" ,;|") for part in text.split(separator)]
            return [part for part in parts if 1 < len(part) <= 120][:20]
    return [text] if len(text) <= 120 else []


def extract_page(html: str, *, max_characters: int = 400_000) -> ExtractedPage:
    """Parse an HTML document into text plus publisher metadata."""
    parser = _PageParser()
    try:
        parser.feed(html[:max_characters])
        parser.close()
    except Exception:  # pragma: no cover - malformed markup should not crash
        pass
    return parser.finish()
