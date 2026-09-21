"""Rendering primitives for Markdown knowledge bases.

A vault is not a passive pile of text. Obsidian renders Markdown *and* raw
HTML inside an Electron window, resolves ``[[wikilinks]]`` into real edges in
the graph, reads YAML frontmatter as structured properties, and lets
community plugins execute fenced blocks - Dataview runs queries, Templater
runs JavaScript.

So the rule that governs acquisition governs export too: retrieved content is
hostile data. Here that means a page cannot smuggle in markup, cannot forge
links to notes it has nothing to do with, cannot open a plugin block, and
cannot break out of the frontmatter into the body.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import yaml

#: Characters no filesystem, and no vault, should be asked to carry.
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")

#: Runs of three or more backticks open a fenced block, which is where plugin
#: execution lives.
_FENCE = re.compile(r"`{3,}")
#: A line that starts with # becomes a heading or a tag.
_LEADING_HASH = re.compile(r"^(\s*)(#+)", re.MULTILINE)
#: Obsidian treats %%...%% as a comment, which would swallow following text.
_COMMENT = re.compile(r"%%")


def escape_external(text: str | None, *, limit: int | None = None) -> str:
    """Neutralise retrieved content for rendering inside a vault.

    The substitutions are deliberately visible rather than silent - a reader
    should be able to tell that a document contained something that was
    defanged, and to compare the note against the stored original by its
    content hash.
    """
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFC", text)
    # HTML first: Obsidian renders it, and it is the only vector that reaches
    # outside Markdown semantics entirely.
    cleaned = cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Wikilinks and embeds: a page must not be able to forge graph edges, and
    # ![[...]] would pull another note's contents into this one.
    cleaned = cleaned.replace("![[", "!\\[\\[").replace("[[", "\\[\\[")
    cleaned = cleaned.replace("]]", "\\]\\]")
    # Fenced blocks: Dataview and Templater execute what is inside them.
    cleaned = _FENCE.sub(lambda match: "\\`" * len(match.group(0)), cleaned)
    cleaned = _COMMENT.sub("\\%\\%", cleaned)
    cleaned = _LEADING_HASH.sub(lambda match: f"{match.group(1)}\\{match.group(2)}", cleaned)
    if limit is not None and len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip() + "\u2026"
    return cleaned


def frontmatter(properties: dict[str, Any]) -> str:
    """Serialise YAML frontmatter.

    Values come from retrieved records, so this goes through a real YAML
    serialiser rather than string formatting: a title containing a colon, a
    newline, or a line reading ``---`` would otherwise end the block early and
    spill into the note body as content.
    """
    cleaned = {key: value for key, value in properties.items() if value not in (None, [], {})}
    if not cleaned:
        return ""
    body = yaml.safe_dump(
        cleaned, default_flow_style=False, allow_unicode=True, sort_keys=True, width=100
    )
    return f"---\n{body}---\n"


def safe_filename(text: str, *, limit: int = 80, fallback: str = "untitled") -> str:
    """A note name that is safe on disk and readable in a graph view."""
    cleaned = unicodedata.normalize("NFC", text or "")
    cleaned = _UNSAFE_FILENAME.sub(" ", cleaned)
    # '[' and '#' and '^' are legal in filenames but confuse Obsidian's link
    # parser when they appear inside one.
    cleaned = cleaned.replace("[", "(").replace("]", ")").replace("#", " ").replace("^", " ")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip(" .")
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rstrip(" .")
    return cleaned or fallback


def wikilink(note: str, alias: str | None = None) -> str:
    """A link to another note, with the alias escaped of pipe characters."""
    target = note.replace("|", " ").replace("[", "(").replace("]", ")")
    if alias is None:
        return f"[[{target}]]"
    return f"[[{target}|{alias.replace('|', ' ')}]]"


def callout(kind: str, title: str, body: str = "", *, fold: str = "") -> str:
    """An Obsidian callout block: native, and visually distinct."""
    lines = [f"> [!{kind}]{fold} {title}".rstrip()]
    for line in (body or "").split("\n"):
        lines.append(f"> {line}" if line else ">")
    return "\n".join(lines)


def blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in (text or "").split("\n"))


def table(headers: list[str], rows: list[list[str]]) -> str:
    """A Markdown table with cell contents escaped of pipes."""
    def cell(value: Any) -> str:
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)
