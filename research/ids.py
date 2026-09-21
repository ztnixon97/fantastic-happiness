"""Stable, readable identifiers.

Identifiers look like ``evidence:412`` or ``claim:17``. They are allocated by
the store (see :meth:`research.storage.database.Database.next_id`) so they are
stable across processes and resumable runs, and short enough that a report
footnote can carry one.
"""

from __future__ import annotations

import re

INVESTIGATION = "investigation"
TASK = "task"
EVIDENCE = "evidence"
CLAIM = "claim"
ENTITY = "entity"
EVENT = "event"
CITATION = "citation"
QUERY = "query"
FETCH = "fetch"
LINK = "link"
RELATIONSHIP = "relationship"

_ID_RE = re.compile(r"^(?P<prefix>[a-z_]+):(?P<number>\d+)$")


def format_id(prefix: str, number: int) -> str:
    return f"{prefix}:{number}"


def parse_id(value: str) -> tuple[str, int]:
    """Split ``evidence:412`` into ``("evidence", 412)``.

    Raises :class:`ValueError` for anything that is not a well-formed id, so
    an identifier arriving from model output or external data cannot be
    mistaken for a valid reference.
    """
    match = _ID_RE.match(value or "")
    if not match:
        raise ValueError(f"malformed identifier: {value!r}")
    return match.group("prefix"), int(match.group("number"))


def is_id(value: str, prefix: str | None = None) -> bool:
    try:
        parsed_prefix, _ = parse_id(value)
    except ValueError:
        return False
    return prefix is None or parsed_prefix == prefix
