"""The layering rules, enforced.

Documentation drifts; an import check does not. Dependencies point downward
only, and in particular a source adapter must never reach into the layer
that persists what it returns.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "research"

#: Lower numbers may be imported by higher ones, never the reverse.
LAYERS = {
    "errors": 0,
    "ids": 0,
    "models": 0,
    "config": 1,
    "normalize": 1,
    "llm": 2,
    "sources": 2,
    "storage": 2,
    # Budgets keep durable counters, so they sit above storage - and below
    # everything that spends, which is nearly everything.
    "budgets": 3,
    "graph": 3,
    "acquisition": 4,
    "operations": 5,
    "agents": 6,
    "orchestration": 7,
    "synthesis": 8,
    "cli": 9,
}


def module_layer(path: Path) -> tuple[str, int] | None:
    parts = path.relative_to(PACKAGE).parts
    name = parts[0] if len(parts) > 1 else path.stem
    if name not in LAYERS:
        return None
    return name, LAYERS[name]


def imported_layers(path: Path) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        for module in modules:
            segments = module.split(".")
            if segments[0] != "research" or len(segments) < 2:
                continue
            if segments[1] in LAYERS:
                found.append((segments[1], module))
    return found


SOURCE_FILES = sorted(PACKAGE.rglob("*.py"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_imports_point_downward(path: Path) -> None:
    own = module_layer(path)
    if own is None:
        return
    own_name, own_layer = own
    for target_name, module in imported_layers(path):
        if target_name == own_name:
            continue
        assert LAYERS[target_name] <= own_layer, (
            f"{path.relative_to(PACKAGE)} ({own_name}, layer {own_layer}) imports "
            f"{module} ({target_name}, layer {LAYERS[target_name]})"
        )


def test_sources_cannot_persist_anything() -> None:
    """Adapters translate; they never write to the store."""
    for path in (PACKAGE / "sources").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "ResearchStore" not in text, f"{path.name} reaches into storage"
        assert "research.storage" not in text, f"{path.name} reaches into storage"


def test_research_loop_has_no_execution_primitives() -> None:
    """No part of the research core may run a command or evaluate code."""
    forbidden = ("subprocess", "os.system", "eval(", "exec(", "pty.spawn")
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text, f"{path.relative_to(PACKAGE)} contains {marker!r}"
