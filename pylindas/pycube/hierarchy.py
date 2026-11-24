"""Pure hierarchy logic for concept tables (no network, no rdflib).

``edges_from_parent_column`` turns a parent column into ``(parent, child)``
edges. ``check_hierarchy`` validates them and raises one ``HierarchyError``
that reports every finding at once.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import pandas as pd

from pylindas.lindas.exceptions import HierarchyError


def _normalize_parent(value: Any) -> Optional[str]:
    """Return the stripped parent ID, or None when the value marks a root."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            # a numeric ID column read with missing values becomes float: 3.0 -> "3"
            return str(int(value))
    text = str(value).strip()
    return text or None


def edges_from_parent_column(
    ids: Sequence[str], parents: Sequence[Optional[str]]
) -> list[tuple[str, str]]:
    """Return ``(parent_id, child_id)`` edges, deduplicated, in input order.

    None, NaN, ``pd.NA`` and strings that are empty after ``strip()`` are
    roots and produce no edge.
    """
    if len(ids) != len(parents):
        raise ValueError("ids and parents must have the same length")
    edges: dict[tuple[str, str], None] = {}  # ordered set
    for child, raw_parent in zip(ids, parents):
        parent = _normalize_parent(raw_parent)
        if parent is None:
            continue
        edges.setdefault((parent, str(child)), None)
    return list(edges)


def check_hierarchy(ids: Sequence[str], parents: Sequence[Optional[str]]) -> None:
    """Raise one ``HierarchyError`` listing all unknown parents, self parents and cycles.

    ``details`` has the keys ``unknown_parents`` (parent -> child IDs),
    ``self_parents`` (concept IDs) and ``cycles`` (lists of concept IDs).
    """
    known = {str(i) for i in ids}
    edges = edges_from_parent_column(ids, parents)

    unknown: dict[str, list[str]] = {}
    self_parents: list[str] = []
    graph: dict[str, list[str]] = {}  # child -> known parents (self edges excluded)
    for parent, child in edges:
        if parent == child:
            self_parents.append(child)
        elif parent not in known:
            unknown.setdefault(parent, []).append(child)
        else:
            graph.setdefault(child, []).append(parent)

    cycles = _find_cycles(graph)

    if not (unknown or self_parents or cycles):
        return

    problems = []
    if unknown:
        problems.append(
            "unknown parents: "
            + "; ".join(f"{p!r} (children: {', '.join(c)})" for p, c in unknown.items())
        )
    if self_parents:
        problems.append("concepts that are their own parent: " + ", ".join(self_parents))
    if cycles:
        problems.append("cycles: " + "; ".join(" -> ".join(c) for c in cycles))
    raise HierarchyError(
        "Invalid hierarchy: " + " | ".join(problems),
        details={"unknown_parents": unknown, "self_parents": self_parents, "cycles": cycles},
    )


def _find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """Iterative 3-colour DFS; returns each distinct cycle (by node set) once."""
    WHITE, GRAY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}
    cycles: list[list[str]] = []
    seen: set[frozenset[str]] = set()

    for start in graph:
        if colour.get(start, WHITE) != WHITE:
            continue
        colour[start] = GRAY
        path = [start]
        stack = [iter(graph.get(start, ()))]
        while stack:
            advanced = False
            for nxt in stack[-1]:
                state = colour.get(nxt, WHITE)
                if state == WHITE:
                    colour[nxt] = GRAY
                    path.append(nxt)
                    stack.append(iter(graph.get(nxt, ())))
                    advanced = True
                    break
                if state == GRAY:
                    cycle = path[path.index(nxt):]
                    key = frozenset(cycle)
                    if key not in seen:
                        seen.add(key)
                        cycles.append(cycle)
            if not advanced:
                colour[path.pop()] = BLACK
                stack.pop()
    return cycles
