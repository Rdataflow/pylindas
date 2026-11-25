"""Property test: invalid hierarchies are rejected completely.

Validates Requirements 3.5, 3.6, 3.7.
"""
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from rdflib import Graph
from rdflib.compare import isomorphic

from pylindas.lindas.exceptions import HierarchyError
from pylindas.lindas.namespaces import SCHEMA
from pylindas.pycube.cube import Cube
from tests.unit.conftest import minimal_description, minimal_frame
from pylindas.pycube.hierarchy import check_hierarchy


def _cube():
    yaml = minimal_description(**{
        "Base-URI": "https://example.org/cube",
        "Concepts": {"assortment": {"URI": "/assortment/{ID}", "name-field": "label",
                                    "parent-field": "parent"}},
    })
    return Cube(minimal_frame(), yaml, environment="TEST", local=True)


@st.composite
def invalid_hierarchies(draw):
    """Valid forest over n unique IDs, then >= 1 injected defects on disjoint nodes.

    Returns (ids, parents, expected) where expected holds the injected defects.
    """
    n = draw(st.integers(min_value=1, max_value=14))
    ids = [f"n{i}" for i in range(n)]
    order = draw(st.permutations(range(n)))  # disjoint node pool for defects
    pool = list(order)

    # valid forest: each node's parent is an earlier node (by index) or none
    parents = [None] * n
    for i in range(1, n):
        if draw(st.booleans()):
            parents[i] = ids[draw(st.integers(0, i - 1))]

    n_unknown = draw(st.integers(0, 3))
    n_self = draw(st.integers(0, 3))
    cycle_lens = draw(st.lists(st.integers(2, 4), min_size=0, max_size=2))
    if n_unknown + n_self + len(cycle_lens) == 0:
        n_unknown = 1

    expected = {"unknown": {}, "self": set(), "cycles": []}
    unknown_names = [f"unk{j}" for j in range(3)]

    for _ in range(n_unknown):
        if not pool:
            break
        i = pool.pop()
        name = draw(st.sampled_from(unknown_names))
        parents[i] = name
        expected["unknown"].setdefault(name, []).append(i)
    for _ in range(n_self):
        if not pool:
            break
        i = pool.pop()
        parents[i] = ids[i]
        expected["self"].add(ids[i])
    for length in cycle_lens:
        if len(pool) < length:
            break
        members = [pool.pop() for _ in range(length)]
        for a, b in zip(members, members[1:] + members[:1]):
            parents[a] = ids[b]
        expected["cycles"].append({ids[m] for m in members})

    # at least one defect must actually have been injected
    if not (expected["unknown"] or expected["self"] or expected["cycles"]):
        parents[0] = "unk0"
        expected["unknown"]["unk0"] = [0]

    expected["unknown"] = {k: sorted(ids[i] for i in v) for k, v in expected["unknown"].items()}
    return ids, parents, expected


# Feature: lindas-publication, Property 7: Invalid hierarchies are rejected completely
# Validates: Requirements 3.5, 3.6, 3.7
@settings(max_examples=100, deadline=None)
@given(invalid_hierarchies())
def test_invalid_hierarchies_are_rejected_completely(case):
    ids, parents, expected = case

    # pure check: one error with complete findings
    with pytest.raises(HierarchyError) as ei:
        check_hierarchy(ids, parents)
    d = ei.value.details
    assert {p: sorted(c) for p, c in d["unknown_parents"].items()} == expected["unknown"]
    assert set(d["self_parents"]) == expected["self"]
    assert len(d["self_parents"]) == len(expected["self"])
    assert sorted(map(frozenset, d["cycles"]), key=sorted) == sorted(
        map(frozenset, expected["cycles"]), key=sorted)
    assert len(d["cycles"]) == len(expected["cycles"])  # each cycle reported once

    # cube level: nothing is written, not even concept triples
    df = pd.DataFrame({"ID": ids, "label": [f"L{i}" for i in ids], "parent": parents})
    cube = _cube()
    snapshot = Graph()
    for t in cube._graph:
        snapshot.add(t)
    before = len(cube._graph)
    with pytest.raises(HierarchyError) as ei2:
        cube.write_concept("assortment", df)
    assert ei2.value.details == d
    assert len(cube._graph) == before
    assert isomorphic(cube._graph, snapshot)
    assert not list(cube._graph.triples((None, SCHEMA.hasPart, None)))
