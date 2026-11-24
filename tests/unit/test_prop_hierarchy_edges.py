"""Property test: hierarchy edges equal the parent column.

Design reference: lindas-publication, Property 6 (Requirements 3.1 - 3.4).
Forests are valid by construction (each parent is an earlier row's ID), so no
``assume``/``filter`` is needed.
"""
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from rdflib import URIRef

from pylindas.lindas.namespaces import SCHEMA
from pylindas.pycube.cube import Cube
from tests.unit.conftest import minimal_description, minimal_frame
from pylindas.pycube.hierarchy import edges_from_parent_column

_ROOT_MARKERS = [None, float("nan"), "", "  ", "\t", pd.NA]


@st.composite
def _forests(draw):
    """Return (ids, parents, expected_edges) with duplicate rows included."""
    ids = draw(st.lists(st.from_regex(r"[A-Za-z0-9]{1,6}", fullmatch=True),
                        min_size=1, max_size=12, unique=True))
    parents, expected = [], set()
    for i, child in enumerate(ids):
        if i == 0 or draw(st.booleans()):
            parents.append(draw(st.sampled_from(_ROOT_MARKERS)))
        else:
            parent = ids[draw(st.integers(0, i - 1))]
            pad = draw(st.sampled_from(["", " ", "  "]))
            parents.append(f"{pad}{parent}{pad}")
            expected.add((parent, child))
    # duplicate rows
    rows = list(zip(ids, parents))
    dup_idx = draw(st.lists(st.integers(0, len(rows) - 1), max_size=5))
    rows += [rows[i] for i in dup_idx]
    return [r[0] for r in rows], [r[1] for r in rows], expected


def _cube(with_parent_field=True):
    concept = {"URI": "/assortment/{ID}", "name-field": "label"}
    if with_parent_field:
        concept["parent-field"] = "parent"
    yaml = minimal_description(**{"Base-URI": "https://example.org/cube", "Concepts": {"assortment": concept}})
    return Cube(minimal_frame(), yaml, environment="TEST", local=True)


def _iri(i):
    return URIRef(f"https://example.org/cube/c/1/concept/assortment/{i}")


# Feature: lindas-publication, Property 6: Hierarchy edges equal the parent column
# Validates: Requirements 3.1, 3.2, 3.3, 3.4
@given(_forests())
def test_pure_edges_equal_parent_column(forest):
    ids, parents, expected = forest
    edges = edges_from_parent_column(ids, parents)
    assert len(edges) == len(set(edges))
    assert set(edges) == expected


# Feature: lindas-publication, Property 6: Hierarchy edges equal the parent column
# Validates: Requirements 3.1, 3.2, 3.3, 3.4
@given(_forests())
def test_cube_has_part_triples_equal_parent_column(forest):
    ids, parents, expected = forest
    df = pd.DataFrame({"ID": ids, "label": ids, "parent": pd.Series(parents, dtype=object)})

    cube = _cube().write_concept("assortment", df)
    triples = list(cube._graph.triples((None, SCHEMA.hasPart, None)))
    assert len(triples) == len(set(triples))  # each written once
    assert {(s, o) for s, _, o in triples} == {(_iri(p), _iri(c)) for p, c in expected}

    # no parent-field -> no hasPart triple, whatever the parent column holds
    plain = _cube(with_parent_field=False).write_concept("assortment", df)
    assert list(plain._graph.triples((None, SCHEMA.hasPart, None))) == []
