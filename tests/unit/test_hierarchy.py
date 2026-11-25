"""Unit tests for ``pycube.hierarchy`` and the ``parent-field`` wiring in ``Cube.write_concept``.

Validates Requirements 3.1 - 3.7.
"""
import pandas as pd
import pytest
from rdflib import URIRef

from pylindas.lindas.exceptions import HierarchyError
from pylindas.lindas.namespaces import SCHEMA
from pylindas.pycube.cube import Cube
from tests.unit.conftest import minimal_description, minimal_frame
from pylindas.pycube.hierarchy import check_hierarchy, edges_from_parent_column

# ---------------------------------------------------------------------------
# pure functions
# ---------------------------------------------------------------------------


def test_edges_roots_are_normalized_and_deduplicated_in_order():
    ids = ["a", "b", "c", "d", "e", "b"]
    parents = [None, "a", float("nan"), "  ", pd.NA, "a"]
    assert edges_from_parent_column(ids, parents) == [("a", "b")]


def test_edges_strip_parent_and_keep_input_order():
    assert edges_from_parent_column(["x", "y", "z"], [" y ", "z", None]) == [("y", "x"), ("z", "y")]


def test_edges_float_integer_parent_becomes_plain_id():
    assert edges_from_parent_column(["1", "2"], [float("nan"), 1.0]) == [("1", "2")]


def test_check_valid_hierarchy_passes():
    check_hierarchy(["a", "b", "c"], [None, "a", "b"])


def test_check_collects_all_findings_in_one_error():
    ids = ["a", "b", "c", "d", "e", "f"]
    parents = [None, "zz", "zz", "d", "f", "e"]  # b,c -> unknown zz; d self; e<->f cycle
    with pytest.raises(HierarchyError) as ei:
        check_hierarchy(ids, parents)
    d = ei.value.details
    assert d["unknown_parents"] == {"zz": ["b", "c"]}
    assert d["self_parents"] == ["d"]
    assert len(d["cycles"]) == 1 and set(d["cycles"][0]) == {"e", "f"}
    assert "zz" in str(ei.value) and "d" in str(ei.value)


def test_check_detects_longer_cycle_and_ignores_acyclic_branches():
    ids = ["r", "a", "b", "c", "leaf"]
    parents = [None, "c", "a", "b", "r"]  # a->c->b->a
    with pytest.raises(HierarchyError) as ei:
        check_hierarchy(ids, parents)
    assert [set(c) for c in ei.value.details["cycles"]] == [{"a", "b", "c"}]
    assert ei.value.details["unknown_parents"] == {}


# ---------------------------------------------------------------------------
# Cube wiring
# ---------------------------------------------------------------------------


def _cube(parent_field="parent"):
    concept = {"URI": "/assortment/{ID}", "name-field": "label"}
    if parent_field:
        concept["parent-field"] = parent_field
    yaml = minimal_description(**{"Base-URI": "https://example.org/cube", "Concepts": {"assortment": concept}})
    return Cube(minimal_frame(), yaml, environment="TEST", local=True)


def _iri(i):
    return URIRef(f"https://example.org/cube/c/1/concept/assortment/{i}")


def _has_part(cube):
    return set(cube._graph.subject_objects(SCHEMA.hasPart))


def test_cube_writes_has_part_from_parent_column_without_mutating_input():
    df = pd.DataFrame({"ID": ["total", "a", "b", "a2"],
                       "label": ["T", "A", "B", "A2"],
                       "parent": [None, "total", " total ", "a"]})
    before = df.copy()
    cube = _cube().write_concept("assortment", df)
    assert _has_part(cube) == {(_iri("total"), _iri("a")), (_iri("total"), _iri("b")), (_iri("a"), _iri("a2"))}
    pd.testing.assert_frame_equal(df, before)
    assert "URI" not in df.columns


def test_cube_dedupes_repeated_pairs():
    df = pd.DataFrame({"ID": ["a", "root", "a"], "label": ["A", "R", "A"], "parent": ["root", None, "root"]})
    cube = _cube().write_concept("assortment", df)
    assert _has_part(cube) == {(_iri("root"), _iri("a"))}
    assert len(list(cube._graph.triples((None, SCHEMA.hasPart, None)))) == 1


def test_cube_without_parent_field_writes_no_has_part():
    df = pd.DataFrame({"ID": ["a", "b"], "label": ["A", "B"], "parent": [None, "a"]})
    cube = _cube(parent_field=None).write_concept("assortment", df)
    assert _has_part(cube) == set()
    assert len(cube._graph) > 0  # concepts themselves were written


def test_cube_invalid_hierarchy_raises_and_adds_no_triple():
    df = pd.DataFrame({"ID": ["a", "b", "c"], "label": ["A", "B", "C"], "parent": ["nope", "c", "b"]})
    cube = _cube()
    before = len(cube._graph)
    with pytest.raises(HierarchyError) as ei:
        cube.write_concept("assortment", df)
    assert ei.value.details["unknown_parents"] == {"nope": ["a"]}
    assert len(cube._graph) == before


def test_cube_missing_parent_column_raises():
    df = pd.DataFrame({"ID": ["a"], "label": ["A"]})
    with pytest.raises(HierarchyError):
        _cube().write_concept("assortment", df)
