"""Hierarchy root resolution through the concept mapping (Requirements 4.1 - 4.4)."""
import pandas as pd
import pytest
from rdflib import Literal, URIRef

from pylindas.lindas.exceptions import HierarchyError
from pylindas.lindas.namespaces import META, SCHEMA
from pylindas.pycube.cube import Cube
from tests.unit.conftest import minimal_description

CONCEPT_BASE = "https://example.org/cube/c/1/concept/assortment/"


def _iri(i):
    return URIRef(CONCEPT_BASE + i)


def _cube(root="total", mapping=None, hierarchy=True):
    dim = {
        "name": {"en": "Assortment"},
        "description": {"en": "Assortment"},
        "dimension-type": "Key Dimension",
        "datatype": "URI",
        "scale-type": "nominal",
        "path": "assortment",
        "mapping": mapping or {"type": "concept", "replacement-automated": "/assortment/{assortment}"},
    }
    if hierarchy:
        dim["hierarchy"] = [{"root": root, "name": "H", "next-in-hierarchy": {"name": "N", "path": str(SCHEMA.hasPart)}}]
    yaml = minimal_description(**{
        "Base-URI": "https://example.org/cube",
        "dimensions": {"assortment": dim},
        "Concepts": {"assortment": {"URI": "/assortment/{ID}", "name-field": "label", "parent-field": "parent"}},
    })
    df = pd.DataFrame({"assortment": ["a", "b"]})
    return Cube(df, yaml, environment="TEST", local=True).prepare_data()


def _table():
    return pd.DataFrame({"ID": ["total", "a", "b"], "label": ["T", "A", "B"], "parent": [None, "total", "total"]})


def _roots(cube):
    return list(cube._graph.objects(None, META.hierarchyRoot))


def test_root_becomes_concept_iri_and_is_subject_of_has_part():
    cube = _cube()
    cube.write_shape()
    assert _roots(cube) == []  # concept table unknown yet: nothing written
    cube.write_concept("assortment", _table())
    assert _roots(cube) == [_iri("total")]
    assert {s for s in cube._graph.subjects(SCHEMA.hasPart, None)} == {_iri("total")}
    assert not any(isinstance(o, Literal) and str(o) == "total" for o in cube._graph.objects(None, META.hierarchyRoot))
    assert URIRef("total") not in set(cube._graph.all_nodes())


def test_root_resolved_immediately_when_concept_written_first():
    cube = _cube()
    cube.write_concept("assortment", _table())
    cube.write_shape()
    assert _roots(cube) == [_iri("total")]


def test_unknown_root_raises_before_any_concept_or_root_triple():
    cube = _cube(root="nope")
    cube.write_shape()
    before = len(cube._graph)
    with pytest.raises(HierarchyError) as ei:
        cube.write_concept("assortment", _table())
    assert "assortment" in str(ei.value) and "nope" in str(ei.value)
    assert ei.value.details["dimension"] == "assortment" and ei.value.details["root"] == "nope"
    assert len(cube._graph) == before
    assert _roots(cube) == []


def test_unknown_root_with_concept_written_first_raises_in_write_shape():
    cube = _cube(root="nope")
    cube.write_concept("assortment", _table())
    with pytest.raises(HierarchyError, match="nope"):
        cube.write_shape()


def test_http_root_used_as_is_with_concept_mapping():
    cube = _cube(root="https://example.org/some/root")
    cube.write_shape()
    assert _roots(cube) == [URIRef("https://example.org/some/root")]


def test_without_concept_mapping_additive_and_replace_unchanged():
    add = _cube(root="x", mapping={"type": "additive", "base": "https://b.example/"})
    add.write_shape()
    assert _roots(add) == [URIRef("https://b.example/x")]
    rep = _cube(root="x", mapping={"type": "replace", "replacements": {"x": "https://r.example/x", "a": "https://r.example/a", "b": "https://r.example/b"}})
    rep.write_shape()
    assert _roots(rep) == [URIRef("https://r.example/x")]
