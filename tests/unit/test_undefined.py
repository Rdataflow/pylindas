"""Missing key values are written as ``cube:Undefined`` (Requirements 6.5, 6.8 - 6.11)."""
import numpy as np
import pandas as pd
import pyshacl
import pytest
from rdflib import Literal, URIRef, XSD, Namespace
from rdflib.collection import Collection
from rdflib.namespace import SH

from pylindas.lindas.exceptions import InputValidationError
from pylindas.pycube.cube import Cube
from pylindas.pycube.validation import validate_dataframe
from tests.unit.conftest import minimal_description

CUBE = Namespace("https://cube.link/")
UNDEFINED = URIRef("https://cube.link/Undefined")
CUBE_URI = "https://example.org/cube/c/1"


def _dim(path, datatype=None, mapping=None, dim_type="Key Dimension", scale="nominal", **extra):
    d = {"name": {"en": path}, "description": {"en": path}, "path": path,
         "dimension-type": dim_type, "scale-type": scale}
    if datatype:
        d["datatype"] = datatype
    if mapping:
        d["mapping"] = mapping
    d.update(extra)
    return d


def _description(dims):
    return minimal_description(
        dimensions=dims,
        Contributor=[{"IRI": "https://example.org/contributor", "Name": "Contributor"}],
        **{"Date Created": "2024-01-01"},
    )


def _build(df, dims, **kw):
    cube = Cube(dataframe=df, cube_yaml=_description(dims), environment="TEST", local=True, **kw)
    return cube.prepare_data().write_cube().write_observations().write_shape()


def _self_consistent(cube):
    """Offline half of ``Cube.validate()``: observations against the cube's own shape.

    The full ``validate()`` also fetches the cube.link profiles over HTTP, which unit tests
    block; this is the step that checks the sh:in / sh:or shape written by this task.
    """
    cube._graph.add((cube._shape_URI, SH.targetClass, CUBE.Observation))
    try:
        ok, _, text = pyshacl.validate(data_graph=cube._graph)
    finally:
        cube._graph.remove((cube._shape_URI, SH.targetClass, CUBE.Observation))
    return ok, text


def _obs_objects(cube, path):
    pred = URIRef(f"https://example.org/cube/{path}")
    return [o for _, _, o in cube._graph.triples((None, pred, None))]


def _prop_shape(cube, path):
    pred = URIRef(f"https://example.org/cube/{path}")
    return next(s for s in cube._graph.subjects(SH.path, pred))


def _list(cube, head):
    return list(Collection(cube._graph, head))


# ---------------------------------------------------------------- (a) literal key
def test_literal_key_missing_becomes_empty_undefined_literal():
    dims = {"k": _dim("k", "string"), "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["a", None, np.nan], "v": [1.0, 2.0, 3.0]})
    df.loc[2, "k"] = pd.NA
    with pytest.raises(InputValidationError) as ei:  # None, NaN, NA are one value -> duplicates
        _build(df, dims)
    assert any("duplicate" in p for p in ei.value.problems)

    df = pd.DataFrame({"k": ["a", None], "v": [1.0, 2.0]})
    cube = _build(df, dims)
    objs = _obs_objects(cube, "k")
    assert Literal("", datatype=CUBE.Undefined) in objs
    assert Literal("a", datatype=XSD.string) in objs

    shape = _prop_shape(cube, "k")
    # plain sh:datatype replaced by sh:or([datatype xsd:string] [datatype cube:Undefined])
    assert (shape, SH.datatype, None) not in cube._graph
    branches = _list(cube, cube._graph.value(shape, SH["or"]))
    datatypes = {cube._graph.value(b, SH.datatype) for b in branches}
    assert datatypes == {XSD.string, CUBE.Undefined}
    # the Undefined literal is allowed in the nominal sh:in list
    assert Literal("", datatype=CUBE.Undefined) in _list(cube, cube._graph.value(shape, SH["in"]))
    ok, text = _self_consistent(cube)
    assert ok, text


# ---------------------------------------------------------------- (b) datatype URI key
def test_uri_key_missing_becomes_undefined_iri():
    dims = {"k": _dim("k", "URI"), "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["https://example.org/x", None], "v": [1.0, 2.0]})
    cube = _build(df, dims)
    objs = _obs_objects(cube, "k")
    assert UNDEFINED in objs and URIRef("https://example.org/x") in objs
    shape = _prop_shape(cube, "k")
    assert UNDEFINED in _list(cube, cube._graph.value(shape, SH["in"]))
    assert (shape, SH.nodeKind, SH.IRI) in cube._graph
    ok, text = _self_consistent(cube)
    assert ok, text


# ---------------------------------------------------------------- (c) mapped key dims
@pytest.mark.parametrize("mapping", [
    {"type": "additive", "base": "https://example.org/base/"},
    {"type": "replace", "replacements": {"a": "https://example.org/r/a", "b": "https://example.org/r/b"}},
    {"type": "concept", "replacement-automated": "/c/{k}"},
])
def test_mapped_key_missing_stays_undefined_and_builds_no_iri(mapping):
    dims = {"k": _dim("k", "URI", mapping=mapping),
            "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["a", None, "b"], "v": [1.0, 2.0, 3.0]})
    cube = _build(df, dims)
    objs = [str(o) for o in _obs_objects(cube, "k")]
    assert objs.count(str(UNDEFINED)) == 1
    assert not any("nan" in o.lower() or "none" in o.lower() for o in objs)
    ok, text = _self_consistent(cube)
    assert ok, text


def test_function_mapping_is_not_called_for_missing(tmp_path):
    f = tmp_path / "fn.py"
    f.write_text("def f(x):\n    return 'https://example.org/f/' + str(x)\n")
    mapping = {"type": "function", "filepath": str(f), "function-name": "f"}
    dims = {"k": _dim("k", "URI", mapping=mapping),
            "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    cube = _build(pd.DataFrame({"k": ["a", None], "v": [1.0, 2.0]}), dims)
    objs = [str(o) for o in _obs_objects(cube, "k")]
    assert sorted(objs) == sorted([str(UNDEFINED), "https://example.org/f/a"])


def test_concept_template_with_missing_other_column_is_undefined():
    dims = {"k": _dim("k", "URI", mapping={"type": "concept", "replacement-automated": "/c/{k}/{j}"}),
            "j": _dim("j", "string"),
            "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["a", "b"], "j": ["x", None], "v": [1.0, 2.0]})
    cube = _build(df, dims)
    objs = [str(o) for o in _obs_objects(cube, "k")]
    assert str(UNDEFINED) in objs and f"{CUBE_URI}/concept/c/a/x" in objs


# ---------------------------------------------------------------- (d) shape
def test_shape_unchanged_without_missing_values():
    dims = {"k": _dim("k", "string"), "u": _dim("u", "URI"),
            "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["a", "b"], "u": ["https://example.org/1", "https://example.org/2"], "v": [1.0, 2.0]})
    cube = _build(df, dims)
    ks = _prop_shape(cube, "k")
    assert (ks, SH.datatype, XSD.string) in cube._graph
    assert (ks, SH["or"], None) not in cube._graph
    assert UNDEFINED not in _list(cube, cube._graph.value(_prop_shape(cube, "u"), SH["in"]))
    assert Literal("", datatype=CUBE.Undefined) not in _list(cube, cube._graph.value(ks, SH["in"]))
    ok, text = _self_consistent(cube)
    assert ok, text


def test_min_max_ignore_missing_values_and_sit_in_real_datatype_branch():
    dims = {"k": _dim("k", "string"),
            "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["a", "b", "c"], "v": [1.0, np.nan, 3.0]})
    cube = _build(df, dims)
    assert Literal("", datatype=CUBE.Undefined) in _obs_objects(cube, "v")
    shape = _prop_shape(cube, "v")
    real, undefined = _list(cube, cube._graph.value(shape, SH["or"]))
    assert cube._graph.value(real, SH.datatype) == XSD.double
    assert float(cube._graph.value(real, SH.minInclusive)) == 1.0
    assert float(cube._graph.value(real, SH.maxInclusive)) == 3.0
    assert (shape, SH.minInclusive, None) not in cube._graph
    ok, text = _self_consistent(cube)
    assert ok, text


def test_sanitize_value_form_follows_dimension_kind():
    assert Cube._sanitize_value(None, "URI") == UNDEFINED
    assert Cube._sanitize_value(np.nan, "string") == Literal("", datatype=CUBE.Undefined)
    assert Cube._sanitize_value(pd.NA, "string", iri=True) == UNDEFINED


# ---------------------------------------------------------------- validation / duplicates / IRIs
def test_missing_key_values_are_no_longer_an_error():
    validate_dataframe(pd.DataFrame({"a": ["x", None], "b": [1, 2]}), ["a", "b"], ["a", "b"])


@pytest.mark.parametrize("a,b", [(None, np.nan), (None, pd.NA), (np.nan, pd.NA)])
def test_duplicates_treat_all_missing_markers_as_equal(a, b):
    df = pd.DataFrame({"k": pd.Series([a, b], dtype=object), "j": [1, 1]})
    with pytest.raises(InputValidationError) as ei:
        validate_dataframe(df, ["k", "j"], ["k", "j"])
    assert any("1 duplicate" in p for p in ei.value.problems)


def test_missing_and_real_value_are_not_duplicates():
    validate_dataframe(pd.DataFrame({"k": ["a", None]}), ["k"], ["k"])


def test_observation_iris_are_deterministic_and_unique():
    dims = {"k": _dim("k", "string"), "j": _dim("j", "string"),
            "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["a", None, None], "j": ["x", "x", None], "v": [1.0, 2.0, 3.0]})
    iris1 = list(_build(df, dims)._dataframe.index)
    iris2 = list(_build(df, dims)._dataframe.index)
    assert iris1 == iris2 and len(set(iris1)) == 3
    assert str(iris1[1]).endswith("/observation/undefined_x")


def test_missing_next_to_real_undefined_value_is_rejected_naming_the_column():
    dims = {"k": _dim("k", "string"),
            "v": _dim("v", "double", dim_type="Measure Dimension", scale="ratio")}
    df = pd.DataFrame({"k": ["undefined", None], "v": [1.0, 2.0]})
    with pytest.raises(InputValidationError) as ei:
        _build(df, dims)
    assert any("'k'" in p and "undefined" in p for p in ei.value.problems)
    # a real "undefined" without any missing value is fine
    _build(pd.DataFrame({"k": ["undefined", "x"], "v": [1.0, 2.0]}), dims)
