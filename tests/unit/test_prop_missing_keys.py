"""Property 14: missing key values become ``cube:Undefined`` (Requirements 6.5, 6.8 - 6.11)."""
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from rdflib import Literal, URIRef
from rdflib.namespace import SH, XSD

from pylindas.lindas.exceptions import InputValidationError
from tests.unit.test_undefined import (
    CUBE,
    UNDEFINED,
    _build,
    _dim,
    _list,
    _obs_objects,
    _prop_shape,
    _self_consistent,
)

UNDEFINED_LITERAL = Literal("", datatype=CUBE.Undefined)
POOL = [f"v{i}" for i in range(6)]
IRI_POOL = [f"https://example.org/x/{i}" for i in range(6)]
MARKERS = [None, float("nan"), pd.NA]

# name -> (dimension kwargs, value pool, is_iri_dimension)
KEY_KINDS = {
    "lit": (dict(datatype="string"), POOL, False),
    "uri": (dict(datatype="URI"), IRI_POOL, True),
    "add": (dict(datatype="URI", mapping={"type": "additive", "base": "https://example.org/base/"}), POOL, True),
    "rep": (dict(datatype="URI", mapping={"type": "replace",
                                          "replacements": {v: f"https://example.org/r/{v}" for v in POOL}}),
            POOL, True),
    "con": (dict(datatype="URI", mapping={"type": "concept", "replacement-automated": "/c/{con}"}), POOL, True),
}


@st.composite
def scenario(draw):
    n = draw(st.integers(1, 6))
    names = draw(st.lists(st.sampled_from(sorted(KEY_KINDS)), min_size=1, max_size=5, unique=True))
    data = {}
    for name in names:
        pool = KEY_KINDS[name][1]
        mask = draw(st.lists(st.booleans(), min_size=n, max_size=n))
        vals = [draw(st.sampled_from(MARKERS)) if m else draw(st.sampled_from(pool)) for m in mask]
        if draw(st.booleans()):
            # string dtype only holds pd.NA as marker
            series = pd.Series([pd.NA if m else v for v, m in zip(vals, mask)], dtype="string")
        else:
            series = pd.Series(vals, dtype=object)
        data[name] = series
    data["v"] = [float(i) for i in range(n)]
    df = pd.DataFrame(data)
    if n > 1 and draw(st.booleans()):  # sometimes inject a duplicate row
        df.iloc[0] = df.iloc[draw(st.integers(1, n - 1))]
    return df, names


def _all_missing_equal_keys(df, names):
    rows = [tuple("<missing>" if pd.isna(x) else x for x in r) for r in df[names].itertuples(index=False)]
    return len(set(rows)) != len(rows)


# Feature: lindas-publication, Property 14: Missing key values become cube:Undefined
# **Validates: Requirements 6.5, 6.8, 6.9, 6.10, 6.11**
@settings(max_examples=150, deadline=None)
@given(scenario())
def test_missing_key_values_become_undefined(sc):
    df, names = sc
    dims = {n: _dim(n, **KEY_KINDS[n][0]) for n in names}
    dims["v"] = _dim("v", "double", dim_type="Measure Dimension", scale="ratio")
    before = df.copy(deep=True)

    if _all_missing_equal_keys(df, names):
        with pytest.raises(InputValidationError) as ei:
            _build(df, dims)
        assert any("duplicate" in p for p in ei.value.problems)
        return

    cube = _build(df, dims)
    assert df.equals(before) or df.astype(object).fillna("<m>").equals(before.astype(object).fillna("<m>"))

    for name in names:
        is_iri = KEY_KINDS[name][2]
        n_missing = int(df[name].isna().sum())
        objs = _obs_objects(cube, name)
        marker = UNDEFINED if is_iri else UNDEFINED_LITERAL
        assert objs.count(marker) == n_missing
        wrong = UNDEFINED_LITERAL if is_iri else UNDEFINED
        assert wrong not in objs

        shape = _prop_shape(cube, name)
        in_list = _list(cube, cube._graph.value(shape, SH["in"])) if (shape, SH["in"], None) in cube._graph else []
        if is_iri:
            assert (UNDEFINED in in_list) == (n_missing > 0)
        else:
            assert (UNDEFINED_LITERAL in in_list) == (n_missing > 0)
            if n_missing:
                branches = _list(cube, cube._graph.value(shape, SH["or"]))
                assert CUBE.Undefined in {cube._graph.value(b, SH.datatype) for b in branches}
            else:
                assert (shape, SH["or"], None) not in cube._graph
                assert (shape, SH.datatype, XSD.string) in cube._graph

    # No IRI built from a missing value
    for term in set(cube._graph.subjects()) | set(cube._graph.objects()):
        if isinstance(term, URIRef):
            assert not any(bad in str(term) for bad in ("nan", "None", "<NA>")), term

    ok, text = _self_consistent(cube)
    assert ok, text
