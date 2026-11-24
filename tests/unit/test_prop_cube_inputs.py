"""Property test: cube construction does not mutate its inputs and is repeatable.

Design reference: lindas-publication, Property 9 (Requirements 5.1, 5.2, 5.3).

Cubes are built offline (``local=True``) from the ``example/Cubes/mock`` description.
"""
import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from rdflib import Graph
from rdflib.compare import isomorphic
from pylindas.lindas.namespaces import SCHEMA

from pylindas.pycube.cube import Cube

_DESCRIPTION_PATH = (
    Path(__file__).resolve().parents[2] / "example" / "Cubes" / "mock" / "description.yml"
)
with open(_DESCRIPTION_PATH, encoding="utf-8") as _f:
    _DESCRIPTION = yaml.safe_load(_f)

_status_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc"))
    | st.sampled_from(['"', "\\", "\n", "ü", "é", "'"]),
    min_size=1,
    max_size=12,
)
_float = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def _dataframes(draw):
    """Unique (year, station) keys by construction; random values."""
    years = draw(st.lists(st.integers(1990, 2030), min_size=1, max_size=3, unique=True))
    stations = draw(
        st.lists(st.sampled_from(["Bern", "Zürich"]), min_size=1, max_size=2, unique=True)
    )
    keys = [(y, s) for y in years for s in stations]
    n = len(keys)
    return pd.DataFrame(
        {
            "Jahr": [k[0] for k in keys],
            "Station": [k[1] for k in keys],
            "Wert": draw(st.lists(_float, min_size=n, max_size=n)),
            "Standardfehler": draw(st.lists(_float, min_size=n, max_size=n)),
            "Status": draw(st.lists(_status_text, min_size=n, max_size=n)),
        }
    )


def _construct(df, cube_yaml, **kwargs):
    return Cube(dataframe=df, cube_yaml=cube_yaml, environment="TEST", local=True, **kwargs)


def _full_build(df, cube_yaml, **kwargs):
    return _construct(df, cube_yaml, **kwargs).prepare_data().write_cube().write_observations().write_shape()


# Known, accepted non-determinism: write_cube stamps schema:datePublished (http://schema.org/, the namespace pylindas uses) (today) and
# schema:dateModified (now, UTC). They are the only triples allowed to differ between two
# builds from the same inputs, so they are removed before the isomorphism check.
_TIMESTAMP_PREDICATES = {SCHEMA.datePublished, SCHEMA.dateModified}


def _without_timestamps(graph):
    stripped = Graph()
    for triple in graph:
        if triple[1] not in _TIMESTAMP_PREDICATES:
            stripped.add(triple)
    return stripped


def _broken_yamls():
    """Descriptions that make construction fail: a required key removed."""
    out = []
    for key in ("Name", "Description", "Publisher", "Creator", "Contact Point", "dimensions"):
        broken = copy.deepcopy(_DESCRIPTION)
        broken.pop(key, None)
        out.append(broken)
    return out


# Feature: lindas-publication, Property 9: Cube construction does not mutate inputs and is repeatable
# Validates: Requirements 5.1, 5.2, 5.3
@settings(max_examples=100, deadline=None)
@given(df=_dataframes())
def test_construction_does_not_mutate_and_is_repeatable(df):
    yaml_in = copy.deepcopy(_DESCRIPTION)
    yaml_snapshot = copy.deepcopy(yaml_in)
    df_snapshot = df.copy(deep=True)

    # Constructor only: inputs unchanged, two constructions isomorphic.
    c1 = _construct(df, yaml_in)
    c2 = _construct(df, yaml_in)
    assert yaml_in == yaml_snapshot
    pd.testing.assert_frame_equal(df, df_snapshot)
    assert isomorphic(c1._graph, c2._graph)

    # Full build: inputs unchanged, graphs isomorphic apart from the two timestamp predicates.
    b1 = _full_build(df, yaml_in)
    b2 = _full_build(df, yaml_in)
    assert yaml_in == yaml_snapshot
    pd.testing.assert_frame_equal(df, df_snapshot)

    g1, g2 = _without_timestamps(b1._graph), _without_timestamps(b2._graph)
    assert isomorphic(g1, g2)

    # Isomorphism after stripping shows the timestamp predicates are the ONLY differing ones;
    # additionally both builds must carry the same number of timestamp triples.
    for p in _TIMESTAMP_PREDICATES:
        assert len(list(b1._graph.triples((None, p, None)))) == len(
            list(b2._graph.triples((None, p, None)))
        )


# Feature: lindas-publication, Property 9: Cube construction does not mutate inputs and is repeatable
# Validates: Requirements 5.1, 5.2, 5.3
@settings(max_examples=100, deadline=None)
@given(df=_dataframes(), broken=st.sampled_from(_broken_yamls()))
def test_failing_construction_does_not_mutate_inputs(df, broken):
    yaml_in = copy.deepcopy(broken)
    yaml_snapshot = copy.deepcopy(broken)
    df_snapshot = df.copy(deep=True)

    with pytest.raises(Exception):
        _full_build(df, yaml_in, validate_description="strict")

    assert yaml_in == yaml_snapshot
    pd.testing.assert_frame_equal(df, df_snapshot)
