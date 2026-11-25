"""Property test: ``Cube.serialize`` is deterministic.

Design reference: lindas-publication, Property 2 (Requirement 2.3).

Cubes are built offline (``local=True``) from the ``example/Cubes/mock``
description. Frames have unique (Jahr, Station) keys by construction, so no
``assume``/``filter`` is needed.
"""
import copy
from pathlib import Path

import pandas as pd
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from rdflib.compare import isomorphic, to_isomorphic

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


def _build_cube(dataframe: pd.DataFrame) -> Cube:
    cube = Cube(
        dataframe=dataframe,
        cube_yaml=copy.deepcopy(_DESCRIPTION),
        environment="TEST",
        local=True,
    )
    return cube.prepare_data().write_cube().write_observations().write_shape()


# Feature: lindas-publication, Property 2: Serialization is deterministic
# Validates: Requirements 2.3
@settings(max_examples=100)
@given(df=_dataframes())
def test_serialize_is_deterministic(df):
    cube = _build_cube(df)

    before = to_isomorphic(cube._graph)
    length_before = len(cube._graph)

    first = cube.serialize()
    second = cube.serialize()

    assert first == second

    # Serializing must not mutate the graph.
    assert len(cube._graph) == length_before
    assert isomorphic(before, cube._graph)
