"""Property test: ``dimension_columns`` is ordered and duplicate-free.

Design reference: lindas-publication, Property 11 (Requirement 6.1).
"""
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from pylindas.pycube.cube import Cube
from pylindas.pycube.validation import dimension_columns
from tests.unit.conftest import minimal_description, minimal_dimensions

_name = st.text(alphabet="abcdefgh", min_size=1, max_size=2)  # small alphabet => overlaps
_names = st.lists(_name, max_size=8)


def _dimension_entry(is_measure):
    if is_measure:
        return {"name": {"en": "M"}, "description": {"en": "M"}, "path": "m",
                "dimension-type": "Measure Dimension", "scale-type": "ratio"}
    return minimal_dimensions()["k"]


@st.composite
def _descriptions(draw):
    """Description with ordered ``dimensions`` and an optional separate ``measures``."""
    dim_names = list(dict.fromkeys(draw(_names)))  # dict keys are unique by nature
    dims = {n: _dimension_entry(draw(st.booleans())) for n in dim_names}
    desc = {"dimensions": dims}
    if draw(st.booleans()):
        # may overlap with dimensions
        desc["measures"] = {n: _dimension_entry(True) for n in dict.fromkeys(draw(_names))}
    return desc


def _expected_first_appearance(desc):
    seen = []
    for section in ("dimensions", "measures"):
        for n in desc.get(section, {}):
            if n not in seen:
                seen.append(n)
    return seen


# Feature: lindas-publication, Property 11: dimension_columns is ordered and duplicate-free
# Validates: Requirements 6.1
@settings(max_examples=100, deadline=None)
@given(desc=_descriptions())
def test_dimension_columns_ordered_and_duplicate_free(desc):
    result = dimension_columns(desc)

    assert len(result) == len(set(result))  # each name exactly once
    all_names = set(desc["dimensions"]) | set(desc.get("measures", {}))
    assert set(result) == all_names
    assert result == _expected_first_appearance(desc)
    # dimensions come first, then measures-only names
    n_dims = len(desc["dimensions"])
    assert result[:n_dims] == list(desc["dimensions"])


# Feature: lindas-publication, Property 11: dimension_columns is ordered and duplicate-free
# Validates: Requirements 6.1
@settings(max_examples=100, deadline=None)
@given(desc=_descriptions().filter(lambda d: d["dimensions"] and any(
    v["dimension-type"] == "Key Dimension" for v in d["dimensions"].values())))
def test_cube_method_matches_pure_function(desc):
    # ``measures`` is not a block Cube builds from; wrap only the dimensions into a valid description.
    desc = minimal_description(**{"Base-URI": "https://example.org/cube", "dimensions": desc["dimensions"]})
    cols = list(desc["dimensions"])
    frame = pd.DataFrame({c: ["a"] for c in cols})

    cube = Cube(frame, desc, environment="TEST", local=True)

    assert cube.dimension_columns() == dimension_columns(desc) == cols
