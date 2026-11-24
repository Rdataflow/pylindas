"""Property test: all input validation failures are reported together.

Design reference: lindas-publication, Property 15 (Requirement 6.7).

Missing key values are no longer a failure kind (they become cube:Undefined); frames
containing them must not add a problem..
"""
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pylindas.lindas.exceptions import InputValidationError
from pylindas.pycube.validation import validate_dataframe

KEYS = ["a", "b"]
EXPECTED = ["a", "b", "v"]

_KIND_PREFIXES = {
    "extra": "Unexpected columns not in the cube description",
    "missing_cols": "Columns required by the cube description are missing",
    "empty": "The DataFrame has no rows",
    "duplicates": None,  # matched by substring below
}


def _kinds(problems: list[str]) -> set[str]:
    found = set()
    for p in problems:
        if p.startswith(_KIND_PREFIXES["extra"]):
            found.add("extra")
        elif p.startswith(_KIND_PREFIXES["missing_cols"]):
            found.add("missing_cols")
        elif p.startswith(_KIND_PREFIXES["empty"]):
            found.add("empty")
        elif "duplicate key combination" in p:
            found.add("duplicates")
        else:
            found.add(f"unknown:{p}")
    return found


@st.composite
def cases(draw):
    """Return (df, expected_kinds); each failure kind is switched on independently."""
    n = draw(st.integers(1, 12))
    # ``a`` is distinct per row, so rows are unique even after nulling at most one value
    # per key column; duplicates are then only the ones injected below.
    a = list(range(n))
    b = draw(st.lists(st.integers(-5, 5), min_size=n, max_size=n))
    df = pd.DataFrame({"a": a, "b": b, "v": range(n)})

    null_a = draw(st.one_of(st.none(), st.integers(0, n - 1)))
    null_b = draw(st.one_of(st.none(), st.integers(0, n - 1)))
    if null_a is not None:
        df["a"] = df["a"].astype(float)
        df.loc[null_a, "a"] = float("nan")
    if null_b is not None:
        df["b"] = df["b"].astype(float)
        df.loc[null_b, "b"] = float("nan")

    dup = draw(st.booleans())
    if dup:
        src = draw(st.integers(0, n - 1))
        df = pd.concat([df, df.iloc[[src] * draw(st.integers(1, 3))]], ignore_index=True)

    dropped = draw(st.lists(st.sampled_from(EXPECTED), unique=True, max_size=3))
    df = df.drop(columns=dropped)
    if draw(st.booleans()):
        df["zzz"] = 0
        extra = True
    else:
        extra = False

    empty = draw(st.booleans())
    if empty:
        df = df.iloc[0:0]

    present_keys = [k for k in KEYS if k not in dropped]
    kinds = set()
    if extra:
        kinds.add("extra")
    if dropped:
        kinds.add("missing_cols")
    if empty:
        kinds.add("empty")
    if not empty:
        if dup and len(present_keys) == len(KEYS):
            kinds.add("duplicates")
    return df, kinds


# Feature: lindas-publication, Property 15: All input validation failures are reported together
# Validates: Requirements 6.7
@settings(max_examples=300, deadline=None)
@given(cases())
def test_all_failures_reported_together(case):
    df, expected_kinds = case
    if not expected_kinds:
        validate_dataframe(df, EXPECTED, KEYS)
        return
    with pytest.raises(InputValidationError) as info:
        validate_dataframe(df, EXPECTED, KEYS)
    reported = _kinds(info.value.problems)
    assert reported == expected_kinds, f"reported {reported}, expected {expected_kinds}"
