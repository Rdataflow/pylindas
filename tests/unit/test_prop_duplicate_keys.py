"""Property test: duplicate keys are reported with counts.

Design reference: lindas-publication, Property 13 (Requirement 6.4).
"""
import re

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pylindas.lindas.exceptions import InputValidationError
from pylindas.pycube.validation import MAX_LISTED_DUPLICATES, validate_dataframe

KEYS = ["a", "b"]
EXPECTED = ["a", "b", "v"]
_DUP = re.compile(r"duplicate key combination", re.IGNORECASE)
_TOTAL = re.compile(r"^(\d+) duplicate key combination")
_COMBO = re.compile(r"\(a=(-?\d+), b=(-?\d+)\)")


@st.composite
def frames(draw, with_duplicates=True):
    """Return (df, injected) where ``injected`` is the set of duplicated (a, b) combinations."""
    k = draw(st.integers(1, 25)) if with_duplicates else 0
    extra_unique = draw(st.integers(0 if with_duplicates else 1, 15))
    combos = draw(
        st.lists(
            st.tuples(st.integers(-1000, 1000), st.integers(-1000, 1000)),
            min_size=k + extra_unique,
            max_size=k + extra_unique,
            unique=True,
        )
    )
    injected = combos[:k]
    rows = list(combos)
    for combo in injected:
        # Some combinations occur more than twice, so row count != combination count.
        rows.extend([combo] * draw(st.integers(1, 3)))
    rows = draw(st.permutations(rows))
    df = pd.DataFrame({"a": [r[0] for r in rows], "b": [r[1] for r in rows], "v": range(len(rows))})
    return df, set(injected)


def _dup_problems(exc: InputValidationError) -> list[str]:
    return [p for p in exc.problems if _DUP.search(p)]


# Feature: lindas-publication, Property 13: Duplicate keys are reported with counts
# Validates: Requirements 6.4
@settings(max_examples=200, deadline=None)
@given(frames())
def test_duplicates_listed_and_counted(case):
    df, injected = case
    k = len(injected)
    with pytest.raises(InputValidationError) as info:
        validate_dataframe(df, EXPECTED, KEYS)
    problems = _dup_problems(info.value)
    assert len(problems) == 1
    problem = problems[0]

    # Requirement 6.4: the total is the number of duplicated COMBINATIONS, not rows.
    total = int(_TOTAL.match(problem).group(1))
    row_count = int(df.duplicated(subset=KEYS, keep=False).sum())
    assert total == k, f"reported {total}, expected {k} combinations (rows involved: {row_count})"

    listed = [(int(a), int(b)) for a, b in _COMBO.findall(problem)]
    assert len(listed) == min(k, MAX_LISTED_DUPLICATES)
    assert len(set(listed)) == len(listed)
    assert set(listed) <= injected


# Feature: lindas-publication, Property 13: Duplicate keys are reported with counts
# Validates: Requirements 6.4
@settings(max_examples=100, deadline=None)
@given(frames(with_duplicates=False))
def test_no_duplicates_no_problem(case):
    df, _ = case
    validate_dataframe(df, EXPECTED, KEYS)
