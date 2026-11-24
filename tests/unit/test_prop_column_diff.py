"""Property 12: column set differences are detected by ``validate_dataframe``."""
import re

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from pandas.testing import assert_frame_equal

from pylindas.lindas.exceptions import InputValidationError
from pylindas.pycube.validation import validate_dataframe

NAMES = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=8)
EXTRA_PREFIX = "Unexpected columns"
MISSING_PREFIX = "Columns required"


def _names(problems, prefix):
    found = [p for p in problems if p.startswith(prefix)]
    assert len(found) <= 1
    return set(re.findall(r"'([^']*)'", found[0])) if found else set()


@st.composite
def scenario(draw):
    pool = draw(st.lists(NAMES, min_size=1, max_size=12, unique=True))
    expected = draw(st.lists(st.sampled_from(pool), min_size=1, unique=True))
    present = draw(st.lists(st.sampled_from(expected), unique=True))
    extras = draw(st.lists(st.sampled_from(pool), unique=True).map(lambda x: [c for c in x if c not in expected]))
    cols = draw(st.permutations(present + extras))
    return expected, list(cols)


# Feature: lindas-publication, Property 12: Column set differences are detected
# **Validates: Requirements 6.2, 6.3**
@settings(max_examples=200, deadline=None)
@given(scenario())
def test_column_set_differences_detected(sc):
    expected, cols = sc
    df = pd.DataFrame({c: [1, 2, 3] if i else [10, 20, 30] for i, c in enumerate(cols)}, columns=cols)
    before = df.copy(deep=True)
    key = expected[0]

    extra = set(cols) - set(expected)
    missing = set(expected) - set(cols)

    if not extra and not missing:
        validate_dataframe(df, expected, [key])  # must not raise
    else:
        with pytest.raises(InputValidationError) as ei:
            validate_dataframe(df, expected, [key])
        problems = ei.value.problems
        assert _names(problems, EXTRA_PREFIX) == extra
        assert _names(problems, MISSING_PREFIX) == missing
        assert (not extra) == (not any(p.startswith(EXTRA_PREFIX) for p in problems))
        assert (not missing) == (not any(p.startswith(MISSING_PREFIX) for p in problems))

    assert_frame_equal(df, before)
