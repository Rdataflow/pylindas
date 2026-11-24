"""Unit tests for ``pycube.validation`` (Requirements 6.1 - 6.7)."""
import pandas as pd
import pytest

from pylindas.lindas.exceptions import InputValidationError
from pylindas.pycube.validation import dimension_columns, validate_dataframe


def _problems(df, expected, keys):
    with pytest.raises(InputValidationError) as ei:
        validate_dataframe(df, expected, keys)
    return ei.value.problems


def test_dimension_columns_order_and_unique():
    y = {"dimensions": {"b": {}, "a": {}}, "measures": {"a": {}, "m": {}}}
    assert dimension_columns(y) == ["b", "a", "m"]
    assert dimension_columns({}) == []


def test_valid_frame_passes_and_is_not_mutated():
    df = pd.DataFrame({"k": [1, 2], "v": [3.0, 4.0]})
    before = df.copy(deep=True)
    validate_dataframe(df, ["k", "v"], ["k"])
    pd.testing.assert_frame_equal(df, before)


def test_extra_and_missing_columns():
    p = _problems(pd.DataFrame({"k": [1], "x": [1]}), ["k", "v"], ["k"])
    assert any("'x'" in m and "Unexpected" in m for m in p)
    assert any("'v'" in m and "missing" in m for m in p)


def test_empty_frame():
    p = _problems(pd.DataFrame({"k": []}), ["k"], ["k"])
    assert any("no rows" in m for m in p)


def test_duplicates_limited_to_ten_with_total():
    df = pd.DataFrame({"k": [i // 2 for i in range(30)]})  # 15 duplicated combinations
    p = _problems(df, ["k"], ["k"])
    msg = next(m for m in p if "duplicate" in m)
    assert msg.startswith("15 duplicate")
    assert msg.count("(k=") == 10


def test_duplicate_check_skipped_for_missing_key_column():
    df = pd.DataFrame({"a": [1, 1]})
    p = _problems(df, ["a", "b"], ["a", "b"])
    assert not any("duplicate" in m for m in p)
    assert any("'b'" in m for m in p)


def test_all_problems_in_one_error():
    df = pd.DataFrame({"k": [1, 1, None], "x": [1, 2, 3]})
    p = _problems(df, ["k", "v"], ["k"])
    assert len(p) == 3  # extra, missing, duplicates (missing key values are valid, Req 6.5)
