"""Pure DataFrame validation for ``Cube`` inputs (Requirement 6). No network, no mutation."""
from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from pylindas.lindas.exceptions import InputValidationError

MAX_LISTED_DUPLICATES = 10

# Token standing for a missing key value in observation IRIs (Requirement 6.11).
# ``quote(..., safe="")`` leaves unreserved characters untouched, so a real value
# ``"undefined"`` would produce the same IRI part; ``validate_dataframe`` rejects that mix.
UNDEFINED_TOKEN = "undefined"


class _Missing:
    """Sentinel making None / NaN / pd.NA compare equal in the duplicate check."""

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def dimension_columns(cube_yaml: Mapping) -> list[str]:
    """Unique column names of the description, in order of first appearance.

    Walks ``dimensions`` first, then ``measures`` (if present). A description normally
    keeps measures inside ``dimensions`` (``dimension-type: Measure Dimension``); the
    ``measures`` walk is for descriptions that list them separately.
    """
    names: list[str] = []
    for section in ("dimensions", "measures"):
        entries = cube_yaml.get(section) or {}
        for name in entries:
            if name not in names:
                names.append(name)
    return names


def _fmt(values) -> str:
    return ", ".join(repr(str(v)) for v in values)


def validate_dataframe(df: pd.DataFrame, expected: Sequence[str], key_columns: Sequence[str]) -> None:
    """Check ``df`` against the expected columns and key columns.

    Collects all problems and raises a single ``InputValidationError``. ``df`` is only read.
    """
    problems: list[str] = []
    actual = set(df.columns)
    expected_set = set(expected)

    extra = sorted(actual - expected_set, key=str)
    if extra:
        problems.append(f"Unexpected columns not in the cube description: {_fmt(extra)}")
    missing = sorted(expected_set - actual, key=str)
    if missing:
        problems.append(f"Columns required by the cube description are missing: {_fmt(missing)}")

    if len(df) == 0:
        problems.append("The DataFrame has no rows.")

    present_keys = [k for k in dict.fromkeys(key_columns) if k in actual]

    # Missing key values are valid (written as cube:Undefined, Requirement 6.5) and count as
    # equal to each other for the duplicate check (Requirement 6.9).
    for col in present_keys:
        mask = df[col].isna()
        if mask.any() and (df.loc[~mask, col].astype(str) == UNDEFINED_TOKEN).any():
            problems.append(
                f"Key column {str(col)!r} mixes missing values with the real value {UNDEFINED_TOKEN!r}; "
                f"their observation IRIs would collide."
            )

    if present_keys and len(present_keys) == len(set(key_columns)) and len(df) > 0:
        keys_df = df[present_keys].astype(object)
        keys_df = keys_df.where(df[present_keys].notna(), _MISSING)
        dup_mask = keys_df.duplicated(keep=False)
        if dup_mask.any():
            combos = list(dict.fromkeys(map(tuple, keys_df.loc[dup_mask].itertuples(index=False, name=None))))
            listed = combos[:MAX_LISTED_DUPLICATES]
            shown = "; ".join(
                "(" + ", ".join(f"{c}={v!r}" for c, v in zip(present_keys, combo)) + ")" for combo in listed
            )
            problems.append(
                f"{len(combos)} duplicate key combination(s) over {_fmt(present_keys)}"
                f" (showing {len(listed)}): {shown}"
            )

    if problems:
        raise InputValidationError(problems)
