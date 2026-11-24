"""Property test: an invalid ``existence_check`` is rejected before any request.

**Validates: Requirements 5.8**
"""
import copy
from unittest import mock

import pandas as pd
import pytest
import responses
from hypothesis import given, strategies as st

from pylindas.lindas.exceptions import ExistenceCheckError
from pylindas.pycube import Cube

QUERY = "pylindas.pycube.cube.query_lindas"
ACCEPTED = ("public", "none")


def _yaml():
    return {
        "Base-URI": "https://example.org/cube/",
        "Identifier": "c",
        "Version": 1,
        "Name": {"en": "n"},
        "Creator": [{"IRI": "https://example.org/c"}],
        "dimensions": {"a": {"dimension-type": "Key Dimension", "nested": [1, {"x": [2]}]}},
    }


# Arbitrary strings other than exactly "public"/"none" (near-misses are covered explicitly),
# plus non-string values.
invalid_strings = st.one_of(
    st.text(),
    st.sampled_from(["PUBLIC", "None", "NONE", " public", "none ", "", "private", "publi", "nonee"]),
).filter(lambda s: s not in ACCEPTED)
non_strings = st.one_of(
    st.none(), st.booleans(), st.integers(), st.floats(), st.binary(),
    st.lists(st.text(), max_size=3), st.tuples(st.text()),
)
invalid_modes = st.one_of(invalid_strings, non_strings)


# Feature: lindas-publication, Property 10: Invalid existence_check is rejected before any request
@given(bad=invalid_modes, local=st.booleans())
def test_invalid_existence_check_rejected_before_any_request(bad, local):
    y = _yaml()
    before = copy.deepcopy(y)
    with mock.patch(QUERY) as q, responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        with pytest.raises(ExistenceCheckError) as e:
            Cube(pd.DataFrame({"a": [1, 2]}), y, "TEST", local=local, existence_check=bad)
        assert len(rsps.calls) == 0
    q.assert_not_called()
    msg = str(e.value)
    assert "existence_check" in msg
    assert '"public"' in msg and '"none"' in msg
    assert y == before
