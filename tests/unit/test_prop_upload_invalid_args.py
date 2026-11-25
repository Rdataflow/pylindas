"""Property test: invalid upload arguments send no request.

**Validates: Requirements 1.8**
"""
import pytest
import responses
from hypothesis import given, settings, strategies as st
from rdflib import Graph

from pylindas.lindas.exceptions import UploadValidationError
from pylindas.lindas.upload import upload_graph

# --- Valid and invalid value generators per argument -------------------------
_nonempty = st.text(min_size=1).filter(lambda s: s.strip())
_blank = st.sampled_from(["", " ", "\t", "\n  "])

valid_data = st.one_of(st.binary(), st.text(), st.just(Graph()))
invalid_data = st.one_of(
    st.none(), st.integers(), st.floats(), st.lists(st.text()), st.dictionaries(st.text(), st.text())
)

valid_endpoint = _nonempty
invalid_endpoint = st.one_of(_blank, st.none(), st.integers(), st.binary())

valid_graph = _nonempty
invalid_graph = st.one_of(_blank.filter(lambda s: not s.strip()), st.none(), st.integers())

valid_auth = st.tuples(st.text(min_size=1), st.text(min_size=1))
invalid_auth = st.one_of(
    st.none(),
    st.text(),
    st.tuples(),
    st.tuples(st.text(min_size=1)),
    st.tuples(st.text(min_size=1), st.text(min_size=1), st.text(min_size=1)),
    st.tuples(st.just(""), st.text(min_size=1)),
    st.tuples(st.text(min_size=1), st.just("")),
    st.tuples(st.integers(), st.text(min_size=1)),
    st.tuples(st.text(min_size=1), st.none()),
)

valid_method = st.sampled_from(["PUT", "POST"])
invalid_method = st.one_of(
    st.sampled_from(["GET", "DELETE", "put", "post", "", "PATCH"]),
    st.none(),
    st.integers(),
).filter(lambda m: m not in ("PUT", "POST"))

valid_timeout = st.one_of(
    st.integers(min_value=1, max_value=10_000),
    st.floats(min_value=0.001, max_value=10_000, allow_nan=False),
)
invalid_timeout = st.one_of(
    st.integers(max_value=0),
    st.floats(max_value=0, allow_nan=False),
    st.just(float("nan")),
    st.booleans(),
    st.none(),
    st.text(),
)

_ARGS = ("data", "endpoint", "graph_uri", "auth", "method", "timeout")
_VALID = dict(
    data=valid_data, endpoint=valid_endpoint, graph_uri=valid_graph,
    auth=valid_auth, method=valid_method, timeout=valid_timeout,
)
_INVALID = dict(
    data=invalid_data, endpoint=invalid_endpoint, graph_uri=invalid_graph,
    auth=invalid_auth, method=invalid_method, timeout=invalid_timeout,
)


@st.composite
def argument_sets(draw):
    """Argument dict with at least one invalid argument; returns (args, bad names)."""
    bad = draw(st.sets(st.sampled_from(_ARGS), min_size=1))
    args = {n: draw(_INVALID[n] if n in bad else _VALID[n]) for n in _ARGS}
    return args, bad


# Feature: lindas-publication, Property 5: Invalid upload arguments send no request
# Validates: Requirements 1.8
@settings(max_examples=100)
@given(argument_sets())
def test_invalid_upload_arguments_send_no_request(case):
    args, bad = case
    data = args.pop("data")
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        with pytest.raises(UploadValidationError) as excinfo:
            upload_graph(data, **args)
        assert len(rsps.calls) == 0
    # every invalid argument is reported (collected, not fail-fast)
    message = str(excinfo.value)
    for name in bad:
        assert name in message
