"""Unit tests for the Cube ``existence_check`` parameter (Requirements 5.5, 5.6, 5.7, 5.9).

The unit-suite conftest blocks sockets and already replaces ``query_lindas``
(the single place that issues the ASK request), so "no request" is asserted by
patching ``pylindas.pycube.cube.query_lindas`` and, as a second guard, the
upload HTTP verbs, which must never be called by the constructor.
"""
from unittest import mock

import pandas as pd
import pytest

from pylindas.lindas.exceptions import CubeAlreadyExistsError
from pylindas.pycube import Cube
from tests.unit.conftest import minimal_description

CUBE_IRI = "https://example.org/cube/c/1"
QUERY = "pylindas.pycube.cube.query_lindas"


def _yaml():
    dim = minimal_description()["dimensions"]["k"]
    return minimal_description(dimensions={"a": dim})


def _df():
    return pd.DataFrame({"a": [1, 2]})


@pytest.fixture
def writes():
    """Mocks of the upload HTTP verbs; must stay uncalled."""
    with mock.patch("pylindas.lindas.upload.requests.put") as put, \
            mock.patch("pylindas.lindas.upload.requests.post") as post:
        yield put, post


def test_public_existing_raises_naming_iri_and_no_write(writes):
    with mock.patch(QUERY, return_value=True) as q:
        with pytest.raises(CubeAlreadyExistsError) as e:
            Cube(_df(), _yaml(), "TEST", existence_check="public")
    assert e.value.uri == CUBE_IRI
    assert CUBE_IRI in str(e.value)
    q.assert_called_once()
    for verb in writes:
        verb.assert_not_called()


def test_public_absent_builds_and_ask_targets_exact_iri(writes):
    with mock.patch(QUERY, return_value=False) as q:
        cube = Cube(_df(), _yaml(), "TEST", existence_check="public")
    assert cube is not None
    q.assert_called_once()
    assert q.call_args.args[0] == f"ASK {{ <{CUBE_IRI}> ?p ?o}}"
    assert q.call_args.kwargs.get("environment") == "TEST"
    for verb in writes:
        verb.assert_not_called()


def test_none_non_local_builds_even_if_cube_exists_no_request():
    with mock.patch(QUERY, return_value=True) as q:
        cube = Cube(_df(), _yaml(), "TEST", local=False, existence_check="none")
    assert cube is not None
    q.assert_not_called()


def test_local_true_sends_no_request_even_with_public():
    with mock.patch(QUERY, return_value=True) as q:
        Cube(_df(), _yaml(), "TEST", local=True, existence_check="public")
    q.assert_not_called()
