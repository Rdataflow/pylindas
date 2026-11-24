import copy
from unittest import mock

import pandas as pd
import pytest

from pylindas.lindas.exceptions import CubeAlreadyExistsError, ExistenceCheckError
from pylindas.pycube import Cube
from tests.unit.conftest import minimal_description

QUERY = "pylindas.pycube.cube.query_lindas"


def _yaml():
    dim = minimal_description()["dimensions"]["k"]
    # schema-allowed free-form nested data (mapping.replacements is an unconstrained object)
    dim["mapping"] = {"type": "replace", "replacements": {"nested": [1, {"x": [2]}]}}
    return minimal_description(dimensions={"a": dim})


def _df():
    return pd.DataFrame({"a": [1, 2]})


def test_deepcopy_not_shared_and_unchanged():
    y = _yaml()
    before = copy.deepcopy(y)
    cube = Cube(_df(), y, "TEST", local=True)
    assert y == before
    assert cube._cube_dict is not y
    assert "dimensions" in y  # the constructor pops it from its own copy only
    assert cube._shape_dict["a"]["mapping"]["replacements"]["nested"][1]["x"] is not y["dimensions"]["a"]["mapping"]["replacements"]["nested"][1]["x"]
    cube._shape_dict["a"]["mapping"]["replacements"]["nested"][1]["x"].append(3)
    assert y == before


def test_yaml_unchanged_on_error():
    y = _yaml()
    before = copy.deepcopy(y)
    with mock.patch(QUERY, return_value=True):
        with pytest.raises(CubeAlreadyExistsError):
            Cube(_df(), y, "TEST")
    assert y == before


def test_dataframe_not_mutated():
    df = _df()
    before = df.copy()
    cube = Cube(df, _yaml(), "TEST", local=True)
    cube._dataframe["z"] = 1
    pd.testing.assert_frame_equal(df, before)


def test_public_is_default_and_asks():
    with mock.patch(QUERY, return_value=False) as q:
        Cube(_df(), _yaml(), "TEST")
    q.assert_called_once()
    assert "https://example.org/cube/c/1" in q.call_args.args[0]


def test_public_existing_raises():
    with mock.patch(QUERY, return_value=True):
        with pytest.raises(CubeAlreadyExistsError):
            Cube(_df(), _yaml(), "TEST", existence_check="public")


def test_none_sends_no_request_even_if_exists():
    with mock.patch(QUERY, return_value=True) as q:
        Cube(_df(), _yaml(), "TEST", existence_check="none")
    q.assert_not_called()


def test_local_sends_no_request():
    with mock.patch(QUERY) as q:
        Cube(_df(), _yaml(), "TEST", local=True)
    q.assert_not_called()


@pytest.mark.parametrize("bad", ["PUBLIC", "", "private", None, True])
def test_invalid_value_raises_before_request(bad):
    y = _yaml()
    before = copy.deepcopy(y)
    with mock.patch(QUERY) as q:
        with pytest.raises(ExistenceCheckError) as e:
            Cube(_df(), y, "TEST", existence_check=bad)
    q.assert_not_called()
    assert "existence_check" in str(e.value)
    assert '"public"' in str(e.value) and '"none"' in str(e.value)
    assert y == before


def test_validate_description_stored():
    assert Cube(_df(), _yaml(), "TEST", local=True, validate_description=False)._validate_description is False
