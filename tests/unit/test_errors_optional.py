"""Unit tests for the new error types and require_extra."""
import dataclasses

import pytest

from pylindas._optional import require_extra
from pylindas.lindas.exceptions import (
    CubeAlreadyExistsError, DescriptionValidationError, ExistenceCheckError,
    HierarchyError, InputValidationError, PylindasError, UploadError,
    UploadTimeoutError, UploadValidationError, Violation,
)


def test_hierarchy_and_value_error_bases():
    for cls in (UploadValidationError, HierarchyError, InputValidationError,
                DescriptionValidationError, ExistenceCheckError):
        assert issubclass(cls, PylindasError) and issubclass(cls, ValueError)
    assert issubclass(UploadTimeoutError, UploadError)
    assert not issubclass(UploadError, ValueError)


def test_upload_error_attrs():
    e = UploadError("boom", status=500, body="x")
    assert (e.status, e.body) == (500, "x")


def test_details_problems_violations():
    assert HierarchyError("h", {"cycles": [["a"]]}).details == {"cycles": [["a"]]}
    assert InputValidationError(["a", "b"]).problems == ["a", "b"]
    v = Violation("dimensions[2].mapping.type", "bad")
    assert DescriptionValidationError([v]).violations == [v]
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.path = "x"


def test_cube_already_exists_unchanged():
    assert "Cube already exists at <u>" in str(CubeAlreadyExistsError("u"))


def test_require_extra():
    assert require_extra("json", "x").dumps
    with pytest.raises(ImportError, match=r"pylindas\[fetch\]"):
        require_extra("no_such_module_xyz", "fetch")
