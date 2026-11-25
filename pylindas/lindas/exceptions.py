"""Custom exception hierarchy for already-existing LINDAS resources.

These replace ``sys.exit()`` calls in the non-local existence checks so that
library consumers receive catchable, non-terminating exceptions instead of a
terminated host process. Each exception carries the constructed URI (via
``.uri`` and the message) so callers can distinguish them from unrelated
failures.
"""


from dataclasses import dataclass


class ResourceExistsError(Exception):
    """Base: a resource already exists at the target URI in non-local mode."""

    def __init__(self, uri: str):
        self.uri = str(uri)
        super().__init__(
            f"Resource already exists at <{self.uri}>. Please update your YAML."
        )


class CubeAlreadyExistsError(ResourceExistsError):
    """Raised when a cube already exists at the constructed cube URI (local=False)."""

    def __init__(self, uri: str):
        self.uri = str(uri)
        super(ResourceExistsError, self).__init__(
            f"Cube already exists at <{self.uri}>. Please update your YAML."
        )


class SharedDimensionAlreadyExistsError(ResourceExistsError):
    """Raised when a shared dimension already exists at the constructed SD URI (local=False)."""

    def __init__(self, uri: str):
        self.uri = str(uri)
        super(ResourceExistsError, self).__init__(
            f"Shared Dimension already exists at <{self.uri}>. Please update your YAML."
        )


# ---------------------------------------------------------------------------
# Pylindas error hierarchy (lindas-publication, Part A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One schema violation, e.g. path ``dimensions[2].mapping.type``."""

    path: str
    message: str


class PylindasError(Exception):
    """Common base for all new Pylindas errors."""


class UploadValidationError(PylindasError, ValueError):
    """Invalid arguments to ``upload_graph``; no request was sent."""


class UploadError(PylindasError):
    """The server answered with an error status (>= 400)."""

    def __init__(self, message: str, status: int = 0, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class UploadTimeoutError(UploadError):
    """The upload request timed out."""


class HierarchyError(PylindasError, ValueError):
    """Invalid hierarchy (unknown parents, self parents, cycles)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details if details is not None else {}


class InputValidationError(PylindasError, ValueError):
    """Input DataFrame problems; all problems are collected in ``problems``."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems) or "Input validation failed")


class DescriptionValidationError(PylindasError, ValueError):
    """Cube description violates the schema; see ``violations``."""

    def __init__(self, violations: list[Violation]):
        self.violations = list(violations)
        super().__init__(
            "; ".join(f"{v.path}: {v.message}" for v in self.violations)
            or "Description validation failed"
        )


class ExistenceCheckError(PylindasError, ValueError):
    """Invalid ``existence_check`` mode."""
