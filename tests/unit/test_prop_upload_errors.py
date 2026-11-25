"""Property test: upload errors report status and body, never credentials.

``upload_graph`` is called against a mocked endpoint (``responses``; no network).
For HTTP status >= 400 the error must carry the status and the (masked) body;
for a timeout it must be an ``UploadTimeoutError``. In both cases neither the
username nor the password may appear in ``str``/``repr``/``args``/``body`` of
the error or of any exception in its ``__cause__``/``__context__`` chain.

Generator design (degenerate secrets):
* Secrets are alphanumeric with a minimum length of 6. Very short secrets
  (e.g. ``"a"``) are substrings of the fixed text of the message (endpoint,
  graph IRI, "failed with status"), so "never contains the secret" would be
  unsatisfiable for any implementation. Secrets that still happen to occur in
  the fixed text are rejected with ``assume``.
* Response bodies are built from non-alphanumeric noise and the secrets
  themselves, so masking is checkable exactly (each echoed secret becomes
  ``***``) and the noise cannot accidentally form a secret.
"""
import requests
import responses
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pylindas.lindas.exceptions import UploadError, UploadTimeoutError
from pylindas.lindas.upload import upload_graph

ENDPOINT = "https://lindas.test/repositories/lindas/rdf-graphs/service"
GRAPH_URI = "https://lindas.admin.ch/foen/cube"
TIMEOUT = 300

_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
secrets_ = st.text(alphabet=_ALNUM, min_size=6, max_size=24)
_noise = st.text(alphabet=" <>:;.,-_/\n", max_size=12)
# Each body part is either noise or an echoed secret (0 = username, 1 = password).
body_parts = st.lists(
    st.one_of(_noise, st.sampled_from([0, 1])), max_size=8
)
statuses = st.integers(min_value=400, max_value=599)


def _chain(exc: BaseException):
    """Yield exc and every exception reachable via __cause__/__context__."""
    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        stack.extend([e.__cause__, e.__context__])


def _texts(exc: BaseException) -> list:
    out = []
    for e in _chain(exc):
        out += [str(e), repr(e), *map(str, e.args)]
        out.append(getattr(e, "body", "") or "")
    return out


def _fixed_text(status: int) -> str:
    return " ".join(
        ["PUT", "POST", ENDPOINT, GRAPH_URI, "failed with status", str(status),
         "timed out after", str(TIMEOUT), "seconds", "***"]
    )


def _call(method: str):
    return upload_graph(
        "<a> <b> <c> .",
        endpoint=ENDPOINT,
        graph_uri=GRAPH_URI,
        auth=(_call.user, _call.pw),
        method=method,
        timeout=TIMEOUT,
    )


def _assert_no_secrets(exc: BaseException, user: str, pw: str):
    for text in _texts(exc):
        assert user not in text
        assert pw not in text


# Feature: lindas-publication, Property 4: Upload errors report status and body but never credentials
# **Validates: Requirements 1.6, 1.7**
@settings(max_examples=100, deadline=None)
@given(
    user=secrets_,
    pw=secrets_,
    parts=body_parts,
    status=statuses,
    method=st.sampled_from(["PUT", "POST"]),
)
def test_http_error_reports_status_and_masked_body_without_credentials(
    user, pw, parts, status, method
):
    fixed = _fixed_text(status)
    assume(user not in fixed and pw not in fixed)

    secrets = (user, pw)
    raw_body = "".join(secrets[p] if isinstance(p, int) else p for p in parts)
    masked_body = "".join("***" if isinstance(p, int) else p for p in parts)

    _call.user, _call.pw = user, pw
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        rsps.add(
            responses.PUT if method == "PUT" else responses.POST,
            ENDPOINT,
            status=status,
            body=raw_body,
        )
        try:
            _call(method)
        except UploadError as err:
            error = err
        else:  # pragma: no cover
            raise AssertionError("UploadError not raised")

    assert not isinstance(error, UploadTimeoutError)
    # Status and masked body are reported.
    assert error.status == status
    assert error.body == masked_body
    assert str(status) in str(error)
    assert masked_body in str(error)
    # No credentials anywhere, including the cause/context chain.
    _assert_no_secrets(error, user, pw)


# Feature: lindas-publication, Property 4: Upload errors report status and body but never credentials
# **Validates: Requirements 1.6, 1.7**
@settings(max_examples=100, deadline=None)
@given(
    user=secrets_,
    pw=secrets_,
    leak=st.booleans(),
    method=st.sampled_from(["PUT", "POST"]),
)
def test_timeout_error_never_contains_credentials(user, pw, leak, method):
    fixed = _fixed_text(0)
    assume(user not in fixed and pw not in fixed)

    # ``leak=True`` is the worst case: the underlying transport error itself
    # echoes the credentials (e.g. a URL with userinfo in its message).
    cause = (
        requests.Timeout(f"read timed out for https://{user}:{pw}@lindas.test/")
        if leak
        else requests.Timeout("read timed out")
    )

    _call.user, _call.pw = user, pw
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        rsps.add(
            responses.PUT if method == "PUT" else responses.POST,
            ENDPOINT,
            body=cause,
        )
        try:
            _call(method)
        except UploadError as err:
            error = err
        else:  # pragma: no cover
            raise AssertionError("UploadTimeoutError not raised")

    assert isinstance(error, UploadTimeoutError)
    assert "timed out" in str(error)
    _assert_no_secrets(error, user, pw)
