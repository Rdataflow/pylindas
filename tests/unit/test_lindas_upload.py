"""Mock-based unit tests for ``pylindas.lindas.upload.upload_ttl``.

Design reference: Testing Strategy — Network isolation (Requirements 18.2,
18.4). These tests exercise the *real* ``upload_ttl`` flow while mocking every
part that would otherwise touch a live GraphDB endpoint or the filesystem:

* ``_load_config`` is patched so no ``lindas.ini`` file is required.
* ``requests.post`` / ``requests.delete`` are patched so no HTTP request is
  sent; each returns a mock response whose ``raise_for_status`` is a no-op.
* ``builtins.open`` is patched so reading the Turtle files never touches disk.

LINDAS runs on GraphDB, which implements the SPARQL 1.1 Graph Store HTTP
Protocol: appending a graph is a ``POST`` with ``Content-Type: text/turtle``
and clearing a graph is a ``DELETE``.

The ``tests/unit`` autouse ``_no_network`` fixture additionally blocks real
sockets, so an accidental outbound connection becomes an immediate failure.
"""

from unittest.mock import MagicMock, mock_open, patch

from pylindas.lindas import upload
from pylindas.lindas.exceptions import UploadValidationError


_DUMMY_CONFIG = {
    "endpoint": "https://example.test/rdf-graphs/service",
    "username": "u",
    "password": "p",
}


def _ok_response():
    """A mock HTTP response whose ``raise_for_status`` does nothing."""
    resp = MagicMock(name="response")
    resp.raise_for_status.return_value = None
    resp.status_code = 200
    resp.text = ""
    return resp


def test_upload_ttl_single_file_posts_once_no_delete():
    """A single-file upload POSTs the Turtle once and does not clear."""
    mock_post = MagicMock(name="post", return_value=_ok_response())
    mock_delete = MagicMock(name="delete", return_value=_ok_response())

    with patch.object(upload, "_load_config", return_value=_DUMMY_CONFIG), patch(
        "pylindas.lindas.upload.requests.request", mock_post
    ), patch("pylindas.lindas.upload.requests.delete", mock_delete), patch(
        "builtins.open", mock_open(read_data=b"<a> <b> <c> .")
    ):
        upload.upload_ttl(
            filename="some.ttl",
            db_file="ignored.ini",
            environment="TEST",
            graph_uri="urn:graph",
            clear_graph=False,
        )

    mock_post.assert_called_once()
    mock_delete.assert_not_called()

    # POST targets the configured endpoint, the right named graph, as text/turtle.
    args, kwargs = mock_post.call_args
    assert args[0] == "POST"
    assert args[1] == _DUMMY_CONFIG["endpoint"]
    assert kwargs["params"] == {"graph": "urn:graph"}
    assert kwargs["headers"]["Content-Type"] == "text/turtle"
    assert kwargs["auth"] is not None


def test_upload_ttl_clear_graph_deletes_then_posts():
    """With ``clear_graph=True`` the named graph is DELETEd before the POST."""
    mock_post = MagicMock(name="post", return_value=_ok_response())
    mock_delete = MagicMock(name="delete", return_value=_ok_response())

    with patch.object(upload, "_load_config", return_value=_DUMMY_CONFIG), patch(
        "pylindas.lindas.upload.requests.request", mock_post
    ), patch("pylindas.lindas.upload.requests.delete", mock_delete), patch(
        "builtins.open", mock_open(read_data=b"<a> <b> <c> .")
    ):
        upload.upload_ttl(
            filename="some.ttl",
            db_file="ignored.ini",
            environment="TEST",
            graph_uri="urn:graph",
            clear_graph=True,
        )

    mock_delete.assert_called_once()
    _, del_kwargs = mock_delete.call_args
    assert del_kwargs["params"] == {"graph": "urn:graph"}
    mock_post.assert_called_once()


def test_upload_ttl_list_of_files_posts_each():
    """A list of files results in one POST per file and no clear."""
    mock_post = MagicMock(name="post", return_value=_ok_response())
    mock_delete = MagicMock(name="delete", return_value=_ok_response())

    with patch.object(upload, "_load_config", return_value=_DUMMY_CONFIG), patch(
        "pylindas.lindas.upload.requests.request", mock_post
    ), patch("pylindas.lindas.upload.requests.delete", mock_delete), patch(
        "builtins.open", mock_open(read_data=b"<a> <b> <c> .")
    ):
        upload.upload_ttl(
            filename=["a.ttl", "b.ttl"],
            db_file="ignored.ini",
            environment="TEST",
            graph_uri="urn:graph",
            clear_graph=False,
        )

    assert mock_post.call_count == 2
    mock_delete.assert_not_called()


def test_upload_ttl_without_credentials_sends_no_auth():
    """An ini section without username/password uploads with ``auth=None``."""
    mock_request = MagicMock(name="request", return_value=_ok_response())
    config = {"endpoint": _DUMMY_CONFIG["endpoint"]}

    with patch.object(upload, "_load_config", return_value=config), patch(
        "pylindas.lindas.upload.requests.request", mock_request
    ), patch("builtins.open", mock_open(read_data=b"<a> <b> <c> .")):
        upload.upload_ttl(
            filename="some.ttl",
            db_file="ignored.ini",
            environment="TEST",
            graph_uri="urn:graph",
        )

    mock_request.assert_called_once()
    assert mock_request.call_args.kwargs["auth"] is None


def test_upload_graph_rejects_missing_auth_without_request():
    """Direct ``upload_graph`` calls require auth and send nothing otherwise."""
    mock_request = MagicMock(name="request", return_value=_ok_response())

    with patch("pylindas.lindas.upload.requests.request", mock_request):
        try:
            upload.upload_graph(
                b"<a> <b> <c> .",
                endpoint=_DUMMY_CONFIG["endpoint"],
                graph_uri="urn:graph",
                auth=None,
            )
        except UploadValidationError:
            pass
        else:
            raise AssertionError("expected UploadValidationError")

    mock_request.assert_not_called()


def test_upload_ttl_missing_endpoint_raises():
    """A config without an ``endpoint`` is a clear error, not a silent no-op."""
    with patch.object(
        upload, "_load_config", return_value={"username": "u", "password": "p"}
    ):
        try:
            upload.upload_ttl(
                filename="some.ttl",
                db_file="ignored.ini",
                environment="TEST",
                graph_uri="urn:graph",
            )
        except Exception as exc:
            assert "endpoint" in str(exc)
        else:
            raise AssertionError("expected an exception for missing endpoint")


_SECRET_CONFIG = {
    "endpoint": "https://example.test/rdf-graphs/service",
    "username": "alice-secret-user",
    "password": "s3cr3t-pw-value",
}


def _run_ttl(request_mock, config=_SECRET_CONFIG):
    with patch.object(upload, "_load_config", return_value=config), patch(
        "pylindas.lindas.upload.requests.request", request_mock
    ), patch("builtins.open", mock_open(read_data=b"<a> <b> <c> .")):
        upload.upload_ttl(
            filename="some.ttl",
            db_file="ignored.ini",
            environment="TEST",
            graph_uri="urn:graph",
        )


def test_upload_ttl_sends_without_timeout():
    """``upload_ttl`` passes ``timeout=None``; ``upload_graph`` keeps 300."""
    mock_request = MagicMock(name="request", return_value=_ok_response())
    _run_ttl(mock_request)
    assert mock_request.call_args.kwargs["timeout"] is None

    mock_request.reset_mock()
    with patch("pylindas.lindas.upload.requests.request", mock_request):
        upload.upload_graph(
            b"<a> <b> <c> .",
            endpoint=_DUMMY_CONFIG["endpoint"],
            graph_uri="urn:graph",
            auth=("u", "p"),
        )
    assert mock_request.call_args.kwargs["timeout"] == 300


def test_upload_ttl_timeout_surfaces_as_requests_timeout_without_secrets():
    """A timeout is a ``requests.exceptions.Timeout`` with no leaked credentials."""
    import requests

    leaky = requests.Timeout(
        "read timed out https://alice-secret-user:s3cr3t-pw-value@example.test/"
    )
    mock_request = MagicMock(name="request", side_effect=leaky)

    try:
        _run_ttl(mock_request)
    except requests.exceptions.Timeout as err:
        error = err
    else:
        raise AssertionError("expected requests.exceptions.Timeout")

    assert not isinstance(error, requests.HTTPError)
    seen = set()
    cur = error
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{cur!s} {cur!r} {cur.args}"
        assert "alice-secret-user" not in text
        assert "s3cr3t-pw-value" not in text
        cur = cur.__cause__ or cur.__context__
    assert error.__cause__ is None
    assert error.__context__ is None


def test_upload_ttl_http_500_raises_http_error_with_status_and_masked_body():
    """HTTP errors keep status and body, with credentials masked."""
    import requests

    resp = MagicMock(name="response")
    resp.status_code = 500
    resp.text = "boom alice-secret-user s3cr3t-pw-value"
    mock_request = MagicMock(name="request", return_value=resp)

    try:
        _run_ttl(mock_request)
    except requests.HTTPError as err:
        error = err
    else:
        raise AssertionError("expected requests.HTTPError")

    assert not isinstance(error, requests.exceptions.Timeout)
    assert error.response.status_code == 500
    body = error.response.text
    assert "boom" in body
    assert "alice-secret-user" not in body and "s3cr3t-pw-value" not in body
    assert "alice-secret-user" not in str(error)
    assert "s3cr3t-pw-value" not in str(error)
