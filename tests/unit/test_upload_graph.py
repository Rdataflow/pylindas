"""Unit tests for ``upload_graph`` and ``upload_ttl`` with HTTP mocked by ``responses``.

Covers Requirements 1.1, 1.2, 1.5, 1.9, 1.10, 8.8.
"""

import base64
import os
from urllib.parse import parse_qs, urlparse

import pytest
import responses
from rdflib import Graph

from pylindas.lindas import upload
from pylindas.lindas.upload import upload_graph, upload_ttl

ENDPOINT = "https://graphdb.test/repositories/lindas/rdf-graphs/service"
GRAPH = "https://lindas.admin.ch/foen/test"
AUTH = ("user", "secret")
TTL = "<http://ex/a> <http://ex/b> <http://ex/c> .\n"


@pytest.fixture
def no_ini_or_env(monkeypatch):
    """Make any ini or environment read fail loudly."""

    def boom(*args, **kwargs):
        raise AssertionError("upload_graph must not read ini/env")

    monkeypatch.setattr(upload, "_load_config", boom)
    monkeypatch.setattr(upload.ConfigParser, "read", boom)

    # ``requests`` itself reads proxy/CA env vars, so only fail for
    # configuration-like keys (credentials, endpoint, graph, LINDAS_*).
    real_getenv, real_get = os.getenv, os.environ.get
    suspicious = ("lindas", "user", "pass", "endpoint", "graph", "secret", "token")

    def guard(real):
        def inner(key, *args, **kwargs):
            if any(s in str(key).lower() for s in suspicious):
                boom()
            return real(key, *args, **kwargs)

        return inner

    monkeypatch.setattr(os, "getenv", guard(real_getenv))
    monkeypatch.setattr(os.environ, "get", guard(real_get))


def _basic(header):
    assert header.startswith("Basic ")
    return base64.b64decode(header[6:]).decode()


@pytest.mark.parametrize("kind", ["bytes", "str", "graph"])
@pytest.mark.parametrize("status", [200, 204, 399])
@responses.activate
def test_accepts_all_input_types_and_returns_none(kind, status, no_ini_or_env):
    responses.add(responses.PUT, ENDPOINT, status=status)
    data = {
        "bytes": TTL.encode("utf-8"),
        "str": TTL,
        "graph": Graph().parse(data=TTL, format="turtle"),
    }[kind]

    result = upload_graph(data, endpoint=ENDPOINT, graph_uri=GRAPH, auth=AUTH)

    assert result is None
    assert len(responses.calls) == 1
    req = responses.calls[0].request
    assert req.method == "PUT"
    assert parse_qs(urlparse(req.url).query) == {"graph": [GRAPH]}
    assert req.headers["Content-Type"] == "text/turtle"
    assert _basic(req.headers["Authorization"]) == "user:secret"
    body = req.body if isinstance(req.body, bytes) else req.body.encode()
    assert len(Graph().parse(data=body.decode("utf-8"), format="turtle")) == 1


@responses.activate
def test_post_method_appends(no_ini_or_env):
    responses.add(responses.POST, ENDPOINT, status=204)
    result = upload_graph(
        TTL, endpoint=ENDPOINT, graph_uri=GRAPH, auth=AUTH, method="POST"
    )
    assert result is None
    assert responses.calls[0].request.method == "POST"


@responses.activate
def test_non_ascii_str_is_utf8_encoded(no_ini_or_env):
    responses.add(responses.PUT, ENDPOINT, status=204)
    text = '<http://ex/a> <http://ex/b> "Zürich" .\n'
    upload_graph(text, endpoint=ENDPOINT, graph_uri=GRAPH, auth=AUTH)
    assert responses.calls[0].request.body == text.encode("utf-8")


@responses.activate
def test_upload_ttl_still_works(tmp_path):
    ini = tmp_path / "lindas.ini"
    ini.write_text(f"[TEST]\nendpoint={ENDPOINT}\nusername=u\npassword=p\n")
    ttl = tmp_path / "a.ttl"
    ttl.write_bytes(TTL.encode("utf-8"))
    responses.add(responses.POST, ENDPOINT, status=204)

    result = upload_ttl(str(ttl), str(ini), "TEST", GRAPH)

    assert result is None
    req = responses.calls[0].request
    assert req.method == "POST"
    assert parse_qs(urlparse(req.url).query) == {"graph": [GRAPH]}
    assert req.headers["Content-Type"] == "text/turtle"
    assert req.body == TTL.encode("utf-8")
    assert _basic(req.headers["Authorization"]) == "u:p"
