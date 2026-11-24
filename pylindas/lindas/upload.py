"""Upload Turtle files to a LINDAS triplestore (GraphDB).

LINDAS migrated from Stardog to GraphDB. GraphDB implements the SPARQL 1.1
Graph Store HTTP Protocol, so uploads are plain authenticated HTTP requests
against the repository's graph-store service rather than a Stardog-specific
client connection:

* append Turtle to a named graph -> ``POST`` with ``Content-Type: text/turtle``
* replace a named graph           -> ``PUT``
* clear a named graph             -> ``DELETE``

Connection details are read from a ``lindas.ini`` section (selected by
``environment``) containing at least an ``endpoint`` and, for a protected
store, ``username`` / ``password``. Example:

    [TEST]
    endpoint=https://graphdb.test.cz-aws.net/repositories/lindas/rdf-graphs/service
    username=a-lindas-user-name
    password=...
"""

import re
from configparser import ConfigParser
from numbers import Real
from typing import Literal, Union

import rdflib
import requests
from requests.auth import HTTPBasicAuth

from pylindas.lindas.exceptions import (
    UploadError,
    UploadTimeoutError,
    UploadValidationError,
)


def _load_config(db_file: str, environment: str) -> dict:
    parser = ConfigParser()
    parser.read(db_file)

    config = {}
    if parser.has_section(environment):
        params = parser.items(environment)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(f"Environment '{environment}' not found in db_file")

    return config


def _auth_from_config(config: dict) -> Union[HTTPBasicAuth, None]:
    """Build basic-auth from the config, or ``None`` when no credentials given."""
    username = config.get("username")
    password = config.get("password")
    if username and password:
        return HTTPBasicAuth(username, password)
    return None


def _mask(text: str, secrets: tuple) -> str:
    """Replace every secret (longest first) in ``text`` by ``***``."""
    found = sorted({x for x in secrets if x}, key=len, reverse=True)
    if not found:
        return text
    # Single left-to-right pass: sequential replace() would let one secret's
    # replacement destroy an overlapping occurrence of the other and leave
    # parts of it unmasked.
    return re.sub("|".join(re.escape(x) for x in found), "***", text)


_USERINFO = re.compile(r"(?<=://)[^/?#\s]*@")


def _mask_url(url: str, secrets: tuple) -> str:
    """Strip URL userinfo (``scheme://user:pw@host``) and mask secrets."""
    return _mask(_USERINFO.sub("***@", url), secrets)


def upload_graph(
    data: Union[bytes, str, rdflib.Graph],
    *,
    endpoint: str,
    graph_uri: str,
    auth: tuple,
    method: Literal["PUT", "POST"] = "PUT",
    timeout: float = 300,
) -> None:
    """Upload Turtle to a GraphDB named graph using explicit arguments only.

    Reads no ini file and no environment variable. ``PUT`` replaces the named
    graph, ``POST`` appends to it (SPARQL 1.1 Graph Store HTTP Protocol).

    Raises:
        UploadValidationError: invalid arguments (no request is sent).
        UploadTimeoutError: the request exceeded ``timeout`` seconds.
        UploadError: the endpoint answered with a status >= 400. Username and
            password are masked in the response body.
    """
    problems = []
    if not isinstance(data, (bytes, str, rdflib.Graph)):
        problems.append(
            f"data must be bytes, str or rdflib.Graph, got {type(data).__name__}"
        )
    if not isinstance(endpoint, str) or not endpoint.strip():
        problems.append("endpoint must be a non-empty string")
    if not isinstance(graph_uri, str) or not graph_uri.strip():
        problems.append("graph_uri must be a non-empty string")
    if (
        not isinstance(auth, (tuple, list))
        or len(auth) != 2
        or not all(isinstance(x, str) and x for x in auth)
    ):
        problems.append("auth must be a (username, password) pair of non-empty strings")
    if method not in ("PUT", "POST"):
        problems.append("method must be 'PUT' or 'POST'")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, Real)
        or not timeout > 0
    ):
        problems.append("timeout must be a number greater than 0")
    if problems:
        raise UploadValidationError("; ".join(problems))

    if isinstance(data, bytes):
        body = data
    elif isinstance(data, str):
        body = data.encode("utf-8")
    else:
        body = data.serialize(format="turtle").encode("utf-8")

    _send_graph(
        body,
        endpoint=endpoint,
        graph_uri=graph_uri,
        auth=HTTPBasicAuth(auth[0], auth[1]),
        method=method,
        timeout=timeout,
    )


def _send_graph(
    body: bytes,
    *,
    endpoint: str,
    graph_uri: str,
    auth: Union[HTTPBasicAuth, None],
    method: str,
    timeout: Union[float, None] = 300,
) -> None:
    """Send already-validated Turtle bytes; ``auth=None`` sends no credentials.

    ``timeout=None`` disables the request timeout (used by ``upload_ttl``).
    """
    secrets = (auth.username, auth.password) if auth is not None else ()
    try:
        resp = requests.request(
            method,
            endpoint,
            params={"graph": graph_uri},
            data=body,
            headers={"Content-Type": "text/turtle"},
            auth=auth,
            timeout=timeout,
        )
    except requests.Timeout:
        timed_out = True
    else:
        timed_out = False

    safe_endpoint = _mask_url(endpoint, secrets)

    if timed_out:
        # Raised outside the ``except`` block so neither __cause__ nor
        # __context__ keeps the original exception (its text may echo
        # credentials). The message is built from masked values only.
        raise UploadTimeoutError(
            f"{method} to <{_mask(graph_uri, secrets)}> at {safe_endpoint} "
            f"timed out"
            + (f" after {timeout} seconds" if timeout is not None else "")
        )

    if resp.status_code >= 400:
        resp_body = _mask(resp.text or "", secrets)
        raise UploadError(
            f"{method} to <{_mask(graph_uri, secrets)}> at {safe_endpoint} "
            f"failed with status {resp.status_code}: {resp_body}",
            status=resp.status_code,
            body=resp_body,
        )


def upload_ttl(
    filename: Union[str, list],
    db_file: str,
    environment: str,
    graph_uri: str,
    clear_graph: bool = False,
):
    """Upload one or more Turtle files into a GraphDB named graph.

    Uses the SPARQL 1.1 Graph Store HTTP Protocol. All files are appended to
    ``graph_uri``; when ``clear_graph`` is set, the named graph is emptied first.

    Args:
        filename: Path to a Turtle file, or a list of paths.
        db_file: Path to the ``lindas.ini`` holding endpoint/credentials.
        environment: Section name in ``lindas.ini`` (e.g. ``TEST``).
        graph_uri: Target named graph IRI.
        clear_graph: When True, delete the named graph before adding (requires
            ``graph_uri`` so the whole store is never cleared by accident).
    """
    config = _load_config(db_file, environment)

    endpoint = config.get("endpoint")
    if not endpoint:
        raise Exception(
            f"No 'endpoint' configured for environment '{environment}' in {db_file}"
        )

    auth = _auth_from_config(config)
    params = {"graph": graph_uri}

    if clear_graph and graph_uri:
        # Guard: only clear when a specific graph is given, so we never wipe the
        # whole store. Mirrors the old connection-based safety check.
        resp = requests.delete(endpoint, params=params, auth=auth)
        resp.raise_for_status()

    def _add_file(file: str):
        print(f"uploading: {file}")
        with open(file, "rb") as f:
            content = f.read()
        failure = None
        try:
            _send_graph(
                content,
                endpoint=endpoint,
                graph_uri=graph_uri,
                auth=auth,
                method="POST",
                timeout=None,
            )
        except UploadError as exc:
            failure = exc
        if failure is None:
            return
        # Raised outside the ``except`` block: no __cause__/__context__ chain.
        if isinstance(failure, UploadTimeoutError):
            raise requests.exceptions.Timeout(str(failure))
        response = requests.Response()
        response.status_code = failure.status
        response._content = failure.body.encode("utf-8")
        raise requests.HTTPError(str(failure), response=response)

    files = [filename] if isinstance(filename, str) else filename
    for f in files:
        _add_file(file=f)
