"""Property test: PUT replaces, POST appends (Graph Store HTTP Protocol).

``upload_graph`` is exercised against an in-memory ``rdflib.Graph`` that sits
behind a ``responses`` callback, so no network is involved.
"""
from urllib.parse import parse_qs, urlparse

import responses
from hypothesis import given, settings
from hypothesis import strategies as st
from rdflib import Graph, Literal, URIRef
from rdflib.compare import isomorphic

from pylindas.lindas.upload import upload_graph

ENDPOINT = "https://lindas.test/repositories/lindas/rdf-graphs/service"
GRAPH_URI = "https://lindas.admin.ch/foen/cube"
AUTH = ("user", "secret")

_uris = st.builds(
    URIRef,
    st.text(alphabet="abcdefghij0123456789", min_size=1, max_size=8).map(
        lambda s: f"https://example.org/{s}"
    ),
)
_literals = st.one_of(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20
    ).map(Literal),
    st.integers(min_value=-1000, max_value=1000).map(Literal),
)
_triples = st.tuples(_uris, _uris, st.one_of(_uris, _literals))
graphs = st.lists(_triples, max_size=8).map(
    lambda ts: _make_graph(ts)
)


def _make_graph(triples) -> Graph:
    g = Graph()
    for t in triples:
        g.add(t)
    return g


class FakeGraphStore:
    """In-memory Graph Store: PUT replaces, POST appends the named graph."""

    def __init__(self, initial: Graph):
        self.graph = Graph()
        for t in initial:
            self.graph.add(t)

    def callback(self, request):
        query = parse_qs(urlparse(request.url).query)
        assert query["graph"] == [GRAPH_URI]
        assert request.headers["Content-Type"] == "text/turtle"
        incoming = Graph().parse(data=request.body, format="turtle")
        if request.method == "PUT":
            self.graph = incoming
        elif request.method == "POST":
            for t in incoming:
                self.graph.add(t)
        else:  # pragma: no cover
            return (405, {}, "method not allowed")
        return (204, {}, "")


def _run(initial: Graph, uploaded: Graph, method: str) -> Graph:
    store = FakeGraphStore(initial)
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps:
        rsps.add_callback(
            responses.PUT if method == "PUT" else responses.POST,
            ENDPOINT,
            callback=store.callback,
        )
        upload_graph(
            uploaded,
            endpoint=ENDPOINT,
            graph_uri=GRAPH_URI,
            auth=AUTH,
            method=method,
        )
        assert len(rsps.calls) == 1
    return store.graph


# Feature: lindas-publication, Property 3: PUT replaces, POST appends
# **Validates: Requirements 1.3, 1.4**
@settings(max_examples=100, deadline=None)
@given(initial=graphs, uploaded=graphs)
def test_put_replaces_and_post_appends(initial: Graph, uploaded: Graph):
    # PUT: stored graph is isomorphic to the uploaded data.
    after_put = _run(initial, uploaded, "PUT")
    assert isomorphic(after_put, uploaded)

    # POST: stored graph equals initial content united with the uploaded data.
    after_post = _run(initial, uploaded, "POST")
    expected = Graph()
    for t in initial:
        expected.add(t)
    for t in uploaded:
        expected.add(t)
    assert isomorphic(after_post, expected)
