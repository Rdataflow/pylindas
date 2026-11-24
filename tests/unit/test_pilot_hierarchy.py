"""Pilot regression: hierarchy edges and roots equal the former manual step (Requirements 3.8, 4.3, 4.5).

The fixtures in ``fixtures/pilot_hierarchy`` are a frozen copy of the notebook
``showcase_cube_v1_hierarchy`` (see the README there). The cube is built with
``parent-field`` only: no manual ``cube._graph.add`` call.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from rdflib import Graph, URIRef

from pylindas.lindas.namespaces import META, SCHEMA
from pylindas.pycube.cube import Cube

FIX = Path(__file__).parent / "fixtures" / "pilot_hierarchy"
DIMS = ("assortment", "origin")
CUBE_URI = "https://environment.ld.admin.ch/foen/wood-showcase-hierarchy/1"


def _expected_edges() -> set:
    g = Graph().parse(FIX / "expected_edges.ttl", format="turtle")
    return set(g.triples((None, SCHEMA.hasPart, None)))


def _expected_roots() -> dict:
    return {k: URIRef(v) for k, v in json.loads((FIX / "expected_roots.json").read_text()).items()}


def _build(absolute_roots: bool = False) -> Cube:
    cube_yaml = yaml.safe_load((FIX / "description.yaml").read_text(encoding="utf-8"))
    if absolute_roots:  # the notebook's original form of the root
        for dim, root in _expected_roots().items():
            cube_yaml["dimensions"][dim]["hierarchy"][0]["root"] = str(root)
    df = pd.read_csv(FIX / "data.csv", dtype={"assortment": str, "origin": str})
    cube = Cube(df, cube_yaml, environment="TEST", local=True)
    assert str(cube._cube_uri) == CUBE_URI
    cube.prepare_data().write_cube().write_observations().write_shape()
    for dim in DIMS:
        table = pd.read_csv(FIX / f"{dim}.csv", dtype=str, keep_default_na=True)
        cube.write_concept(dim, table)
    return cube


def _roots_by_dimension(cube: Cube) -> dict:
    q = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    PREFIX meta: <https://cube.link/meta/>
    SELECT ?path ?root WHERE { ?prop sh:path ?path ; meta:inHierarchy/meta:hierarchyRoot ?root . }
    """
    return {str(r.path).rsplit("/", 1)[-1]: r.root for r in cube._graph.query(q)}


@pytest.mark.parametrize("absolute_roots", [False, True], ids=["root-id", "root-iri"])
def test_has_part_triples_equal_stored_notebook_set(absolute_roots):
    cube = _build(absolute_roots)
    produced = set(cube._graph.triples((None, SCHEMA.hasPart, None)))
    expected = _expected_edges()
    assert len(expected) == 6
    assert produced == expected, (
        f"missing: {sorted(expected - produced)}\nunexpected: {sorted(produced - expected)}"
    )


@pytest.mark.parametrize("absolute_roots", [False, True], ids=["root-id", "root-iri"])
def test_hierarchy_root_equals_stored_root_and_is_edge_subject(absolute_roots):
    cube = _build(absolute_roots)
    roots = _roots_by_dimension(cube)
    expected = _expected_roots()
    assert roots == expected
    for dim in DIMS:
        root = roots[dim]
        assert list(cube._graph.objects(None, META.hierarchyRoot)).count(root) == 1
        assert (root, SCHEMA.hasPart, None) in cube._graph
        # the root is the subject of the top-level edges and never a child
        assert (None, SCHEMA.hasPart, root) not in cube._graph


def test_walking_has_part_from_root_reaches_all_used_concepts():
    cube = _build()
    df = pd.read_csv(FIX / "data.csv", dtype=str)
    for dim, root in _roots_by_dimension(cube).items():
        reachable = {str(o) for o in cube._graph.transitive_objects(root, SCHEMA.hasPart)}
        used = {f"{CUBE_URI}/concept/{dim}/{cid}" for cid in df[dim].unique()}
        assert reachable == used
