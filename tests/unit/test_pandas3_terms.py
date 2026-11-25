"""Regression: graph nodes stay rdflib terms when pandas stores strings as pyarrow ``str`` (pandas 3)."""
from pathlib import Path

import pandas as pd
import pytest
import yaml
from rdflib import BNode, URIRef
from rdflib.term import Node

from pylindas.pycube.cube import Cube
from tests.unit.test_pilot_hierarchy import _build as build_pilot

MOCK = Path(__file__).parents[2] / "example" / "Cubes" / "mock"


def _build_mock() -> Cube:
    cube_yaml = yaml.safe_load((MOCK / "description.yml").read_text(encoding="utf-8"))
    cube = Cube(pd.read_csv(MOCK / "data.csv"), cube_yaml, environment="TEST", local=True)
    return cube.prepare_data().write_cube().write_observations().write_shape()


@pytest.mark.parametrize("build", [_build_mock, build_pilot], ids=["mock", "pilot"])
def test_all_graph_nodes_are_rdflib_terms(build):
    graph = build()._graph
    assert len(graph) > 0
    for s, p, o in graph:
        assert isinstance(s, (URIRef, BNode)), s
        assert isinstance(p, URIRef), p
        assert isinstance(o, Node), o
