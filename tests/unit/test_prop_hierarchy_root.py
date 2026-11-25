"""Property test: hierarchy root resolution through the concept mapping."""
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from rdflib import Literal, URIRef

from pylindas.lindas.exceptions import HierarchyError
from pylindas.lindas.namespaces import META, SCHEMA

from tests.unit.test_hierarchy_root import CONCEPT_BASE, _cube

_ids = st.lists(st.text(alphabet="abcdefghij", min_size=1, max_size=6), min_size=1, max_size=8, unique=True)


@st.composite
def _scenario(draw):
    ids = draw(_ids)
    parents = [None]
    for i in range(1, len(ids)):
        parents.append(draw(st.one_of(st.none(), st.sampled_from(ids[:i]))))
    top = [ids[i] for i, p in enumerate(parents) if p is None]
    kind = draw(st.sampled_from(["valid", "valid", "missing", "empty"]))
    if kind == "valid":
        root = draw(st.sampled_from(top))
    elif kind == "empty":
        root = ""
    else:
        root = draw(st.text(alphabet="klmnopqrstuvwxyz", min_size=1, max_size=6))  # disjoint from ids
    df = pd.DataFrame({"ID": ids, "label": [i.upper() for i in ids], "parent": parents})
    return df, root, kind


def _roots(cube):
    return list(cube._graph.objects(None, META.hierarchyRoot))


# Feature: lindas-publication, Property 8: Hierarchy root resolves through the concept mapping
# Validates: Requirements 4.1, 4.2, 4.4
@pytest.mark.parametrize("concept_first", [False, True])
@settings(max_examples=100, deadline=None)
@given(sc=_scenario())
def test_hierarchy_root_resolves_through_concept_mapping(concept_first, sc):
    df, root, kind = sc
    cube = _cube(root=root)

    def run():
        if concept_first:
            cube.write_concept("assortment", df)
            cube.write_shape()
        else:
            cube.write_shape()
            cube.write_concept("assortment", df)

    if kind == "valid":
        run()
        expected = URIRef(CONCEPT_BASE + root)
        assert _roots(cube) == [expected]
        # literal root never written as IRI or literal
        for o in cube._graph.objects(None, META.hierarchyRoot):
            assert not isinstance(o, Literal)
        assert URIRef(root) not in set(cube._graph.all_nodes())
        if (df["parent"] == root).any():
            assert expected in set(cube._graph.subjects(SCHEMA.hasPart, None))
    else:
        if not concept_first:
            cube.write_shape()
            before = set(cube._graph)
            with pytest.raises(HierarchyError) as ei:
                cube.write_concept("assortment", df)
            assert set(cube._graph) == before  # rolled back
        else:
            with pytest.raises(HierarchyError) as ei:
                run()
        assert "assortment" in str(ei.value)
        assert root in str(ei.value)
        assert _roots(cube) == []
