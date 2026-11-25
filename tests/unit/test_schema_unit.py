"""Requirement 7: unit tests for the cube description schema (task 7.3).

Complements ``test_schema_validate.py`` (task 7.1). Covers:

* the pilot description has zero violations (7.1, 7.2),
* the schema loads from an installed wheel with core dependencies only (7.7),
* the subset validator agrees with ``jsonschema`` on generated descriptions (7.2),
* ``validate_description=False`` builds as before (7.8).

The pilot Lindas_Block cube description (``forest_industrial_wood_survey``) is written
later (Part B). Until then the notebook-derived pilot fixture
``tests/unit/fixtures/pilot_hierarchy/description.yaml`` stands in for it.
"""
import copy
import glob
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml
from hypothesis import HealthCheck, given, settings, strategies as st

from pylindas.pycube import Cube
from pylindas.pycube.schema import load_schema, validate_description

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "tests" / "unit" / "fixtures" / "pilot_hierarchy" / "description.yaml"
EXAMPLES = sorted(glob.glob(str(ROOT / "example" / "Cubes" / "*" / "description.yml")))
MOCK = ROOT / "example" / "Cubes" / "mock" / "description.yml"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- pilot description ---------------------------------------------------------------------

def test_pilot_description_has_zero_violations():
    """Stand-in for ``forest_industrial_wood_survey`` (see module docstring)."""
    assert validate_description(_load(PILOT)) == []


def test_pilot_description_builds_with_validation_enabled():
    df = pd.read_csv(PILOT.parent / "data.csv", dtype={"assortment": str, "origin": str})
    cube = Cube(df, _load(PILOT), "TEST", local=True).prepare_data().write_cube()
    assert len(cube._graph) > 0


# --- validate_description=False ------------------------------------------------------------

def _mock_cube(desc, **kwargs):
    df = pd.DataFrame({c: ["x"] for c in desc["dimensions"]})
    # write_cube() emits the cube metadata triples; the constructor alone adds none.
    return Cube(df, desc, "TEST", local=True, **kwargs).write_cube()


def _without_timestamps(graph):
    """Copy of ``graph`` without the build-time timestamps.

    ``write_cube`` writes ``schema:datePublished`` and ``schema:dateModified`` from the
    clock, so two builds differ whenever they straddle a second boundary.
    """
    from rdflib import Graph
    from pylindas.lindas.namespaces import SCHEMA

    stripped = Graph()
    for s, p, o in graph:
        if p not in (SCHEMA.datePublished, SCHEMA.dateModified):
            stripped.add((s, p, o))
    return stripped


def test_validation_off_builds_description_with_unknown_key():
    desc = _load(MOCK)
    desc["UnknownKey"] = "ignored"
    cube = _mock_cube(desc, validate_description=False)
    assert len(cube._graph) > 0


def test_validation_off_graph_equals_validated_build_of_valid_description():
    """With the flag off, a valid description produces the same graph as the default build."""
    from rdflib.compare import isomorphic

    desc = _load(MOCK)
    on = _mock_cube(copy.deepcopy(desc))
    off = _mock_cube(copy.deepcopy(desc), validate_description=False)
    assert isomorphic(_without_timestamps(on._graph), _without_timestamps(off._graph))


def test_validation_off_unknown_key_does_not_change_graph():
    """An unknown key is not read by the builder, so the graph equals the build without it."""
    from rdflib.compare import isomorphic

    desc = _load(MOCK)
    with_key = copy.deepcopy(desc)
    with_key["UnknownKey"] = "ignored"
    a = _mock_cube(desc, validate_description=False)
    b = _mock_cube(with_key, validate_description=False)
    assert isomorphic(_without_timestamps(a._graph), _without_timestamps(b._graph))


# --- agreement with jsonschema -------------------------------------------------------------

BASES = [_load(p) for p in EXAMPLES + [str(PILOT)]]

_wrong_values = st.one_of(
    st.none(), st.booleans(), st.integers(-5, 5), st.floats(0.25, 9.75).filter(lambda f: f != int(f)),
    st.text(max_size=8), st.lists(st.integers(0, 3), max_size=2),
    st.dictionaries(st.text("abc", min_size=1, max_size=3), st.integers(0, 3), max_size=2),
)


def _nodes(obj, path=()):
    """All (path, container) pairs where container is a dict or list inside obj."""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from _nodes(v, path + (k,))
    elif isinstance(obj, list):
        yield path, obj
        for i, v in enumerate(obj):
            yield from _nodes(v, path + (i,))


@st.composite
def mutated_descriptions(draw):
    desc = copy.deepcopy(draw(st.sampled_from(BASES)))
    for _ in range(draw(st.integers(0, 3))):
        nodes = list(_nodes(desc))
        _, node = nodes[draw(st.integers(0, len(nodes) - 1))]
        keys = list(node.keys()) if isinstance(node, dict) else list(range(len(node)))
        op = draw(st.sampled_from(["drop", "add_unknown", "wrong_type", "bad_enum"]))
        if op == "add_unknown" and isinstance(node, dict):
            node[draw(st.sampled_from(["bogus", "Typo", "x-extra"]))] = draw(_wrong_values)
        elif keys:
            key = keys[draw(st.integers(0, len(keys) - 1))]
            if op == "drop" and isinstance(node, dict):
                del node[key]
            elif op == "wrong_type":
                node[key] = draw(_wrong_values)
            elif op == "bad_enum" and isinstance(node[key], str):
                node[key] = draw(st.sampled_from(["not-an-enum-value", "", "NOMINAL?"]))
    return desc


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(mutated_descriptions())
def test_subset_validator_agrees_with_jsonschema(desc):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(load_schema())
    reference_valid = not any(True for _ in validator.iter_errors(desc))
    assert (validate_description(desc) == []) == reference_valid


# --- installed wheel, core dependencies only -----------------------------------------------

CORE_DEPS = ["pandas", "numpy", "rdflib", "pyshacl", "PyYAML", "requests", "SPARQLWrapper"]

_WHEEL_SCRIPT = textwrap.dedent(
    """
    import importlib.util, json, sys
    import pylindas
    assert "site-packages" in pylindas.__file__, pylindas.__file__
    assert importlib.util.find_spec("jsonschema") is None, "jsonschema must not be installed"
    from importlib import resources
    text = resources.files("pylindas.pycube.schema").joinpath("cube_description.schema.json").read_text(encoding="utf-8")
    assert json.loads(text)["$schema"].endswith("2020-12/schema")
    from pylindas.pycube.schema import validate_description
    ok = {"Name": {"en": "n"}}
    bad = {"Unknown": 1}
    violations = validate_description(bad)
    assert any(v.path == "Unknown" for v in violations), violations
    assert validate_description(ok) != []  # incomplete description is reported, not crashed on
    print("WHEEL_OK")
    """
)


@pytest.mark.slow
def test_schema_loads_from_installed_wheel_with_core_dependencies_only():
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    tmp = Path(tempfile.mkdtemp(prefix="pylindas-wheel-"))
    try:
        def run(cmd, **kw):
            return subprocess.run(cmd, capture_output=True, text=True, timeout=180, **kw)

        build = run([uv, "build", "--wheel", "--out-dir", str(tmp / "dist"), str(ROOT)])
        assert build.returncode == 0, build.stdout + build.stderr
        wheels = list((tmp / "dist").glob("pylindas-*.whl"))
        assert len(wheels) == 1, wheels

        venv = tmp / "venv"
        created = run([uv, "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", str(venv)])
        if created.returncode != 0:
            created = run([uv, "venv", str(venv)])
        assert created.returncode == 0, created.stderr
        py = str(venv / "bin" / "python")

        no_deps = run([uv, "pip", "install", "--python", py, "--no-deps", str(wheels[0])])
        assert no_deps.returncode == 0, no_deps.stderr
        deps = run([uv, "pip", "install", "--python", py, *CORE_DEPS])
        if deps.returncode != 0:
            pytest.skip("core dependencies could not be installed (offline?): " + deps.stderr[-300:])

        # cwd is the tmp dir so the repository checkout cannot shadow the installed package.
        result = run([py, "-c", _WHEEL_SCRIPT], cwd=str(tmp))
        assert result.returncode == 0 and "WHEEL_OK" in result.stdout, result.stdout + result.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
