"""Property 16: schema violations are reported with paths.

# Feature: lindas-publication, Property 16: Schema violations are reported with paths
**Validates: Requirements 7.5, 7.6**

Strategy: start from a schema-valid description (the packaged mock example or
``minimal_description()``), inject 1..n *independent* defects at known locations and
track each defect's expected path. Then check that

(a) ``validate_description`` reports a violation at exactly each expected path (and nothing else),
(b) ``Cube(...)`` raises ONE ``DescriptionValidationError`` carrying all of them,
(c) the caller's dict is unchanged.

Avoiding masked defects (generator design)
------------------------------------------
Defects are drawn greedily from a shuffled list of candidate sites; a candidate is accepted
only if it is compatible with everything accepted so far:

* Target paths are never equal or in a prefix relation. A wrong-typed / dropped parent would
  hide errors inside it, and an unknown key can only be at a path that does not yet exist.
* A defect on a mapping's ``type`` (dropped, invalid enum value, wrong type) breaks the
  ``oneOf`` branch selection, so nothing else at or below that mapping is reported. Such a
  defect must therefore be the only one inside its mapping.
* ``data-kind`` (``oneOf`` without discriminator), ``dimensions`` itself (an unknown key
  there is a dimension), free-form ``replacements`` and ``Date Created`` get no defects.
"""
import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from hypothesis import given, settings, strategies as st

from pylindas.lindas.exceptions import DescriptionValidationError
from pylindas.pycube.cube import Cube
from pylindas.pycube.schema.validate import validate_description

from tests.unit.conftest import minimal_description

MOCK = Path(__file__).resolve().parents[2] / "example" / "Cubes" / "mock" / "description.yml"

UNKNOWN_KEYS = ["bogus", "extra key", "a.b", "x[0]"]  # last two exercise ["key"] path quoting

ROOT_REQUIRED = ["Name", "Description", "Publisher", "Creator", "Contact Point",
                 "Base-URI", "Identifier", "Version", "dimensions"]
DIM_REQUIRED = ["name", "description", "path", "dimension-type", "scale-type"]
MAPPING_REQUIRED = {"additive": ["type", "base"], "replace": ["type", "replacements"],
                    "regex": ["type", "pattern", "replacement"],
                    "concept": ["type", "replacement-automated"],
                    "function": ["type", "filepath", "function-name"]}
ROOT_WRONG = {"Name": 5, "Description": "s", "Publisher": "s", "Creator": "s", "Contact Point": [],
              "Base-URI": 5, "Identifier": 5, "Version": [1], "Visualize": "yes", "dimensions": [],
              "Contributor": "s"}
ROOT_ENUM = {"Work Status": "Bogus", "Accrual Periodicity": "hourly"}
DIM_WRONG = {"name": "s", "description": "s", "path": 5, "unit": 5, "relates-to": 5}
DIM_ENUM = ["dimension-type", "scale-type", "datatype"]


def fmt(path):
    """Independent re-implementation of the documented path format. Root is '$'."""
    out = ""
    for seg in path:
        if isinstance(seg, int):
            out += f"[{seg}]"
        elif "." in seg or "[" in seg or "]" in seg:
            out += f'["{seg}"]'
        else:
            out += f".{seg}" if out else seg
    return out or "$"


def _get(d, path):
    for seg in path:
        d = d[seg]
    return d


def candidates(desc):
    """All (kind, target_path) injection sites for ``desc``; kind in unknown/drop/wrong/enum."""
    out = []
    unknown_parents = [(), ("Name",), ("Description",), ("Contact Point",)]
    for org in ("Publisher", "Creator", "Contributor"):
        for i, o in enumerate(desc.get(org, [])):
            unknown_parents.append((org, i))
            out.append(("wrong", (org, i, "IRI")))
            if org != "Contributor":
                out.append(("drop", (org, i, "IRI")))
    for k in ROOT_REQUIRED:
        out.append(("drop", (k,)))
    for k in ROOT_WRONG:
        if k in desc:
            out.append(("wrong", (k,)))
    for k in ROOT_ENUM:
        if k in desc:
            out.append(("enum", (k,)))
    for dn, dim in desc["dimensions"].items():
        base = ("dimensions", dn)
        unknown_parents.append(base)
        out.append(("wrong", base))
        for k in DIM_REQUIRED:
            out.append(("drop", base + (k,)))
        for k in DIM_WRONG:
            if k in dim:
                out.append(("wrong", base + (k,)))
        for k in DIM_ENUM:
            if k in dim:
                out.append(("enum", base + (k,)))
        for k in ("name", "description"):
            unknown_parents.append(base + (k,))
        if "mapping" in dim:
            m = base + ("mapping",)
            mp = dim["mapping"]
            unknown_parents.append(m)
            out.append(("wrong", m))
            out.append(("enum", m + ("type",)))
            out.append(("wrong", m + ("type",)))
            for k in MAPPING_REQUIRED[mp["type"]]:
                out.append(("drop", m + (k,)))
            for k in ("base", "replacements", "pattern", "replacement", "filepath"):
                if k in mp:
                    out.append(("wrong", m + (k,)))
    for p in unknown_parents:
        for name in UNKNOWN_KEYS:
            out.append(("unknown", p + (name,)))
    return [(k, p) for k, p in out if k != "drop" or p[-1] in _get_keys(desc, p[:-1])]


def _get_keys(desc, parent):
    return list(_get(desc, parent).keys())


def _wrong_value(path):
    key = path[-1]
    if key == "replacements":
        return "s"
    if key == "type":
        return 5
    return {**ROOT_WRONG, **DIM_WRONG}.get(key, 5)


def _is_mapping_type(kind, path):
    return kind != "unknown" and path[-1] == "type" and "mapping" in path


def _mapping_region(path):
    return path[:path.index("mapping") + 1] if "mapping" in path else None


def compatible(a, b):
    (ka, pa), (kb, pb) = a, b
    n = min(len(pa), len(pb))
    if pa[:n] == pb[:n]:  # equal or prefix-related
        return False
    ra, rb = _mapping_region(pa), _mapping_region(pb)
    if ra is not None and ra == rb and (_is_mapping_type(ka, pa) or _is_mapping_type(kb, pb)):
        return False
    return True


def apply(desc, kind, path):
    parent = _get(desc, path[:-1])
    key = path[-1]
    if kind == "unknown":
        assert key not in parent
        parent[key] = "x"
    elif kind == "drop":
        del parent[key]
    elif kind == "wrong":
        parent[key] = _wrong_value(path)
    elif kind == "enum":
        parent[key] = "bogus-enum-value"


@st.composite
def scenario(draw):
    base = draw(st.sampled_from(["mock", "minimal"]))
    if base == "mock":
        desc = yaml.safe_load(MOCK.read_text(encoding="utf-8"))
    else:
        desc = minimal_description()
    cands = draw(st.permutations(candidates(desc)))
    n = draw(st.integers(1, 6))
    chosen = []
    for c in cands:
        if all(compatible(c, o) for o in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return desc, chosen


def test_bases_are_valid():
    assert validate_description(yaml.safe_load(MOCK.read_text(encoding="utf-8"))) == []
    assert validate_description(minimal_description()) == []


# Feature: lindas-publication, Property 16: Schema violations are reported with paths
# Validates: Requirements 7.5, 7.6
@settings(max_examples=200, deadline=None)
@given(scenario())
def test_schema_violations_reported_with_paths(sc):
    desc, defects = sc
    for kind, path in defects:
        apply(desc, kind, path)
    expected = sorted(fmt(p) for _, p in defects)
    snapshot = copy.deepcopy(desc)

    violations = validate_description(desc)
    # (a) one violation per defect, at exactly the expected path, with a rule description
    assert sorted(v.path for v in violations) == expected
    assert all(v.message for v in violations)
    unknown = {fmt(p) for k, p in defects if k == "unknown"}
    for v in violations:
        if v.path in unknown:
            assert "unknown key" in v.message

    # (b) one error from the constructor lists every violation with path and message
    with pytest.raises(DescriptionValidationError) as exc:
        Cube(pd.DataFrame({"k": ["a"]}), desc, "TEST", local=True, validate_description="strict")
    assert sorted(v.path for v in exc.value.violations) == expected
    for v in exc.value.violations:
        assert f"{v.path}: {v.message}" in str(exc.value)

    # (c) the caller's description is untouched
    assert desc == snapshot
