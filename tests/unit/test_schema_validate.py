"""Requirement 7: cube description schema and core validator (task 7.1)."""
import copy
import glob
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pylindas.lindas.exceptions import DescriptionValidationError, Violation
from pylindas.pycube import Cube
from pylindas.pycube.schema import load_schema, validate_description

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted(glob.glob(str(ROOT / "example" / "Cubes" / "*" / "description.yml")))
OTHERS = [ROOT / "tests" / "test.yml", ROOT / "tests" / "unit" / "fixtures" / "pilot_hierarchy" / "description.yaml"]
MOCK = ROOT / "example" / "Cubes" / "mock" / "description.yml"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mock():
    return _load(MOCK)


def _by_path(violations):
    return {v.path: v.message for v in violations}


@pytest.mark.parametrize("path", EXAMPLES + OTHERS, ids=lambda p: Path(p).parent.name + "/" + Path(p).name)
def test_real_descriptions_have_zero_violations(path):
    assert validate_description(_load(path)) == []


def test_examples_found():
    assert len(EXAMPLES) >= 5


def test_schema_is_packaged_resource_draft_2020_12():
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False


def test_unknown_top_level_key_reported_with_path():
    d = _mock()
    d["Typo"] = 1
    assert _by_path(validate_description(d)) == {"Typo": "unknown key (not defined by the schema)"}


def test_unknown_nested_key_path_and_dict_keys():
    d = _mock()
    d["dimensions"]["Jahr"]["bogus"] = 1
    assert "dimensions.Jahr.bogus" in _by_path(validate_description(d))


def test_missing_required_key_path():
    d = _mock()
    del d["Publisher"]
    del d["dimensions"]["Jahr"]["path"]
    paths = _by_path(validate_description(d))
    assert set(paths) == {"Publisher", "dimensions.Jahr.path"}


def test_mapping_type_violation_path():
    d = _mock()
    name = next(n for n, v in d["dimensions"].items() if "mapping" in v)
    d["dimensions"][name]["mapping"]["type"] = "nope"
    paths = _by_path(validate_description(d))
    assert f"dimensions.{name}.mapping.type" in paths


def test_mapping_missing_branch_key_path():
    d = _mock()
    name = next(n for n, v in d["dimensions"].items() if v.get("mapping", {}).get("type") == "additive")
    del d["dimensions"][name]["mapping"]["base"]
    assert f"dimensions.{name}.mapping.base" in _by_path(validate_description(d))


def test_array_index_path_and_enum():
    d = _mock()
    d["Creator"] = [{"IRI": "x"}, {"IRI": 5}]
    assert "Creator[1].IRI" in _by_path(validate_description(d))
    d = _mock()
    name = next(iter(d["dimensions"]))
    d["dimensions"][name]["scale-type"] = "weird"
    assert f"dimensions.{name}.scale-type" in _by_path(validate_description(d))


def test_concept_parent_field_and_unknown_concept_key():
    d = _load(OTHERS[1])
    key = next(iter(d["Concepts"]))
    d["Concepts"][key]["parent-field"] = "p"
    d["Concepts"][key]["nope"] = 1
    assert _by_path(validate_description(d)) == {f"Concepts.{key}.nope": "unknown key (not defined by the schema)"}


def test_all_violations_collected():
    d = _mock()
    d["A"] = 1
    d["B"] = 2
    del d["Name"]
    assert len(validate_description(d)) == 3


def test_non_mapping_root():
    assert validate_description([]) == [Violation("$", "must be of type object, got array")]


def test_validator_does_not_mutate_input():
    d = _mock()
    d["X"] = 1
    before = copy.deepcopy(d)
    validate_description(d)
    assert d == before


def test_agrees_with_jsonschema_on_examples_and_broken_copies():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(load_schema())
    for path in EXAMPLES + OTHERS:
        d = _load(path)
        broken = copy.deepcopy(d)
        broken["Typo"] = 1
        del broken["Name"]
        for desc in (d, broken):
            assert (not list(validator.iter_errors(desc))) == (validate_description(desc) == [])


def _frame(d):
    return pd.DataFrame({c: [1] for c in d["dimensions"]})


def test_constructor_raises_single_error_with_all_violations_and_leaves_yaml_unchanged():
    d = _mock()
    d["A"] = 1
    del d["Name"]
    before = copy.deepcopy(d)
    with pytest.raises(DescriptionValidationError) as e:
        Cube(_frame(d), d, "TEST", local=True, validate_description="strict")
    assert {v.path for v in e.value.violations} == {"A", "Name"}
    assert d == before


def test_constructor_skips_validation_when_false():
    d = _mock()
    d["A"] = 1
    Cube(pd.DataFrame({c: ["x"] for c in d["dimensions"]}), d, "TEST", local=True, validate_description=False)


def test_validation_default_is_warn():
    # Requirement 7.4
    import inspect

    default = inspect.signature(Cube.__init__).parameters["validate_description"].default
    assert default == "warn"


def _violating():
    d = _mock()
    d["A"] = 1
    del d["Name"]
    return d


def _pycube_warnings(caplog):
    return [r for r in caplog.records if r.name == "pycube" and r.levelname == "WARNING"]


def test_warn_logs_exactly_one_warning_with_all_violations_and_builds(caplog):
    # Requirements 7.3, 7.5, 7.6
    d = _violating()
    before = copy.deepcopy(d)
    expected = validate_description(d)
    assert {v.path for v in expected} == {"A", "Name"}
    with caplog.at_level("WARNING", logger="pycube"):
        cube = Cube(_frame(d), d, "TEST", local=True)  # default mode
    records = _pycube_warnings(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    for v in expected:
        assert f"{v.path}: {v.message}" in msg
    assert cube._graph is not None
    assert d == before


def test_explicit_warn_behaves_like_default(caplog):
    d = _violating()
    with caplog.at_level("WARNING", logger="pycube"):
        Cube(_frame(d), d, "TEST", local=True, validate_description="warn")
    assert len(_pycube_warnings(caplog)) == 1


def test_warn_valid_description_logs_nothing(caplog):
    d = _mock()
    with caplog.at_level("WARNING", logger="pycube"):
        Cube(_frame(d), d, "TEST", local=True)
    assert _pycube_warnings(caplog) == []


def test_strict_raises_and_logs_no_warning(caplog):
    d = _violating()
    with caplog.at_level("WARNING", logger="pycube"):
        with pytest.raises(DescriptionValidationError) as e:
            Cube(_frame(d), d, "TEST", local=True, validate_description="strict")
    assert {v.path for v in e.value.violations} == {"A", "Name"}
    assert _pycube_warnings(caplog) == []


def test_true_is_alias_for_strict():
    d = _violating()
    with pytest.raises(DescriptionValidationError) as e_true:
        Cube(_frame(d), d, "TEST", local=True, validate_description=True)
    with pytest.raises(DescriptionValidationError) as e_strict:
        Cube(_frame(d), d, "TEST", local=True, validate_description="strict")
    assert e_true.value.violations == e_strict.value.violations


def test_false_skips_validation_and_logs_nothing(caplog):
    d = _violating()
    with caplog.at_level("WARNING", logger="pycube"):
        Cube(_frame(d), d, "TEST", local=True, validate_description=False)
    assert _pycube_warnings(caplog) == []


@pytest.mark.parametrize("bad", ["loose", "", "STRICT", 1, 0, None, 1.0])
def test_invalid_mode_rejected_before_any_request(bad):
    from unittest import mock

    d = _mock()
    with mock.patch.object(Cube, "_setup_cube_uri") as setup_uri:
        with pytest.raises(ValueError) as e:
            Cube(_frame(d), d, "TEST", local=False, validate_description=bad)
    assert "validate_description" in str(e.value)
    for accepted in ('"warn"', '"strict"', "False"):
        assert accepted in str(e.value)
    setup_uri.assert_not_called()


def test_validation_runs_before_any_triple_is_created():
    # Requirement 7.3: graph set-up (the first place triples are created) is never reached.
    from unittest import mock

    d = _mock()
    d["A"] = 1
    with mock.patch.object(Cube, "_setup_graph") as setup_graph, mock.patch.object(
        Cube, "_setup_cube_uri"
    ) as setup_uri:
        with pytest.raises(DescriptionValidationError):
            Cube(_frame(d), d, "TEST", local=True, validate_description="strict")
    setup_graph.assert_not_called()
    setup_uri.assert_not_called()
