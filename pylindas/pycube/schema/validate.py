"""Minimal recursive JSON-schema checker for cube descriptions.

Core dependencies only (no ``jsonschema``). It supports exactly the keyword subset the
packaged schema uses: ``type``, ``properties``, ``required``, ``additionalProperties``,
``enum``, ``items``, ``oneOf`` and ``$ref`` to ``#/$defs/...``. Pure and side-effect free:
the instance is only read.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from pylindas.lindas.exceptions import Violation

SCHEMA_FILE = "cube_description.schema.json"
ROOT_PATH = "$"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """Load the packaged schema (works from an installed wheel)."""
    text = resources.files("pylindas.pycube.schema").joinpath(SCHEMA_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def _is_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "null":
        return value is None
    return False


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    for name in ("boolean", "integer", "number", "string", "array", "object"):
        if _is_type(value, name):
            return name
    return type(value).__name__


def _key(path: str, key: Any) -> str:
    key = str(key)
    if "." in key or "[" in key or "]" in key:
        return f'{"" if path == ROOT_PATH else path}["{key}"]'
    return key if path == ROOT_PATH else f"{path}.{key}"


def _resolve(schema: dict, root: dict) -> dict:
    seen = 0
    while "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Unsupported $ref {ref!r}")
        schema = root["$defs"][ref[len("#/$defs/"):]]
        seen += 1
        if seen > 50:
            raise ValueError("$ref cycle")
    return schema


def _fmt_enum(values) -> str:
    return ", ".join(repr(v) for v in values)


def _check(instance: Any, schema: dict, root: dict, path: str, out: list[Violation]) -> None:
    schema = _resolve(schema, root)
    types = schema.get("type")
    if types is not None:
        allowed = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(instance, t) for t in allowed):
            out.append(Violation(path, f"must be of type {' or '.join(allowed)}, got {_type_name(instance)}"))
            return
    if "enum" in schema and instance not in schema["enum"]:
        out.append(Violation(path, f"{instance!r} is not one of: {_fmt_enum(schema['enum'])}"))
    if "oneOf" in schema:
        _check_one_of(instance, schema["oneOf"], root, path, out)
    if isinstance(instance, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                out.append(Violation(_key(path, name), "required key is missing"))
        extra = schema.get("additionalProperties", True)
        for name, value in instance.items():
            if name in props:
                _check(value, props[name], root, _key(path, name), out)
            elif extra is False:
                out.append(Violation(_key(path, name), "unknown key (not defined by the schema)"))
            elif isinstance(extra, dict):
                _check(value, extra, root, _key(path, name), out)
    elif isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            _check(item, schema["items"], root, f"{'' if path == ROOT_PATH else path}[{i}]", out)


def _discriminator_values(branch: dict, root: dict) -> list | None:
    type_schema = _resolve(branch, root).get("properties", {}).get("type")
    if type_schema is None:
        return None
    return _resolve(type_schema, root).get("enum")


def _check_one_of(instance: Any, branches: list, root: dict, path: str, out: list[Violation]) -> None:
    results = []
    for branch in branches:
        found: list[Violation] = []
        _check(instance, branch, root, path, found)
        results.append(found)
    matching = sum(1 for r in results if not r)
    if matching == 1:
        return
    if matching > 1:
        out.append(Violation(path, "matches more than one allowed alternative"))
        return
    # No branch matches. For objects with a "type" discriminator, report against the branch the
    # discriminator selects; if it selects none, report the allowed discriminator values.
    if isinstance(instance, dict):
        discriminators = [_discriminator_values(b, root) for b in branches]
        if all(d is not None for d in discriminators):
            tag = instance.get("type")
            for found, values in zip(results, discriminators):
                if tag in values:
                    out.extend(found)
                    return
            allowed = [v for values in discriminators for v in values]
            out.append(Violation(_key(path, "type"), f"{tag!r} is not one of: {_fmt_enum(allowed)}"
                                 if "type" in instance else f"required key is missing; one of: {_fmt_enum(allowed)}"))
            return
    out.append(Violation(path, f"does not match any allowed alternative (got {_type_name(instance)})"))


def validate_description(description: Any) -> list[Violation]:
    """Return every schema violation of ``description`` (empty list means valid). Never raises for bad input."""
    root = load_schema()
    violations: list[Violation] = []
    _check(description, root, root, ROOT_PATH, violations)
    return violations
