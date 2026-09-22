"""Small dependency-free JSON Schema subset used at trust boundaries."""

from __future__ import annotations

import re
from typing import Any, Dict, List


class SchemaValidationError(ValueError):
    """Raised when data does not satisfy a declared JSON schema."""


def validate_json_schema(value: Any, schema: Dict[str, Any], path: str = "$") -> None:
    errors: List[str] = []
    _validate(value, schema, path, errors)
    if errors:
        raise SchemaValidationError("; ".join(errors))


def _validate(value: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> None:
    for alternative_key in ("oneOf", "anyOf"):
        if alternative_key in schema:
            matches = 0
            alternative_errors = []
            for alternative in schema[alternative_key]:
                candidate_errors: List[str] = []
                _validate(value, alternative, path, candidate_errors)
                if not candidate_errors:
                    matches += 1
                alternative_errors.extend(candidate_errors)
            valid = matches == 1 if alternative_key == "oneOf" else matches >= 1
            if not valid:
                errors.append(f"{path}: failed {alternative_key}")
            return

    expected = schema.get("type")
    if expected and not _matches_type(value, expected):
        errors.append(f"{path}: expected {expected}")
        return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']!r}")

    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: is required")
        properties = schema.get("properties", {})
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append(f"{path}: property count must be >= {schema['minProperties']}")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            errors.append(f"{path}: property count must be <= {schema['maxProperties']}")
        for name, child in value.items():
            if name in properties:
                _validate(child, properties[name], f"{path}.{name}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{name}: is not allowed")

    if isinstance(value, list) and "items" in schema:
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: item count must be >= {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: item count must be <= {schema['maxItems']}")
        for index, child in enumerate(value):
            _validate(child, schema["items"], f"{path}[{index}]", errors)

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: length must be >= {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: length must be <= {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match required pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: must be < {schema['exclusiveMaximum']}")


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    return any({
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(type_name, False) for type_name in expected_types)