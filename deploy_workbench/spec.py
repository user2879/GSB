from __future__ import annotations

import ast
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from .errors import E_BIND, E_ENV, E_GRID, E_SPEC, WorkbenchError
from .model import (
    EnvironmentSpec,
    FormulaTerm,
    SourceBinding,
    SourceDef,
    SUPPORTED_OVERFLOW_POLICIES,
    SUPPORTED_SOURCE_TYPES,
    TaskSpec,
    WorkbenchSpec,
)


SUPPORTED_FORMAT_VERSION = 1
MIN_SCALE_POWER = 0
MAX_SCALE_POWER = 9


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON number {token}")


def parse_json_text(text: str) -> Any:
    try:
        return json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, _DuplicateJsonKey, ValueError) as exc:
        raise WorkbenchError(E_ENV, f"environment declaration JSON is malformed: {exc}", path="$") from exc


def read_json_file(path: str | Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(E_ENV, f"cannot read environment declaration: {exc}", path="$") from exc
    return parse_json_text(text)


class FormulaParseError(ValueError):
    pass


def _as_decimal_constant(node: ast.AST) -> Decimal | None:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return Decimal(node.value)
        if isinstance(node.value, float):
            value = Decimal(str(node.value))
            return value if value.is_finite() else None
    return None


def _linearize(node: ast.AST) -> tuple[dict[str, Decimal], Decimal]:
    if isinstance(node, ast.Expression):
        return _linearize(node.body)
    constant = _as_decimal_constant(node)
    if constant is not None:
        return {}, constant
    if isinstance(node, ast.Name):
        return {node.id: Decimal(1)}, Decimal(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        coefs, const = _linearize(node.operand)
        factor = Decimal(-1) if isinstance(node.op, ast.USub) else Decimal(1)
        return {key: value * factor for key, value in coefs.items()}, const * factor
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left_coefs, left_const = _linearize(node.left)
        right_coefs, right_const = _linearize(node.right)
        factor = Decimal(-1) if isinstance(node.op, ast.Sub) else Decimal(1)
        merged = dict(left_coefs)
        for key, value in right_coefs.items():
            merged[key] = merged.get(key, Decimal(0)) + factor * value
        return merged, left_const + factor * right_const
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left_coefs, left_const = _linearize(node.left)
        right_coefs, right_const = _linearize(node.right)
        left_has_keys = bool(left_coefs)
        right_has_keys = bool(right_coefs)
        if left_has_keys and right_has_keys:
            raise FormulaParseError("formula is not linear: key-by-key multiplication is unsupported")
        if left_has_keys:
            return (
                {key: value * right_const for key, value in left_coefs.items()},
                left_const * right_const,
            )
        if right_has_keys:
            return (
                {key: value * left_const for key, value in right_coefs.items()},
                left_const * right_const,
            )
        return {}, left_const * right_const
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        raise FormulaParseError("division is unsupported; only integer fixed-point multiplication/addition is allowed")
    raise FormulaParseError(f"unsupported formula construct: {type(node).__name__}")


def parse_formula(formula: str) -> tuple[tuple[FormulaTerm, ...], tuple[str, ...]]:
    try:
        tree = ast.parse(formula, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise FormulaParseError(f"formula syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not node.id.replace("_", "a").isalnum():
            raise FormulaParseError(f"unsupported key identifier {node.id!r}")
    try:
        coefficients, constant = _linearize(tree)
    except FormulaParseError:
        raise
    except Exception as exc:
        raise FormulaParseError(f"formula could not be linearized: {exc}") from exc
    referenced = tuple(sorted(coefficients))
    if not referenced:
        raise FormulaParseError("formula must reference at least one numeric key")
    terms = [FormulaTerm(key=key, coefficient=coefficients[key], original_order=index) for index, key in enumerate(referenced)]
    if constant != 0:
        # A constant does not count as a referenced key, but it participates in
        # fixed-point truncation and formula-quantization error.
        terms.append(FormulaTerm(key=None, coefficient=constant, original_order=len(terms)))
    return tuple(terms), referenced


def _fail(
    code: str,
    message: str,
    *,
    path: str,
    environment: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raise WorkbenchError(code, message, environment=environment, path=path, details=details or {})


def _require_object(value: Any, path: str, *, environment: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(E_SPEC, "expected an object", path=path, environment=environment)
    return value


def _require_list(value: Any, path: str, *, environment: str | None = None, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _fail(E_SPEC, "expected an array", path=path, environment=environment)
    if nonempty and not value:
        _fail(E_SPEC, "array must not be empty", path=path, environment=environment)
    return value


def _require_string(value: Any, path: str, *, environment: str | None = None, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(E_SPEC, "expected a string", path=path, environment=environment)
    if nonempty and not value.strip():
        _fail(E_SPEC, "string must not be empty", path=path, environment=environment)
    return value


def _require_int(value: Any, path: str, *, environment: str | None = None, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(E_SPEC, "expected an integer", path=path, environment=environment)
    if minimum is not None and value < minimum:
        _fail(E_SPEC, f"integer must be >= {minimum}", path=path, environment=environment)
    return value


def _numeric_decimal(value: Any, path: str, *, environment: str | None = None) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        _fail(E_SPEC, "supported numeric key values are int, finite float, or decimal", path=path, environment=environment)
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise WorkbenchError(E_SPEC, "numeric value is invalid", environment=environment, path=path) from exc
    if not result.is_finite():
        _fail(E_SPEC, "numeric value must be finite", path=path, environment=environment)
    return result


def _string_list(value: Any, path: str, *, environment: str | None = None, nonempty: bool = False) -> tuple[str, ...]:
    items = _require_list(value, path, environment=environment, nonempty=nonempty)
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _require_string(item, f"{path}[{index}]", environment=environment)
        if text in seen:
            _fail(E_SPEC, f"duplicate value {text!r}", path=f"{path}[{index}]", environment=environment)
        seen.add(text)
        result.append(text)
    return tuple(result)


def _validate_threshold(value: Any) -> Decimal:
    threshold = _numeric_decimal(value, "$.error_dominance_threshold")
    if threshold <= 0 or threshold > 1:
        _fail(E_SPEC, "error dominance threshold must be in (0, 1]", path="$.error_dominance_threshold")
    return threshold


def _validate_scale_power(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkbenchError(E_GRID, "scale power must be an integer", path=path)
    if value < MIN_SCALE_POWER or value > MAX_SCALE_POWER:
        raise WorkbenchError(
            E_GRID,
            f"scale power must be an integer in [{MIN_SCALE_POWER}, {MAX_SCALE_POWER}]",
            path=path,
            details={"minimum": MIN_SCALE_POWER, "maximum": MAX_SCALE_POWER, "given": value},
        )
    return value


def load_spec(path: str | Path, *, scale_power_override: int | str | None = None) -> WorkbenchSpec:
    raw = read_json_file(path)
    root = _require_object(raw, "$")
    allowed_root = {
        "format_version",
        "scale_power",
        "overflow_policy",
        "error_dominance_threshold",
        "source_type_bias",
        "sources",
        "environments",
        "tasks",
    }
    unknown = sorted(set(root) - allowed_root)
    if unknown:
        _fail(E_SPEC, f"unknown top-level field {unknown[0]!r}", path=f"$.{unknown[0]}")

    format_version = _require_int(root.get("format_version"), "$.format_version")
    if format_version != SUPPORTED_FORMAT_VERSION:
        _fail(E_SPEC, f"unsupported format_version {format_version}", path="$.format_version")

    scale_value = root.get("scale_power")
    if scale_power_override is not None:
        scale_value = scale_power_override
    scale_power = _validate_scale_power(scale_value, "$.scale_power")

    overflow_policy = _require_string(root.get("overflow_policy"), "$.overflow_policy")
    if overflow_policy not in SUPPORTED_OVERFLOW_POLICIES:
        _fail(E_SPEC, f"unsupported overflow policy {overflow_policy!r}", path="$.overflow_policy")

    threshold = _validate_threshold(root.get("error_dominance_threshold"))

    bias_raw = _require_object(root.get("source_type_bias"), "$.source_type_bias")
    bias: dict[str, int] = {}
    for source_type in SUPPORTED_SOURCE_TYPES:
        path = f"$.source_type_bias.{source_type}"
        if source_type not in bias_raw:
            _fail(E_SPEC, f"source type bias missing for {source_type!r}", path=path)
        bias[source_type] = _require_int(bias_raw[source_type], path, minimum=0)
    unknown_bias = sorted(set(bias_raw) - set(SUPPORTED_SOURCE_TYPES))
    if unknown_bias:
        _fail(E_SPEC, f"unknown source type {unknown_bias[0]!r}", path=f"$.source_type_bias.{unknown_bias[0]}")

    source_items = _require_list(root.get("sources"), "$.sources", nonempty=True)
    sources: dict[str, SourceDef] = {}
    for index, item in enumerate(source_items):
        path = f"$.sources[{index}]"
        obj = _require_object(item, path)
        source_id = _require_string(obj.get("id"), path + ".id")
        if source_id in sources:
            _fail(E_SPEC, f"duplicate source id {source_id!r}", path=path + ".id")
        source_type = _require_string(obj.get("type"), path + ".type")
        if source_type not in SUPPORTED_SOURCE_TYPES:
            _fail(E_SPEC, f"unsupported source type {source_type!r}", path=path + ".type")
        sources[source_id] = SourceDef(id=source_id, type=source_type)

    environment_items = _require_list(root.get("environments"), "$.environments", nonempty=True)
    environments: dict[str, EnvironmentSpec] = {}
    for index, item in enumerate(environment_items):
        path = f"$.environments[{index}]"
        obj = _require_object(item, path)
        env_id = _require_string(obj.get("id"), path + ".id")
        if env_id in environments:
            _fail(E_SPEC, f"duplicate environment id {env_id!r}", path=path + ".id", environment=env_id)
        depth = _require_int(obj.get("depth"), path + ".depth", environment=env_id, minimum=0)
        available = _string_list(obj.get("available_sources"), path + ".available_sources", environment=env_id, nonempty=True)
        for source_index, source_id in enumerate(available):
            if source_id not in sources:
                _fail(
                    E_SPEC,
                    f"available source {source_id!r} is not declared",
                    path=f"{path}.available_sources[{source_index}]",
                    environment=env_id,
                )
        priority = _string_list(obj.get("source_priority"), path + ".source_priority", environment=env_id, nonempty=True)
        if set(priority) != set(available):
            _fail(
                E_SPEC,
                "source_priority must contain every available source exactly once",
                path=path + ".source_priority",
                environment=env_id,
            )

        entries_raw = _require_object(obj.get("entries"), path + ".entries", environment=env_id)
        unknown_entry_sources = sorted(set(entries_raw) - set(available))
        if unknown_entry_sources:
            _fail(
                E_SPEC,
                f"entries reference non-available source {unknown_entry_sources[0]!r}",
                path=f"{path}.entries.{unknown_entry_sources[0]}",
                environment=env_id,
            )
        entries: dict[str, dict[str, Decimal]] = {}
        for source_id in available:
            source_entries_raw = entries_raw.get(source_id, {})
            source_entries = _require_object(source_entries_raw, f"{path}.entries.{source_id}", environment=env_id)
            entries[source_id] = {}
            for key, value in source_entries.items():
                if not isinstance(key, str) or not key.strip():
                    _fail(E_SPEC, "numeric key name must be a non-empty string", path=f"{path}.entries.{source_id}", environment=env_id)
                entries[source_id][key] = _numeric_decimal(
                    value,
                    f"{path}.entries.{source_id}.{key}",
                    environment=env_id,
                )

        bindings_raw = obj.get("bindings", {})
        bindings_obj = _require_object(bindings_raw, path + ".bindings", environment=env_id)
        bindings: dict[str, SourceBinding] = {}
        for binding_key, binding_value in bindings_obj.items():
            binding_path = f"{path}.bindings.{binding_key}"
            if not isinstance(binding_key, str) or not binding_key.strip():
                _fail(E_BIND, "binding key must be a non-empty string", path=binding_path, environment=env_id)
            binding_obj = _require_object(binding_value, binding_path, environment=env_id)
            unknown_fields = sorted(set(binding_obj) - {"allowed_source_ids", "excluded_source_ids", "allowed_source_types"})
            if unknown_fields:
                _fail(E_BIND, f"unknown binding field {unknown_fields[0]!r}", path=f"{binding_path}.{unknown_fields[0]}", environment=env_id)
            allowed_ids = (
                _string_list(binding_obj["allowed_source_ids"], binding_path + ".allowed_source_ids", environment=env_id)
                if "allowed_source_ids" in binding_obj
                else None
            )
            excluded_ids = (
                _string_list(binding_obj["excluded_source_ids"], binding_path + ".excluded_source_ids", environment=env_id)
                if "excluded_source_ids" in binding_obj
                else ()
            )
            allowed_types = (
                _string_list(binding_obj["allowed_source_types"], binding_path + ".allowed_source_types", environment=env_id)
                if "allowed_source_types" in binding_obj
                else None
            )
            for field_name, ids in (("allowed_source_ids", allowed_ids or ()), ("excluded_source_ids", excluded_ids)):
                for source_index, source_id in enumerate(ids):
                    if source_id not in sources:
                        raise WorkbenchError(
                            E_BIND,
                            f"binding references missing source {source_id!r}",
                            environment=env_id,
                            path=f"{binding_path}.{field_name}[{source_index}]",
                            details={"binding_key": binding_key, "missing_source_id": source_id},
                        )
            if allowed_types is not None:
                for type_index, source_type in enumerate(allowed_types):
                    if source_type not in SUPPORTED_SOURCE_TYPES:
                        _fail(
                            E_BIND,
                            f"binding references unsupported source type {source_type!r}",
                            path=f"{binding_path}.allowed_source_types[{type_index}]",
                            environment=env_id,
                        )
            key_available = any(binding_key in entries[source_id] for source_id in available)
            if not key_available:
                _fail(
                    E_BIND,
                    f"binding key {binding_key!r} appears in no available source",
                    path=binding_path,
                    environment=env_id,
                )
            bindings[binding_key] = SourceBinding(
                allowed_source_ids=allowed_ids,
                excluded_source_ids=excluded_ids,
                allowed_source_types=allowed_types,
            )
        environments[env_id] = EnvironmentSpec(
            id=env_id,
            depth=depth,
            available_sources=tuple(sorted(available)),
            source_priority=priority,
            bindings=bindings,
            entries=entries,
        )

    available_key_set = {key for env in environments.values() for key in env.all_entry_keys()}
    task_items = _require_list(root.get("tasks"), "$.tasks", nonempty=True)
    tasks: dict[str, TaskSpec] = {}
    for index, item in enumerate(task_items):
        path = f"$.tasks[{index}]"
        obj = _require_object(item, path)
        task_id = _require_string(obj.get("id"), path + ".id")
        if task_id in tasks:
            _fail(E_SPEC, f"duplicate task id {task_id!r}", path=path + ".id")
        formula = _require_string(obj.get("formula"), path + ".formula")
        try:
            terms, keys = parse_formula(formula)
        except FormulaParseError as exc:
            raise WorkbenchError(E_SPEC, str(exc), path=path + ".formula") from exc
        if not keys:
            _fail(E_SPEC, "task key set must not be empty", path=path + ".formula")
        for key in keys:
            if key not in available_key_set:
                _fail(
                    E_SPEC,
                    f"formula key {key!r} occurs in no available source",
                    path=path + ".formula",
                    details={"missing_key": key},
                )
        tasks[task_id] = TaskSpec(id=task_id, formula=formula, terms=terms, keys=keys)

    return WorkbenchSpec(
        format_version=format_version,
        scale_power=scale_power,
        overflow_policy=overflow_policy,
        error_dominance_threshold=threshold,
        source_type_bias=bias,
        sources=sources,
        environments=environments,
        tasks=tasks,
        origin_path=str(Path(path).resolve()),
    )


def write_injected_spec(source_path: str | Path, destination_path: str | Path, injection: str) -> Path:
    """Create a separate bad input file; runtime validation logic stays unchanged."""

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_text = Path(source_path).read_text(encoding="utf-8")
    if injection == "env":
        destination.write_text('{"format_version": 1, "environments": [', encoding="utf-8")
        return destination
    if injection == "bind":
        raw = json.loads(original_text)
        environments = raw.get("environments")
        if not isinstance(environments, list) or not environments:
            raise WorkbenchError(E_SPEC, "cannot inject bind failure into an empty environment list", path="$.environments")
        first = environments[0]
        if not isinstance(first, dict):
            raise WorkbenchError(E_SPEC, "cannot inject bind failure into malformed environment", path="$.environments[0]")
        bindings = first.setdefault("bindings", {})
        if not isinstance(bindings, dict):
            raise WorkbenchError(E_SPEC, "cannot inject bind failure into malformed bindings", path="$.environments[0].bindings")
        bindings["cpu_cores"] = {"allowed_source_ids": ["__missing_injected_source__"]}
        destination.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return destination
    if injection == "grid":
        destination.write_text(original_text, encoding="utf-8")
        return destination
    raise WorkbenchError(E_SPEC, f"unknown injection type {injection!r}", path="$")
