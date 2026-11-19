"""Strictly validated local change sets for incremental recomputation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_SPEC, fail
from .spec import Spec, _is_int


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        fail(E_SPEC, path, "must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        fail(E_SPEC, path, "must be an array")
    return value


def _keys(obj: Mapping[str, Any], path: str, required: tuple[str, ...]) -> None:
    for key in obj:
        if key not in required:
            fail(E_SPEC, f"{path}.{key}", "unknown field")
    for key in required:
        if key not in obj:
            fail(E_SPEC, f"{path}.{key}", "required field is missing")


@dataclass(frozen=True)
class ValueChange:
    source: str
    row: int
    column: str
    value: Any
    index: int


@dataclass(frozen=True)
class NullChange:
    source: str
    row: int
    column: str
    is_null: bool
    index: int


@dataclass(frozen=True)
class ChangeSet:
    values: tuple[ValueChange, ...]
    nulls: tuple[NullChange, ...]
    rule_updates: Mapping[str, tuple[int, ...]]
    delete_rules: tuple[str, ...]

    def value_overrides(self) -> dict[str, dict[tuple[int, str], Any]]:
        out: dict[str, dict[tuple[int, str], Any]] = {}
        for change in self.values:
            out.setdefault(change.source, {})[(change.row, change.column)] = change.value
        return {key: out[key] for key in sorted(out)}

    def null_overrides(self) -> dict[str, dict[tuple[int, str], bool]]:
        out: dict[str, dict[tuple[int, str], bool]] = {}
        for change in self.nulls:
            out.setdefault(change.source, {})[(change.row, change.column)] = change.is_null
        return {key: out[key] for key in sorted(out)}

    def changed_source_ids(self) -> tuple[str, ...]:
        return tuple(sorted({change.source for change in self.values + self.nulls}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": [
                {
                    "source": change.source,
                    "row": change.row,
                    "column": change.column,
                    "value": change.value,
                }
                for change in self.values
            ],
            "nulls": [
                {
                    "source": change.source,
                    "row": change.row,
                    "column": change.column,
                    "is_null": change.is_null,
                }
                for change in self.nulls
            ],
            "rules": [
                {"rule": rule_id, "boundaries": list(self.rule_updates[rule_id])}
                for rule_id in sorted(self.rule_updates)
            ],
            "delete_rules": list(self.delete_rules),
        }


def _validate_boundaries(value: Any, path: str) -> tuple[int, ...]:
    raw = _array(value, path)
    if not raw:
        fail(E_SPEC, path, "at least one integer boundary is required")
    boundaries: list[int] = []
    previous: int | None = None
    for idx, boundary in enumerate(raw):
        if not _is_int(boundary):
            fail(E_SPEC, f"{path}[{idx}]", "bucket boundary must be an integer")
        if previous is not None and boundary <= previous:
            fail(
                E_SPEC,
                f"{path}[{idx}]",
                "boundaries must be strictly increasing and unique",
                previous=previous,
                value=boundary,
            )
        boundaries.append(boundary)
        previous = boundary
    return tuple(boundaries)


def parse_changes(raw: Any, spec: Spec) -> ChangeSet:
    obj = _mapping(raw, "$changes")
    for key in obj:
        if key not in ("values", "nulls", "rules", "delete_rules"):
            fail(E_SPEC, f"$changes.{key}", "unknown field")

    values_raw = _array(obj.get("values", []), "$changes.values")
    values: list[ValueChange] = []
    seen_overrides: dict[tuple[str, int, str], str] = {}
    for idx, item in enumerate(values_raw):
        path = f"$changes.values[{idx}]"
        change = _mapping(item, path)
        _keys(change, path, ("source", "row", "column", "value"))
        source_id = change["source"]
        row = change["row"]
        column = change["column"]
        if not isinstance(source_id, str) or spec.source(source_id) is None:
            fail(E_SPEC, f"{path}.source", "source does not exist")
        source = spec.source(source_id)
        assert source is not None
        if not _is_int(row) or row < 0 or row >= source.rows:
            fail(E_SPEC, f"{path}.row", "row index must be within the source")
        if not isinstance(column, str) or source.column(column) is None:
            fail(E_SPEC, f"{path}.column", "column does not exist")
        key = (source_id, row, column)
        if key in seen_overrides:
            fail(E_SPEC, f"{path}.column", "cell has more than one value change")
        seen_overrides[key] = path
        values.append(ValueChange(source_id, row, column, change["value"], idx))

    nulls_raw = _array(obj.get("nulls", []), "$changes.nulls")
    nulls: list[NullChange] = []
    for idx, item in enumerate(nulls_raw):
        path = f"$changes.nulls[{idx}]"
        change = _mapping(item, path)
        _keys(change, path, ("source", "row", "column", "is_null"))
        source_id = change["source"]
        row = change["row"]
        column = change["column"]
        is_null = change["is_null"]
        if not isinstance(source_id, str) or spec.source(source_id) is None:
            fail(E_SPEC, f"{path}.source", "source does not exist")
        source = spec.source(source_id)
        assert source is not None
        if not _is_int(row) or row < 0 or row >= source.rows:
            fail(E_SPEC, f"{path}.row", "row index must be within the source")
        if not isinstance(column, str) or source.column(column) is None:
            fail(E_SPEC, f"{path}.column", "column does not exist")
        if not isinstance(is_null, bool):
            fail(E_SPEC, f"{path}.is_null", "is_null must be a boolean")
        key = (source_id, row, column)
        if key in seen_overrides:
            fail(E_SPEC, f"{path}.column", "the same cell has both value and null changes")
        seen_overrides[key] = path
        nulls.append(NullChange(source_id, row, column, is_null, idx))

    rules_raw = _array(obj.get("rules", []), "$changes.rules")
    rule_updates: dict[str, tuple[int, ...]] = {}
    for idx, item in enumerate(rules_raw):
        path = f"$changes.rules[{idx}]"
        change = _mapping(item, path)
        _keys(change, path, ("rule", "boundaries"))
        rule_id = change["rule"]
        if not isinstance(rule_id, str) or spec.rule(rule_id) is None:
            fail(E_SPEC, f"{path}.rule", "bucket rule does not exist")
        if rule_id in rule_updates:
            fail(E_SPEC, f"{path}.rule", "rule has more than one boundary change")
        rule_updates[rule_id] = _validate_boundaries(change["boundaries"], f"{path}.boundaries")

    deletes_raw = obj.get("delete_rules", [])
    deletes_list = _array(deletes_raw, "$changes.delete_rules")
    delete_rules: list[str] = []
    seen_deletes: set[str] = set()
    for idx, rule_id in enumerate(deletes_list):
        if not isinstance(rule_id, str) or spec.rule(rule_id) is None:
            fail(E_SPEC, f"$changes.delete_rules[{idx}]", "bucket rule does not exist")
        if rule_id in seen_deletes:
            fail(E_SPEC, f"$changes.delete_rules[{idx}]", "duplicate deleted rule")
        if rule_id in rule_updates:
            fail(
                E_SPEC,
                f"$changes.delete_rules[{idx}]",
                "a rule cannot be updated and deleted in the same change set",
            )
        seen_deletes.add(rule_id)
        delete_rules.append(rule_id)
    return ChangeSet(
        values=tuple(sorted(values, key=lambda item: (item.source, item.row, item.column))),
        nulls=tuple(sorted(nulls, key=lambda item: (item.source, item.row, item.column))),
        rule_updates={key: rule_updates[key] for key in sorted(rule_updates)},
        delete_rules=tuple(sorted(delete_rules)),
    )


def load_changes(path: str | Path, spec: Spec) -> ChangeSet:
    change_path = Path(path)
    try:
        def reject_constant(value: str) -> Any:
            fail(
                E_SPEC,
                "$changes",
                "non-finite JSON numbers are not allowed in a change set",
                value=value,
            )

        raw = json.loads(
            change_path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except OSError as exc:
        fail(E_SPEC, "$changes", "cannot read change set", file=str(change_path), error=str(exc))
    except json.JSONDecodeError as exc:
        fail(
            E_SPEC,
            "$changes",
            "invalid JSON change set",
            line=exc.lineno,
            column=exc.colno,
            reason=exc.msg,
        )
    return parse_changes(raw, spec)
