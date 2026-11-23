"""Strict, deterministic JSON and NDJSON loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import DeclError
from .models import (
    CallEvent,
    Declaration,
    FunctionDecl,
    PreemptionRelation,
    SessionConfig,
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeclError(field, "must be a JSON object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DeclError(field, "must be a JSON array")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeclError(field, "must be a non-empty string")
    return value


def _require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not _is_int(value):
        raise DeclError(field, "must be an integer (booleans are not integers)")
    if minimum is not None and value < minimum:
        raise DeclError(field, f"must be >= {minimum}", actual=value)
    if maximum is not None and value > maximum:
        raise DeclError(field, f"must be <= {maximum}", actual=value)
    return value


def _reject_unknown_keys(obj: dict[str, Any], allowed: set[str], field: str) -> None:
    for key in sorted(obj):
        if key not in allowed:
            raise DeclError(f"{field}.{key}", "unknown field")


def _read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise DeclError("declaration", f"file not found: {source}", path=str(source)) from exc
    except json.JSONDecodeError as exc:
        raise DeclError(
            "declaration",
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
            path=str(source),
        ) from exc
    except OSError as exc:
        raise DeclError("declaration", f"cannot read file: {exc}", path=str(source)) from exc


def _parse_function(raw: Any, index: int) -> dict[str, Any]:
    field = f"functions[{index}]"
    obj = _require_object(raw, field)
    required = {
        "function_id",
        "memory_mib",
        "instance_concurrency",
        "lease_ms",
        "min_concurrency",
        "max_concurrency",
    }
    allowed = required | {"priority"}
    _reject_unknown_keys(obj, allowed, field)
    for key in (
        "function_id",
        "memory_mib",
        "instance_concurrency",
        "lease_ms",
        "min_concurrency",
        "max_concurrency",
    ):
        if key not in obj:
            raise DeclError(f"{field}.{key}", "required field is missing")
    function_id = _require_str(obj["function_id"], f"{field}.function_id")
    memory_mib = _require_int(obj["memory_mib"], f"{field}.memory_mib", minimum=1)
    instance_concurrency = _require_int(
        obj["instance_concurrency"], f"{field}.instance_concurrency", minimum=1
    )
    lease_ms = _require_int(obj["lease_ms"], f"{field}.lease_ms", minimum=1)
    min_concurrency = _require_int(
        obj["min_concurrency"], f"{field}.min_concurrency", minimum=0
    )
    max_concurrency = _require_int(
        obj["max_concurrency"], f"{field}.max_concurrency", minimum=0
    )
    if min_concurrency > max_concurrency:
        raise DeclError(
            f"{field}.min_concurrency",
            "must not exceed max_concurrency",
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
        )
    priority_raw = obj.get("priority")
    priority: int | None = None
    if priority_raw is not None:
        priority = _require_int(priority_raw, f"{field}.priority", minimum=0)
    return {
        "function_id": function_id,
        "memory_mib": memory_mib,
        "instance_concurrency": instance_concurrency,
        "lease_ms": lease_ms,
        "min_concurrency": min_concurrency,
        "max_concurrency": max_concurrency,
        "priority": priority,
        "field": field,
    }


def _parse_relation(raw: Any, index: int) -> dict[str, Any]:
    field = f"preemptions[{index}]"
    obj = _require_object(raw, field)
    required = {"higher", "lower", "return_deadline_ms"}
    _reject_unknown_keys(obj, required, field)
    for key in ("higher", "lower", "return_deadline_ms"):
        if key not in obj:
            raise DeclError(f"{field}.{key}", "required field is missing")
    return {
        "higher": _require_str(obj["higher"], f"{field}.higher"),
        "lower": _require_str(obj["lower"], f"{field}.lower"),
        "return_deadline_ms": _require_int(
            obj["return_deadline_ms"], f"{field}.return_deadline_ms", minimum=0
        ),
        "field": field,
    }


def _detect_cycle(
    function_ids: Iterable[str], edges: Iterable[PreemptionRelation]
) -> tuple[str, ...] | None:
    adjacency: dict[str, list[str]] = {function_id: [] for function_id in function_ids}
    for edge in edges:
        adjacency[edge.higher].append(edge.lower)
    for values in adjacency.values():
        values.sort()
    state: dict[str, int] = {key: 0 for key in adjacency}
    stack: list[str] = []

    def visit(node: str) -> tuple[str, ...] | None:
        state[node] = 1
        stack.append(node)
        for neighbour in adjacency[node]:
            if state[neighbour] == 0:
                found = visit(neighbour)
                if found is not None:
                    return found
            elif state[neighbour] == 1:
                start = stack.index(neighbour)
                return tuple(stack[start:] + [neighbour])
        stack.pop()
        state[node] = 2
        return None

    for function_id in sorted(adjacency):
        if state[function_id] == 0:
            found = visit(function_id)
            if found is not None:
                return found
    return None


def _derive_priorities(raw_functions: list[dict[str, Any]], edges: list[PreemptionRelation]) -> dict[str, int]:
    present = [item["priority"] is not None for item in raw_functions]
    if any(present) and not all(present):
        for item, has_priority in zip(raw_functions, present):
            if not has_priority:
                raise DeclError(
                    f"{item['field']}.priority",
                    "priority must be supplied for every function or omitted for every function",
                )
    ids = [item["function_id"] for item in raw_functions]
    if all(present):
        return {item["function_id"]: int(item["priority"]) for item in raw_functions}

    adjacency: dict[str, list[str]] = {function_id: [] for function_id in ids}
    for edge in edges:
        adjacency[edge.higher].append(edge.lower)
    for values in adjacency.values():
        values.sort()
    memo: dict[str, int] = {function_id: 0 for function_id in ids}

    def rank(node: str) -> int:
        if adjacency[node]:
            memo[node] = 1 + max(rank(child) for child in adjacency[node])
        return memo[node]

    for function_id in sorted(ids):
        rank(function_id)
    return memo


def load_declaration(path: str | Path) -> Declaration:
    return parse_declaration(_read_json(path))


def parse_declaration(value: Any) -> Declaration:
    raw = _require_object(value, "declaration")
    allowed = {
        "schema_version",
        "declaration_version",
        "global_concurrency",
        "preemption_checkpoint_ms",
        "functions",
        "preemptions",
        "description",
    }
    _reject_unknown_keys(raw, allowed, "declaration")
    for key in ("declaration_version", "global_concurrency", "functions", "preemptions"):
        if key not in raw:
            raise DeclError(f"declaration.{key}", "required field is missing")
    schema_version = _require_int(
        raw.get("schema_version", 1), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise DeclError("schema_version", "unsupported schema version", supported=[1])
    declaration_version = _require_int(
        raw["declaration_version"], "declaration_version", minimum=1
    )
    global_concurrency = _require_int(
        raw["global_concurrency"], "global_concurrency", minimum=0
    )
    checkpoint_ms = _require_int(
        raw.get("preemption_checkpoint_ms", 100),
        "preemption_checkpoint_ms",
        minimum=1,
    )
    if "description" in raw and not isinstance(raw["description"], str):
        raise DeclError("description", "must be a string")

    raw_functions = _require_list(raw["functions"], "functions")
    if not raw_functions:
        raise DeclError("functions", "must contain at least one function")
    parsed_functions = [_parse_function(item, index) for index, item in enumerate(raw_functions)]
    seen: set[str] = set()
    for item in parsed_functions:
        function_id = item["function_id"]
        if function_id in seen:
            raise DeclError(
                f"{item['field']}.function_id",
                "duplicate function identifier",
                function_id=function_id,
            )
        seen.add(function_id)

    raw_relations = _require_list(raw["preemptions"], "preemptions")
    parsed_relations = [_parse_relation(item, index) for index, item in enumerate(raw_relations)]
    relation_keys: set[tuple[str, str]] = set()
    relations: list[PreemptionRelation] = []
    for item in parsed_relations:
        field = item["field"]
        higher = item["higher"]
        lower = item["lower"]
        if higher not in seen:
            raise DeclError(f"{field}.higher", "unknown function reference", function_id=higher)
        if lower not in seen:
            raise DeclError(f"{field}.lower", "unknown function reference", function_id=lower)
        if higher == lower:
            raise DeclError(f"{field}.lower", "self preemption edge is not allowed")
        key = (higher, lower)
        if key in relation_keys:
            raise DeclError(f"{field}.lower", "duplicate preemption relation", relation=key)
        relation_keys.add(key)
        relations.append(
            PreemptionRelation(
                higher=higher,
                lower=lower,
                return_deadline_ms=item["return_deadline_ms"],
            )
        )

    cycle = _detect_cycle(seen, relations)
    if cycle is not None:
        raise DeclError(
            "preemptions",
            "preemption graph contains a cycle",
            cycle=list(cycle),
        )
    priorities = _derive_priorities(parsed_functions, relations)
    functions: dict[str, FunctionDecl] = {}
    for item in parsed_functions:
        function_id = item["function_id"]
        functions[function_id] = FunctionDecl(
            function_id=function_id,
            memory_mib=item["memory_mib"],
            instance_concurrency=item["instance_concurrency"],
            lease_ms=item["lease_ms"],
            min_concurrency=item["min_concurrency"],
            max_concurrency=item["max_concurrency"],
            priority=priorities[function_id],
        )
    for index, relation in enumerate(relations):
        higher_priority = functions[relation.higher].priority
        lower_priority = functions[relation.lower].priority
        if higher_priority <= lower_priority:
            raise DeclError(
                f"preemptions[{index}]",
                "higher endpoint must have strictly greater priority",
                higher=relation.higher,
                lower=relation.lower,
                higher_priority=higher_priority,
                lower_priority=lower_priority,
            )

    declaration = Declaration(
        declaration_version=declaration_version,
        global_concurrency=global_concurrency,
        functions=functions,
        preemptions=tuple(
            sorted(relations, key=lambda edge: (edge.higher, edge.lower))
        ),
        preemption_checkpoint_ms=checkpoint_ms,
        source=raw,
    )
    return declaration


def load_events(path: str | Path, declaration: Declaration) -> list[CallEvent]:
    source = Path(path)
    events: list[CallEvent] = []
    previous_arrival: int | None = None
    try:
        handle = source.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise DeclError("events", f"file not found: {source}", path=str(source)) from exc
    except OSError as exc:
        raise DeclError("events", f"cannot read file: {exc}", path=str(source)) from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            field = f"events[{line_number}]"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DeclError(
                    field,
                    f"invalid NDJSON at column {exc.colno}: {exc.msg}",
                    line=line_number,
                ) from exc
            obj = _require_object(raw, field)
            allowed = {
                "event_seq",
                "arrival_ms",
                "function_id",
                "expected_occupancy_ms",
                "concurrency_demand",
            }
            _reject_unknown_keys(obj, allowed, field)
            for key in (
                "event_seq",
                "arrival_ms",
                "function_id",
                "expected_occupancy_ms",
                "concurrency_demand",
            ):
                if key not in obj:
                    raise DeclError(f"{field}.{key}", "required field is missing")
            event_seq = _require_int(obj["event_seq"], f"{field}.event_seq", minimum=1)
            expected_seq = len(events) + 1
            if event_seq != expected_seq:
                raise DeclError(
                    f"{field}.event_seq",
                    f"event sequence must be contiguous and start at 1; expected {expected_seq}",
                    actual=event_seq,
                )
            arrival_ms = _require_int(obj["arrival_ms"], f"{field}.arrival_ms", minimum=0)
            if previous_arrival is not None and arrival_ms < previous_arrival:
                raise DeclError(
                    f"{field}.arrival_ms",
                    "arrival_ms must be monotonically non-decreasing",
                    previous=previous_arrival,
                    actual=arrival_ms,
                )
            function_id = _require_str(obj["function_id"], f"{field}.function_id")
            if function_id not in declaration.functions:
                raise DeclError(
                    f"{field}.function_id",
                    "unknown function reference",
                    function_id=function_id,
                )
            occupancy = _require_int(
                obj["expected_occupancy_ms"],
                f"{field}.expected_occupancy_ms",
                minimum=0,
            )
            demand = _require_int(
                obj["concurrency_demand"],
                f"{field}.concurrency_demand",
                minimum=0,
            )
            events.append(
                CallEvent(
                    event_seq=event_seq,
                    arrival_ms=arrival_ms,
                    function_id=function_id,
                    expected_occupancy_ms=occupancy,
                    concurrency_demand=demand,
                )
            )
            previous_arrival = arrival_ms
    if not events:
        raise DeclError("events", "event stream must contain at least one event")
    return events


def load_session(
    watermark_scale: int = 1000, borrow_return_error_ms: int = 50
) -> SessionConfig:
    try:
        return SessionConfig(
            watermark_scale=watermark_scale,
            borrow_return_error_ms=borrow_return_error_ms,
        )
    except ValueError as exc:
        raise DeclError("watermark_scale", str(exc)) from exc
