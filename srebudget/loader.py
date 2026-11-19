"""Strict local JSON/NDJSON loading with stable field paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .errors import chain_error, trace_error
from .models import (
    BudgetPolicy,
    ChainSpec,
    CircuitPolicy,
    Outcome,
    ServiceSpec,
    TraceCall,
    TraceStream,
    canonical_json,
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def _required(obj: dict[str, Any], key: str, path: str) -> Any:
    if key not in obj:
        raise chain_error("缺少必填字段", f"{path}.{key}")
    return obj[key]


def _read_int(
    obj: dict[str, Any],
    key: str,
    path: str,
    *,
    positive: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = _required(obj, key, path)
    if not _is_int(value):
        raise chain_error("字段必须是整数", f"{path}.{key}")
    if positive and value <= 0:
        raise chain_error("字段必须是正整数", f"{path}.{key}")
    if minimum is not None and value < minimum:
        raise chain_error(f"字段不得小于 {minimum}", f"{path}.{key}")
    if maximum is not None and value > maximum:
        raise chain_error(f"字段不得大于 {maximum}", f"{path}.{key}")
    return value


def _nested_or_flat(
    node: dict[str, Any], nested_name: str, key: str, path: str
) -> tuple[dict[str, Any], str]:
    nested = node.get(nested_name)
    if nested is not None:
        if not isinstance(nested, dict):
            raise chain_error("字段必须是对象", f"{path}.{nested_name}")
        return nested, f"{path}.{nested_name}"
    if any(name in node for name in (key,)):
        return node, path
    raise chain_error("缺少必填字段", f"{path}.{nested_name}")


def load_chain_data(data: Any) -> ChainSpec:
    if not isinstance(data, dict):
        raise chain_error("服务链顶层必须是对象", "$")

    version = data.get("version", data.get("chain_version"))
    if not isinstance(version, str) or not version.strip():
        raise chain_error("version 必须是非空字符串", "$.version")

    raw_services = data.get("services")
    if not isinstance(raw_services, list):
        raise chain_error("services 必须是数组", "$.services")
    if not raw_services:
        raise chain_error("services 不得为空", "$.services")

    services: list[ServiceSpec] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(raw_services):
        path = f"$.services[{index}]"
        if not isinstance(raw, dict):
            raise chain_error("服务节点必须是对象", path)

        service_id = _required(raw, "id", path)
        if not isinstance(service_id, str) or not service_id.strip():
            raise chain_error("id 必须是非空字符串", f"{path}.id")
        if service_id in seen:
            raise chain_error(
                "服务标识重复",
                f"{path}.id",
                duplicate=service_id,
                first_index=seen[service_id],
            )
        seen[service_id] = index

        upstream = _required(raw, "upstream", path)
        if not isinstance(upstream, list):
            raise chain_error("upstream 必须是数组", f"{path}.upstream")
        upstream_values: list[str] = []
        for up_index, upstream_id in enumerate(upstream):
            up_path = f"{path}.upstream[{up_index}]"
            if not isinstance(upstream_id, str) or not upstream_id.strip():
                raise chain_error("上游服务标识必须是非空字符串", up_path)
            if upstream_id in upstream_values:
                raise chain_error("上游服务标识重复", up_path)
            upstream_values.append(upstream_id)

        timeout_ms = _read_int(raw, "timeout_ms", path, positive=True)

        budget_obj, budget_path = _nested_or_flat(raw, "budget", "window_hours", path)
        window_hours = _read_int(
            budget_obj, "window_hours", budget_path, positive=True
        )
        ratio_bp = _read_int(
            budget_obj,
            "ratio_bp",
            budget_path,
            minimum=0,
            maximum=10_000,
        )
        min_samples = _read_int(
            budget_obj, "min_samples", budget_path, positive=True
        )

        circuit_obj, circuit_path = _nested_or_flat(
            raw, "circuit", "failure_threshold", path
        )
        failure_threshold = _read_int(
            circuit_obj, "failure_threshold", circuit_path, positive=True
        )
        cooldown_ms = _read_int(
            circuit_obj, "cooldown_ms", circuit_path, positive=True
        )
        half_open_probes = _read_int(
            circuit_obj, "half_open_probes", circuit_path, positive=True
        )

        services.append(
            ServiceSpec(
                id=service_id,
                upstream=tuple(upstream_values),
                timeout_ms=timeout_ms,
                budget=BudgetPolicy(window_hours, ratio_bp, min_samples),
                circuit=CircuitPolicy(
                    failure_threshold, cooldown_ms, half_open_probes
                ),
            )
        )

    for index, service in enumerate(services):
        for up_index, upstream_id in enumerate(service.upstream):
            if upstream_id not in seen:
                raise chain_error(
                    "上游引用不存在",
                    f"$.services[{index}].upstream[{up_index}]",
                    service=service.id,
                    reference=upstream_id,
                )

    _validate_acyclic(services)
    return ChainSpec(version=version, services=tuple(services))


def _validate_acyclic(services: list[ServiceSpec]) -> None:
    graph = {service.id: tuple(service.upstream) for service in services}
    state: dict[str, int] = {service.id: 0 for service in services}
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for dependency in graph[node]:
            if state[dependency] == 0:
                visit(dependency)
            elif state[dependency] == 1:
                cycle_start = stack.index(dependency)
                cycle = stack[cycle_start:] + [dependency]
                raise chain_error(
                    "服务链存在环",
                    "$.services.upstream",
                    cycle=cycle,
                )
        stack.pop()
        state[node] = 2

    for service in sorted(services, key=lambda item: item.id):
        if state[service.id] == 0:
            visit(service.id)


def load_chain(path: str | Path) -> ChainSpec:
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise chain_error("无法读取服务链文件", "$", reason=str(exc)) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise chain_error(
            "服务链 JSON 语法错误",
            "$",
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    return load_chain_data(data)


def load_trace_lines(
    lines: Iterable[str], chain: ChainSpec, *, content_digest: str | None = None
) -> TraceStream:
    known = set(chain.service_ids)
    calls: list[TraceCall] = []
    previous_timestamp: int | None = None
    raw_hasher = hashlib.sha256()
    saw_content = False

    for line_number, raw_line in enumerate(lines, start=1):
        raw_hasher.update(raw_line.encode("utf-8"))
        saw_content = saw_content or bool(raw_line)
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise trace_error(
                "调用行不是合法 JSON",
                line_number,
                "$",
                column=exc.colno,
            ) from exc
        if not isinstance(item, dict):
            raise trace_error("调用行必须是对象", line_number, "$")

        timestamp = item.get("timestamp_ms")
        if not _is_int(timestamp) or timestamp < 0:
            raise trace_error(
                "timestamp_ms 必须是非负整数",
                line_number,
                "$.timestamp_ms",
            )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise trace_error(
                "调用时间戳倒退",
                line_number,
                "$.timestamp_ms",
                previous=previous_timestamp,
                actual=timestamp,
            )
        previous_timestamp = timestamp

        path = item.get("path")
        if not isinstance(path, list) or not path:
            raise trace_error(
                "path 必须是非空服务标识数组", line_number, "$.path"
            )
        path_values: list[str] = []
        for path_index, service_id in enumerate(path):
            field = f"$.path[{path_index}]"
            if not isinstance(service_id, str) or not service_id:
                raise trace_error(
                    "path 中的服务标识必须是非空字符串",
                    line_number,
                    field,
                )
            if service_id not in known:
                raise trace_error(
                    "path 引用了未声明的服务",
                    line_number,
                    field,
                    service=service_id,
                )
            if service_id in path_values:
                raise trace_error(
                    "path 中服务标识不得重复",
                    line_number,
                    field,
                    service=service_id,
                )
            path_values.append(service_id)

        outcome_raw = item.get("outcome")
        if not isinstance(outcome_raw, str):
            raise trace_error(
                "outcome 必须是受支持的字符串", line_number, "$.outcome"
            )
        try:
            outcome = Outcome(outcome_raw)
        except ValueError as exc:
            raise trace_error(
                "outcome 必须是 success、failure 或 timeout",
                line_number,
                "$.outcome",
                actual=outcome_raw,
            ) from exc

        duration_ms = item.get("duration_ms")
        if not _is_int(duration_ms) or duration_ms < 0:
            raise trace_error(
                "duration_ms 必须是非负整数",
                line_number,
                "$.duration_ms",
            )

        calls.append(
            TraceCall(
                index=len(calls) + 1,
                timestamp_ms=timestamp,
                path=tuple(path_values),
                outcome=outcome,
                duration_ms=duration_ms,
                source_line=line_number,
            )
        )

    digest = content_digest or raw_hasher.hexdigest()
    if content_digest is None and not saw_content:
        digest = hashlib.sha256(b"").hexdigest()
    return TraceStream(calls=tuple(calls), content_digest=digest)


def load_trace(path: str | Path, chain: ChainSpec) -> TraceStream:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise trace_error("无法读取调用流文件", 0, "$", reason=str(exc)) from exc
    lines = text.splitlines(keepends=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return load_trace_lines(lines, chain, content_digest=digest)


def trace_digest(calls: Iterable[TraceCall]) -> str:
    return hashlib.sha256(
        canonical_json([call.as_dict() for call in calls]).encode("utf-8")
    ).hexdigest()
