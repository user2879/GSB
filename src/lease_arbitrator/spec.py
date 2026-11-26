from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import SpecError


EVENT_TYPES = {
    "apply",
    "renew",
    "release",
    "expire",
    "upgrade_quota",
    "upsert_node",
    "remove_node",
    "delete_lease",
}


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json_file(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read declaration: {exc}", path=str(path)) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            path=f"{path}:$.line[{exc.lineno}].col[{exc.colno}]",
        ) from exc


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("expected JSON object", path=path)
    return value


def _require_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if not is_int(value):
        raise SpecError("expected integer", path=path)
    if minimum is not None and value < minimum:
        raise SpecError(f"expected integer >= {minimum}", path=path)
    return value


def _require_nonnegative_int(value: Any, path: str) -> int:
    return _require_int(value, path, minimum=0)


def _require_positive_int(value: Any, path: str) -> int:
    return _require_int(value, path, minimum=1)


def _require_nonempty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError("expected non-empty string", path=path)
    return value


def _validate_node(node: Any, path: str, *, partial: bool = False) -> dict[str, Any]:
    node_obj = _require_object(node, path)
    node_id = _require_nonempty_str(node_obj.get("node_id"), f"{path}.node_id")
    host = _require_nonempty_str(node_obj.get("host"), f"{path}.host")
    if host != "127.0.0.1":
        raise SpecError("host must be the literal 127.0.0.1", path=f"{path}.host")
    port = _require_int(node_obj.get("port"), f"{path}.port", minimum=1)
    if port > 65535:
        raise SpecError("port must be <= 65535", path=f"{path}.port")
    quota = _require_nonnegative_int(node_obj.get("quota"), f"{path}.quota")
    max_active = _require_nonnegative_int(node_obj.get("max_active"), f"{path}.max_active")
    seed = _require_int(node_obj.get("seed"), f"{path}.seed")
    quota_version = _require_positive_int(
        node_obj.get("quota_version"), f"{path}.quota_version"
    )
    result = {
        "node_id": node_id,
        "host": host,
        "port": port,
        "quota": quota,
        "max_active": max_active,
        "seed": seed,
        "quota_version": quota_version,
    }
    if partial:
        result["_partial"] = True
    return result


def _policy_node_map(raw: Any, path: str, known_nodes: set[str]) -> dict[str, dict[str, int]]:
    if isinstance(raw, dict):
        items: list[tuple[str, Any]] = list(raw.items())
        normalized: dict[str, dict[str, int]] = {}
        for key, value in items:
            item_path = f"{path}.{key}"
            obj = _require_object(value, item_path)
            quota = _require_nonnegative_int(obj.get("quota"), f"{item_path}.quota")
            max_active = _require_nonnegative_int(
                obj.get("max_active"), f"{item_path}.max_active"
            )
            if key not in known_nodes:
                raise SpecError("quota policy references unknown node", path=item_path)
            normalized[key] = {"quota": quota, "max_active": max_active}
        return normalized
    if isinstance(raw, list):
        normalized = {}
        for index, item in enumerate(raw):
            item_path = f"{path}[{index}]"
            obj = _require_object(item, item_path)
            node_id = _require_nonempty_str(obj.get("node_id"), f"{item_path}.node_id")
            if node_id not in known_nodes:
                raise SpecError("quota policy references unknown node", path=f"{item_path}.node_id")
            quota = _require_nonnegative_int(obj.get("quota"), f"{item_path}.quota")
            max_active = _require_nonnegative_int(
                obj.get("max_active"), f"{item_path}.max_active"
            )
            normalized[node_id] = {"quota": quota, "max_active": max_active}
        return normalized
    raise SpecError("expected object or array", path=path)


def validate_spec(raw: Any) -> dict[str, Any]:
    spec = _require_object(raw, "$")

    format_version = spec.get("format_version", 1)
    _require_positive_int(format_version, "$.format_version")

    scale = spec.get("scale", 0)
    if not is_int(scale) or scale not in (0, 1, 2, 3):
        raise SpecError("scale must be integer 0, 1, 2, or 3", path="$.scale")

    priority_range = spec.get("priority_range", [0, 100])
    if not isinstance(priority_range, list) or len(priority_range) != 2:
        raise SpecError("priority_range must be [min, max]", path="$.priority_range")
    priority_min = _require_int(priority_range[0], "$.priority_range[0]")
    priority_max = _require_int(priority_range[1], "$.priority_range[1]")
    if priority_min > priority_max:
        raise SpecError("priority_range min must be <= max", path="$.priority_range")

    global_budget = _require_nonnegative_int(spec.get("global_budget"), "$.global_budget")

    nodes_raw = spec.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise SpecError("nodes must be a non-empty array", path="$.nodes")
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    port_keys: set[tuple[str, int]] = set()
    for index, raw_node in enumerate(nodes_raw):
        path = f"$.nodes[{index}]"
        node = _validate_node(raw_node, path)
        if node["node_id"] in node_ids:
            raise SpecError("duplicate node_id", path=f"{path}.node_id")
        port_key = (node["host"], node["port"])
        if port_key in port_keys:
            raise SpecError("duplicate 127.0.0.1 port", path=f"{path}.port")
        node_ids.add(node["node_id"])
        port_keys.add(port_key)
        nodes.append(node)

    leases_raw = spec.get("leases")
    if not isinstance(leases_raw, list):
        raise SpecError("leases must be an array", path="$.leases")
    leases: list[dict[str, Any]] = []
    lease_ids: set[str] = set()
    for index, raw_lease in enumerate(leases_raw):
        path = f"$.leases[{index}]"
        lease = _require_object(raw_lease, path)
        lease_id = _require_nonempty_str(lease.get("lease_id"), f"{path}.lease_id")
        if lease_id in lease_ids:
            raise SpecError("duplicate lease_id", path=f"{path}.lease_id")
        node_id = _require_nonempty_str(lease.get("node_id"), f"{path}.node_id")
        if node_id not in node_ids:
            raise SpecError("lease references unknown node", path=f"{path}.node_id")
        units = _require_positive_int(lease.get("units"), f"{path}.units")
        priority = _require_int(lease.get("priority"), f"{path}.priority")
        if not (priority_min <= priority <= priority_max):
            raise SpecError(
                f"priority must be in declared interval [{priority_min}, {priority_max}]",
                path=f"{path}.priority",
            )
        hold_ms = _require_positive_int(lease.get("hold_ms"), f"{path}.hold_ms")
        quota_version = lease.get("quota_version", 1)
        _require_positive_int(quota_version, f"{path}.quota_version")
        references_raw = lease.get("references", [])
        if not isinstance(references_raw, list):
            raise SpecError("references must be an array", path=f"{path}.references")
        references = [
            _require_nonempty_str(item, f"{path}.references[{i}]")
            for i, item in enumerate(references_raw)
        ]
        for ref in references:
            if ref == lease_id:
                raise SpecError("lease cannot reference itself", path=f"{path}.references")
        lease_ids.add(lease_id)
        leases.append(
            {
                "lease_id": lease_id,
                "node_id": node_id,
                "units": units,
                "priority": priority,
                "hold_ms": hold_ms,
                "quota_version": quota_version,
                "references": references,
            }
        )

    for lease in leases:
        for ref in lease["references"]:
            if ref not in lease_ids:
                raise SpecError(
                    "lease reference points to unknown lease",
                    path=f"$.leases[?lease_id={lease['lease_id']}].references",
                )

    quota_versions_raw = spec.get("quota_versions", [])
    if not isinstance(quota_versions_raw, list):
        raise SpecError("quota_versions must be an array", path="$.quota_versions")
    quota_versions: list[dict[str, Any]] = []
    previous_version = 0
    for index, raw_version in enumerate(quota_versions_raw):
        path = f"$.quota_versions[{index}]"
        item = _require_object(raw_version, path)
        version = _require_positive_int(item.get("quota_version"), f"{path}.quota_version")
        if version < previous_version:
            raise SpecError(
                "quota_version sequence must be monotonically non-decreasing",
                path=f"{path}.quota_version",
            )
        previous_version = version
        entry: dict[str, Any] = {"quota_version": version}
        if "global_budget" in item:
            entry["global_budget"] = _require_nonnegative_int(
                item["global_budget"], f"{path}.global_budget"
            )
        if "nodes" in item:
            entry["nodes"] = _policy_node_map(item["nodes"], f"{path}.nodes", node_ids)
        quota_versions.append(entry)

    replay_horizon_ms = spec.get("replay_horizon_ms", 0)
    _require_nonnegative_int(replay_horizon_ms, "$.replay_horizon_ms")
    node_timeout_seconds = spec.get("node_timeout_seconds", 2.0)
    if (
        not isinstance(node_timeout_seconds, (int, float))
        or isinstance(node_timeout_seconds, bool)
        or node_timeout_seconds <= 0
    ):
        raise SpecError("node_timeout_seconds must be a positive number", path="$.node_timeout_seconds")

    result = dict(spec)
    result.update(
        {
            "format_version": format_version,
            "scale": scale,
            "priority_range": [priority_min, priority_max],
            "global_budget": global_budget,
            "nodes": nodes,
            "leases": leases,
            "quota_versions": quota_versions,
            "replay_horizon_ms": replay_horizon_ms,
            "node_timeout_seconds": float(node_timeout_seconds),
        }
    )
    return result


def load_spec(path: str | Path) -> dict[str, Any]:
    return validate_spec(load_json_file(path))


def validate_event(event: Any, path: str, spec: dict[str, Any], seen_seqs: set[int]) -> dict[str, Any]:
    obj = _require_object(event, path)
    seq = _require_positive_int(obj.get("seq"), f"{path}.seq")
    if seq in seen_seqs:
        raise SpecError("duplicate event seq", path=f"{path}.seq")
    seen_seqs.add(seq)
    event_type = _require_nonempty_str(obj.get("type"), f"{path}.type")
    if event_type not in EVENT_TYPES:
        raise SpecError(f"unsupported event type {event_type}", path=f"{path}.type")
    arrival_ms = obj.get("arrival_ms", seq - 1)
    _require_nonnegative_int(arrival_ms, f"{path}.arrival_ms")
    normalized: dict[str, Any] = {"seq": seq, "type": event_type, "arrival_ms": arrival_ms}

    lease_ids = {lease["lease_id"] for lease in spec["leases"]}
    if event_type in {"apply", "renew", "release", "expire", "delete_lease"}:
        lease_id = _require_nonempty_str(obj.get("lease_id"), f"{path}.lease_id")
        if lease_id not in lease_ids:
            raise SpecError("event references unknown lease_id", path=f"{path}.lease_id")
        normalized["lease_id"] = lease_id

    if event_type in {"renew", "release", "expire"} and "generation" in obj:
        generation = _require_nonnegative_int(obj.get("generation"), f"{path}.generation")
        normalized["generation"] = generation

    if event_type == "upgrade_quota":
        version = _require_positive_int(obj.get("quota_version"), f"{path}.quota_version")
        known_versions = {item["quota_version"] for item in spec.get("quota_versions", [])}
        if known_versions and version not in known_versions:
            raise SpecError(
                "upgrade references undeclared quota_version",
                path=f"{path}.quota_version",
            )
        normalized["quota_version"] = version

    if event_type in {"upsert_node", "remove_node"}:
        if event_type == "upsert_node":
            node = _validate_node(obj.get("node"), f"{path}.node")
            normalized["node"] = node
        else:
            normalized["node_id"] = _require_nonempty_str(
                obj.get("node_id"), f"{path}.node_id"
            )

    if "expect_error" in obj:
        expected = obj["expect_error"]
        if expected is not None and not isinstance(expected, str):
            raise SpecError("expect_error must be a string or null", path=f"{path}.expect_error")
        normalized["expect_error"] = expected
    if "note" in obj:
        normalized["note"] = str(obj["note"])
    return normalized


def load_events(path: str | Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SpecError(f"cannot read event stream: {exc}", path=str(path)) from exc
    events: list[dict[str, Any]] = []
    seen_seqs: set[int] = set()
    last_seq = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid NDJSON at line {line_number}: {exc.msg}",
                path=f"{path}:$.events[{line_number - 1}]",
            ) from exc
        event_path = f"$.events[{line_number - 1}]"
        event = validate_event(raw, event_path, spec, seen_seqs)
        if event["seq"] <= last_seq:
            raise SpecError("events must be ordered by strictly increasing seq", path=f"{event_path}.seq")
        last_seq = event["seq"]
        events.append(event)
    if not events:
        raise SpecError("event stream must contain at least one event", path=str(path))
    return events


def load_bundle(spec_path: str | Path, events_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = load_spec(spec_path)
    events = load_events(events_path, spec)
    return spec, events
