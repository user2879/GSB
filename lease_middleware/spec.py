"""Loading and strict validation of local JSON specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError


EVENT_TYPES = {
    "grant",
    "renew",
    "release",
    "expire_scan",
    "node_generation_bump",
    "version_evolved",
}


@dataclass(frozen=True)
class SpecBundle:
    nodes_doc: dict[str, Any]
    requests_doc: dict[str, Any]
    events_doc: dict[str, Any]
    nodes_path: Path | None = None
    requests_path: Path | None = None
    events_path: Path | None = None

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return self.nodes_doc["nodes"]

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.requests_doc["requests"]

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.events_doc["events"]

    @property
    def node_ids(self) -> set[str]:
        return {node["node_id"] for node in self.nodes}

    def node_by_id(self, node_id: str) -> dict[str, Any]:
        for node in self.nodes:
            if node["node_id"] == node_id:
                return node
        raise KeyError(node_id)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{field} must be a JSON object", field=field)
    return value


def _require_nonempty_str(obj: dict[str, Any], key: str, field: str) -> str:
    if key not in obj:
        raise SpecError(f"missing required field {key!r}", field=f"{field}.{key}")
    value = obj[key]
    if not isinstance(value, str) or not value:
        raise SpecError(f"{field}.{key} must be a non-empty string", field=f"{field}.{key}")
    return value


def _require_positive_int(
    obj: dict[str, Any], key: str, field: str, *, default: int | None = None
) -> int:
    if key not in obj and default is not None:
        return default
    if key not in obj:
        raise SpecError(f"missing required field {key!r}", field=f"{field}.{key}")
    value = obj[key]
    if not _is_int(value) or value <= 0:
        raise SpecError(
            f"{field}.{key} must be a positive integer",
            field=f"{field}.{key}",
            details={"value": value},
        )
    return value


def _require_nonnegative_int(obj: dict[str, Any], key: str, field: str) -> int:
    if key not in obj:
        raise SpecError(f"missing required field {key!r}", field=f"{field}.{key}")
    value = obj[key]
    if not _is_int(value) or value < 0:
        raise SpecError(
            f"{field}.{key} must be a non-negative integer",
            field=f"{field}.{key}",
            details={"value": value},
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SpecError(f"duplicate JSON key {key!r}", field=f"$.{key}")
        result[key] = value
    return result


def load_json_file(path: str | Path) -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read JSON file: {exc}", field=str(path)) from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            field=f"{path}:{exc.lineno}:{exc.colno}",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc


def validate_nodes_doc(doc: Any) -> dict[str, Any]:
    root = _require_object(doc, "$")
    if "format_version" in root:
        version = root["format_version"]
        if not _is_int(version) or version <= 0:
            raise SpecError("format_version must be a positive integer", field="format_version")

    quotas = root.get("quotas", {})
    quotas = _require_object(quotas, "quotas")
    global_limit = _require_positive_int(quotas, "global_in_flight", "quotas")
    node_default = _require_positive_int(
        quotas, "node_in_flight_default", "quotas", default=global_limit
    )

    topology_version = _require_positive_int(root, "topology_version", "$", default=1)
    quota_version = _require_positive_int(root, "quota_version", "$", default=1)

    nodes_value = root.get("nodes")
    if not isinstance(nodes_value, list) or not nodes_value:
        raise SpecError("nodes must be a non-empty array", field="nodes")

    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    normalized_nodes: list[dict[str, Any]] = []

    for index, raw_node in enumerate(nodes_value):
        field = f"nodes[{index}]"
        node = _require_object(raw_node, field)
        node_id = _require_nonempty_str(node, "node_id", field)
        if node_id in seen_ids:
            raise SpecError("duplicate node_id", field=f"{field}.node_id")
        seen_ids.add(node_id)

        host = _require_nonempty_str(node, "host", field)
        if host != "127.0.0.1":
            raise SpecError(
                "host must be the literal loopback address 127.0.0.1",
                field=f"{field}.host",
                details={"host": host},
            )

        port = _require_positive_int(node, "port", field)
        if port > 65535:
            raise SpecError("port must be <= 65535", field=f"{field}.port")
        if port in seen_ports:
            raise SpecError("loopback port is duplicated", field=f"{field}.port")
        seen_ports.add(port)

        lease_limit = _require_positive_int(node, "lease_limit", field)
        in_flight_limit = _require_positive_int(
            node, "in_flight_limit", field, default=min(lease_limit, node_default)
        )
        if in_flight_limit > lease_limit:
            raise SpecError(
                "in_flight_limit must not exceed lease_limit in this version",
                field=f"{field}.in_flight_limit",
            )
        node_generation = _require_positive_int(node, "node_generation", field, default=1)
        processing_delay_ms = node.get("processing_delay_ms", 0)
        if not _is_int(processing_delay_ms) or processing_delay_ms < 0:
            raise SpecError(
                f"{field}.processing_delay_ms must be a non-negative integer",
                field=f"{field}.processing_delay_ms",
            )

        normalized = dict(node)
        normalized.update(
            {
                "node_id": node_id,
                "host": host,
                "port": port,
                "lease_limit": lease_limit,
                "in_flight_limit": in_flight_limit,
                "node_generation": node_generation,
                "processing_delay_ms": processing_delay_ms,
            }
        )
        normalized_nodes.append(normalized)

    normalized_root = dict(root)
    normalized_root.update(
        {
            "format_version": int(root.get("format_version", 1)),
            "quotas": {
                "global_in_flight": global_limit,
                "node_in_flight_default": node_default,
            },
            "topology_version": topology_version,
            "quota_version": quota_version,
            "nodes": normalized_nodes,
        }
    )
    return normalized_root


def validate_requests_doc(doc: Any, node_ids: Iterable[str] | None = None) -> dict[str, Any]:
    root = _require_object(doc, "$")
    if "format_version" in root:
        version = root["format_version"]
        if not _is_int(version) or version <= 0:
            raise SpecError("format_version must be a positive integer", field="format_version")
    requests_value = root.get("requests")
    if not isinstance(requests_value, list):
        raise SpecError("requests must be an array", field="requests")

    known_nodes = set(node_ids or [])
    seen_request_ids: set[str] = set()
    seen_lease_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for index, raw_request in enumerate(requests_value):
        field = f"requests[{index}]"
        request = _require_object(raw_request, field)
        op = request.get("op", "grant")
        if op != "grant":
            raise SpecError("lease manifest requests must use op=grant", field=f"{field}.op")

        request_id = _require_nonempty_str(request, "request_id", field)
        if request_id in seen_request_ids:
            raise SpecError("duplicate request_id", field=f"{field}.request_id")
        seen_request_ids.add(request_id)

        lease_id = _require_nonempty_str(request, "lease_id", field)
        if lease_id in seen_lease_ids:
            raise SpecError("duplicate lease_id in request manifest", field=f"{field}.lease_id")
        seen_lease_ids.add(lease_id)

        equity_id = _require_nonempty_str(request, "equity_id", field)
        node_id = _require_nonempty_str(request, "node_id", field)
        if known_nodes and node_id not in known_nodes:
            raise SpecError(
                "request refers to undeclared node",
                field=f"{field}.node_id",
                details={"node_id": node_id},
            )
        ttl_ms = _require_positive_int(request, "ttl_ms", field)
        at_ms = _require_nonnegative_int(request, "at_ms", field) if "at_ms" in request else 0
        priority = request.get("priority", 100)
        if not _is_int(priority):
            raise SpecError("priority must be an integer", field=f"{field}.priority")

        item = dict(request)
        item.update(
            {
                "op": "grant",
                "request_id": request_id,
                "lease_id": lease_id,
                "equity_id": equity_id,
                "node_id": node_id,
                "ttl_ms": ttl_ms,
                "at_ms": at_ms,
                "priority": priority,
            }
        )
        normalized.append(item)

    normalized_root = dict(root)
    normalized_root["requests"] = normalized
    return normalized_root


def validate_events_doc(
    doc: Any,
    *,
    lease_ids: Iterable[str] | None = None,
    node_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = _require_object(doc, "$")
    if "format_version" in root:
        version = root["format_version"]
        if not _is_int(version) or version <= 0:
            raise SpecError("format_version must be a positive integer", field="format_version")
    events_value = root.get("events")
    if not isinstance(events_value, list):
        raise SpecError("events must be an array", field="events")

    known_leases = set(lease_ids or [])
    known_nodes = set(node_ids or [])
    previous_seq = 0
    normalized_events: list[dict[str, Any]] = []

    for index, raw_event in enumerate(events_value):
        field = f"events[{index}]"
        event = _require_object(raw_event, field)
        seq = _require_positive_int(event, "seq", field)
        if seq <= previous_seq:
            raise SpecError(
                "event seq must be strictly increasing and unique",
                field=f"{field}.seq",
                details={"seq": seq, "previous_seq": previous_seq},
            )
        previous_seq = seq

        event_type = _require_nonempty_str(event, "type", field)
        if event_type not in EVENT_TYPES:
            raise SpecError(
                "unsupported event type",
                field=f"{field}.type",
                details={"type": event_type, "allowed": sorted(EVENT_TYPES)},
            )

        normalized = dict(event)
        normalized["seq"] = seq
        normalized["type"] = event_type

        if event_type in {"grant", "renew", "release"}:
            lease_id = _require_nonempty_str(event, "lease_id", field)
            if known_leases and lease_id not in known_leases:
                raise SpecError(
                    "event refers to undeclared lease",
                    field=f"{field}.lease_id",
                    details={"lease_id": lease_id},
                )
            if "version_tag" in event and event["version_tag"] is not None:
                tag = event["version_tag"]
                if not _is_int(tag) or tag <= 0:
                    raise SpecError(
                        "version_tag must be a positive integer or null",
                        field=f"{field}.version_tag",
                    )
            if "at_ms" in event:
                _require_nonnegative_int(event, "at_ms", field)

        elif event_type == "expire_scan":
            if "at_ms" in event:
                at_ms = _require_nonnegative_int(event, "at_ms", field)
            elif "as_of_ms" in event:
                at_ms = _require_nonnegative_int(event, "as_of_ms", field)
            else:
                raise SpecError(
                    "expire_scan requires at_ms",
                    field=f"{field}.at_ms",
                )
            normalized["at_ms"] = at_ms

        elif event_type == "node_generation_bump":
            node_id = _require_nonempty_str(event, "node_id", field)
            if known_nodes and node_id not in known_nodes:
                raise SpecError(
                    "event refers to undeclared node",
                    field=f"{field}.node_id",
                )
            if "node_generation" in event:
                _require_positive_int(event, "node_generation", field)
            if "at_ms" in event:
                _require_nonnegative_int(event, "at_ms", field)

        elif event_type == "version_evolved":
            has_topology = "topology_version" in event
            has_quota = "quota_version" in event
            if not has_topology and not has_quota:
                raise SpecError(
                    "version_evolved requires topology_version or quota_version",
                    field=field,
                )
            if has_topology:
                _require_positive_int(event, "topology_version", field)
            if has_quota:
                _require_positive_int(event, "quota_version", field)
            for list_field in ("changed_node_ids", "removed_node_ids"):
                node_list = event.get(list_field, [])
                if not isinstance(node_list, list) or not all(
                    isinstance(x, str) and x for x in node_list
                ):
                    raise SpecError(
                        f"{list_field} must be an array of non-empty strings",
                        field=f"{field}.{list_field}",
                    )
                if len(node_list) != len(set(node_list)):
                    raise SpecError(
                        f"{list_field} must be unique",
                        field=f"{field}.{list_field}",
                    )
                if known_nodes:
                    for node_id in node_list:
                        if node_id not in known_nodes:
                            raise SpecError(
                                "version event refers to undeclared node",
                                field=f"{field}.{list_field}",
                                details={"node_id": node_id},
                            )
            if event.get("scope") not in (None, "node_subset", "global"):
                raise SpecError(
                    "scope must be node_subset or global",
                    field=f"{field}.scope",
                )
            if "at_ms" in event:
                _require_nonnegative_int(event, "at_ms", field)

        normalized_events.append(normalized)

    normalized_root = dict(root)
    normalized_root["events"] = normalized_events
    return normalized_root


def load_bundle(
    nodes_path: str | Path,
    requests_path: str | Path,
    events_path: str | Path,
) -> SpecBundle:
    nodes_path = Path(nodes_path)
    requests_path = Path(requests_path)
    events_path = Path(events_path)
    nodes_doc = validate_nodes_doc(load_json_file(nodes_path))
    requests_doc = validate_requests_doc(load_json_file(requests_path), nodes_doc["nodes"] and {n["node_id"] for n in nodes_doc["nodes"]})
    lease_ids = {item["lease_id"] for item in requests_doc["requests"]}
    node_ids = {node["node_id"] for node in nodes_doc["nodes"]}
    events_doc = validate_events_doc(
        load_json_file(events_path), lease_ids=lease_ids, node_ids=node_ids
    )
    return SpecBundle(
        nodes_doc=nodes_doc,
        requests_doc=requests_doc,
        events_doc=events_doc,
        nodes_path=nodes_path,
        requests_path=requests_path,
        events_path=events_path,
    )


def load_sample_dir(directory: str | Path) -> SpecBundle:
    directory = Path(directory)
    return load_bundle(
        directory / "nodes.json",
        directory / "lease_requests.json",
        directory / "events.json",
    )
