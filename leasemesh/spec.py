from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

from .canonical import sha256_hex
from .errors import E_SPEC, SpecError
from .jsonio import JsonDocument, first_problem, load_json_document


GRANT = "grant"
RENEW = "renew"
RELEASE = "release"
PING = "ping"
LEASE_OPS = {GRANT, RENEW, RELEASE}


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _problem(
    problems: list[SpecError],
    doc: Optional[JsonDocument],
    field: str,
    message: str,
) -> None:
    line = doc.line(field) if doc is not None else None
    problems.append(SpecError(message, code=E_SPEC, field=field, line=line or 1))


def _require_positive_int(
    data: dict[str, Any],
    field: str,
    problems: list[SpecError],
    doc: Optional[JsonDocument],
) -> Optional[int]:
    value = data.get(field)
    if not is_int(value) or value <= 0:
        _problem(problems, doc, field, f"{field} must be a positive integer")
        return None
    return value


def _require_nonnegative_int(
    data: dict[str, Any],
    field: str,
    problems: list[SpecError],
    doc: Optional[JsonDocument],
) -> Optional[int]:
    value = data.get(field)
    if not is_int(value) or value < 0:
        _problem(problems, doc, field, f"{field} must be a non-negative integer")
        return None
    return value


def load_declaration(path: str | Path) -> dict[str, Any]:
    doc = load_json_document(path)
    spec = validate_declaration(doc)
    spec["_path"] = str(Path(path).resolve())
    spec["declaration_digest"] = digest_declaration(spec)
    return spec


def digest_declaration(spec: dict[str, Any]) -> str:
    clean = {
        key: value
        for key, value in spec.items()
        if not key.startswith("_") and key != "declaration_digest"
    }
    return sha256_hex(clean)


def validate_declaration(doc: JsonDocument) -> dict[str, Any]:
    problems: list[SpecError] = []
    data = doc.data
    if not isinstance(data, dict):
        raise SpecError("declaration root must be a JSON object", field="$", line=1)

    quota_version = _require_nonnegative_int(data, "quota_version", problems, doc)
    duration = _require_positive_int(data, "lease_duration_ms", problems, doc)
    renew_interval = _require_positive_int(data, "renew_interval_ms", problems, doc)
    global_concurrency = _require_positive_int(data, "global_concurrency", problems, doc)
    per_node_limit = _require_positive_int(data, "per_node_lease_limit", problems, doc)

    if duration is not None and renew_interval is not None and renew_interval >= duration:
        _problem(
            problems,
            doc,
            "renew_interval_ms",
            "renew_interval_ms must be strictly less than lease_duration_ms",
        )

    for field_name in ("socket_timeout_ms", "handshake_timeout_ms", "io_buffer_bytes"):
        if field_name in data:
            value = _require_positive_int(data, field_name, problems, doc)
            if field_name == "io_buffer_bytes" and value is not None and value < 1024:
                _problem(problems, doc, field_name, "io_buffer_bytes is too small")

    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        _problem(problems, doc, "nodes", "nodes must be a non-empty array")
        nodes = []

    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    normalized_nodes: list[dict[str, Any]] = []

    for index, node in enumerate(nodes):
        base = f"nodes[{index}]"
        if not isinstance(node, dict):
            _problem(problems, doc, base, "node declaration must be an object")
            continue

        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            _problem(problems, doc, f"{base}.node_id", "node_id must be a non-empty string")
        elif node_id in seen_node_ids:
            _problem(problems, doc, f"{base}.node_id", f"duplicate node_id: {node_id}")
        else:
            seen_node_ids.add(node_id)

        host = node.get("host")
        if host != "127.0.0.1":
            _problem(
                problems,
                doc,
                f"{base}.host",
                "host must be the literal loopback address 127.0.0.1",
            )

        port = node.get("port")
        if not is_int(port) or not (1024 <= int(port) <= 65535):
            _problem(
                problems,
                doc,
                f"{base}.port",
                "port must be an unprivileged high port in [1024, 65535]",
            )
        elif port in seen_ports:
            _problem(problems, doc, f"{base}.port", f"duplicate loopback port: {port}")
        else:
            seen_ports.add(int(port))

        profile = node.get("checksum", {})
        if profile in (None, {}):
            profile = {}
        if not isinstance(profile, dict):
            _problem(problems, doc, f"{base}.checksum", "checksum must be an object")
            profile = {}

        seed = profile.get("seed", 20261126)
        sample_count = profile.get("sample_count", 64)
        group_count = profile.get("group_count", 4)
        scale = profile.get("scale", 1_000_000)
        for field_name, value in (
            ("seed", seed),
            ("sample_count", sample_count),
            ("group_count", group_count),
            ("scale", scale),
        ):
            if field_name == "seed":
                valid = is_int(value) and 0 <= int(value) <= 2**63 - 1
            else:
                valid = is_int(value) and int(value) > 0
            if not valid:
                _problem(problems, doc, f"{base}.checksum.{field_name}", f"{field_name} must be valid")

        if (
            is_int(sample_count)
            and is_int(group_count)
            and int(group_count) > int(sample_count)
        ):
            _problem(
                problems,
                doc,
                f"{base}.checksum.group_count",
                "group_count must not exceed sample_count",
            )

        if not problems or all(p.field != f"{base}.node_id" for p in problems):
            normalized_nodes.append(
                {
                    "node_id": node_id if isinstance(node_id, str) else f"invalid-{index}",
                    "host": "127.0.0.1",
                    "port": int(port) if is_int(port) else 0,
                    "checksum": {
                        "algorithm": "numpy-fixed-point-group-sum-v1",
                        "seed": int(seed) if is_int(seed) else 0,
                        "sample_count": int(sample_count) if is_int(sample_count) else 1,
                        "group_count": int(group_count) if is_int(group_count) else 1,
                        "scale": int(scale) if is_int(scale) else 1,
                    },
                }
            )

    slots = data.get("slots")
    normalized_slots: list[str] = []
    if slots is None:
        capacity = len(normalized_nodes) * (per_node_limit or 1)
        normalized_slots = [f"slot-{index + 1}" for index in range(max(1, capacity))]
    elif not isinstance(slots, list) or not slots:
        _problem(problems, doc, "slots", "slots must be a non-empty array when provided")
    else:
        seen_slots: set[str] = set()
        for index, slot in enumerate(slots):
            field = f"slots[{index}]"
            slot_id = slot.get("slot_id") if isinstance(slot, dict) else slot
            if not isinstance(slot_id, str) or not slot_id.strip():
                _problem(problems, doc, field, "slot_id must be a non-empty string")
            elif slot_id in seen_slots:
                _problem(problems, doc, field, f"duplicate slot_id: {slot_id}")
            else:
                seen_slots.add(slot_id)
                normalized_slots.append(slot_id)

    first = first_problem(problems)
    if first is not None:
        raise first

    normalized = {
        "schema": data.get("schema", "leasemesh.declaration/v1"),
        "quota_version": quota_version,
        "lease_duration_ms": duration,
        "renew_interval_ms": renew_interval,
        "global_concurrency": global_concurrency,
        "per_node_lease_limit": per_node_limit,
        "handshake_timeout_ms": int(data.get("handshake_timeout_ms", 1000)),
        "socket_timeout_ms": int(data.get("socket_timeout_ms", 2000)),
        "io_buffer_bytes": int(data.get("io_buffer_bytes", 1_048_576)),
        "nodes": normalized_nodes,
        "slots": normalized_slots,
    }
    for key, value in data.items():
        if key not in normalized:
            normalized[key] = copy.deepcopy(value)
    normalized.pop("declaration_digest", None)
    return normalized


def load_requests(
    path: str | Path, spec: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    doc = load_json_document(path)
    requests = validate_requests(doc, spec)
    return requests


def validate_requests(
    doc: JsonDocument, spec: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    problems: list[SpecError] = []
    data = doc.data
    if isinstance(data, list):
        raw_requests = data
        root = ""
    elif isinstance(data, dict) and isinstance(data.get("requests") or data.get("applications"), list):
        raw_requests = data.get("requests") or data.get("applications")
        root = "requests" if "requests" in data else "applications"
    else:
        raise SpecError(
            "request manifest must be a list or contain a requests/applications array",
            field="requests",
            line=doc.line("requests") or 1,
        )

    node_ids = {node["node_id"] for node in (spec or {}).get("nodes", [])}
    slot_ids = set((spec or {}).get("slots", []))
    seen_request_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_requests):
        base = f"{root}[{index}]" if root else f"[{index}]"
        if not isinstance(raw, dict):
            _problem(problems, doc, base, "request must be an object")
            continue

        request_id = raw.get("request_id") or raw.get("id")
        if not isinstance(request_id, str) or not request_id.strip():
            _problem(problems, doc, f"{base}.request_id", "request_id must be a non-empty string")
        elif request_id in seen_request_ids:
            _problem(problems, doc, f"{base}.request_id", f"duplicate request_id: {request_id}")
        else:
            seen_request_ids.add(request_id)

        op = raw.get("op") or raw.get("operation")
        if op not in LEASE_OPS and op != PING:
            _problem(problems, doc, f"{base}.op", "op must be grant, renew, release, or ping")

        node_id = raw.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            _problem(problems, doc, f"{base}.node_id", "node_id must be a non-empty string")
        elif node_ids and node_id not in node_ids:
            _problem(problems, doc, f"{base}.node_id", f"unknown node_id: {node_id}")

        lease_id = raw.get("lease_id")
        if op in LEASE_OPS and (not isinstance(lease_id, str) or not lease_id.strip()):
            _problem(problems, doc, f"{base}.lease_id", "lease_id must be a non-empty string")

        slot_id = raw.get("slot_id")
        if op == GRANT:
            if not isinstance(slot_id, str) or not slot_id.strip():
                _problem(problems, doc, f"{base}.slot_id", "grant requires a slot_id string")
            elif slot_ids and slot_id not in slot_ids:
                _problem(problems, doc, f"{base}.slot_id", f"unknown slot_id: {slot_id}")

        generation = raw.get("generation")
        label = raw.get("label")
        if op in (RENEW, RELEASE):
            if generation is None and label is None:
                _problem(
                    problems,
                    doc,
                    f"{base}.generation",
                    "renew/release requires generation or label",
                )
            elif generation is not None and (not is_int(generation) or generation <= 0):
                _problem(problems, doc, f"{base}.generation", "generation must be a positive integer")
            elif label is not None:
                if not isinstance(label, str) or "@" not in label:
                    _problem(problems, doc, f"{base}.label", "label must have form <lease_id>@<generation>")
                else:
                    label_lease, suffix = label.rsplit("@", 1)
                    if not suffix.isdigit() or int(suffix) <= 0:
                        _problem(problems, doc, f"{base}.label", "label generation must be positive")
                    elif generation is not None and int(suffix) != int(generation):
                        _problem(problems, doc, f"{base}.label", "label does not match generation")
                    elif not isinstance(lease_id, str) or label_lease != lease_id:
                        _problem(problems, doc, f"{base}.label", "label does not match lease_id")
                    else:
                        generation = int(suffix)
            if generation is None and isinstance(label, str) and "@" in label:
                suffix = label.rsplit("@", 1)[1]
                if suffix.isdigit():
                    generation = int(suffix)

        stage = raw.get("stage", 1)
        if not is_int(stage) or int(stage) <= 0:
            _problem(problems, doc, f"{base}.stage", "stage must be a positive integer when provided")
            stage = 1

        normalized.append(
            {
                "request_id": request_id if isinstance(request_id, str) else f"request-{index}",
                "op": op,
                "node_id": node_id if isinstance(node_id, str) else "",
                "lease_id": lease_id if isinstance(lease_id, str) else None,
                "slot_id": slot_id if isinstance(slot_id, str) else None,
                "generation": int(generation) if is_int(generation) else generation,
                "label": label if isinstance(label, str) else None,
                "quota_version": raw.get("quota_version"),
                "expect_code": raw.get("expect_code"),
                "stage": int(stage),
                "metadata": raw.get("metadata", {}),
            }
        )

    first = first_problem(problems)
    if first is not None:
        raise first
    return normalized


EVENT_KINDS = {
    "lease_granted",
    "lease_renewed",
    "lease_released",
    "lease_expired",
    "reclaim_proposed",
    "quota_epoch_changed",
    "node_added",
    "node_removed",
}


def load_events(path: str | Path) -> list[dict[str, Any]]:
    doc = load_json_document(path)
    return validate_events(doc)


def validate_events(doc: JsonDocument) -> list[dict[str, Any]]:
    problems: list[SpecError] = []
    data = doc.data
    if isinstance(data, list):
        raw_events = data
        root = ""
    elif isinstance(data, dict) and isinstance(data.get("events"), list):
        raw_events = data["events"]
        root = "events"
    else:
        raise SpecError(
            "event stream must be a list or contain an events array",
            field="events",
            line=doc.line("events") or 1,
        )

    previous_seq: Optional[int] = None
    seen_seq: set[int] = set()
    normalized: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_events):
        base = f"{root}[{index}]" if root else f"[{index}]"
        if not isinstance(raw, dict):
            _problem(problems, doc, base, "event must be an object")
            continue

        seq = raw.get("seq")
        if not is_int(seq) or int(seq) < 0:
            _problem(problems, doc, f"{base}.seq", "seq must be a non-negative integer")
        else:
            seq = int(seq)
            if seq in seen_seq:
                _problem(problems, doc, f"{base}.seq", f"duplicate event seq: {seq}")
            elif previous_seq is not None and seq <= previous_seq:
                _problem(
                    problems,
                    doc,
                    f"{base}.seq",
                    f"seq {seq} is not strictly greater than previous seq {previous_seq}",
                )
            seen_seq.add(seq)
            previous_seq = seq

        kind = raw.get("kind")
        if kind not in EVENT_KINDS:
            _problem(problems, doc, f"{base}.kind", f"unknown event kind: {kind!r}")

        if kind in {"lease_granted", "lease_renewed", "lease_released", "lease_expired"}:
            for field_name in ("lease_id", "node_id", "slot_id"):
                value = raw.get(field_name)
                if not isinstance(value, str) or not value:
                    _problem(problems, doc, f"{base}.{field_name}", f"{field_name} must be a non-empty string")
            generation = raw.get("generation")
            if not is_int(generation) or int(generation) <= 0:
                _problem(problems, doc, f"{base}.generation", "generation must be a positive integer")

        if kind == "reclaim_proposed":
            for field_name in ("claim_id", "slot_id", "lease_id", "node_id"):
                value = raw.get(field_name)
                if not isinstance(value, str) or not value:
                    _problem(problems, doc, f"{base}.{field_name}", f"{field_name} must be a non-empty string")
            generation = raw.get("generation")
            if not is_int(generation) or int(generation) <= 0:
                _problem(problems, doc, f"{base}.generation", "generation must be a positive integer")
            if raw.get("reason") not in {"expired", "quota_fenced"}:
                _problem(
                    problems,
                    doc,
                    f"{base}.reason",
                    "reason must be expired or quota_fenced",
                )

        if kind == "quota_epoch_changed":
            for field_name in ("from_version", "to_version"):
                value = raw.get(field_name)
                if not is_int(value) or int(value) < 0:
                    _problem(problems, doc, f"{base}.{field_name}", f"{field_name} must be non-negative")

        if kind in {"node_added", "node_removed"}:
            node_id = raw.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                _problem(problems, doc, f"{base}.node_id", "node_id must be a non-empty string")

        normalized.append(copy.deepcopy(raw))

    first = first_problem(problems)
    if first is not None:
        raise first
    normalized.sort(key=lambda event: event.get("seq", 0))
    return normalized
