from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest
from .errors import SpecError


_IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TYPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_OPS = frozenset({"grant", "renew", "release", "quota_invalidate"})
_NODE_KEYS = frozenset({"node_id", "host", "port", "max_leases"})
_LEASE_KEYS = frozenset(
    {
        "lease_id",
        "node_id",
        "lease_type",
        "ttl_ms",
        "renew_interval_ms",
        "quota_version",
    }
)
_COMMON_EVENT_KEYS = frozenset({"seq", "at_ms", "op", "reason"})
_EVENT_KEYS = {
    "grant": _COMMON_EVENT_KEYS | {"lease_id", "expected_label"},
    "renew": _COMMON_EVENT_KEYS | {"lease_id", "expected_label"},
    "release": _COMMON_EVENT_KEYS | {"lease_id", "expected_label"},
    "quota_invalidate": _COMMON_EVENT_KEYS | {"node_id", "lease_type", "quota_version"},
}


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    max_leases: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "max_leases": self.max_leases,
        }


@dataclass(frozen=True)
class LeaseSpec:
    lease_id: str
    node_id: str
    lease_type: str
    ttl_ms: int
    renew_interval_ms: int
    quota_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "node_id": self.node_id,
            "lease_type": self.lease_type,
            "ttl_ms": self.ttl_ms,
            "renew_interval_ms": self.renew_interval_ms,
            "quota_version": self.quota_version,
        }


@dataclass(frozen=True)
class Event:
    seq: int
    at_ms: int
    op: str
    line: int
    lease_id: str | None = None
    expected_label: int | None = None
    node_id: str | None = None
    lease_type: str | None = None
    quota_version: int | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seq": self.seq,
            "at_ms": self.at_ms,
            "op": self.op,
        }
        if self.lease_id is not None:
            payload["lease_id"] = self.lease_id
        if self.expected_label is not None:
            payload["expected_label"] = self.expected_label
        if self.node_id is not None:
            payload["node_id"] = self.node_id
        if self.lease_type is not None:
            payload["lease_type"] = self.lease_type
        if self.quota_version is not None:
            payload["quota_version"] = self.quota_version
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class Spec:
    schema_version: int
    declaration_version: int
    global_concurrency: int
    connect_timeout_ms: int
    read_timeout_ms: int
    worker_threads: int
    nodes: tuple[NodeSpec, ...]
    leases: tuple[LeaseSpec, ...]
    events: tuple[Event, ...]
    digest: str
    nodes_file: str
    leases_file: str
    events_file: str

    @property
    def node_by_id(self) -> dict[str, NodeSpec]:
        return {node.node_id: node for node in self.nodes}

    @property
    def lease_by_id(self) -> dict[str, LeaseSpec]:
        return {lease.lease_id: lease for lease in self.leases}

    def declaration_payload(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "declaration_version": self.declaration_version,
            "global_concurrency": self.global_concurrency,
            "connect_timeout_ms": self.connect_timeout_ms,
            "read_timeout_ms": self.read_timeout_ms,
            "worker_threads": self.worker_threads,
            "nodes": [node.to_dict() for node in self.nodes],
            "leases": [lease.to_dict() for lease in self.leases],
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _key_lines(text: str, key: str) -> list[int]:
    pattern = re.compile(rf'"{re.escape(key)}"\s*:')
    return [text.count("\n", 0, match.start()) + 1 for match in pattern.finditer(text)]


def _line_for_key(text: str, key: str, occurrence: int = 0) -> int:
    lines = _key_lines(text, key)
    if not lines:
        return 1
    if occurrence < 0:
        occurrence = 0
    if occurrence >= len(lines):
        occurrence = len(lines) - 1
    return lines[occurrence]


def _line_for_value(text: str, key: str, value: Any, fallback: int = 1) -> int:
    if isinstance(value, str):
        pattern = re.compile(
            rf'"{re.escape(key)}"\s*:\s*"{re.escape(value)}"'
        )
        match = pattern.search(text)
        if match:
            return text.count("\n", 0, match.start()) + 1
    return _line_for_key(text, key)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass
class _JsonDoc:
    path: Path
    text: str
    value: Any

    @property
    def name(self) -> str:
        return str(self.path)

    def line(self, key: str, occurrence: int = 0) -> int:
        return _line_for_key(self.text, key, occurrence)

    def value_line(self, key: str, value: Any) -> int:
        return _line_for_value(self.text, key, value)


def _read_json_doc(path: str | Path) -> _JsonDoc:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError("$file", 1, f"cannot read JSON file: {exc}", file=str(source)) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        key = "JSON"
        message = f"invalid JSON at column {exc.colno}: {exc.msg}"
        raise SpecError(key, exc.lineno, message, file=str(source)) from exc
    except ValueError as exc:
        message = str(exc)
        key = message.rsplit(":", 1)[-1].strip() if "key:" in message else "JSON"
        line = _line_for_key(text, key)
        raise SpecError(key, line, message, file=str(source)) from exc
    if not isinstance(value, dict):
        raise SpecError("$", 1, "top-level JSON value must be an object", file=str(source))
    return _JsonDoc(source, text, value)


def _require_exact_keys(
    doc: _JsonDoc,
    value: dict[str, Any],
    allowed: frozenset[str],
    prefix: str,
) -> None:
    for key in value:
        if key not in allowed:
            raise SpecError(
                f"{prefix}.{key}" if prefix else key,
                doc.line(key),
                f"unknown field {key!r}",
                file=doc.name,
                value=value[key],
            )
    for key in sorted(allowed):
        if key not in value:
            raise SpecError(
                f"{prefix}.{key}" if prefix else key,
                doc.line(key),
                f"missing required field {key!r}",
                file=doc.name,
            )


def _positive_int(doc: _JsonDoc, value: Any, field: str, line: int) -> int:
    if not _is_int(value) or value <= 0:
        raise SpecError(field, line, "must be a positive integer", file=doc.name, value=value)
    return value


def _node_specs(doc: _JsonDoc) -> tuple[int, int, int, int, int, int, tuple[NodeSpec, ...]]:
    raw = doc.value
    allowed_top = frozenset(
        {
            "schema_version",
            "declaration_version",
            "global_concurrency",
            "connect_timeout_ms",
            "read_timeout_ms",
            "worker_threads",
            "nodes",
        }
    )
    _require_exact_keys(doc, raw, allowed_top, "")
    schema_version = raw["schema_version"]
    if not _is_int(schema_version) or schema_version != 1:
        raise SpecError("schema_version", doc.line("schema_version"), "must equal 1", file=doc.name, value=schema_version)
    declaration_version = _positive_int(
        doc, raw["declaration_version"], "declaration_version", doc.line("declaration_version")
    )
    if declaration_version > 999_999:
        raise SpecError(
            "declaration_version",
            doc.line("declaration_version"),
            "must be at most 999999",
            file=doc.name,
            value=declaration_version,
        )
    global_concurrency = _positive_int(
        doc, raw["global_concurrency"], "global_concurrency", doc.line("global_concurrency")
    )
    connect_timeout_ms = _positive_int(
        doc, raw["connect_timeout_ms"], "connect_timeout_ms", doc.line("connect_timeout_ms")
    )
    read_timeout_ms = _positive_int(
        doc, raw["read_timeout_ms"], "read_timeout_ms", doc.line("read_timeout_ms")
    )
    worker_threads = _positive_int(
        doc, raw["worker_threads"], "worker_threads", doc.line("worker_threads")
    )
    nodes_raw = raw["nodes"]
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise SpecError("nodes", doc.line("nodes"), "must be a non-empty array", file=doc.name, value=nodes_raw)

    nodes: list[NodeSpec] = []
    node_ids: set[str] = set()
    ports: set[int] = set()
    for index, item in enumerate(nodes_raw):
        prefix = f"nodes[{index}]"
        if not isinstance(item, dict):
            raise SpecError(prefix, doc.line("nodes"), "must be an object", file=doc.name)
        _require_exact_keys(doc, item, _NODE_KEYS, prefix)
        node_id = item["node_id"]
        node_line = doc.line("node_id", index)
        if not _nonempty_str(node_id) or not _IDENT_RE.fullmatch(node_id):
            raise SpecError(f"{prefix}.node_id", node_line, "must be a non-empty identifier", file=doc.name, value=node_id)
        if node_id in node_ids:
            raise SpecError(f"{prefix}.node_id", node_line, "node_id must be unique", file=doc.name, value=node_id)
        node_ids.add(node_id)
        host = item["host"]
        if host != "127.0.0.1":
            raise SpecError(
                f"{prefix}.host",
                doc.line("host", index),
                "host must be the literal 127.0.0.1",
                file=doc.name,
                value=host,
            )
        port = item["port"]
        if not _is_int(port) or not (1 <= port <= 65535):
            raise SpecError(
                f"{prefix}.port",
                doc.line("port", index),
                "must be an integer in [1, 65535]",
                file=doc.name,
                value=port,
            )
        if port in ports:
            raise SpecError(f"{prefix}.port", doc.line("port", index), "port must be globally unique", file=doc.name, value=port)
        ports.add(port)
        max_leases = _positive_int(doc, item["max_leases"], f"{prefix}.max_leases", doc.line("max_leases", index))
        nodes.append(NodeSpec(node_id, "127.0.0.1", port, max_leases))
    return (
        schema_version,
        declaration_version,
        global_concurrency,
        connect_timeout_ms,
        read_timeout_ms,
        worker_threads,
        tuple(nodes),
    )


def _lease_specs(doc: _JsonDoc, node_ids: set[str]) -> tuple[int, int, tuple[LeaseSpec, ...]]:
    raw = doc.value
    allowed_top = frozenset({"schema_version", "declaration_version", "leases"})
    _require_exact_keys(doc, raw, allowed_top, "")
    schema_version = raw["schema_version"]
    if not _is_int(schema_version) or schema_version != 1:
        raise SpecError("schema_version", doc.line("schema_version"), "must equal 1", file=doc.name, value=schema_version)
    declaration_version = _positive_int(
        doc, raw["declaration_version"], "declaration_version", doc.line("declaration_version")
    )
    if declaration_version > 999_999:
        raise SpecError(
            "declaration_version",
            doc.line("declaration_version"),
            "must be at most 999999",
            file=doc.name,
            value=declaration_version,
        )
    leases_raw = raw["leases"]
    if not isinstance(leases_raw, list) or not leases_raw:
        raise SpecError("leases", doc.line("leases"), "must be a non-empty array", file=doc.name, value=leases_raw)
    leases: list[LeaseSpec] = []
    lease_ids: set[str] = set()
    for index, item in enumerate(leases_raw):
        prefix = f"leases[{index}]"
        if not isinstance(item, dict):
            raise SpecError(prefix, doc.line("leases"), "must be an object", file=doc.name)
        _require_exact_keys(doc, item, _LEASE_KEYS, prefix)
        lease_id = item["lease_id"]
        lease_line = doc.line("lease_id", index)
        if not _nonempty_str(lease_id) or not _IDENT_RE.fullmatch(lease_id):
            raise SpecError(f"{prefix}.lease_id", lease_line, "must be a non-empty identifier", file=doc.name, value=lease_id)
        if lease_id in lease_ids:
            raise SpecError(f"{prefix}.lease_id", lease_line, "lease_id must be unique", file=doc.name, value=lease_id)
        lease_ids.add(lease_id)
        node_id = item["node_id"]
        if not _nonempty_str(node_id) or node_id not in node_ids:
            raise SpecError(
                f"{prefix}.node_id",
                doc.line("node_id", index),
                "node_id must reference a declared node",
                file=doc.name,
                value=node_id,
            )
        lease_type = item["lease_type"]
        if not _nonempty_str(lease_type) or not _TYPE_RE.fullmatch(lease_type):
            raise SpecError(
                f"{prefix}.lease_type",
                doc.line("lease_type", index),
                "must be a non-empty lease type identifier",
                file=doc.name,
                value=lease_type,
            )
        ttl_ms = _positive_int(doc, item["ttl_ms"], f"{prefix}.ttl_ms", doc.line("ttl_ms", index))
        renew_interval_ms = _positive_int(
            doc,
            item["renew_interval_ms"],
            f"{prefix}.renew_interval_ms",
            doc.line("renew_interval_ms", index),
        )
        if renew_interval_ms >= ttl_ms:
            raise SpecError(
                f"{prefix}.renew_interval_ms",
                doc.line("renew_interval_ms", index),
                "renew_interval_ms must be strictly smaller than ttl_ms",
                file=doc.name,
                value=renew_interval_ms,
            )
        quota_version = _positive_int(
            doc, item["quota_version"], f"{prefix}.quota_version", doc.line("quota_version", index)
        )
        leases.append(
            LeaseSpec(
                lease_id=lease_id,
                node_id=node_id,
                lease_type=lease_type,
                ttl_ms=ttl_ms,
                renew_interval_ms=renew_interval_ms,
                quota_version=quota_version,
            )
        )
    return schema_version, declaration_version, tuple(leases)


def _event_specs(path: str | Path, lease_by_id: dict[str, LeaseSpec], node_ids: set[str]) -> tuple[Event, ...]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError("$file", 1, f"cannot read event stream: {exc}", file=str(source)) from exc
    lines = text.splitlines()
    if not lines:
        raise SpecError("events", 1, "event stream must contain at least one event", file=str(source))
    events: list[Event] = []
    previous_seq: int | None = None
    previous_at: int | None = None
    seen_seq: set[int] = set()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            raise SpecError("$line", line_no, "blank lines are not allowed", file=str(source))
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise SpecError("JSON", line_no, f"invalid JSON at column {exc.colno}: {exc.msg}", file=str(source)) from exc
        except ValueError as exc:
            message = str(exc)
            field = message.rsplit(":", 1)[-1].strip() if "key:" in message else "JSON"
            raise SpecError(field, line_no, message, file=str(source)) from exc
        if not isinstance(raw, dict):
            raise SpecError("$", line_no, "event must be a JSON object", file=str(source))
        op = raw.get("op")
        if not isinstance(op, str) or op not in _OPS:
            raise SpecError("op", line_no, "op must be one of grant, renew, release, quota_invalidate", file=str(source), value=op)
        allowed = _EVENT_KEYS[op]
        for key in raw:
            if key not in allowed:
                raise SpecError(key, line_no, f"unknown field {key!r} for op {op}", file=str(source), value=raw[key])
        for key in sorted(allowed - {"reason"}):
            if key not in raw:
                raise SpecError(key, line_no, f"missing required field {key!r}", file=str(source))
        seq = raw["seq"]
        if not _is_int(seq) or seq <= 0:
            raise SpecError("seq", line_no, "seq must be a positive integer", file=str(source), value=seq)
        if seq in seen_seq or (previous_seq is not None and seq <= previous_seq):
            raise SpecError("seq", line_no, "event seq must be strictly increasing and unique", file=str(source), value=seq)
        seen_seq.add(seq)
        previous_seq = seq
        at_ms = raw["at_ms"]
        if not _is_int(at_ms) or at_ms < 0:
            raise SpecError("at_ms", line_no, "at_ms must be a non-negative integer", file=str(source), value=at_ms)
        if previous_at is not None and at_ms < previous_at:
            raise SpecError("at_ms", line_no, "at_ms must be monotonically non-decreasing", file=str(source), value=at_ms)
        previous_at = at_ms
        reason = raw.get("reason", "")
        if not isinstance(reason, str):
            raise SpecError("reason", line_no, "reason must be a string", file=str(source), value=reason)
        if op == "quota_invalidate":
            node_id = raw["node_id"]
            if not _nonempty_str(node_id) or node_id not in node_ids:
                raise SpecError("node_id", line_no, "node_id must reference a declared node", file=str(source), value=node_id)
            lease_type = raw["lease_type"]
            if not _nonempty_str(lease_type) or not _TYPE_RE.fullmatch(lease_type):
                raise SpecError("lease_type", line_no, "must be a non-empty lease type identifier", file=str(source), value=lease_type)
            quota_version = raw["quota_version"]
            if not _is_int(quota_version) or quota_version <= 0:
                raise SpecError("quota_version", line_no, "quota_version must be a positive integer", file=str(source), value=quota_version)
            events.append(
                Event(
                    seq=seq,
                    at_ms=at_ms,
                    op=op,
                    line=line_no,
                    node_id=node_id,
                    lease_type=lease_type,
                    quota_version=quota_version,
                    reason=reason,
                )
            )
            continue
        lease_id = raw["lease_id"]
        if not _nonempty_str(lease_id) or lease_id not in lease_by_id:
            raise SpecError("lease_id", line_no, "lease_id must reference a declared lease", file=str(source), value=lease_id)
        expected_label = raw["expected_label"]
        if not _is_int(expected_label) or expected_label < 0:
            raise SpecError("expected_label", line_no, "expected_label must be a non-negative integer", file=str(source), value=expected_label)
        events.append(
            Event(
                seq=seq,
                at_ms=at_ms,
                op=op,
                line=line_no,
                lease_id=lease_id,
                expected_label=expected_label,
                reason=reason,
            )
        )
    return tuple(events)


def load_spec(nodes_path: str | Path, leases_path: str | Path, events_path: str | Path) -> Spec:
    nodes_doc = _read_json_doc(nodes_path)
    leases_doc = _read_json_doc(leases_path)
    (
        nodes_schema,
        nodes_version,
        global_concurrency,
        connect_timeout_ms,
        read_timeout_ms,
        worker_threads,
        nodes,
    ) = _node_specs(nodes_doc)
    leases_schema, leases_version, leases = _lease_specs(leases_doc, {node.node_id for node in nodes})
    if nodes_schema != leases_schema:
        raise SpecError("schema_version", leases_doc.line("schema_version"), "node and lease schema versions must match", file=leases_doc.name)
    lease_by_id = {lease.lease_id: lease for lease in leases}
    events = _event_specs(events_path, lease_by_id, {node.node_id for node in nodes})
    declaration_version = nodes_version * 1_000_000 + leases_version
    combined = {
        "format_version": 1,
        "declaration_version": declaration_version,
        "global_concurrency": global_concurrency,
        "connect_timeout_ms": connect_timeout_ms,
        "read_timeout_ms": read_timeout_ms,
        "worker_threads": worker_threads,
        "nodes": [node.to_dict() for node in nodes],
        "leases": [lease.to_dict() for lease in leases],
    }
    return Spec(
        schema_version=1,
        declaration_version=declaration_version,
        global_concurrency=global_concurrency,
        connect_timeout_ms=connect_timeout_ms,
        read_timeout_ms=read_timeout_ms,
        worker_threads=worker_threads,
        nodes=nodes,
        leases=leases,
        events=events,
        digest=digest(combined),
        nodes_file=str(Path(nodes_path)),
        leases_file=str(Path(leases_path)),
        events_file=str(Path(events_path)),
    )


def _lease_version(doc: _JsonDoc) -> int:
    # _lease_specs already validated this field; reading it again keeps the
    # combined declaration version explicit and deterministic.
    return int(doc.value["declaration_version"])


def validate_files(nodes_path: str | Path, leases_path: str | Path, events_path: str | Path) -> Spec:
    return load_spec(nodes_path, leases_path, events_path)
