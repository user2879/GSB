from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_hash, sha256_hex
from .errors import SpecError
from .jsonloc import load_json_file, load_jsonl_file


ALLOWED_FIXED_SCALES = (1, 10, 100, 1000)
SPEC_FILES = (
    "nodes.json",
    "content_manifest.json",
    "lease_policy.json",
    "event_stream.jsonl",
)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(obj: dict[str, Any], key: str, path: str, lines: dict[str, int]) -> int:
    if key not in obj:
        raise SpecError(f"missing integer field {key!r}", f"{path}.{key}", _line(lines, f"{path}.{key}"))
    value = obj[key]
    if not is_int(value):
        raise SpecError(f"{key} must be an integer", f"{path}.{key}", _line(lines, f"{path}.{key}"))
    return value


def require_str(obj: dict[str, Any], key: str, path: str, lines: dict[str, int]) -> str:
    if key not in obj or not isinstance(obj[key], str) or not obj[key]:
        json_path = f"{path}.{key}"
        raise SpecError(f"{key} must be a non-empty string", json_path, _line(lines, json_path))
    return obj[key]


def _line(lines: dict[str, int], path: str) -> int:
    if path in lines:
        return lines[path]
    prefix = path
    while prefix:
        if prefix in lines:
            return lines[prefix]
        if "[" in prefix:
            prefix = prefix.rsplit("[", 1)[0]
        elif "." in prefix:
            prefix = prefix.rsplit(".", 1)[0]
        else:
            break
    return lines.get("$", 1)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    max_node_leases: int
    seed: int

    @property
    def effective_lease_cap(self) -> int:
        return min(self.capacity_slots, self.max_node_leases)


@dataclass(frozen=True)
class InvalidRefSpec:
    ref_id: str
    shard_id: str
    reason: str


@dataclass(frozen=True)
class ContentSpec:
    content_key: str
    total_bytes: int
    shard_count: int
    sample_stride: int
    integer_scale: int
    seed: int
    invalid_refs: tuple[InvalidRefSpec, ...] = ()

    @property
    def shard_size(self) -> int:
        return self.total_bytes // self.shard_count

    def shard_id(self, seq: int) -> str:
        return f"shard-{seq:04d}"

    def shard_ids(self) -> tuple[str, ...]:
        return tuple(self.shard_id(i) for i in range(self.shard_count))

    def byte_range(self, seq: int) -> tuple[int, int]:
        size = self.shard_size
        return seq * size, (seq + 1) * size

    def sample_count(self, seq: int) -> int:
        start, end = self.byte_range(seq)
        length = end - start
        return (length + self.sample_stride - 1) // self.sample_stride


@dataclass(frozen=True)
class LeasePolicy:
    quota_version: int
    ttl_ms: int
    max_renewals: int
    global_max_active_leases: int
    global_byte_budget: int
    reaper_interval_ms: int = 25
    queue_timeout_ms: int = 5000
    socket_timeout_ms: int = 2000
    legacy_node_id: str = "legacy-v1"


@dataclass(frozen=True)
class EventSpec:
    seq: int
    event_type: str
    line: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecBundle:
    root: Path
    nodes: tuple[NodeSpec, ...]
    contents: tuple[ContentSpec, ...]
    policy: LeasePolicy
    events: tuple[EventSpec, ...]
    raw_hashes: dict[str, str]

    @property
    def quota_version(self) -> int:
        return self.policy.quota_version

    @property
    def content(self) -> ContentSpec:
        return self.contents[0]

    @property
    def content_digest(self) -> str:
        return canonical_hash(
            [
                {
                    "content_key": item.content_key,
                    "total_bytes": item.total_bytes,
                    "shard_count": item.shard_count,
                    "sample_stride": item.sample_stride,
                    "integer_scale": item.integer_scale,
                    "seed": item.seed,
                    "invalid_refs": [asdict(ref) for ref in item.invalid_refs],
                }
                for item in self.contents
            ]
        )

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise SpecError(f"unknown node {node_id!r}", "node_id")

    def content_for(self, content_key: str) -> ContentSpec:
        for item in self.contents:
            if item.content_key == content_key:
                return item
        raise SpecError(f"unknown content key {content_key!r}", "content_key")

    def canonical(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in sorted(self.nodes, key=lambda n: n.node_id)],
            "contents": [asdict(item) for item in sorted(self.contents, key=lambda c: c.content_key)],
            "policy": asdict(self.policy),
            "events": [
                {"seq": event.seq, "event_type": event.event_type, "line": event.line}
                for event in self.events
            ],
        }


def _validate_nodes(value: Any, lines: dict[str, int]) -> tuple[NodeSpec, ...]:
    if not isinstance(value, dict) or "nodes" not in value:
        raise SpecError("nodes.json must contain a nodes array", "nodes", lines.get("$.nodes", 1))
    raw_nodes = value["nodes"]
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SpecError("nodes must be a non-empty array", "$.nodes", _line(lines, "$.nodes"))

    nodes: list[NodeSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_nodes):
        base = f"$.nodes[{index}]"
        if not isinstance(raw, dict):
            raise SpecError("node declaration must be an object", base, _line(lines, base))
        node_id = require_str(raw, "node_id", base, lines)
        if node_id in seen_ids:
            raise SpecError(f"duplicate node_id {node_id!r}", f"{base}.node_id", _line(lines, f"{base}.node_id"))
        seen_ids.add(node_id)

        host = require_str(raw, "host", base, lines)
        if host != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", f"{base}.host", _line(lines, f"{base}.host"))

        port = require_int(raw, "port", base, lines)
        if not (1024 < port <= 65535):
            raise SpecError("port must be an unprivileged high port in 1025..65535", f"{base}.port", _line(lines, f"{base}.port"))
        endpoint = (host, port)
        if endpoint in seen_ports:
            raise SpecError("loopback host/port must be globally unique", f"{base}.port", _line(lines, f"{base}.port"))
        seen_ports.add(endpoint)

        capacity = require_int(raw, "capacity_slots", base, lines)
        if capacity < 0:
            raise SpecError("capacity_slots must be a non-negative integer", f"{base}.capacity_slots", _line(lines, f"{base}.capacity_slots"))
        max_leases = require_int(raw, "max_node_leases", base, lines)
        if max_leases < 0:
            raise SpecError("max_node_leases must be a non-negative integer", f"{base}.max_node_leases", _line(lines, f"{base}.max_node_leases"))
        seed = require_int(raw, "seed", base, lines)
        if not (0 <= seed <= 0xFFFFFFFF):
            raise SpecError("seed must fit in an unsigned 32-bit integer", f"{base}.seed", _line(lines, f"{base}.seed"))

        nodes.append(NodeSpec(node_id, host, port, capacity, max_leases, seed))
    return tuple(nodes)


def _validate_contents(value: Any, lines: dict[str, int]) -> tuple[ContentSpec, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("contents"), list) or not value["contents"]:
        raise SpecError("content_manifest.json must contain a non-empty contents array", "contents", lines.get("$.contents", 1))
    result: list[ContentSpec] = []
    seen_keys: set[str] = set()
    for index, raw in enumerate(value["contents"]):
        base = f"$.contents[{index}]"
        if not isinstance(raw, dict):
            raise SpecError("content declaration must be an object", base, _line(lines, base))
        key = require_str(raw, "content_key", base, lines)
        if key in seen_keys:
            raise SpecError(f"duplicate content_key {key!r}", f"{base}.content_key", _line(lines, f"{base}.content_key"))
        if len(key) > 40:
            raise SpecError("content_key exceeds v1 fixed width of 40", f"{base}.content_key", _line(lines, f"{base}.content_key"))
        seen_keys.add(key)

        total_bytes = require_int(raw, "total_bytes", base, lines)
        if total_bytes <= 0:
            raise SpecError("total_bytes must be positive", f"{base}.total_bytes", _line(lines, f"{base}.total_bytes"))
        shard_count = require_int(raw, "shard_count", base, lines)
        if shard_count <= 0:
            raise SpecError("shard_count must be positive", f"{base}.shard_count", _line(lines, f"{base}.shard_count"))
        if total_bytes % shard_count != 0:
            raise SpecError("shard_count must divide total_bytes exactly", f"{base}.shard_count", _line(lines, f"{base}.shard_count"))

        stride = require_int(raw, "sample_stride", base, lines)
        if stride <= 0:
            raise SpecError("sample_stride must be positive", f"{base}.sample_stride", _line(lines, f"{base}.sample_stride"))
        scale = require_int(raw, "integer_scale", base, lines)
        if scale < 0:
            raise SpecError("integer_scale must be a non-negative integer", f"{base}.integer_scale", _line(lines, f"{base}.integer_scale"))
        if scale not in ALLOWED_FIXED_SCALES:
            raise SpecError(
                f"integer_scale must be one of {ALLOWED_FIXED_SCALES}",
                f"{base}.integer_scale",
                _line(lines, f"{base}.integer_scale"),
            )
        seed = require_int(raw, "seed", base, lines)
        if not (0 <= seed <= 0x7FFFFFFFFFFFFFFF):
            raise SpecError("seed must be a non-negative signed 64-bit integer", f"{base}.seed", _line(lines, f"{base}.seed"))

        refs: list[InvalidRefSpec] = []
        raw_refs = raw.get("invalid_refs", [])
        if raw_refs is None:
            raw_refs = []
        refs_path = f"{base}.invalid_refs"
        refs_line = _line(lines, refs_path)
        if not isinstance(raw_refs, list):
            raise SpecError("invalid_refs must be an array", refs_path, refs_line)
        valid_shards = {f"shard-{i:04d}" for i in range(shard_count)}
        seen_refs: set[str] = set()
        for ref_index, raw_ref in enumerate(raw_refs):
            ref_base = f"{refs_path}[{ref_index}]"
            if not isinstance(raw_ref, dict):
                raise SpecError("invalid ref must be an object", ref_base, _line(lines, ref_base))
            ref_id = require_str(raw_ref, "ref_id", ref_base, lines)
            if ref_id in seen_refs:
                raise SpecError("duplicate invalid ref_id", f"{ref_base}.ref_id", _line(lines, f"{ref_base}.ref_id"))
            seen_refs.add(ref_id)
            shard_id = require_str(raw_ref, "shard_id", ref_base, lines)
            if shard_id not in valid_shards:
                raise SpecError("invalid ref target is dangling or outside manifest shards", f"{ref_base}.shard_id", _line(lines, f"{ref_base}.shard_id"))
            reason = raw_ref.get("reason", "declared-invalid")
            if not isinstance(reason, str) or not reason:
                raise SpecError("invalid ref reason must be a non-empty string", f"{ref_base}.reason", _line(lines, f"{ref_base}.reason"))
            refs.append(InvalidRefSpec(ref_id, shard_id, reason))

        result.append(
            ContentSpec(key, total_bytes, shard_count, stride, scale, seed, tuple(refs))
        )
    return tuple(result)


def _validate_policy(value: Any, lines: dict[str, int], nodes: tuple[NodeSpec, ...]) -> LeasePolicy:
    if not isinstance(value, dict):
        raise SpecError("lease_policy.json must be an object", "$", lines.get("$", 1))
    base = "$"
    quota_version = require_int(value, "quota_version", base, lines)
    if quota_version < 0:
        raise SpecError("quota_version must be non-negative", "$.quota_version", _line(lines, "$.quota_version"))
    ttl_ms = require_int(value, "ttl_ms", base, lines)
    if ttl_ms <= 0:
        raise SpecError("ttl_ms must be positive", "$.ttl_ms", _line(lines, "$.ttl_ms"))
    max_renewals = require_int(value, "max_renewals", base, lines)
    if max_renewals < 0:
        raise SpecError("max_renewals must be non-negative", "$.max_renewals", _line(lines, "$.max_renewals"))
    global_max = require_int(value, "global_max_active_leases", base, lines)
    if global_max < 0:
        raise SpecError("global_max_active_leases must be non-negative", "$.global_max_active_leases", _line(lines, "$.global_max_active_leases"))
    byte_budget = require_int(value, "global_byte_budget", base, lines)
    if byte_budget < 0:
        raise SpecError("global_byte_budget must be non-negative", "$.global_byte_budget", _line(lines, "$.global_byte_budget"))

    reaper = value.get("reaper_interval_ms", 25)
    if not is_int(reaper) or reaper <= 0:
        raise SpecError("reaper_interval_ms must be a positive integer", "$.reaper_interval_ms", _line(lines, "$.reaper_interval_ms"))
    queue_timeout = value.get("queue_timeout_ms", 5000)
    if not is_int(queue_timeout) or queue_timeout <= 0:
        raise SpecError("queue_timeout_ms must be a positive integer", "$.queue_timeout_ms", _line(lines, "$.queue_timeout_ms"))
    socket_timeout = value.get("socket_timeout_ms", 2000)
    if not is_int(socket_timeout) or socket_timeout <= 0:
        raise SpecError("socket_timeout_ms must be a positive integer", "$.socket_timeout_ms", _line(lines, "$.socket_timeout_ms"))
    legacy_node_id = value.get("legacy_node_id", "legacy-v1")
    if not isinstance(legacy_node_id, str) or not legacy_node_id:
        raise SpecError("legacy_node_id must be a non-empty string", "$.legacy_node_id", _line(lines, "$.legacy_node_id"))
    if legacy_node_id != "legacy-v1" and all(node.node_id != legacy_node_id for node in nodes):
        raise SpecError("legacy_node_id does not match a declared node", "$.legacy_node_id", _line(lines, "$.legacy_node_id"))

    return LeasePolicy(
        quota_version=quota_version,
        ttl_ms=ttl_ms,
        max_renewals=max_renewals,
        global_max_active_leases=global_max,
        global_byte_budget=byte_budget,
        reaper_interval_ms=reaper,
        queue_timeout_ms=queue_timeout,
        socket_timeout_ms=socket_timeout,
        legacy_node_id=legacy_node_id,
    )


def _validate_events(path: Path) -> tuple[EventSpec, ...]:
    jsonl_path = path / "event_stream.jsonl"
    json_path = path / "event_stream.json"
    events: list[EventSpec] = []
    if jsonl_path.exists():
        records, _ = load_jsonl_file(jsonl_path)
        previous: int | None = None
        for record in records:
            line = int(record.pop("__line__", 1))
            if "seq" not in record or not is_int(record["seq"]):
                raise SpecError("event seq must be an integer", "seq", line)
            seq = int(record["seq"])
            if seq < 0:
                raise SpecError("event seq must be non-negative", "seq", line)
            if previous is not None and seq <= previous:
                raise SpecError(f"event seq {seq} is not strictly greater than {previous}", "seq", line)
            previous = seq
            event_type = record.get("event_type")
            if not isinstance(event_type, str) or not event_type:
                raise SpecError("event_type must be a non-empty string", "event_type", line)
            events.append(EventSpec(seq, event_type, line, {k: v for k, v in record.items() if k not in ("seq", "event_type")}))
    elif json_path.exists():
        value, lines, _ = load_json_file(json_path)
        if not isinstance(value, dict) or not isinstance(value.get("events"), list):
            raise SpecError("event_stream.json must contain an events array", "events", lines.get("$.events", 1))
        previous = None
        for index, record in enumerate(value["events"]):
            base = f"$.events[{index}]"
            if not isinstance(record, dict):
                raise SpecError("event must be an object", base, _line(lines, base))
            seq = require_int(record, "seq", base, lines)
            if previous is not None and seq <= previous:
                raise SpecError(f"event seq {seq} is not strictly greater than {previous}", f"{base}.seq", _line(lines, f"{base}.seq"))
            event_type = require_str(record, "event_type", base, lines)
            previous = seq
            events.append(EventSpec(seq, event_type, _line(lines, f"{base}.seq"), {k: v for k, v in record.items() if k not in ("seq", "event_type")}))
    else:
        raise SpecError("missing event_stream.jsonl or event_stream.json", "$", 1)
    return tuple(events)


def load_spec_dir(root: str | Path) -> SpecBundle:
    path = Path(root)
    nodes_value, nodes_lines, nodes_text = load_json_file(path / "nodes.json")
    nodes = _validate_nodes(nodes_value, nodes_lines)

    content_value, content_lines, content_text = load_json_file(path / "content_manifest.json")
    contents = _validate_contents(content_value, content_lines)

    policy_value, policy_lines, policy_text = load_json_file(path / "lease_policy.json")
    policy = _validate_policy(policy_value, policy_lines, nodes)

    events = _validate_events(path)

    raw_hashes = {
        "nodes.json": sha256_hex(nodes_text),
        "content_manifest.json": sha256_hex(content_text),
        "lease_policy.json": sha256_hex(policy_text),
    }
    event_path = path / "event_stream.jsonl"
    if event_path.exists():
        raw_hashes["event_stream.jsonl"] = sha256_hex(event_path.read_text(encoding="utf-8"))
    else:
        raw_hashes["event_stream.json"] = sha256_hex((path / "event_stream.json").read_text(encoding="utf-8"))

    return SpecBundle(path, nodes, contents, policy, events, raw_hashes)


def create_readonly_snapshot(spec_dir: str | Path, snapshot_dir: str | Path | None = None) -> Path:
    src = Path(spec_dir)
    dest = Path(snapshot_dir) if snapshot_dir is not None else src / "raw_snapshot"
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for name in SPEC_FILES:
        source = src / name
        if not source.exists():
            continue
        target = dest / name
        text = source.read_text(encoding="utf-8")
        if target.exists():
            if target.read_text(encoding="utf-8") != text:
                raise SpecError(f"read-only snapshot {name} already exists with different content", "snapshot", 1)
        else:
            target.write_text(text, encoding="utf-8")
            target.chmod(0o444)
        manifest[name] = sha256_hex(text)
    manifest_text = json.dumps({"files": manifest}, sort_keys=True, indent=2) + "\n"
    manifest_path = dest / "SNAPSHOT_MANIFEST.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("files") != manifest:
                raise SpecError("raw snapshot already exists with different content", "snapshot", 1)
        except (OSError, json.JSONDecodeError) as exc:
            raise SpecError("raw snapshot manifest is unreadable", "snapshot", 1) from exc
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_path.chmod(0o444)
    return dest
