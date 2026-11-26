from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError
from .hashing import sha256_hex
from .topology import topological_order


REQUIRED_FILES = (
    "session.json",
    "nodes.json",
    "shards.json",
    "capacity_versions.json",
    "events.json",
)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("must be a JSON object", path=path)
    return value


def _require_str(obj: dict[str, Any], key: str, path: str, *, nonempty: bool = True) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or (nonempty and not value):
        raise SpecError(f"{key} must be a non-empty string", path=f"{path}.{key}")
    return value


def _require_int(
    obj: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    nonnegative: bool = False,
    positive: bool = False,
) -> int:
    value = obj.get(key)
    field_path = f"{path}.{key}"
    if not is_int(value):
        raise SpecError(f"{key} must be an integer", path=field_path)
    assert isinstance(value, int)
    low = 0 if nonnegative else (1 if positive else minimum)
    if low is not None and value < low:
        adjective = "non-negative" if nonnegative else ("positive" if positive else f">= {low}")
        raise SpecError(f"{key} must be {adjective}", path=field_path)
    if maximum is not None and value > maximum:
        raise SpecError(f"{key} must be <= {maximum}", path=field_path)
    return value


def _optional_int(obj: dict[str, Any], key: str, path: str, default: int, **kwargs: Any) -> int:
    if key not in obj:
        return default
    return _require_int(obj, key, path, **kwargs)


def _reject_unknown(obj: dict[str, Any], allowed: set[str], path: str) -> None:
    for key in obj:
        if key not in allowed:
            raise SpecError(f"unknown field {key!r}", path=f"{path}.{key}")


@dataclass(frozen=True)
class SessionConfig:
    spec_version: str
    scale_exp: int
    global_resource_budget: int
    lease_ttl_ms: int
    request_timeout_ms: int = 2000
    handshake_timeout_ms: int = 1000
    wait_poll_ms: int = 10
    queue_window_ms: int = 64


@dataclass(frozen=True)
class NodeDecl:
    node_id: str
    host: str
    port: int
    seed: int
    concurrency_quota: int
    max_leases: int
    fixed_point_scale: int = 100
    processing_delay_ms: int = 0


@dataclass(frozen=True)
class ShardDecl:
    shard_id: str
    node_id: str
    upstreams: tuple[str, ...]
    resource_units: int
    capacity_version: str
    sample_rows: int = 96
    batch_size: int = 32
    state: str = "active"


@dataclass(frozen=True)
class CapacityVersion:
    version: str
    parent: str | None
    budget_units: int
    revision: int
    note: str = ""


@dataclass(frozen=True)
class ScheduleEvent:
    event_id: str
    event_type: str
    sequence: int
    at_monotonic_ms: int
    shard_id: str | None = None
    node_id: str | None = None
    capacity_version: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpecBundle:
    session: SessionConfig
    nodes: tuple[NodeDecl, ...]
    shards: tuple[ShardDecl, ...]
    capacity_versions: tuple[CapacityVersion, ...]
    events: tuple[ScheduleEvent, ...]
    raw: dict[str, Any]
    source_path: Path | None = None
    spec_digest: str = ""

    @property
    def scale(self) -> int:
        return 10 ** self.session.scale_exp

    def node(self, node_id: str) -> NodeDecl:
        return {item.node_id: item for item in self.nodes}[node_id]

    def shard(self, shard_id: str) -> ShardDecl:
        return {item.shard_id: item for item in self.shards}[shard_id]

    def capacity(self, version: str) -> CapacityVersion:
        return {item.version: item for item in self.capacity_versions}[version]


def _load_json_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read {path.name}: {exc}", path=f"$/{path.name}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON in {path.name}: line {exc.lineno} column {exc.colno}: {exc.msg}",
            path=f"$/{path.name}",
        ) from exc


def load_spec(path: str | Path) -> SpecBundle:
    source = Path(path)
    if source.is_dir():
        raw = {
            "session": _load_json_file(source / "session.json"),
            "nodes": _load_json_file(source / "nodes.json"),
            "shards": _load_json_file(source / "shards.json"),
            "capacity_versions": _load_json_file(source / "capacity_versions.json"),
            "events": _load_json_file(source / "events.json"),
        }
        for key in ("nodes", "shards", "capacity_versions", "events"):
            raw[key] = raw[key].get(key, []) if isinstance(raw[key], dict) else raw[key]
    else:
        combined = _load_json_file(source)
        raw = combined

    bundle = parse_spec(raw, source_path=source if source.exists() else None)
    return bundle


def parse_spec(raw: dict[str, Any], *, source_path: Path | None = None) -> SpecBundle:
    root = _require_dict(raw, "$")
    _reject_unknown(
        root,
        {"session", "nodes", "shards", "capacity_versions", "events", "format_version"},
        "$",
    )

    session_raw = _require_dict(root.get("session"), "$.session")
    _reject_unknown(
        session_raw,
        {
            "spec_version",
            "scale_exp",
            "global_resource_budget",
            "lease_ttl_ms",
            "request_timeout_ms",
            "handshake_timeout_ms",
            "wait_poll_ms",
            "queue_window_ms",
        },
        "$.session",
    )
    session = SessionConfig(
        spec_version=_require_str(session_raw, "spec_version", "$.session"),
        scale_exp=_require_int(session_raw, "scale_exp", "$.session", minimum=0, maximum=3),
        global_resource_budget=_require_int(
            session_raw, "global_resource_budget", "$.session", positive=True
        ),
        lease_ttl_ms=_require_int(session_raw, "lease_ttl_ms", "$.session", positive=True),
        request_timeout_ms=_optional_int(
            session_raw, "request_timeout_ms", "$.session", 2000, positive=True
        ),
        handshake_timeout_ms=_optional_int(
            session_raw, "handshake_timeout_ms", "$.session", 1000, positive=True
        ),
        wait_poll_ms=_optional_int(session_raw, "wait_poll_ms", "$.session", 10, positive=True),
        queue_window_ms=_optional_int(
            session_raw, "queue_window_ms", "$.session", 64, positive=True
        ),
    )

    nodes_raw = root.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise SpecError("nodes must be a non-empty list", path="$.nodes")
    nodes: list[NodeDecl] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, item in enumerate(nodes_raw):
        path = f"$.nodes[{index}]"
        obj = _require_dict(item, path)
        _reject_unknown(
            obj,
            {
                "node_id",
                "host",
                "port",
                "seed",
                "concurrency_quota",
                "max_leases",
                "fixed_point_scale",
                "processing_delay_ms",
            },
            path,
        )
        node_id = _require_str(obj, "node_id", path)
        if node_id in seen_node_ids:
            raise SpecError(f"duplicate node_id {node_id!r}", path=f"{path}.node_id")
        seen_node_ids.add(node_id)
        host = _require_str(obj, "host", path)
        if host != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", path=f"{path}.host")
        port = _require_int(obj, "port", path, minimum=1, maximum=65535)
        if port in seen_ports:
            raise SpecError(f"duplicate loopback port {port}", path=f"{path}.port")
        seen_ports.add(port)
        seed = _require_int(obj, "seed", path, minimum=0, maximum=2**32 - 1)
        quota = _require_int(obj, "concurrency_quota", path, nonnegative=True)
        max_leases = _require_int(obj, "max_leases", path, nonnegative=True)
        fp_scale = _optional_int(obj, "fixed_point_scale", path, 100, minimum=1)
        if fp_scale not in {1, 10, 100, 1000, 10000, 100000, 1000000}:
            raise SpecError(
                "fixed_point_scale must be a power of ten from 1 to 1000000",
                path=f"{path}.fixed_point_scale",
            )
        delay = _optional_int(obj, "processing_delay_ms", path, 0, nonnegative=True)
        nodes.append(
            NodeDecl(node_id, host, port, seed, quota, max_leases, fp_scale, delay)
        )

    capacities_raw = root.get("capacity_versions")
    if not isinstance(capacities_raw, list) or not capacities_raw:
        raise SpecError("capacity_versions must be a non-empty list", path="$.capacity_versions")
    capacities: list[CapacityVersion] = []
    seen_versions: set[str] = set()
    for index, item in enumerate(capacities_raw):
        path = f"$.capacity_versions[{index}]"
        obj = _require_dict(item, path)
        _reject_unknown(obj, {"version", "parent", "budget_units", "revision", "note"}, path)
        version = _require_str(obj, "version", path)
        if version in seen_versions:
            raise SpecError(f"duplicate capacity version {version!r}", path=f"{path}.version")
        seen_versions.add(version)
        parent = obj.get("parent")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise SpecError("parent must be null or a non-empty string", path=f"{path}.parent")
        budget = _require_int(obj, "budget_units", path, positive=True)
        revision = _require_int(obj, "revision", path, positive=True)
        note = obj.get("note", "")
        if not isinstance(note, str):
            raise SpecError("note must be a string", path=f"{path}.note")
        capacities.append(CapacityVersion(version, parent, budget, revision, note))

    cap_map = {item.version: item for item in capacities}
    for cap in capacities:
        path = f"$.capacity_versions"
        seen_chain: set[str] = set()
        current = cap
        while current.parent is not None:
            if current.parent not in cap_map:
                raise SpecError(
                    f"parent capacity version {current.parent!r} does not exist",
                    path=f"{path}[{cap.version}].parent",
                )
            if current.version in seen_chain:
                raise SpecError("capacity version parent chain contains a cycle", path=path)
            seen_chain.add(current.version)
            current = cap_map[current.parent]

    shards_raw = root.get("shards")
    if not isinstance(shards_raw, list) or not shards_raw:
        raise SpecError("shards must be a non-empty list", path="$.shards")
    node_map = {item.node_id: item for item in nodes}
    shards: list[ShardDecl] = []
    seen_shard_ids: set[str] = set()
    for index, item in enumerate(shards_raw):
        path = f"$.shards[{index}]"
        obj = _require_dict(item, path)
        _reject_unknown(
            obj,
            {
                "shard_id",
                "node_id",
                "upstreams",
                "resource_units",
                "capacity_version",
                "sample_rows",
                "batch_size",
                "state",
            },
            path,
        )
        shard_id = _require_str(obj, "shard_id", path)
        if shard_id in seen_shard_ids:
            raise SpecError(f"duplicate shard_id {shard_id!r}", path=f"{path}.shard_id")
        seen_shard_ids.add(shard_id)
        node_id = _require_str(obj, "node_id", path)
        if node_id not in node_map:
            raise SpecError(f"node_id {node_id!r} does not exist", path=f"{path}.node_id")
        upstreams_raw = obj.get("upstreams", [])
        if not isinstance(upstreams_raw, list):
            raise SpecError("upstreams must be a list", path=f"{path}.upstreams")
        upstreams: list[str] = []
        seen_upstream: set[str] = set()
        for up_index, upstream in enumerate(upstreams_raw):
            up_path = f"{path}.upstreams[{up_index}]"
            if not isinstance(upstream, str) or not upstream:
                raise SpecError("upstream id must be a non-empty string", path=up_path)
            if upstream in seen_upstream:
                raise SpecError(f"duplicate upstream {upstream!r}", path=up_path)
            if upstream == shard_id:
                raise SpecError("shard cannot depend on itself", path=up_path)
            seen_upstream.add(upstream)
            upstreams.append(upstream)
        resource_units = _require_int(obj, "resource_units", path, positive=True)
        capacity_version = _require_str(obj, "capacity_version", path)
        if capacity_version not in cap_map:
            raise SpecError(
                f"capacity_version {capacity_version!r} does not exist",
                path=f"{path}.capacity_version",
            )
        sample_rows = _optional_int(obj, "sample_rows", path, 96, minimum=1, maximum=1_000_000)
        batch_size = _optional_int(obj, "batch_size", path, 32, positive=True)
        if batch_size > sample_rows:
            raise SpecError("batch_size cannot exceed sample_rows", path=f"{path}.batch_size")
        state = obj.get("state", "active")
        if state not in {"active", "frozen"}:
            raise SpecError("state must be active or frozen", path=f"{path}.state")
        shards.append(
            ShardDecl(
                shard_id,
                node_id,
                tuple(upstreams),
                resource_units,
                capacity_version,
                sample_rows,
                batch_size,
                state,
            )
        )

    shard_map = {item.shard_id: item for item in shards}
    for shard in shards:
        for upstream in shard.upstreams:
            if upstream not in shard_map:
                raise SpecError(
                    f"upstream shard {upstream!r} does not exist",
                    path=f"$.shards[{shard.shard_id}].upstreams",
                )
    topological_order({item.shard_id: item for item in shards})

    events_raw = root.get("events", [])
    if not isinstance(events_raw, list):
        raise SpecError("events must be a list", path="$.events")
    events: list[ScheduleEvent] = []
    seen_event_ids: set[str] = set()
    for index, item in enumerate(events_raw):
        path = f"$.events[{index}]"
        obj = _require_dict(item, path)
        _reject_unknown(
            obj,
            {
                "event_id",
                "event_type",
                "sequence",
                "at_monotonic_ms",
                "shard_id",
                "node_id",
                "capacity_version",
                "detail",
            },
            path,
        )
        event_id = _require_str(obj, "event_id", path)
        if event_id in seen_event_ids:
            raise SpecError(f"duplicate event_id {event_id!r}", path=f"{path}.event_id")
        seen_event_ids.add(event_id)
        event_type = _require_str(obj, "event_type", path)
        allowed_types = {
            "spec_loaded",
            "quota_changed",
            "capacity_revised",
            "shard_replaced",
            "scheduler_tick",
            "lease_event",
            "manual",
        }
        if event_type not in allowed_types:
            raise SpecError(
                f"event_type must be one of {sorted(allowed_types)}",
                path=f"{path}.event_type",
            )
        sequence = _require_int(obj, "sequence", path, nonnegative=True)
        at_ms = _require_int(obj, "at_monotonic_ms", path, nonnegative=True)
        shard_id = obj.get("shard_id")
        if shard_id is not None and shard_id not in shard_map:
            raise SpecError("shard_id does not exist", path=f"{path}.shard_id")
        node_id = obj.get("node_id")
        if node_id is not None and node_id not in node_map:
            raise SpecError("node_id does not exist", path=f"{path}.node_id")
        cap_id = obj.get("capacity_version")
        if cap_id is not None and cap_id not in cap_map:
            raise SpecError("capacity_version does not exist", path=f"{path}.capacity_version")
        detail = obj.get("detail", {})
        if not isinstance(detail, dict):
            raise SpecError("detail must be an object", path=f"{path}.detail")
        events.append(
            ScheduleEvent(
                event_id,
                event_type,
                sequence,
                at_ms,
                shard_id if isinstance(shard_id, str) else None,
                node_id if isinstance(node_id, str) else None,
                cap_id if isinstance(cap_id, str) else None,
                detail,
            )
        )

    digest_document = {
        "session": session_raw,
        "nodes": nodes_raw,
        "shards": shards_raw,
        "capacity_versions": [item.__dict__ for item in capacities],
        "events": events_raw,
    }
    digest = sha256_hex(digest_document)
    normalized_raw = {
        "session": session.__dict__,
        "nodes": [item.__dict__ for item in nodes],
        "shards": [item.__dict__ for item in shards],
        "capacity_versions": [item.__dict__ for item in capacities],
        "events": [item.__dict__ for item in events],
    }
    return SpecBundle(
        session=session,
        nodes=tuple(nodes),
        shards=tuple(shards),
        capacity_versions=tuple(capacities),
        events=tuple(events),
        raw=normalized_raw,
        source_path=source_path,
        spec_digest=digest,
    )


def dump_bundle_json(bundle: SpecBundle) -> dict[str, Any]:
    return bundle.raw
