"""本地 JSON 规格的加载与强校验。

规格包含四个顶层段落：nodes / shards / revisions / events。
任何一条不合法都抛出 :class:`SpecError`（E_SPEC），携带首个违规的
``location`` 与 ``field``，调用方不会拿到部分加载结果。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SpecError
from .fixedpoint import ALLOWED_SCALES

EVENT_KINDS = ("dispatch", "renew", "cancel", "revise", "exhausted")


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    rate: int
    burst: int
    bucket_key: int
    seed: int
    param_version: int


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    upstream: tuple[str, ...]
    sample_rows: int
    weight: int
    priority: int


@dataclass(frozen=True)
class Revision:
    version: int
    parent: int | None
    global_rate: int | None
    global_burst: int | None
    node_rates: dict[str, int]
    node_bursts: dict[str, int]
    shard_deps: dict[str, tuple[str, ...]]
    scale: int | None


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    ts_ms: int
    shard_id: str | None
    node_id: str | None
    version: int | None
    expected_version: int | None = None


@dataclass
class Spec:
    nodes: tuple[NodeSpec, ...]
    shards: tuple[ShardSpec, ...]
    revisions: tuple[Revision, ...]
    events: tuple[Event, ...]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def shard(self, shard_id: str) -> ShardSpec:
        for shard in self.shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(shard_id)


def _fail(location: str, field_name: str, reason: str) -> None:
    raise SpecError(f"{location}.{field_name}: {reason}",
                    location=location, field=field_name)


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_ident(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require(obj: Any, key: str, location: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        _fail(location, key, "required field missing")
    return obj[key]


def _check_nodes(raw_nodes: Any) -> list[NodeSpec]:
    if not isinstance(raw_nodes, list) or not raw_nodes:
        _fail("nodes", "<list>", "must be a non-empty list")
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    result: list[NodeSpec] = []
    for index, item in enumerate(raw_nodes):
        loc = f"nodes[{index}]"
        if not isinstance(item, dict):
            _fail(loc, "<object>", "node entry must be an object")
        node_id = _require(item, "node_id", loc)
        if not _is_ident(node_id):
            _fail(loc, "node_id", "must be a non-empty string")
        if node_id in seen_ids:
            _fail(loc, "node_id", f"duplicate node id {node_id!r}")
        seen_ids.add(node_id)
        host = _require(item, "host", loc)
        if host != "127.0.0.1":
            _fail(loc, "host", "must be the literal 127.0.0.1")
        port = _require(item, "port", loc)
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            _fail(loc, "port", "must be an integer in [1, 65535]")
        if port in seen_ports:
            _fail(loc, "port", f"port {port} already used by another node")
        seen_ports.add(port)
        for field_name in ("rate", "burst", "bucket_key", "seed", "param_version"):
            value = _require(item, field_name, loc)
            if not _is_nonneg_int(value):
                _fail(loc, field_name, "must be a non-negative integer")
        result.append(NodeSpec(
            node_id=node_id, host=host, port=port, rate=item["rate"],
            burst=item["burst"], bucket_key=item["bucket_key"],
            seed=item["seed"], param_version=item["param_version"],
        ))
    return result


def _check_shards(raw_shards: Any, node_ids: set[str]) -> list[ShardSpec]:
    if not isinstance(raw_shards, list) or not raw_shards:
        _fail("shards", "<list>", "must be a non-empty list")
    seen_ids: set[str] = set()
    result: list[ShardSpec] = []
    for index, item in enumerate(raw_shards):
        loc = f"shards[{index}]"
        if not isinstance(item, dict):
            _fail(loc, "<object>", "shard entry must be an object")
        shard_id = _require(item, "shard_id", loc)
        if not _is_ident(shard_id):
            _fail(loc, "shard_id", "must be a non-empty string")
        if shard_id in seen_ids:
            _fail(loc, "shard_id", f"duplicate shard id {shard_id!r}")
        seen_ids.add(shard_id)
        upstream = _require(item, "upstream", loc)
        if not isinstance(upstream, list) or not all(_is_ident(u) for u in upstream):
            _fail(loc, "upstream", "must be a list of shard id strings")
        if len(set(upstream)) != len(upstream):
            _fail(loc, "upstream", "duplicate upstream reference")
        for field_name in ("sample_rows", "weight", "priority"):
            value = _require(item, field_name, loc)
            if not _is_nonneg_int(value):
                _fail(loc, field_name, "must be a non-negative integer")
        result.append(ShardSpec(
            shard_id=shard_id, upstream=tuple(upstream),
            sample_rows=item["sample_rows"], weight=item["weight"],
            priority=item["priority"],
        ))
    shard_ids = seen_ids
    for shard in result:
        for ref in shard.upstream:
            if ref not in shard_ids:
                _fail(f"shards[{shard.shard_id}]", "upstream",
                      f"unknown upstream shard {ref!r}")
            if ref == shard.shard_id:
                _fail(f"shards[{shard.shard_id}]", "upstream", "self dependency")
    _assert_acyclic(result)
    return result


def _assert_acyclic(shards: list[ShardSpec]) -> None:
    deps = {s.shard_id: set(s.upstream) for s in shards}
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visiting:
            _fail("shards", "upstream", f"dependency cycle at {node_id!r}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for ref in sorted(deps[node_id]):
            walk(ref)
        visiting.discard(node_id)
        visited.add(node_id)

    for shard_id in sorted(deps):
        walk(shard_id)


def _check_revisions(raw_revisions: Any, node_ids: set[str],
                     shard_ids: set[str]) -> list[Revision]:
    if not isinstance(raw_revisions, list) or not raw_revisions:
        _fail("revisions", "<list>", "must be a non-empty list")
    result: list[Revision] = []
    seen_versions: set[int] = set()
    previous_version: int | None = None
    for index, item in enumerate(raw_revisions):
        loc = f"revisions[{index}]"
        if not isinstance(item, dict):
            _fail(loc, "<object>", "revision entry must be an object")
        version = _require(item, "version", loc)
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            _fail(loc, "version", "must be a non-negative integer")
        if previous_version is not None and version <= previous_version:
            _fail(loc, "version",
                  f"must be strictly increasing; previous={previous_version}")
        if version in seen_versions:
            _fail(loc, "version", f"duplicate version {version}")
        seen_versions.add(version)
        previous_version = version
        parent = item.get("parent")
        if parent is not None:
            if not isinstance(parent, int) or isinstance(parent, bool):
                _fail(loc, "parent", "must be an integer or null")
            if parent not in seen_versions or parent == version:
                _fail(loc, "parent", f"unknown parent revision {parent}")
        elif index != 0:
            _fail(loc, "parent", "only the first revision may omit parent")
        for optional in ("global_rate", "global_burst", "scale"):
            value = item.get(optional)
            if value is not None and not _is_nonneg_int(value):
                _fail(loc, optional, "must be a non-negative integer or null")
        scale = item.get("scale")
        if scale is not None and scale not in ALLOWED_SCALES:
            _fail(loc, "scale", f"must be a power of ten in {ALLOWED_SCALES}")
        node_rates = _check_int_map(item.get("node_rates", {}), loc, "node_rates",
                                    node_ids)
        node_bursts = _check_int_map(item.get("node_bursts", {}), loc,
                                     "node_bursts", node_ids)
        raw_deps = item.get("shard_deps", {})
        if not isinstance(raw_deps, dict):
            _fail(loc, "shard_deps", "must be an object of shard id -> list")
        shard_deps: dict[str, tuple[str, ...]] = {}
        for shard_id, deps in raw_deps.items():
            if shard_id not in shard_ids:
                _fail(loc, "shard_deps", f"unknown shard id {shard_id!r}")
            if not isinstance(deps, list) or not all(_is_ident(d) for d in deps):
                _fail(loc, "shard_deps",
                      f"deps of {shard_id!r} must be a list of shard id strings")
            unknown = [d for d in deps if d not in shard_ids]
            if unknown:
                _fail(loc, "shard_deps",
                      f"unknown dependency {unknown[0]!r} for shard {shard_id!r}")
            if shard_id in deps:
                _fail(loc, "shard_deps", f"shard {shard_id!r} cannot depend on itself")
            shard_deps[shard_id] = tuple(deps)
        result.append(Revision(
            version=version, parent=parent,
            global_rate=item.get("global_rate"),
            global_burst=item.get("global_burst"),
            node_rates=node_rates, node_bursts=node_bursts,
            shard_deps=shard_deps, scale=scale,
        ))
    _assert_revision_chain(result)
    return result


def _check_int_map(raw: Any, loc: str, field_name: str,
                  allowed_ids: set[str]) -> dict[str, int]:
    if not isinstance(raw, dict):
        _fail(loc, field_name, "must be an object of id -> non-negative integer")
    mapped: dict[str, int] = {}
    for key, value in raw.items():
        if key not in allowed_ids:
            _fail(loc, field_name, f"unknown node id {key!r}")
        if not _is_nonneg_int(value):
            _fail(loc, field_name, f"value for {key!r} must be a non-negative integer")
        mapped[key] = value
    return mapped


def _assert_revision_chain(revisions: list[Revision]) -> None:
    by_version = {r.version: r for r in revisions}
    roots: set[int] = set()
    for revision in revisions:
        seen: set[int] = set()
        current = revision
        while True:
            if current.version in seen:
                _fail("revisions", "parent",
                      f"parent chain cycle at version {current.version}")
            seen.add(current.version)
            if current.parent is None:
                roots.add(current.version)
                break
            current = by_version[current.parent]
    if len(roots) != 1:
        _fail("revisions", "parent",
              "revision chain must have exactly one root")


def _check_events(raw_events: Any, node_ids: set[str],
                  shard_ids: set[str], versions: set[int]) -> list[Event]:
    if not isinstance(raw_events, list):
        _fail("events", "<list>", "must be a list")
    seen_seq: set[int] = set()
    result: list[Event] = []
    last_ts = -1
    for index, item in enumerate(raw_events):
        loc = f"events[{index}]"
        if not isinstance(item, dict):
            _fail(loc, "<object>", "event entry must be an object")
        seq = _require(item, "seq", loc)
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            _fail(loc, "seq", "must be a non-negative integer")
        if seq in seen_seq:
            _fail(loc, "seq", f"duplicate arrival seq {seq}")
        seen_seq.add(seq)
        kind = _require(item, "kind", loc)
        if kind not in EVENT_KINDS:
            _fail(loc, "kind", f"must be one of {EVENT_KINDS}")
        ts_ms = _require(item, "ts_ms", loc)
        if not isinstance(ts_ms, int) or isinstance(ts_ms, bool) or ts_ms < 0:
            _fail(loc, "ts_ms", "must be a non-negative integer")
        if ts_ms < last_ts:
            _fail(loc, "ts_ms", "event timestamps must be non-decreasing")
        last_ts = ts_ms
        shard_id = item.get("shard_id")
        if shard_id is not None:
            if not _is_ident(shard_id) or shard_id not in shard_ids:
                _fail(loc, "shard_id", f"unknown shard id {shard_id!r}")
        if kind in ("dispatch", "cancel", "exhausted") and shard_id is None:
            _fail(loc, "shard_id", f"{kind} event requires shard_id")
        node_id = item.get("node_id")
        if node_id is not None and node_id not in node_ids:
            _fail(loc, "node_id", f"unknown node id {node_id!r}")
        if kind == "dispatch" and node_id is None:
            _fail(loc, "node_id", "dispatch event requires node_id")
        version = item.get("version")
        if version is not None and (not isinstance(version, int)
                                    or isinstance(version, bool)
                                    or version not in versions):
            _fail(loc, "version", f"unknown revision version {version}")
        if kind == "revise" and version is None:
            _fail(loc, "version", "revise event requires version")
        expected_version = item.get("expected_version")
        if expected_version is not None and (
                not isinstance(expected_version, int)
                or isinstance(expected_version, bool)
                or expected_version <= 0):
            _fail(loc, "expected_version",
                  "must be a positive integer or null")
        if kind == "renew":
            if shard_id is None:
                _fail(loc, "shard_id", "renew event requires shard_id")
            if expected_version is None:
                _fail(loc, "expected_version",
                      "renew event requires expected_version")
        result.append(Event(seq=seq, kind=kind, ts_ms=ts_ms,
                            shard_id=shard_id, node_id=node_id, version=version,
                            expected_version=expected_version))
    seqs = [event.seq for event in result]
    if seqs != sorted(seqs):
        _fail("events", "seq", "events must be ordered by arrival seq")
    return result


def load_spec(path: str | Path) -> Spec:
    """从 JSON 文件加载规格；失败时抛 E_SPEC，不返回部分结果。"""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"spec file not found: {path}", location="<file>",
                        field="path")
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc.msg} at line {exc.lineno}",
                        location="<json>", field="<parse>")
    return parse_spec(raw)


def parse_spec(raw: dict[str, Any]) -> Spec:
    if not isinstance(raw, dict):
        _fail("<root>", "<object>", "spec root must be a JSON object")
    nodes = _check_nodes(raw.get("nodes"))
    node_ids = {node.node_id for node in nodes}
    shards = _check_shards(raw.get("shards"), node_ids)
    shard_ids = {shard.shard_id for shard in shards}
    revisions = _check_revisions(raw.get("revisions"), node_ids, shard_ids)
    versions = {revision.version for revision in revisions}
    events = _check_events(raw.get("events", []), node_ids, shard_ids, versions)
    return Spec(nodes=tuple(nodes), shards=tuple(shards),
                revisions=tuple(revisions), events=tuple(events), raw=raw)
