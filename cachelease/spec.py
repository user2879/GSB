from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_bytes
from .errors import SpecError


FORMAT_VERSION = 1
LOOPBACK_HOST = "127.0.0.1"
NODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    lease_limit: int
    seed: int
    quota_version: int
    request_concurrency: int = 2


@dataclass(frozen=True)
class StageSpec:
    stage: str
    concurrency: int


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    stage: str
    refs: tuple[str, ...]
    shards: int
    shard_bytes: int

    def references_closure(self, by_key: dict[str, "ArtifactSpec"]) -> tuple[str, ...]:
        seen: set[str] = set()
        stack = list(reversed(self.refs))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(reversed(by_key[current].refs))
        return tuple(sorted(seen))


@dataclass(frozen=True)
class RuntimeSpec:
    global_concurrency: int
    handshake_timeout_ms: int
    reply_timeout_ms: int
    grace_ms: int


@dataclass(frozen=True)
class Config:
    format_version: int
    scale: int
    runtime: RuntimeSpec
    nodes: tuple[NodeSpec, ...]
    stages: tuple[StageSpec, ...]
    artifacts: tuple[ArtifactSpec, ...]
    config_path: str
    config_digest: str
    node_by_id: dict[str, NodeSpec] = field(init=False, repr=False)
    artifact_by_key: dict[str, ArtifactSpec] = field(init=False, repr=False)
    stage_by_name: dict[str, StageSpec] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_by_id", {node.node_id: node for node in self.nodes})
        object.__setattr__(self, "artifact_by_key", {item.key: item for item in self.artifacts})
        object.__setattr__(self, "stage_by_name", {stage.stage: stage for stage in self.stages})


@dataclass(frozen=True)
class Event:
    ordinal: int
    seq: int
    batch_id: int
    event_type: str
    line_no: int
    artifact: str | None = None
    lease_ms: int | None = None
    lease_id: str | None = None
    node_id: str | None = None
    quota_version: int | None = None
    capacity_slots: int | None = None
    lease_limit: int | None = None
    policy: str | None = None
    hold_ms: int = 0

    def canonical_payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seq": self.seq,
            "batch": self.batch_id,
            "type": self.event_type,
        }
        for key in (
            "artifact",
            "lease_ms",
            "lease_id",
            "node",
            "quota_version",
            "capacity_slots",
            "lease_limit",
            "policy",
            "hold_ms",
        ):
            value = getattr(self, key if key != "node" else "node_id")
            if value is not None and not (key == "hold_ms" and value == 0):
                result[key] = value
        return result

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_bytes(self.canonical_payload())).hexdigest()


@dataclass(frozen=True)
class EventBatch:
    batch_id: int
    events: tuple[Event, ...]

    @property
    def all_requests(self) -> bool:
        return all(event.event_type in {"write", "read"} for event in self.events)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_file(path: str | Path, location_prefix: str = "") -> Any:
    source = Path(path)
    location = location_prefix or str(source)
    try:
        raw = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SpecError(f"cannot read JSON: {exc}", location) from exc
    try:
        return json.loads(raw, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SpecError(f"invalid JSON: {exc}", location) from exc


def _object(value: Any, location: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("expected a JSON object", location)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecError(f"unknown field {unknown[0]!r}", f"{location}.{unknown[0]}")
    return value


def _required(obj: dict[str, Any], key: str, location: str) -> Any:
    if key not in obj:
        raise SpecError(f"missing required field {key!r}", f"{location}.{key}")
    return obj[key]


def _string(value: Any, location: str, *, min_length: int = 1, max_length: int = 4096) -> str:
    if not isinstance(value, str):
        raise SpecError("expected a string", location)
    if len(value) < min_length or len(value) > max_length:
        raise SpecError(f"string length must be in [{min_length}, {max_length}]", location)
    return value


def _int(value: Any, location: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError("expected an integer", location)
    if value < minimum:
        raise SpecError(f"integer must be >= {minimum}", location)
    if maximum is not None and value > maximum:
        raise SpecError(f"integer must be <= {maximum}", location)
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError("expected an array", location)
    return value


def load_config(path: str | Path) -> Config:
    source = Path(path)
    root_location = str(source)
    raw = load_json_file(source)
    root = _object(
        raw,
        root_location,
        {"format_version", "session", "runtime", "nodes", "pipeline"},
    )

    format_version = _int(_required(root, "format_version", root_location), f"{root_location}.format_version")
    if format_version != FORMAT_VERSION:
        raise SpecError(f"format_version must be {FORMAT_VERSION}", f"{root_location}.format_version")

    session_raw = _object(_required(root, "session", root_location), f"{root_location}.session", {"scale"})
    scale = _int(_required(session_raw, "scale", f"{root_location}.session"), f"{root_location}.session.scale", minimum=2)

    runtime_raw = _object(
        _required(root, "runtime", root_location),
        f"{root_location}.runtime",
        {"global_concurrency", "handshake_timeout_ms", "reply_timeout_ms", "grace_ms"},
    )
    runtime_location = f"{root_location}.runtime"
    runtime = RuntimeSpec(
        global_concurrency=_int(
            _required(runtime_raw, "global_concurrency", runtime_location),
            f"{runtime_location}.global_concurrency",
            minimum=1,
        ),
        handshake_timeout_ms=_int(
            _required(runtime_raw, "handshake_timeout_ms", runtime_location),
            f"{runtime_location}.handshake_timeout_ms",
            minimum=1,
        ),
        reply_timeout_ms=_int(
            _required(runtime_raw, "reply_timeout_ms", runtime_location),
            f"{runtime_location}.reply_timeout_ms",
            minimum=1,
        ),
        grace_ms=_int(
            _required(runtime_raw, "grace_ms", runtime_location),
            f"{runtime_location}.grace_ms",
            minimum=1,
        ),
    )

    nodes: list[NodeSpec] = []
    node_ids: set[str] = set()
    ports: set[int] = set()
    nodes_location = f"{root_location}.nodes"
    for index, node_value in enumerate(_list(_required(root, "nodes", root_location), nodes_location)):
        location = f"{nodes_location}[{index}]"
        node = _object(
            node_value,
            location,
            {
                "node_id",
                "host",
                "port",
                "capacity_slots",
                "lease_limit",
                "seed",
                "quota_version",
                "request_concurrency",
            },
        )
        node_id = _string(_required(node, "node_id", location), f"{location}.node_id", max_length=64)
        if not NODE_ID_RE.fullmatch(node_id):
            raise SpecError("node_id must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", f"{location}.node_id")
        if node_id in node_ids:
            raise SpecError(f"duplicate node_id {node_id!r}", f"{location}.node_id")
        node_ids.add(node_id)

        host = _string(_required(node, "host", location), f"{location}.host")
        if host != LOOPBACK_HOST:
            raise SpecError("host must be the loopback literal 127.0.0.1", f"{location}.host")
        port = _int(_required(node, "port", location), f"{location}.port", minimum=1024, maximum=65535)
        if port in ports:
            raise SpecError(f"duplicate port {port}", f"{location}.port")
        ports.add(port)

        capacity_slots = _int(
            _required(node, "capacity_slots", location), f"{location}.capacity_slots", minimum=1
        )
        lease_limit = _int(_required(node, "lease_limit", location), f"{location}.lease_limit", minimum=1)
        seed = _int(_required(node, "seed", location), f"{location}.seed", minimum=0)
        quota_version = _int(
            _required(node, "quota_version", location), f"{location}.quota_version", minimum=1
        )
        request_concurrency = _int(
            node.get("request_concurrency", 2), f"{location}.request_concurrency", minimum=1
        )
        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=port,
                capacity_slots=capacity_slots,
                lease_limit=lease_limit,
                seed=seed,
                quota_version=quota_version,
                request_concurrency=request_concurrency,
            )
        )
    if not nodes:
        raise SpecError("nodes must not be empty", nodes_location)

    pipeline_raw = _object(
        _required(root, "pipeline", root_location),
        f"{root_location}.pipeline",
        {"stages", "artifacts"},
    )
    pipeline_location = f"{root_location}.pipeline"
    stages: list[StageSpec] = []
    stage_names: set[str] = set()
    stages_location = f"{pipeline_location}.stages"
    for index, stage_value in enumerate(_list(_required(pipeline_raw, "stages", pipeline_location), stages_location)):
        location = f"{stages_location}[{index}]"
        stage_obj = _object(stage_value, location, {"stage", "concurrency"})
        stage = _string(_required(stage_obj, "stage", location), f"{location}.stage", max_length=128)
        if stage in stage_names:
            raise SpecError(f"duplicate stage {stage!r}", f"{location}.stage")
        stage_names.add(stage)
        concurrency = _int(
            _required(stage_obj, "concurrency", location), f"{location}.concurrency", minimum=1
        )
        stages.append(StageSpec(stage=stage, concurrency=concurrency))
    if not stages:
        raise SpecError("stages must not be empty", stages_location)

    artifacts: list[ArtifactSpec] = []
    artifact_keys: set[str] = set()
    artifacts_location = f"{pipeline_location}.artifacts"
    for index, artifact_value in enumerate(
        _list(_required(pipeline_raw, "artifacts", pipeline_location), artifacts_location)
    ):
        location = f"{artifacts_location}[{index}]"
        artifact_obj = _object(artifact_value, location, {"key", "stage", "refs", "shards"})
        key = _string(_required(artifact_obj, "key", location), f"{location}.key", max_length=256)
        if key in artifact_keys:
            raise SpecError(f"duplicate artifact key {key!r}", f"{location}.key")
        artifact_keys.add(key)
        stage = _string(_required(artifact_obj, "stage", location), f"{location}.stage")
        if stage not in stage_names:
            raise SpecError(f"unknown stage {stage!r}", f"{location}.stage")
        refs_value = _list(_required(artifact_obj, "refs", location), f"{location}.refs")
        refs: list[str] = []
        seen_refs: set[str] = set()
        for ref_index, ref_value in enumerate(refs_value):
            ref_location = f"{location}.refs[{ref_index}]"
            ref = _string(ref_value, ref_location, max_length=256)
            if ref in seen_refs:
                raise SpecError(f"duplicate reference {ref!r}", ref_location)
            if ref == key:
                raise SpecError("an artifact cannot reference itself", ref_location)
            seen_refs.add(ref)
            refs.append(ref)
        shards_raw = artifact_obj.get("shards")
        shards = 3
        if isinstance(shards_raw, dict):
            shards_obj = _object(shards_raw, f"{location}.shards", {"count", "bytes"})
            shards = _int(
                _required(shards_obj, "count", f"{location}.shards"),
                f"{location}.shards.count",
                minimum=1,
                maximum=64,
            )
            shard_bytes = _int(
                _required(shards_obj, "bytes", f"{location}.shards"),
                f"{location}.shards.bytes",
                minimum=1,
                maximum=65536,
            )
        elif shards_raw is None:
            shard_bytes = 128
        else:
            raise SpecError("shards must be an object with count and bytes", f"{location}.shards")
        artifacts.append(
            ArtifactSpec(key=key, stage=stage, refs=tuple(refs), shards=shards, shard_bytes=shard_bytes)
        )
    if not artifacts:
        raise SpecError("artifacts must not be empty", artifacts_location)

    artifact_by_key = {artifact.key: artifact for artifact in artifacts}
    for artifact in artifacts:
        for ref_index, ref in enumerate(artifact.refs):
            if ref not in artifact_by_key:
                index = artifacts.index(artifact)
                raise SpecError(f"unknown artifact reference {ref!r}", f"{artifacts_location}[{index}].refs[{ref_index}]")

    _validate_acyclic(artifacts, artifact_by_key, root_location)

    digest = hashlib.sha256(
        canonical_bytes(
            {
                "format_version": format_version,
                "scale": scale,
                "runtime": runtime.__dict__,
                "nodes": [node.__dict__ for node in nodes],
                "stages": [stage.__dict__ for stage in stages],
                "artifacts": [
                    {
                        "key": artifact.key,
                        "stage": artifact.stage,
                        "refs": list(artifact.refs),
                        "shards": {"count": artifact.shards, "bytes": artifact.shard_bytes},
                    }
                    for artifact in artifacts
                ],
            }
        )
    ).hexdigest()

    return Config(
        format_version=format_version,
        scale=scale,
        runtime=runtime,
        nodes=tuple(nodes),
        stages=tuple(stages),
        artifacts=tuple(artifacts),
        config_path=str(source),
        config_digest=digest,
    )


def _validate_acyclic(
    artifacts: list[ArtifactSpec], by_key: dict[str, ArtifactSpec], root_location: str
) -> None:
    state: dict[str, int] = {artifact.key: 0 for artifact in artifacts}
    stack: list[str] = []

    def visit(key: str) -> None:
        state[key] = 1
        stack.append(key)
        artifact = by_key[key]
        for ref_index, ref in enumerate(artifact.refs):
            if state[ref] == 1:
                cycle_start = stack.index(ref)
                cycle = stack[cycle_start:] + [ref]
                owner_index = next(i for i, item in enumerate(artifacts) if item.key == key)
                raise SpecError(
                    f"artifact reference cycle: {' -> '.join(cycle)}",
                    f"{root_location}.pipeline.artifacts[{owner_index}].refs[{ref_index}]",
                )
            if state[ref] == 0:
                visit(ref)
        stack.pop()
        state[key] = 2

    for artifact in artifacts:
        if state[artifact.key] == 0:
            visit(artifact.key)


def load_events(path: str | Path, config: Config) -> tuple[Event, ...]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise SpecError(f"cannot read NDJSON: {exc}", str(source)) from exc
    if not lines:
        raise SpecError("event stream is empty", str(source))

    events: list[Event] = []
    previous_seq: int | None = None
    previous_batch: int | None = None
    seen_batches: set[int] = set()
    batch_members: dict[int, int] = {}
    current_quota: dict[str, int] = {node.node_id: node.quota_version for node in config.nodes}

    for line_index, raw_line in enumerate(lines, start=1):
        location = f"{source}:{line_index}"
        if not raw_line.strip():
            raise SpecError("blank NDJSON lines are not allowed", location)
        try:
            value = json.loads(raw_line, object_pairs_hook=_strict_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SpecError(f"invalid event JSON: {exc}", location) from exc
        obj = _object(value, location, {
            "seq", "batch", "type", "artifact", "lease_ms", "lease_id", "node",
            "quota_version", "capacity_slots", "lease_limit", "policy", "hold_ms",
        })
        seq = _int(_required(obj, "seq", location), f"{location}.seq", minimum=1)
        if previous_seq is not None and seq < previous_seq:
            raise SpecError("seq must be monotonically nondecreasing", f"{location}.seq")
        previous_seq = seq

        default_batch = seq
        batch_id = _int(obj.get("batch", default_batch), f"{location}.batch", minimum=1)
        if previous_batch is not None and batch_id < previous_batch:
            raise SpecError("batch ids must be monotonically nondecreasing", f"{location}.batch")
        if batch_id not in seen_batches and seen_batches:
            if batch_id == previous_batch:
                pass
        if batch_id != previous_batch and batch_id in seen_batches:
            raise SpecError("batch ids must be contiguous", f"{location}.batch")
        seen_batches.add(batch_id)
        previous_batch = batch_id
        batch_members[batch_id] = batch_members.get(batch_id, 0) + 1

        event_type = _string(_required(obj, "type", location), f"{location}.type")
        if event_type not in {"write", "read", "release", "quota"}:
            raise SpecError("type must be one of write, read, release, quota", f"{location}.type")

        ordinal = len(events) + 1
        common = {"ordinal": ordinal, "seq": seq, "batch_id": batch_id, "event_type": event_type, "line_no": line_index}
        if event_type in {"write", "read"}:
            artifact = _string(_required(obj, "artifact", location), f"{location}.artifact")
            if artifact not in config.artifact_by_key:
                raise SpecError(f"unknown artifact {artifact!r}", f"{location}.artifact")
            lease_ms = _int(_required(obj, "lease_ms", location), f"{location}.lease_ms", minimum=1)
            hold_ms = _int(obj.get("hold_ms", 0), f"{location}.hold_ms", minimum=0)
            if any(key in obj for key in ("lease_id", "node", "quota_version", "capacity_slots", "lease_limit", "policy")):
                raise SpecError("write/read event contains fields for another event type", location)
            events.append(Event(**common, artifact=artifact, lease_ms=lease_ms, hold_ms=hold_ms))
        elif event_type == "release":
            if ("lease_id" in obj) == ("artifact" in obj):
                raise SpecError("release requires exactly one of lease_id or artifact", location)
            if any(key in obj for key in ("lease_ms", "node", "quota_version", "capacity_slots", "lease_limit", "policy", "hold_ms")):
                raise SpecError("release event contains fields for another event type", location)
            if "lease_id" in obj:
                lease_id = _string(obj["lease_id"], f"{location}.lease_id")
                events.append(Event(**common, lease_id=lease_id))
            else:
                artifact = _string(obj["artifact"], f"{location}.artifact")
                if artifact not in config.artifact_by_key:
                    raise SpecError(f"unknown artifact {artifact!r}", f"{location}.artifact")
                events.append(Event(**common, artifact=artifact))
        else:
            if any(key in obj for key in ("artifact", "lease_ms", "lease_id", "hold_ms")):
                raise SpecError("quota event contains fields for another event type", location)
            node_id = _string(_required(obj, "node", location), f"{location}.node")
            if node_id not in config.node_by_id:
                raise SpecError(f"unknown node {node_id!r}", f"{location}.node")
            quota_version = _int(
                _required(obj, "quota_version", location), f"{location}.quota_version", minimum=1
            )
            if quota_version <= current_quota[node_id]:
                raise SpecError(
                    f"quota_version must increase above {current_quota[node_id]}",
                    f"{location}.quota_version",
                )
            current_quota[node_id] = quota_version
            capacity_slots = obj.get("capacity_slots")
            if capacity_slots is not None:
                capacity_slots = _int(capacity_slots, f"{location}.capacity_slots", minimum=1)
            lease_limit = obj.get("lease_limit")
            if lease_limit is not None:
                lease_limit = _int(lease_limit, f"{location}.lease_limit", minimum=1)
            policy = _string(_required(obj, "policy", location), f"{location}.policy")
            if policy not in {"renew", "revoke"}:
                raise SpecError("policy must be renew or revoke", f"{location}.policy")
            events.append(
                Event(
                    **common,
                    node_id=node_id,
                    quota_version=quota_version,
                    capacity_slots=capacity_slots,
                    lease_limit=lease_limit,
                    policy=policy,
                )
            )

    for batch_id, count in batch_members.items():
        members = [event for event in events if event.batch_id == batch_id]
        if count > 1 and any(event.event_type in {"release", "quota"} for event in members):
            first_barrier = next(event for event in members if event.event_type in {"release", "quota"})
            raise SpecError(
                "a release/quota batch must contain exactly one event",
                f"{source}:{first_barrier.line_no}.batch",
            )
    return tuple(events)


def group_batches(events: Iterable[Event]) -> tuple[EventBatch, ...]:
    grouped: list[EventBatch] = []
    current_id: int | None = None
    current: list[Event] = []
    for event in events:
        if current_id is None:
            current_id = event.batch_id
        if event.batch_id != current_id:
            grouped.append(EventBatch(current_id, tuple(current)))
            current_id = event.batch_id
            current = []
        current.append(event)
    if current_id is not None:
        grouped.append(EventBatch(current_id, tuple(current)))
    return tuple(grouped)
