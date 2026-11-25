from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .content import canonical_digest
from .errors import SpecError


NODE_FILE = "nodes.json"
PIPELINE_FILE = "pipeline.json"
EVENT_FILE = "events.jsonl"

ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
EVENT_TYPES = {
    "write",
    "read",
    "renew",
    "release",
    "quota_change",
    "wait",
}


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_type(
    condition: bool,
    *,
    entry_id: str,
    field: str,
    message: str,
) -> None:
    if not condition:
        raise SpecError(message, entry_id=entry_id, field=field)


def _duplicate_key_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise SpecError(
                f"duplicate JSON field {key!r}",
                entry_id="$",
                field=key,
            )
        seen.add(key)
        result[key] = value
    return result


def load_json_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(
            f"cannot read declaration {path.name}: {exc}",
            entry_id=path.name,
            field="file",
        ) from exc
    try:
        return json.loads(text, object_pairs_hook=_duplicate_key_hook)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON in {path.name}: line {exc.lineno} column {exc.colno}",
            entry_id=f"{path.name}:{exc.lineno}",
            field="json",
        ) from exc


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    lease_limit: int
    seed: int


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    concurrency_quota: int


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    stage: str
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    stages: tuple[StageSpec, ...]
    artifacts: tuple[ArtifactSpec, ...]
    watermark_scale: int = 1000
    high_watermark_fixed: int = 800


@dataclass(frozen=True)
class EventSpec:
    seq: int
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass(frozen=True)
class Bundle:
    sample_dir: Path
    nodes: tuple[NodeSpec, ...]
    pipeline: PipelineSpec
    events: tuple[EventSpec, ...]
    global_concurrency: int
    initial_quota_version: int
    shard_bytes: int
    request_timeout_ms: int
    handshake_timeout_ms: int
    node_concurrency: dict[str, int]
    raw_digests: dict[str, str]
    spec_digest: str

    @property
    def node_ids(self) -> list[str]:
        return [node.node_id for node in self.nodes]

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def artifact(self, key: str) -> ArtifactSpec:
        for artifact in self.pipeline.artifacts:
            if artifact.key == key:
                return artifact
        raise KeyError(key)


def validate_nodes(raw: Any) -> tuple[list[NodeSpec], dict[str, Any]]:
    entry = "nodes.json"
    require_type(isinstance(raw, dict), entry_id=entry, field="$", message="nodes document must be an object")
    require_type(isinstance(raw.get("nodes"), list) and raw["nodes"], entry_id="nodes", field="nodes", message="nodes must be a non-empty array")

    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    nodes: list[NodeSpec] = []

    for index, item in enumerate(raw["nodes"]):
        provisional_id = f"nodes[{index}]"
        require_type(isinstance(item, dict), entry_id=provisional_id, field="$", message="node entry must be an object")
        node_id = item.get("node_id")
        entry_id = str(node_id) if node_id is not None else provisional_id

        require_type(isinstance(node_id, str) and bool(ID_PATTERN.fullmatch(node_id)), entry_id=entry_id, field="node_id", message="node_id must be a non-empty identifier using letters, digits, '_', '.' or '-'")
        require_type(node_id not in seen_ids, entry_id=entry_id, field="node_id", message="node_id must be unique")
        seen_ids.add(node_id)

        host = item.get("host")
        require_type(host == "127.0.0.1", entry_id=entry_id, field="host", message="host must be the literal 127.0.0.1")

        port = item.get("port")
        require_type(is_int(port), entry_id=entry_id, field="port", message="port must be an integer")
        require_type(1 <= int(port) <= 65535, entry_id=entry_id, field="port", message="port must be in 1..65535")
        require_type(port not in seen_ports, entry_id=entry_id, field="port", message="loopback port must be globally unique")
        seen_ports.add(int(port))

        for field_name in ("capacity_slots", "lease_limit"):
            value = item.get(field_name)
            require_type(is_int(value), entry_id=entry_id, field=field_name, message=f"{field_name} must be an integer")
            require_type(int(value) > 0, entry_id=entry_id, field=field_name, message=f"{field_name} must be a positive integer")

        seed = item.get("seed")
        require_type(is_int(seed), entry_id=entry_id, field="seed", message="seed must be an integer")

        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=int(port),
                capacity_slots=int(item["capacity_slots"]),
                lease_limit=int(item["lease_limit"]),
                seed=int(seed),
            )
        )

    global_concurrency = raw.get("global_concurrency", len(nodes) * 2)
    require_type(is_int(global_concurrency) and int(global_concurrency) > 0, entry_id="runtime", field="global_concurrency", message="global_concurrency must be a positive integer")

    initial_version = raw.get("initial_quota_version", 1)
    require_type(is_int(initial_version) and int(initial_version) > 0, entry_id="runtime", field="initial_quota_version", message="initial_quota_version must be a positive integer")

    shard_bytes = raw.get("shard_bytes", 64)
    require_type(is_int(shard_bytes) and 0 <= int(shard_bytes) <= 1_048_576, entry_id="runtime", field="shard_bytes", message="shard_bytes must be a non-negative integer not greater than 1048576")

    handshake_ms = raw.get("handshake_timeout_ms", 2000)
    request_ms = raw.get("request_timeout_ms", 3000)
    require_type(is_int(handshake_ms) and int(handshake_ms) > 0, entry_id="runtime", field="handshake_timeout_ms", message="handshake_timeout_ms must be a positive integer")
    require_type(is_int(request_ms) and int(request_ms) > 0, entry_id="runtime", field="request_timeout_ms", message="request_timeout_ms must be a positive integer")

    raw_node_concurrency = raw.get("node_concurrency", {})
    require_type(isinstance(raw_node_concurrency, dict), entry_id="runtime", field="node_concurrency", message="node_concurrency must be an object")
    node_concurrency: dict[str, int] = {}
    for node_id, value in raw_node_concurrency.items():
        require_type(node_id in seen_ids, entry_id=str(node_id), field="node_concurrency", message="node_concurrency refers to an unknown node")
        require_type(is_int(value) and int(value) > 0, entry_id=str(node_id), field="concurrency", message="per-node concurrency must be a positive integer")
        node_concurrency[node_id] = int(value)
    for node in nodes:
        node_concurrency.setdefault(node.node_id, node.lease_limit)

    runtime = {
        "global_concurrency": int(global_concurrency),
        "initial_quota_version": int(initial_version),
        "shard_bytes": int(shard_bytes),
        "handshake_timeout_ms": int(handshake_ms),
        "request_timeout_ms": int(request_ms),
        "node_concurrency": node_concurrency,
    }
    return nodes, runtime


def validate_pipeline(raw: Any) -> PipelineSpec:
    entry = "pipeline.json"
    require_type(isinstance(raw, dict), entry_id=entry, field="$", message="pipeline document must be an object")
    pipeline_id = raw.get("pipeline_id")
    require_type(isinstance(pipeline_id, str) and pipeline_id, entry_id="pipeline", field="pipeline_id", message="pipeline_id must be a non-empty string")

    raw_stages = raw.get("stages")
    require_type(isinstance(raw_stages, list) and raw_stages, entry_id=pipeline_id, field="stages", message="stages must be a non-empty array")
    stages: list[StageSpec] = []
    seen_stages: set[str] = set()
    for index, stage in enumerate(raw_stages):
        provisional = f"stages[{index}]"
        require_type(isinstance(stage, dict), entry_id=provisional, field="$", message="stage must be an object")
        stage_id = stage.get("stage_id")
        stage_entry = str(stage_id) if stage_id is not None else provisional
        require_type(isinstance(stage_id, str) and ID_PATTERN.fullmatch(stage_id), entry_id=stage_entry, field="stage_id", message="stage_id must be a valid identifier")
        require_type(stage_id not in seen_stages, entry_id=stage_entry, field="stage_id", message="stage_id must be unique")
        quota = stage.get("concurrency_quota")
        require_type(is_int(quota) and int(quota) > 0, entry_id=stage_entry, field="concurrency_quota", message="stage concurrency_quota must be a positive integer")
        seen_stages.add(stage_id)
        stages.append(StageSpec(stage_id, int(quota)))

    raw_artifacts = raw.get("artifacts")
    require_type(isinstance(raw_artifacts, list) and raw_artifacts, entry_id=pipeline_id, field="artifacts", message="artifacts must be a non-empty array")
    artifacts: list[ArtifactSpec] = []
    seen_keys: set[str] = set()
    stage_ids = {stage.stage_id for stage in stages}

    for index, artifact in enumerate(raw_artifacts):
        provisional = f"artifacts[{index}]"
        require_type(isinstance(artifact, dict), entry_id=provisional, field="$", message="artifact must be an object")
        key = artifact.get("key")
        artifact_entry = str(key) if key is not None else provisional
        require_type(isinstance(key, str) and key, entry_id=artifact_entry, field="key", message="artifact key must be a non-empty string")
        require_type(key not in seen_keys, entry_id=artifact_entry, field="key", message="artifact key must be unique")
        seen_keys.add(key)

        stage = artifact.get("stage")
        require_type(stage in stage_ids, entry_id=artifact_entry, field="stage", message="artifact stage must reference a declared stage")

        references = artifact.get("references", [])
        require_type(isinstance(references, list), entry_id=artifact_entry, field="references", message="references must be an array")
        ref_tuple: list[str] = []
        for ref_index, reference in enumerate(references):
            require_type(isinstance(reference, str) and reference, entry_id=artifact_entry, field=f"references[{ref_index}]", message="artifact reference must be a non-empty key string")
            ref_tuple.append(reference)
        artifacts.append(ArtifactSpec(key=key, stage=stage, references=tuple(ref_tuple)))

    key_set = {artifact.key for artifact in artifacts}
    for artifact in artifacts:
        for reference in artifact.references:
            require_type(reference in key_set, entry_id=artifact.key, field="references", message=f"reference {reference!r} does not exist")

    state: dict[str, int] = {}

    def visit(key: str, trail: tuple[str, ...]) -> None:
        if state.get(key) == 2:
            return
        if state.get(key) == 1:
            cycle = " -> ".join([*trail, key])
            raise SpecError(f"artifact reference graph contains a cycle: {cycle}", entry_id=key, field="references")
        state[key] = 1
        artifact = next(item for item in artifacts if item.key == key)
        for reference in artifact.references:
            visit(reference, (*trail, key))
        state[key] = 2

    for artifact in artifacts:
        visit(artifact.key, ())

    scale = raw.get("watermark_scale", 1000)
    require_type(is_int(scale) and int(scale) >= 10, entry_id=pipeline_id, field="watermark_scale", message="watermark_scale must be an integer >= 10")
    high = raw.get("high_watermark_fixed", int(scale) * 8 // 10 if is_int(scale) else 800)
    require_type(is_int(high) and 0 <= int(high) <= int(scale), entry_id=pipeline_id, field="high_watermark_fixed", message="high_watermark_fixed must be between 0 and watermark_scale")

    return PipelineSpec(
        pipeline_id=pipeline_id,
        stages=tuple(stages),
        artifacts=tuple(artifacts),
        watermark_scale=int(scale),
        high_watermark_fixed=int(high),
    )


def _validate_quota_maps(event_entry: str, event: dict[str, Any], bundle_context: dict[str, Any]) -> None:
    for field_name in ("stage_quotas",):
        values = event.get(field_name)
        if values is None:
            continue
        require_type(isinstance(values, dict), entry_id=event_entry, field=field_name, message=f"{field_name} must be an object")
        valid_keys = bundle_context["stages"]
        for key, value in values.items():
            require_type(key in valid_keys, entry_id=event_entry, field=f"{field_name}.{key}", message=f"{field_name} references unknown stage")
            require_type(is_int(value) and int(value) > 0, entry_id=event_entry, field=f"{field_name}.{key}", message="quota must be a positive integer")

    values = event.get("node_concurrency")
    if values is not None:
        require_type(isinstance(values, dict), entry_id=event_entry, field="node_concurrency", message="node_concurrency must be an object")
        valid_nodes = bundle_context["nodes"]
        for key, value in values.items():
            require_type(key in valid_nodes, entry_id=event_entry, field=f"node_concurrency.{key}", message="node_concurrency references unknown node")
            require_type(is_int(value) and int(value) > 0, entry_id=event_entry, field=f"node_concurrency.{key}", message="concurrency must be a positive integer")

    active_nodes = event.get("active_nodes")
    if active_nodes is not None:
        require_type(isinstance(active_nodes, list) and active_nodes, entry_id=event_entry, field="active_nodes", message="active_nodes must be a non-empty array when provided")
        valid_nodes = bundle_context["nodes"]
        seen: set[str] = set()
        for node_id in active_nodes:
            require_type(isinstance(node_id, str), entry_id=event_entry, field="active_nodes", message="active_nodes entries must be node identifiers")
            require_type(node_id in valid_nodes, entry_id=event_entry, field=f"active_nodes.{node_id}", message="active_nodes references unknown node")
            require_type(node_id not in seen, entry_id=event_entry, field="active_nodes", message="active_nodes must not contain duplicates")
            seen.add(node_id)


def validate_events(
    raw_lines: Iterable[str],
    *,
    artifact_keys: set[str],
    node_ids: set[str],
    stage_ids: set[str],
) -> list[EventSpec]:
    events: list[EventSpec] = []
    previous_seq = 0
    context = {"artifacts": artifact_keys, "nodes": node_ids, "stages": stage_ids}

    for line_number, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        line_entry = f"events.jsonl:{line_number}"
        try:
            event = json.loads(stripped, object_pairs_hook=_duplicate_key_hook)
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid JSONL at line {line_number}: column {exc.colno}",
                entry_id=line_entry,
                field="json",
            ) from exc

        require_type(isinstance(event, dict), entry_id=line_entry, field="$", message="event must be a JSON object")
        seq = event.get("seq")
        event_entry = f"seq={seq}" if is_int(seq) else line_entry
        require_type(is_int(seq) and int(seq) > 0, entry_id=event_entry, field="seq", message="event seq must be a positive integer")
        require_type(int(seq) > previous_seq, entry_id=event_entry, field="seq", message=f"event seq must be strictly greater than {previous_seq}")
        previous_seq = int(seq)

        event_type = event.get("type")
        require_type(event_type in EVENT_TYPES, entry_id=event_entry, field="type", message=f"type must be one of {sorted(EVENT_TYPES)}")

        if event_type in {"write", "read"}:
            key = event.get("artifact_key")
            require_type(isinstance(key, str) and key in artifact_keys, entry_id=event_entry, field="artifact_key", message="artifact_key must reference a declared artifact")
            shard_index = event.get("shard_index", 0)
            require_type(is_int(shard_index) and int(shard_index) >= 0, entry_id=event_entry, field="shard_index", message="shard_index must be a non-negative integer")

        if event_type == "write":
            lease_ms = event.get("lease_ms", 60_000)
            require_type(is_int(lease_ms) and int(lease_ms) > 0, entry_id=event_entry, field="lease_ms", message="lease_ms must be a positive integer")
            candidates = event.get("candidate_nodes")
            if candidates is not None:
                require_type(isinstance(candidates, list) and candidates, entry_id=event_entry, field="candidate_nodes", message="candidate_nodes must be a non-empty array")
                seen_candidates: set[str] = set()
                for node_id in candidates:
                    require_type(isinstance(node_id, str) and node_id in node_ids, entry_id=event_entry, field="candidate_nodes", message="candidate_nodes must reference declared nodes")
                    require_type(node_id not in seen_candidates, entry_id=event_entry, field="candidate_nodes", message="candidate_nodes must be unique")
                    seen_candidates.add(node_id)
            lease_ref = event.get("lease_ref")
            require_type(lease_ref is None or (isinstance(lease_ref, str) and lease_ref), entry_id=event_entry, field="lease_ref", message="lease_ref must be a non-empty string")
            fail_write = event.get("fail_write", False)
            require_type(isinstance(fail_write, bool), entry_id=event_entry, field="fail_write", message="fail_write must be a boolean")
            cancel_after_reserve = event.get("cancel_after_reserve", False)
            require_type(isinstance(cancel_after_reserve, bool), entry_id=event_entry, field="cancel_after_reserve", message="cancel_after_reserve must be a boolean")

        if event_type in {"release", "renew"}:
            lease_ref = event.get("lease_ref")
            require_type(isinstance(lease_ref, str) and lease_ref, entry_id=event_entry, field="lease_ref", message="lease_ref must reference a granted lease label")

        if event_type == "renew":
            version = event.get("expected_version")
            require_type(is_int(version) and int(version) > 0, entry_id=event_entry, field="expected_version", message="expected_version must be a positive integer")
            lease_ms = event.get("lease_ms", 60_000)
            require_type(is_int(lease_ms) and int(lease_ms) > 0, entry_id=event_entry, field="lease_ms", message="lease_ms must be a positive integer")

        if event_type == "quota_change":
            version = event.get("quota_version")
            require_type(is_int(version) and int(version) > 0, entry_id=event_entry, field="quota_version", message="quota_version must be a positive integer")
            global_value = event.get("global_concurrency")
            if global_value is not None:
                require_type(is_int(global_value) and int(global_value) > 0, entry_id=event_entry, field="global_concurrency", message="global_concurrency must be a positive integer")
            _validate_quota_maps(event_entry, event, context)

        if event_type == "wait":
            duration = event.get("duration_ms", 0)
            require_type(is_int(duration) and 0 <= int(duration) <= 60_000, entry_id=event_entry, field="duration_ms", message="duration_ms must be an integer in 0..60000")

        events.append(EventSpec(seq=int(seq), type=event_type, data=dict(event)))

    require_type(bool(events), entry_id=EVENT_FILE, field="events", message="events.jsonl must contain at least one event")
    return events


def load_bundle(sample_dir: str | Path) -> Bundle:
    directory = Path(sample_dir)
    nodes_path = directory / NODE_FILE
    pipeline_path = directory / PIPELINE_FILE
    events_path = directory / EVENT_FILE

    for path in (nodes_path, pipeline_path, events_path):
        if not path.is_file():
            raise SpecError(f"missing required declaration file: {path.name}", entry_id=path.name, field="file")

    raw_nodes = load_json_file(nodes_path)
    raw_pipeline = load_json_file(pipeline_path)
    event_text = events_path.read_text(encoding="utf-8")

    nodes, runtime = validate_nodes(raw_nodes)
    pipeline = validate_pipeline(raw_pipeline)
    events = validate_events(
        event_text.splitlines(),
        artifact_keys={artifact.key for artifact in pipeline.artifacts},
        node_ids={node.node_id for node in nodes},
        stage_ids={stage.stage_id for stage in pipeline.stages},
    )

    raw_digests = {
        NODE_FILE: canonical_digest(raw_nodes),
        PIPELINE_FILE: canonical_digest(raw_pipeline),
        EVENT_FILE: canonical_digest([event.data for event in events]),
    }
    spec_digest = canonical_digest(raw_digests)

    return Bundle(
        sample_dir=directory,
        nodes=tuple(nodes),
        pipeline=pipeline,
        events=tuple(events),
        global_concurrency=runtime["global_concurrency"],
        initial_quota_version=runtime["initial_quota_version"],
        shard_bytes=runtime["shard_bytes"],
        request_timeout_ms=runtime["request_timeout_ms"],
        handshake_timeout_ms=runtime["handshake_timeout_ms"],
        node_concurrency=runtime["node_concurrency"],
        raw_digests=raw_digests,
        spec_digest=spec_digest,
    )
