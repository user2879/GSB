"""Strict loading and validation for node declarations and pipeline manifests."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError
from .util import is_int, positive_int, read_json_file, sha256_value

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_SPEC_VERSION = 1


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    lease_limit: int
    seed: int


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    expected_digest: str
    refs: tuple[str, ...]
    stage_id: str


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    concurrency: int
    artifact_keys: tuple[str, ...]


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    global_concurrency: int
    stages: tuple[StageSpec, ...]
    artifacts: dict[str, ArtifactSpec]
    ordered_artifact_keys: tuple[str, ...]


@dataclass(frozen=True)
class Spec:
    format_version: int
    source_path: str
    nodes: tuple[NodeSpec, ...]
    pipeline: PipelineSpec
    artifacts: dict[str, ArtifactSpec] = field(init=False)
    node_by_id: dict[str, NodeSpec] = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", dict(self.pipeline.artifacts))
        object.__setattr__(self, "node_by_id", {node.node_id: node for node in self.nodes})
        object.__setattr__(self, "digest", sha256_value(self.to_canonical_dict()))

    @property
    def ordered_artifacts(self) -> tuple[ArtifactSpec, ...]:
        return tuple(self.artifacts[key] for key in self.pipeline.ordered_artifact_keys)

    @property
    def all_artifact_keys(self) -> tuple[str, ...]:
        return self.pipeline.ordered_artifact_keys

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "capacity_slots": n.capacity_slots,
                    "lease_limit": n.lease_limit,
                    "seed": n.seed,
                }
                for n in self.nodes
            ],
            "pipeline": {
                "pipeline_id": self.pipeline.pipeline_id,
                "global_concurrency": self.pipeline.global_concurrency,
                "stages": [
                    {
                        "stage_id": s.stage_id,
                        "concurrency": s.concurrency,
                        "artifacts": [
                            {
                                "key": self.artifacts[key].key,
                                "expected_digest": self.artifacts[key].expected_digest,
                                "refs": list(self.artifacts[key].refs),
                                "stage_id": self.artifacts[key].stage_id,
                            }
                            for key in s.artifact_keys
                        ],
                    }
                    for s in self.pipeline.stages
                ],
            },
        }

    def topological_stages(self) -> tuple[StageSpec, ...]:
        """Topologically order stages using cross-stage artifact references."""
        stage_map = {stage.stage_id: stage for stage in self.pipeline.stages}
        deps: dict[str, set[str]] = {stage.stage_id: set() for stage in self.pipeline.stages}
        for artifact in self.ordered_artifacts:
            for ref in artifact.refs:
                ref_stage = self.artifacts[ref].stage_id
                if ref_stage != artifact.stage_id:
                    deps[artifact.stage_id].add(ref_stage)
        result: list[StageSpec] = []
        remaining = {stage_id: set(values) for stage_id, values in deps.items()}
        emitted: set[str] = set()
        order = {stage.stage_id: index for index, stage in enumerate(self.pipeline.stages)}
        while remaining:
            ready = [stage_id for stage_id, values in remaining.items() if values <= emitted]
            if not ready:
                raise SpecError(
                    "stage dependency cycle",
                    artifact_key=sorted(remaining)[0],
                    field="stages",
                )
            ready.sort(key=lambda sid: order[sid])
            for stage_id in ready:
                result.append(stage_map[stage_id])
                emitted.add(stage_id)
                del remaining[stage_id]
        return tuple(result)


def _require_object(value: Any, field_name: str, artifact_key: str | None = None) -> dict:
    if not isinstance(value, dict):
        raise SpecError("expected JSON object", field=field_name, artifact_key=artifact_key)
    return value


def _require_nonempty_str(value: Any, field_name: str, artifact_key: str | None = None, node_id: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(
            f"{field_name} must be a non-empty string",
            field=field_name,
            artifact_key=artifact_key,
            node_id=node_id,
        )
    return value


def _require_positive_int(value: Any, field_name: str, node_id: str | None = None, artifact_key: str | None = None) -> int:
    if not positive_int(value):
        raise SpecError(
            f"{field_name} must be a positive integer",
            field=field_name,
            node_id=node_id,
            artifact_key=artifact_key,
        )
    return int(value)


def load_spec(path: str | Path) -> Spec:
    """Load and strongly validate a combined JSON specification.

    Validation stops at the first violation in document order and reports the
    offending node id or artifact key.  References and cycles are checked after
    the entire artifact set is known.
    """
    source = Path(path)
    try:
        raw = read_json_file(source)
    except FileNotFoundError as exc:
        raise SpecError("spec file not found", path=str(source), field="path") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            "invalid JSON",
            field="json",
            line=exc.lineno,
            column=exc.colno,
            path=str(source),
        ) from exc
    return parse_spec(raw, source_path=str(source))


def parse_spec(raw: Any, source_path: str = "<memory>") -> Spec:
    root = _require_object(raw, "$")
    version = root.get("format_version", 1)
    if not is_int(version) or version != SUPPORTED_SPEC_VERSION:
        raise SpecError(
            "unsupported spec format_version; expected integer 1",
            field="format_version",
            expected=SUPPORTED_SPEC_VERSION,
            actual=version,
        )

    nodes_raw = root.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise SpecError("nodes must be a non-empty array", field="nodes")

    nodes: list[NodeSpec] = []
    node_ids: set[str] = set()
    ports: dict[int, str] = {}
    for index, item in enumerate(nodes_raw):
        entry = _require_object(item, f"nodes[{index}]")
        node_id = _require_nonempty_str(entry.get("node_id"), "node_id")
        if node_id in node_ids:
            raise SpecError("duplicate node_id", node_id=node_id, field="node_id")
        host = entry.get("host")
        if host != "127.0.0.1":
            raise SpecError(
                "node host must be the loopback literal 127.0.0.1",
                node_id=node_id,
                field="host",
                expected="127.0.0.1",
                actual=host,
            )
        port = entry.get("port")
        if not is_int(port) or not 1 <= port <= 65535:
            raise SpecError(
                "port must be an integer in 1..65535",
                node_id=node_id,
                field="port",
                actual=port,
            )
        if port in ports:
            raise SpecError(
                "duplicate node port",
                node_id=node_id,
                field="port",
                conflicting_node_id=ports[port],
                actual=port,
            )
        capacity = _require_positive_int(entry.get("capacity_slots"), "capacity_slots", node_id=node_id)
        lease_limit = _require_positive_int(entry.get("lease_limit"), "lease_limit", node_id=node_id)
        seed = entry.get("seed")
        if not is_int(seed):
            raise SpecError("seed must be an integer", node_id=node_id, field="seed", actual=seed)
        node_ids.add(node_id)
        ports[port] = node_id
        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=port,
                capacity_slots=capacity,
                lease_limit=lease_limit,
                seed=int(seed),
            )
        )

    pipeline_raw = _require_object(root.get("pipeline"), "pipeline")
    pipeline_id = _require_nonempty_str(pipeline_raw.get("pipeline_id"), "pipeline_id")
    global_concurrency = _require_positive_int(
        pipeline_raw.get("global_concurrency"), "global_concurrency"
    )
    stages_raw = pipeline_raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise SpecError("pipeline.stages must be a non-empty array", field="stages")

    stage_ids: set[str] = set()
    artifacts: dict[str, ArtifactSpec] = {}
    artifact_order: list[str] = []
    stages: list[StageSpec] = []
    for stage_index, item in enumerate(stages_raw):
        entry = _require_object(item, f"stages[{stage_index}]")
        stage_id = _require_nonempty_str(entry.get("stage_id"), "stage_id")
        if stage_id in stage_ids:
            raise SpecError("duplicate stage_id", field="stage_id", stage_id=stage_id)
        concurrency = _require_positive_int(entry.get("concurrency"), "concurrency", artifact_key=None)
        artifacts_raw = entry.get("artifacts")
        if not isinstance(artifacts_raw, list) or not artifacts_raw:
            raise SpecError("stage artifacts must be a non-empty array", field="artifacts", stage_id=stage_id)
        stage_keys: list[str] = []
        for artifact_index, artifact_raw in enumerate(artifacts_raw):
            art = _require_object(artifact_raw, f"stages[{stage_index}].artifacts[{artifact_index}]")
            key = _require_nonempty_str(art.get("key"), "artifact_key")
            if key in artifacts:
                raise SpecError(
                    "duplicate artifact key",
                    artifact_key=key,
                    field="artifact_key",
                    stage_id=stage_id,
                )
            digest = art.get("expected_digest")
            if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
                raise SpecError(
                    "expected_digest must be 64 lowercase hexadecimal characters",
                    artifact_key=key,
                    field="expected_digest",
                    actual=digest,
                )
            refs_raw = art.get("refs", [])
            if not isinstance(refs_raw, list):
                raise SpecError("refs must be an array", artifact_key=key, field="refs")
            refs: list[str] = []
            for ref_index, ref in enumerate(refs_raw):
                if not isinstance(ref, str) or not ref:
                    raise SpecError(
                        "reference must be a non-empty artifact key string",
                        artifact_key=key,
                        field=f"refs[{ref_index}]",
                    )
                refs.append(ref)
            artifacts[key] = ArtifactSpec(
                key=key,
                expected_digest=digest,
                refs=tuple(refs),
                stage_id=stage_id,
            )
            artifact_order.append(key)
            stage_keys.append(key)
        stages.append(
            StageSpec(stage_id=stage_id, concurrency=concurrency, artifact_keys=tuple(stage_keys))
        )
        stage_ids.add(stage_id)

    for artifact in artifacts.values():
        for ref in artifact.refs:
            if ref not in artifacts:
                raise SpecError(
                    "artifact reference target does not exist",
                    artifact_key=artifact.key,
                    field="refs",
                    missing_reference=ref,
                )

    # Iterative depth-first cycle check, preserving manifest order so the first
    # reported key is stable.
    state: dict[str, int] = {key: 0 for key in artifact_order}

    def visit(start: str) -> None:
        stack: list[tuple[str, int]] = [(start, 0)]
        while stack:
            key, index = stack[-1]
            if index == 0:
                if state[key] == 1:
                    raise SpecError("artifact reference cycle", artifact_key=key, field="refs")
                if state[key] == 2:
                    stack.pop()
                    continue
                state[key] = 1
            refs = artifacts[key].refs
            if index >= len(refs):
                state[key] = 2
                stack.pop()
                continue
            ref = refs[index]
            stack[-1] = (key, index + 1)
            if state[ref] == 1:
                raise SpecError(
                    "artifact reference cycle",
                    artifact_key=ref,
                    field="refs",
                    cycle_at=key,
                )
            if state[ref] == 0:
                stack.append((ref, 0))

    for artifact_key in artifact_order:
        if state[artifact_key] == 0:
            visit(artifact_key)

    pipeline = PipelineSpec(
        pipeline_id=pipeline_id,
        global_concurrency=global_concurrency,
        stages=tuple(stages),
        artifacts=artifacts,
        ordered_artifact_keys=tuple(artifact_order),
    )
    return Spec(
        format_version=version,
        source_path=source_path,
        nodes=tuple(nodes),
        pipeline=pipeline,
    )


def iter_stage_artifacts(spec: Spec) -> Iterable[tuple[StageSpec, ArtifactSpec]]:
    for stage in spec.pipeline.stages:
        for key in stage.artifact_keys:
            yield stage, spec.artifacts[key]
