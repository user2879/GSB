from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LeaseError
from .util import canonical_sha256, is_int


LOOPBACK_HOST = "127.0.0.1"
NODE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ARTIFACT_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    lease_limit: int
    random_seed: int


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    references: tuple[str, ...]
    stage_id: str
    stage_index: int
    artifact_index: int


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    index: int
    concurrency: int
    artifact_keys: tuple[str, ...]


@dataclass(frozen=True)
class PipelineSpec:
    path: Path
    raw: dict[str, Any]
    format_version: int
    pipeline_id: str
    global_concurrency: int
    lease_ms: int
    nodes: tuple[NodeSpec, ...]
    stages: tuple[StageSpec, ...]
    artifacts: tuple[ArtifactSpec, ...]
    content_sha256: str

    @property
    def node_by_id(self) -> dict[str, NodeSpec]:
        return {node.node_id: node for node in self.nodes}

    @property
    def artifact_by_key(self) -> dict[str, ArtifactSpec]:
        return {artifact.key: artifact for artifact in self.artifacts}

    @property
    def stage_by_id(self) -> dict[str, StageSpec]:
        return {stage.stage_id: stage for stage in self.stages}

    @property
    def sorted_node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(node.node_id for node in self.nodes))

    @property
    def artifact_keys(self) -> tuple[str, ...]:
        return tuple(artifact.key for artifact in self.artifacts)

    def topological_layers(self) -> list[list[str]]:
        pending = {artifact.key: set(artifact.references) for artifact in self.artifacts}
        layers: list[list[str]] = []
        while pending:
            ready = sorted(key for key, refs in pending.items() if not refs)
            if not ready:
                raise LeaseError(
                    "E_SPEC",
                    "artifact reference graph contains a cycle",
                    field="pipeline.stages[].artifacts[].references",
                    expected="acyclic artifact references",
                )
            layers.append(ready)
            for key in ready:
                del pending[key]
            for refs in pending.values():
                refs.difference_update(ready)
        return layers


def _fail(field: str, expected: str, actual: Any = None) -> None:
    raise LeaseError(
        "E_SPEC",
        f"invalid specification at {field}",
        field=field,
        expected=expected,
        actual=actual,
    )


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(field, "JSON object", type(value).__name__)
    return value


def _list(value: Any, field: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        _fail(field, "JSON array", type(value).__name__)
    if nonempty and not value:
        _fail(field, "non-empty JSON array", value)
    return value


def _string(value: Any, field: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        _fail(field, "non-empty string", value)
    if pattern is not None and not pattern.fullmatch(value):
        _fail(field, f"string matching {pattern.pattern}", value)
    return value


def _positive_int(value: Any, field: str) -> int:
    if not is_int(value) or value <= 0:
        _fail(field, "positive integer", value)
    return value


def _required(obj: dict[str, Any], name: str, field: str) -> Any:
    if name not in obj:
        _fail(f"{field}.{name}", "required field", "<missing>")
    return obj[name]


def load_spec(path: str | Path) -> PipelineSpec:
    spec_path = Path(path).resolve()
    try:
        raw_bytes = spec_path.read_bytes()
    except OSError as exc:
        raise LeaseError(
            "E_SPEC",
            f"cannot read specification: {exc}",
            field="$",
            expected="readable UTF-8 JSON file",
        ) from exc
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LeaseError(
            "E_SPEC",
            "specification is not valid UTF-8",
            field="$",
            expected="UTF-8 encoded JSON",
            actual=str(exc),
        ) from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LeaseError(
            "E_SPEC",
            "specification is not valid JSON",
            field=f"$ (line {exc.lineno}, column {exc.colno})",
            expected="valid JSON document",
        ) from exc
    root = _object(raw, "$")

    format_version = _required(root, "format_version", "$")
    if not is_int(format_version) or format_version != 1:
        _fail("format_version", "integer 1", format_version)

    nodes_raw = _list(_required(root, "nodes", "$"), "nodes", nonempty=True)
    nodes: list[NodeSpec] = []
    node_ids: set[str] = set()
    ports: dict[int, str] = {}
    for index, item in enumerate(nodes_raw):
        field = f"nodes[{index}]"
        node = _object(item, field)
        node_id = _string(
            _required(node, "id", field), f"{field}.id", pattern=NODE_ID_RE
        )
        if node_id in node_ids:
            _fail(f"{field}.id", "globally unique node identifier", node_id)
        node_ids.add(node_id)
        host = _string(_required(node, "host", field), f"{field}.host")
        if host != LOOPBACK_HOST:
            _fail(f"{field}.host", f"loopback literal {LOOPBACK_HOST}", host)
        port = _positive_int(_required(node, "port", field), f"{field}.port")
        if port < 1024 or port > 65535:
            _fail(f"{field}.port", "high integer port in [1024, 65535]", port)
        if port in ports:
            _fail(
                f"{field}.port",
                f"globally unique loopback port (already used by {ports[port]})",
                port,
            )
        ports[port] = node_id
        capacity_slots = _positive_int(
            _required(node, "capacity_slots", field), f"{field}.capacity_slots"
        )
        lease_limit = _positive_int(
            _required(node, "lease_limit", field), f"{field}.lease_limit"
        )
        random_seed = _required(node, "random_seed", field)
        if not is_int(random_seed) or random_seed < 0 or random_seed > 0xFFFFFFFF:
            _fail(f"{field}.random_seed", "integer in [0, 4294967295]", random_seed)
        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=port,
                capacity_slots=capacity_slots,
                lease_limit=lease_limit,
                random_seed=random_seed,
            )
        )

    pipeline_raw = _object(_required(root, "pipeline", "$"), "pipeline")
    pipeline_id = _string(
        _required(pipeline_raw, "id", "pipeline"), "pipeline.id", pattern=NODE_ID_RE
    )
    global_concurrency = _positive_int(
        _required(pipeline_raw, "global_concurrency", "pipeline"),
        "pipeline.global_concurrency",
    )
    lease_ms = _positive_int(
        _required(pipeline_raw, "lease_ms", "pipeline"), "pipeline.lease_ms"
    )
    stages_raw = _list(
        _required(pipeline_raw, "stages", "pipeline"), "pipeline.stages", nonempty=True
    )

    stages: list[StageSpec] = []
    artifacts: list[ArtifactSpec] = []
    stage_ids: set[str] = set()
    artifact_keys: set[str] = set()
    for stage_index, item in enumerate(stages_raw):
        stage_field = f"pipeline.stages[{stage_index}]"
        stage_obj = _object(item, stage_field)
        stage_id = _string(
            _required(stage_obj, "id", stage_field),
            f"{stage_field}.id",
            pattern=NODE_ID_RE,
        )
        if stage_id in stage_ids:
            _fail(f"{stage_field}.id", "globally unique stage identifier", stage_id)
        stage_ids.add(stage_id)
        concurrency = _positive_int(
            _required(stage_obj, "concurrency", stage_field),
            f"{stage_field}.concurrency",
        )
        stage_artifacts_raw = _list(
            _required(stage_obj, "artifacts", stage_field),
            f"{stage_field}.artifacts",
            nonempty=True,
        )
        stage_artifact_keys: list[str] = []
        for artifact_index, artifact_item in enumerate(stage_artifacts_raw):
            artifact_field = f"{stage_field}.artifacts[{artifact_index}]"
            artifact_obj = _object(artifact_item, artifact_field)
            key = _string(
                _required(artifact_obj, "key", artifact_field),
                f"{artifact_field}.key",
                pattern=ARTIFACT_KEY_RE,
            )
            if key in artifact_keys:
                _fail(f"{artifact_field}.key", "globally unique artifact key", key)
            artifact_keys.add(key)
            references_raw = _list(
                _required(artifact_obj, "references", artifact_field),
                f"{artifact_field}.references",
            )
            references: list[str] = []
            seen_refs: set[str] = set()
            for ref_index, reference in enumerate(references_raw):
                ref_field = f"{artifact_field}.references[{ref_index}]"
                ref = _string(reference, ref_field)
                if ref in seen_refs:
                    _fail(ref_field, "unique artifact reference", ref)
                seen_refs.add(ref)
                references.append(ref)
            stage_artifact_keys.append(key)
            artifacts.append(
                ArtifactSpec(
                    key=key,
                    references=tuple(sorted(references)),
                    stage_id=stage_id,
                    stage_index=stage_index,
                    artifact_index=artifact_index,
                )
            )
        stages.append(
            StageSpec(
                stage_id=stage_id,
                index=stage_index,
                concurrency=concurrency,
                artifact_keys=tuple(stage_artifact_keys),
            )
        )

    for artifact_index, artifact in enumerate(artifacts):
        for ref in artifact.references:
            if ref not in artifact_keys:
                _fail(
                    f"pipeline.stages[{artifact.stage_index}].artifacts"
                    f"[{artifact.artifact_index}].references",
                    f"existing artifact key (missing {ref})",
                    ref,
                )

    artifact_lookup = {artifact.key: artifact for artifact in artifacts}
    colors: dict[str, int] = {}
    path: list[str] = []

    def check_cycle(key: str) -> None:
        color = colors.get(key, 0)
        if color == 1:
            cycle_start = path.index(key)
            cycle = path[cycle_start:] + [key]
            offending = artifact_lookup[key]
            _fail(
                f"pipeline.stages[{offending.stage_index}].artifacts"
                f"[{offending.artifact_index}].references",
                "acyclic artifact references",
                cycle,
            )
        if color == 2:
            return
        colors[key] = 1
        path.append(key)
        for reference in artifact_lookup[key].references:
            check_cycle(reference)
        path.pop()
        colors[key] = 2

    for artifact in artifacts:
        check_cycle(artifact.key)

    # Validate and cache the DAG ordering now; it is also used by the scheduler.
    provisional = PipelineSpec(
        path=spec_path,
        raw=root,
        format_version=format_version,
        pipeline_id=pipeline_id,
        global_concurrency=global_concurrency,
        lease_ms=lease_ms,
        nodes=tuple(nodes),
        stages=tuple(stages),
        artifacts=tuple(artifacts),
        content_sha256=canonical_sha256(root),
    )
    provisional.topological_layers()
    return provisional


def write_spec(path: str | Path, raw: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
