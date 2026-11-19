"""Scene file loading with strict validation.

A scene declares three sections:

* ``nodes``      - deterministic local mock-service targets on 127.0.0.1
* ``task``       - payload, per-node attempt cap, timeout and backoff series
* ``slo``        - integer success-rate threshold (percent) and P95 ceiling
* ``scheduling`` - global/per-node concurrency and time budgets (optional)

Any violation raises :class:`SceneError` (code ``E_SCENE``) and the details
payload lists every offending JSON field path; the scene is rejected as a
whole, i.e. no partial configuration object is ever returned.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SceneError

LOOPBACK = "127.0.0.1"
SCENE_SCHEMA_VERSION = 1
DEFAULT_DELIVERIES_PER_NODE = 1
DEFAULT_NODE_CONCURRENCY = 1
DEFAULT_SESSION_BUDGET_MS = 60_000
OUTCOMES = ("success", "failure", "timeout", "unreachable", "unknown")


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    process_ms: int
    failure_rate: int          # basis points, 0..10000, deterministic rule
    slow_rate: int             # basis points, 0..10000, deterministic rule
    slow_extra_ms: int
    seed: int


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    payload_bytes: int
    max_attempts: int
    attempt_timeout_ms: int
    backoff_ms: tuple[int, ...]
    deliveries_per_node: int = DEFAULT_DELIVERIES_PER_NODE


@dataclass(frozen=True)
class SchedulingSpec:
    global_concurrency: int
    node_concurrency: int
    session_budget_ms: int


@dataclass(frozen=True)
class SloSpec:
    success_rate_percent: int
    p95_latency_ms: int


@dataclass(frozen=True)
class Scene:
    scene_version: str
    nodes: tuple[NodeSpec, ...]
    task: TaskSpec
    scheduling: SchedulingSpec
    slo: SloSpec
    payload_digest: str
    source_path: Path | None = field(default=None, compare=False)

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def canonical_dict(self) -> dict[str, Any]:
        """JSON-safe canonical representation (keys sorted, tuples expanded)."""
        return {
            "scene_version": self.scene_version,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "process_ms": n.process_ms,
                    "failure_rate": n.failure_rate,
                    "slow_rate": n.slow_rate,
                    "slow_extra_ms": n.slow_extra_ms,
                    "seed": n.seed,
                }
                for n in self.nodes
            ],
            "task": {
                "task_id": self.task.task_id,
                "payload_bytes": self.task.payload_bytes,
                "max_attempts": self.task.max_attempts,
                "attempt_timeout_ms": self.task.attempt_timeout_ms,
                "backoff_ms": list(self.task.backoff_ms),
                "deliveries_per_node": self.task.deliveries_per_node,
            },
            "scheduling": {
                "global_concurrency": self.scheduling.global_concurrency,
                "node_concurrency": self.scheduling.node_concurrency,
                "session_budget_ms": self.scheduling.session_budget_ms,
            },
            "slo": {
                "success_rate_percent": self.slo.success_rate_percent,
                "p95_latency_ms": self.slo.p95_latency_ms,
            },
            "payload_digest": self.payload_digest,
        }


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; reject it explicitly.
    return isinstance(value, int) and not isinstance(value, bool)


def _digest_payload(task_id: str, payload_bytes: int) -> str:
    material = f"{task_id}|{payload_bytes}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def load_scene(path: str | Path) -> Scene:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SceneError(f"scene file not found: {path}", {"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise SceneError(
            "scene file is not valid JSON",
            {"path": str(path), "json_error": exc.msg, "line": exc.lineno, "col": exc.colno},
        ) from exc
    scene = parse_scene(raw, source_path=path)
    return scene


def parse_scene(raw: Any, source_path: Path | None = None) -> Scene:
    errors: list[dict[str, str]] = []

    def err(field_path: str, reason: str) -> None:
        errors.append({"field": field_path, "reason": reason})

    if not isinstance(raw, dict):
        raise SceneError("scene root must be a JSON object", {"field": "$", "reason": "not an object"})

    scene_version = raw.get("scene_version")
    if not isinstance(scene_version, str) or not scene_version.strip():
        err("scene_version", "must be a non-empty string")
    else:
        scene_version = scene_version.strip()

    nodes_raw = raw.get("nodes")
    nodes: list[NodeSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    if not isinstance(nodes_raw, list) or not nodes_raw:
        err("nodes", "must be a non-empty array")
        nodes_raw = []
    for index, item in enumerate(nodes_raw):
        prefix = f"nodes[{index}]"
        if not isinstance(item, dict):
            err(prefix, "node entry must be an object")
            continue
        node_id = item.get("node_id")
        if not isinstance(node_id, str) or not node_id.strip():
            err(f"{prefix}.node_id", "must be a non-empty string")
        elif node_id in seen_ids:
            err(f"{prefix}.node_id", f"duplicate node_id {node_id!r}")
        else:
            seen_ids.add(node_id)
        host = item.get("host", LOOPBACK)
        if host != LOOPBACK:
            err(f"{prefix}.host", "must be the loopback literal 127.0.0.1")
        port = item.get("port")
        if not _is_int(port) or not (1 <= port <= 65535):
            err(f"{prefix}.port", "must be an integer in [1, 65535]")
        elif port in seen_ports:
            err(f"{prefix}.port", f"duplicate port {port}")
        else:
            seen_ports.add(port)
        process_ms = item.get("process_ms", 0)
        if not _is_int(process_ms) or process_ms < 0:
            err(f"{prefix}.process_ms", "must be a non-negative integer")
        slow_extra_ms = item.get("slow_extra_ms", 0)
        if not _is_int(slow_extra_ms) or slow_extra_ms < 0:
            err(f"{prefix}.slow_extra_ms", "must be a non-negative integer")
        failure_rate = item.get("failure_rate", 0)
        if not _is_int(failure_rate) or not (0 <= failure_rate <= 10000):
            err(f"{prefix}.failure_rate", "must be an integer in [0, 10000] basis points")
        slow_rate = item.get("slow_rate", 0)
        if not _is_int(slow_rate) or not (0 <= slow_rate <= 10000):
            err(f"{prefix}.slow_rate", "must be an integer in [0, 10000] basis points")
        seed = item.get("seed", 0)
        if not _is_int(seed) or seed < 0:
            err(f"{prefix}.seed", "must be a non-negative integer")
        if not errors or all(e["field"].split(".")[0] != prefix for e in errors):
            nodes.append(
                NodeSpec(
                    node_id=node_id if isinstance(node_id, str) else f"__invalid_{index}",
                    host=LOOPBACK,
                    port=port if _is_int(port) else 0,
                    process_ms=process_ms if _is_int(process_ms) else 0,
                    failure_rate=failure_rate if _is_int(failure_rate) else 0,
                    slow_rate=slow_rate if _is_int(slow_rate) else 0,
                    slow_extra_ms=slow_extra_ms if _is_int(slow_extra_ms) else 0,
                    seed=seed if _is_int(seed) else 0,
                )
            )

    task_raw = raw.get("task")
    if not isinstance(task_raw, dict):
        err("task", "must be an object")
        task_raw = {}
    task_id = task_raw.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        err("task.task_id", "must be a non-empty string")
    payload_bytes = task_raw.get("payload_bytes")
    if not _is_int(payload_bytes) or payload_bytes <= 0:
        err("task.payload_bytes", "must be a positive integer")
    max_attempts = task_raw.get("max_attempts")
    if not _is_int(max_attempts) or max_attempts <= 0:
        err("task.max_attempts", "must be a positive integer")
    attempt_timeout_ms = task_raw.get("attempt_timeout_ms")
    if not _is_int(attempt_timeout_ms) or attempt_timeout_ms <= 0:
        err("task.attempt_timeout_ms", "must be a positive integer (ms)")
    backoff_raw = task_raw.get("backoff_ms", [])
    backoff: list[int] = []
    if not isinstance(backoff_raw, list) or not backoff_raw:
        err("task.backoff_ms", "must be a non-empty array of integers")
    else:
        previous = -1
        monotone = True
        for j, value in enumerate(backoff_raw):
            if not _is_int(value) or value < 0:
                err(f"task.backoff_ms[{j}]", "must be a non-negative integer")
                monotone = False
                break
            if value < previous:
                monotone = False
            previous = value
            backoff.append(value)
        if not monotone:
            err("task.backoff_ms", "must be monotonically non-decreasing")
    deliveries = task_raw.get("deliveries_per_node", DEFAULT_DELIVERIES_PER_NODE)
    if not _is_int(deliveries) or deliveries <= 0:
        err("task.deliveries_per_node", "must be a positive integer")

    sched_raw = raw.get("scheduling", {})
    if not isinstance(sched_raw, dict):
        err("scheduling", "must be an object")
        sched_raw = {}
    global_concurrency = sched_raw.get("global_concurrency", 1)
    if not _is_int(global_concurrency) or global_concurrency <= 0:
        err("scheduling.global_concurrency", "must be a positive integer")
    node_concurrency = sched_raw.get("node_concurrency", DEFAULT_NODE_CONCURRENCY)
    if not _is_int(node_concurrency) or node_concurrency <= 0:
        err("scheduling.node_concurrency", "must be a positive integer")
    session_budget = sched_raw.get("session_budget_ms", DEFAULT_SESSION_BUDGET_MS)
    if not _is_int(session_budget) or session_budget <= 0:
        err("scheduling.session_budget_ms", "must be a positive integer")

    slo_raw = raw.get("slo")
    if not isinstance(slo_raw, dict):
        err("slo", "must be an object")
        slo_raw = {}
    rate = slo_raw.get("success_rate_percent")
    if not _is_int(rate) or not (0 <= rate <= 100):
        err("slo.success_rate_percent", "must be an integer in [0, 100]")
    p95 = slo_raw.get("p95_latency_ms")
    if not _is_int(p95) or p95 <= 0:
        err("slo.p95_latency_ms", "must be a positive integer (ms)")

    if errors:
        # Reject as a whole; list every offending field path.
        raise SceneError(
            f"scene validation failed with {len(errors)} error(s)",
            {"errors": errors},
        )

    if node_concurrency > global_concurrency:
        raise SceneError(
            "node_concurrency cannot exceed global_concurrency",
            {"errors": [
                {"field": "scheduling.node_concurrency", "reason": "exceeds global_concurrency"}
            ]},
        )

    task = TaskSpec(
        task_id=task_id.strip(),
        payload_bytes=payload_bytes,
        max_attempts=max_attempts,
        attempt_timeout_ms=attempt_timeout_ms,
        backoff_ms=tuple(backoff),
        deliveries_per_node=deliveries,
    )
    scheduling = SchedulingSpec(global_concurrency, node_concurrency, session_budget)
    slo = SloSpec(rate, p95)
    payload_digest = _digest_payload(task.task_id, task.payload_bytes)
    return Scene(
        scene_version=scene_version,
        nodes=tuple(nodes),
        task=task,
        scheduling=scheduling,
        slo=slo,
        payload_digest=payload_digest,
        source_path=source_path,
    )
