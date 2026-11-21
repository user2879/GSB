"""Stable presentation models shared by the CLI, tests and Tkinter UI."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .deps import build_dependency_graph
from .models import DiagnosticRun
from .snapshots import SnapshotInfo


def _short_digest(value: str, length: int = 12) -> str:
    return value[:length]


def build_report(
    run: DiagnosticRun,
    *,
    snapshots: Iterable[SnapshotInfo | Mapping[str, Any]] = (),
    recovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    graph = build_dependency_graph(run.plan)
    trace_by_layer = {step.layer_id: step for step in run.invalidation}
    invalidated = {step.layer_id for step in run.invalidation}
    layers: list[dict[str, Any]] = []
    for layer_id in graph.topo:
        result = run.layers[layer_id]
        trace = trace_by_layer.get(layer_id)
        layers.append(
            {
                "layer_id": layer_id,
                "parents": list(graph.parents(layer_id)),
                "children": list(graph.children(layer_id)),
                "precision": result.precision,
                "scale": result.scale,
                "input_dim": result.spec["input_dim"],
                "output_dim": result.spec["output_dim"],
                "param_count": result.spec["param_count"],
                "version": result.version,
                "invalidated": layer_id in invalidated,
                "invalidation": None if trace is None else trace.to_dict(),
                "block_ids": list(result.block_ids),
                "allocation_mode": result.allocation_mode,
                "allocation_runs": [list(item) for item in result.allocation_runs],
                "request_bytes": result.request_bytes,
                "output_digest": result.output_digest,
                "float_digest": result.float_digest,
                "error_bound": result.error_bound,
                "max_abs_error": result.max_abs_error,
                "error_components": {
                    key: result.error_components[key]
                    for key in sorted(result.error_components)
                },
                "dominant_segment": result.dominant_segment,
            }
        )
    snapshot_list = [
        item.to_dict() if isinstance(item, SnapshotInfo) else dict(item)
        for item in snapshots
    ]
    snapshot_list.sort(key=lambda item: (int(item["plan_version"]), str(item["snapshot_id"])))
    return {
        "ok": True,
        "format_version": 3,
        "plan_version": run.plan.plan_version,
        "plan_digest": run.plan.digest,
        "content_digest": run.content_digest,
        "samples_digest": run.samples.digest,
        "sample_count": len(run.samples.samples),
        "max_batch_size": run.plan.max_batch_size,
        "render_order": "stable_topological_then_lexical",
        "invalidated_layers": [step.layer_id for step in run.invalidation],
        "invalidation_trace": [step.to_dict() for step in run.invalidation],
        "layers": layers,
        "memory": run.memory_summary,
        "batches": [batch.to_dict() for batch in run.batches],
        "attribution": run.attribution,
        "snapshots": snapshot_list,
        "recovery": None if recovery is None else dict(recovery),
    }


def build_view_model(
    run: DiagnosticRun,
    *,
    snapshots: Iterable[SnapshotInfo | Mapping[str, Any]] = (),
    sessions: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    report = build_report(run, snapshots=snapshots)
    block_owners: dict[int, str] = {}
    for layer in report["layers"]:
        for block in layer["block_ids"]:
            block_owners[int(block)] = layer["layer_id"]
    blocks = [
        {
            "index": index,
            "owner": block_owners.get(index),
            "state": "allocated" if index in block_owners else "free",
        }
        for index in range(run.pool_capacity_blocks)
    ]
    error_ranking = report["attribution"].get("rankings", [])
    error_chain = [
        {
            "rank": entry["rank"],
            "layer_id": entry["layer_id"],
            "error_bound": entry["error_bound"],
            "max_abs_error": entry["max_abs_error"],
            "cumulative_error_ratio": entry["cumulative_error_ratio"],
            "dominant_segment": entry["dominant_segment"],
            "tie_affected": entry["tie_affected"],
        }
        for entry in error_ranking
    ]
    session_list = [dict(item) for item in sessions]
    session_list.sort(key=lambda item: (str(item.get("session_id", "")),))
    return {
        "header": {
            "plan_version": report["plan_version"],
            "content_digest": report["content_digest"],
            "plan_digest": report["plan_digest"],
            "sample_count": report["sample_count"],
            "render_order": report["render_order"],
        },
        "layers": report["layers"],
        "invalidation_trace": report["invalidation_trace"],
        "memory": report["memory"],
        "blocks": blocks,
        "error_chain": error_chain,
        "top_error_layer": report["attribution"].get("top_layer"),
        "attribution": report["attribution"],
        "batches": report["batches"],
        "snapshots": report["snapshots"],
        "sessions": session_list,
        "report": report,
    }


def format_layer_chain(layer: Mapping[str, Any]) -> str:
    parents = ",".join(layer.get("parents", [])) or "sample"
    children = ",".join(layer.get("children", [])) or "sink"
    return f"{parents} -> [{layer['layer_id']}] -> {children}"


def block_map_text(run: DiagnosticRun) -> str:
    owner = {block: layer_id for layer_id, blocks in run.allocations.items() for block in blocks}
    return " ".join(f"{index}:{owner.get(index, '-')}" for index in range(run.pool_capacity_blocks))
