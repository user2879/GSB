"""Incremental DAG execution with exact invalidation closure and full fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig, changed_parameter_keys
from .errors import closure_mismatch, dependency_stale, region_bounds, snapshot_version
from .features import (
    Selection,
    SelectionResolver,
    build_conclusion,
    build_cross_gop_summary,
    build_gop_summary,
    compute_frame_payload,
)
from .graph import DependencyGraph, GraphEdge, GraphNode, compare_graphs
from .runcontrol import RunSession
from .source import SourceBundle, load_source
from .state import CommitResult, Snapshot, SnapshotStore
from .util import sha256_json


SOURCE_NODE = "source:root"
CONCLUSION_NODE = "conclusion"
PARAMETER_KEYS = (
    "source",
    "sampling_region",
    "watermark_mask",
    "analysis_window",
    "threshold",
    "gop_size",
    "feature_version",
)


@dataclass
class BuildResult:
    graph: DependencyGraph
    audit: dict[str, Any]


@dataclass
class RunResult:
    snapshot: Snapshot | None
    commit: CommitResult | None
    build: BuildResult
    full_verified: bool
    run_session: RunSession | None = None

    @property
    def graph(self) -> DependencyGraph:
        return self.build.graph

    @property
    def audit(self) -> dict[str, Any]:
        return self.build.audit

    @property
    def conclusion_hash(self) -> str:
        return self.graph.nodes[CONCLUSION_NODE].content_hash


def frame_node(frame_index: int) -> str:
    return f"frame:{frame_index:06d}"


def gop_node(gop_index: int) -> str:
    return f"gop:{gop_index:03d}"


def cross_node(left: int, right: int) -> str:
    return f"xgop:{left:03d}-{right:03d}"


def _source_payload(source: SourceBundle) -> dict[str, Any]:
    return {
        "manifest_sha256": source.manifest_digest,
        "video_sha256": source.video_sha256,
        "truth_sha256": source.truth_sha256,
        "pixel_content_sha256": source.content_sha256,
        "width": source.spec.width,
        "height": source.spec.height,
        "fps": source.spec.fps,
        "frame_count": source.frame_count,
        "gop_size": source.spec.gop_size,
        "shot_boundaries": sorted(source.shot_boundaries),
    }


def _frame_param_versions(source: SourceBundle, selection: Selection, config: PipelineConfig) -> dict[str, str]:
    return {
        "source": source.manifest_digest,
        "sampling_region": selection.sampling_digest,
        "watermark_mask": selection.mask_digest,
        "feature_version": sha256_json(config.feature_version),
    }


def _window_param(frame_start: int, frame_end: int) -> str:
    return sha256_json({"frame_start": frame_start, "frame_end": frame_end})


def _gop_param_versions(config: PipelineConfig, frame_start: int, frame_end: int) -> dict[str, str]:
    return {
        "analysis_window": _window_param(frame_start, frame_end),
        "threshold": sha256_json(config.suspicion_threshold),
        "feature_version": sha256_json(config.feature_version),
    }


def _cross_param_versions(config: PipelineConfig) -> dict[str, str]:
    return {
        "threshold": sha256_json(config.suspicion_threshold),
        "feature_version": sha256_json(config.feature_version),
    }


def _conclusion_param_versions(config: PipelineConfig) -> dict[str, str]:
    return {
        "analysis_window": sha256_json(
            {
                "start_frame": config.analysis_window.start_frame,
                "end_frame": config.analysis_window.end_frame,
            }
        ),
        "threshold": sha256_json(config.suspicion_threshold),
        "feature_version": sha256_json(config.feature_version),
    }


def _edge(
    source: str,
    target: str,
    relation: str,
    trigger_keys: tuple[str, ...],
) -> GraphEdge:
    return GraphEdge(source=source, target=target, trigger_keys=trigger_keys, relation=relation)


def build_graph(
    source: SourceBundle,
    config: PipelineConfig,
    *,
    base_snapshot: Snapshot | None,
    target_version: int,
    force_full: bool = False,
    run_id: str = "run",
) -> BuildResult:
    config.validate(source.spec.width, source.spec.height, source.frame_count)
    if config.gop_size != source.spec.gop_size:
        raise region_bounds(
            "pipeline gop_size must match the generated source",
            config_gop_size=config.gop_size,
            source_gop_size=source.spec.gop_size,
        )
    previous_graph = None if base_snapshot is None else base_snapshot.graph
    current_param_versions = config.parameter_versions()
    current_param_versions["source"] = source.manifest_digest
    if previous_graph is None:
        changed_keys = list(PARAMETER_KEYS)
    else:
        changed_keys = changed_parameter_keys(previous_graph.param_versions, current_param_versions)
    window = config.analysis_window
    active_frames = list(range(window.start_frame, window.end_frame))
    resolver = SelectionResolver(source.spec.width, source.spec.height, config)
    selections = {index: resolver.for_frame(index) for index in active_frames}
    active_gops = sorted({index // config.gop_size for index in active_frames})
    gop_frames: dict[int, list[int]] = {}
    for gop_index in active_gops:
        start = max(window.start_frame, gop_index * config.gop_size)
        end = min(window.end_frame, (gop_index + 1) * config.gop_size)
        gop_frames[gop_index] = list(range(start, end))
    cross_items = list(zip(active_gops, active_gops[1:]))

    previous_nodes = {} if previous_graph is None else previous_graph.nodes
    dirty_seed: set[str] = set()
    dirty_reason: dict[str, list[str]] = {}
    if force_full or previous_graph is None:
        dirty_seed.add(SOURCE_NODE)
    elif SOURCE_NODE not in previous_nodes or previous_nodes[SOURCE_NODE].payload != _source_payload(source):
        dirty_seed.add(SOURCE_NODE)

    for index in active_frames:
        node_id = frame_node(index)
        selection = selections[index]
        expected_params = _frame_param_versions(source, selection, config)
        reasons: list[str] = []
        previous = previous_nodes.get(node_id)
        if force_full or previous is None:
            reasons.append("new_or_forced")
        else:
            if previous.payload.get("selection_digest") != selection.selection_digest:
                reasons.append("selection_changed")
            if previous.param_versions != expected_params:
                reasons.append("frame_params_changed")
        if reasons:
            dirty_seed.add(node_id)
            dirty_reason[node_id] = reasons

    for gop_index, members in gop_frames.items():
        node_id = gop_node(gop_index)
        expected_params = _gop_param_versions(config, members[0], members[-1] + 1)
        reasons: list[str] = []
        previous = previous_nodes.get(node_id)
        if force_full or previous is None:
            reasons.append("new_or_forced")
        else:
            if previous.param_versions != expected_params:
                reasons.append("gop_params_changed")
            if set(previous.input_hashes) != {frame_node(index) for index in members}:
                reasons.append("gop_membership_changed")
        if reasons:
            dirty_seed.add(node_id)
            dirty_reason[node_id] = reasons

    for left, right in cross_items:
        node_id = cross_node(left, right)
        expected_params = _cross_param_versions(config)
        previous = previous_nodes.get(node_id)
        if force_full or previous is None:
            dirty_seed.add(node_id)
            dirty_reason[node_id] = ["new_or_forced"]
        elif previous.param_versions != expected_params:
            dirty_seed.add(node_id)
            dirty_reason[node_id] = ["cross_gop_params_changed"]

    if force_full or CONCLUSION_NODE not in previous_nodes:
        dirty_seed.add(CONCLUSION_NODE)
        dirty_reason[CONCLUSION_NODE] = ["new_or_forced"]
    elif previous_nodes[CONCLUSION_NODE].param_versions != _conclusion_param_versions(config):
        dirty_seed.add(CONCLUSION_NODE)
        dirty_reason[CONCLUSION_NODE] = ["conclusion_params_changed"]

    dirty = set(dirty_seed)
    changed = True
    while changed:
        changed = False
        for gop_index, members in gop_frames.items():
            node_id = gop_node(gop_index)
            if node_id not in dirty and any(frame_node(index) in dirty for index in members):
                dirty.add(node_id)
                dirty_reason.setdefault(node_id, []).append("member_frame_invalidated")
                changed = True
        for left, right in cross_items:
            node_id = cross_node(left, right)
            if node_id not in dirty and (
                gop_node(left) in dirty or gop_node(right) in dirty
            ):
                dirty.add(node_id)
                dirty_reason.setdefault(node_id, []).append("adjacent_gop_invalidated")
                changed = True
        has_dirty_gop = any(gop_node(index) in dirty for index in gop_frames)
        has_dirty_cross = any(cross_node(left, right) in dirty for left, right in cross_items)
        if CONCLUSION_NODE not in dirty and (has_dirty_gop or has_dirty_cross):
            dirty.add(CONCLUSION_NODE)
            dirty_reason.setdefault(CONCLUSION_NODE, []).append("dependency_invalidated")
            changed = True

    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    reused: list[dict[str, Any]] = []
    recomputed: list[str] = []
    source_node = GraphNode(
        node_id=SOURCE_NODE,
        node_type="source",
        payload=_source_payload(source),
        param_versions={"source": source.manifest_digest},
        input_hashes={},
        produced_in_version=target_version,
    ).seal()
    if SOURCE_NODE not in dirty and SOURCE_NODE in previous_nodes:
        source_node = previous_nodes[SOURCE_NODE]
        _record_reuse(reused, base_snapshot, source_node)
    else:
        recomputed.append(SOURCE_NODE)
    nodes[SOURCE_NODE] = source_node

    frame_reader_needed = any(frame_node(index) in dirty for index in active_frames)
    new_frame_payloads: dict[int, dict[str, Any]] = {}
    if frame_reader_needed:
        with source.open_reader() as reader:
            for index in active_frames:
                node_id = frame_node(index)
                if node_id not in dirty:
                    continue
                frame = reader.read_frame(index)
                try:
                    payload = compute_frame_payload(
                        frame_index=index,
                        frame=frame,
                        selection=selections[index],
                        shot_boundaries=source.shot_boundaries,
                        gop_size=config.gop_size,
                        feature_version=config.feature_version,
                    )
                except Exception:
                    raise
                new_frame_payloads[index] = payload

    for index in active_frames:
        node_id = frame_node(index)
        selection = selections[index]
        expected_params = _frame_param_versions(source, selection, config)
        expected_inputs = {SOURCE_NODE: source_node.content_hash}
        if node_id in dirty:
            payload = new_frame_payloads[index]
            node = GraphNode(
                node_id=node_id,
                node_type="frame_feature",
                payload=payload,
                param_versions=expected_params,
                input_hashes=expected_inputs,
                produced_in_version=target_version,
            ).seal()
            recomputed.append(node_id)
        else:
            node = _reuse_checked(
                previous_nodes,
                node_id,
                expected_params,
                expected_inputs,
                base_snapshot,
            )
            _record_reuse(reused, base_snapshot, node)
        nodes[node_id] = node
        edges.append(
            _edge(
                SOURCE_NODE,
                node_id,
                "source_frame_to_frame_feature",
                ("source", "sampling_region", "watermark_mask"),
            )
        )

    for gop_index, members in gop_frames.items():
        node_id = gop_node(gop_index)
        feature_payloads = [nodes[frame_node(index)].payload for index in members]
        expected_inputs = {frame_node(index): nodes[frame_node(index)].content_hash for index in members}
        expected_params = _gop_param_versions(config, members[0], members[-1] + 1)
        if node_id in dirty:
            payload = build_gop_summary(
                gop_index=gop_index,
                features=feature_payloads,
                threshold=config.suspicion_threshold,
            )
            node = GraphNode(
                node_id=node_id,
                node_type="gop_summary",
                payload=payload,
                param_versions=expected_params,
                input_hashes=expected_inputs,
                produced_in_version=target_version,
            ).seal()
            recomputed.append(node_id)
        else:
            node = _reuse_checked(
                previous_nodes,
                node_id,
                expected_params,
                expected_inputs,
                base_snapshot,
            )
            _record_reuse(reused, base_snapshot, node)
        nodes[node_id] = node
        for index in members:
            edges.append(
                _edge(
                    frame_node(index),
                    node_id,
                    "frame_feature_to_gop_summary",
                    ("sampling_region", "watermark_mask", "analysis_window"),
                )
            )

    for left, right in cross_items:
        node_id = cross_node(left, right)
        left_node = nodes[gop_node(left)]
        right_node = nodes[gop_node(right)]
        expected_inputs = {
            left_node.node_id: left_node.content_hash,
            right_node.node_id: right_node.content_hash,
        }
        expected_params = _cross_param_versions(config)
        if node_id in dirty:
            payload = build_cross_gop_summary(
                left_gop=left,
                right_gop=right,
                left=left_node.payload,
                right=right_node.payload,
                threshold=config.suspicion_threshold,
            )
            node = GraphNode(
                node_id=node_id,
                node_type="cross_gop",
                payload=payload,
                param_versions=expected_params,
                input_hashes=expected_inputs,
                produced_in_version=target_version,
            ).seal()
            recomputed.append(node_id)
        else:
            node = _reuse_checked(
                previous_nodes,
                node_id,
                expected_params,
                expected_inputs,
                base_snapshot,
            )
            _record_reuse(reused, base_snapshot, node)
        nodes[node_id] = node
        edges.append(
            _edge(
                gop_node(left),
                node_id,
                "gop_summary_to_cross_gop_left",
                ("sampling_region", "watermark_mask", "analysis_window", "threshold"),
            )
        )
        edges.append(
            _edge(
                gop_node(right),
                node_id,
                "gop_summary_to_cross_gop_right",
                ("sampling_region", "watermark_mask", "analysis_window", "threshold"),
            )
        )

    gop_summaries = [nodes[gop_node(index)].payload for index in active_gops]
    cross_summaries = [nodes[cross_node(left, right)].payload for left, right in cross_items]
    conclusion_inputs: dict[str, str] = {}
    for gop_index in active_gops:
        gop_id = gop_node(gop_index)
        conclusion_inputs[gop_id] = nodes[gop_id].content_hash
        edges.append(
            _edge(
                gop_id,
                CONCLUSION_NODE,
                "gop_summary_to_conclusion",
                ("analysis_window", "threshold", "sampling_region", "watermark_mask"),
            )
        )
    for left, right in cross_items:
        cross_id = cross_node(left, right)
        conclusion_inputs[cross_id] = nodes[cross_id].content_hash
        edges.append(
            _edge(
                cross_id,
                CONCLUSION_NODE,
                "cross_gop_to_conclusion",
                ("analysis_window", "threshold", "sampling_region", "watermark_mask"),
            )
        )
    expected_conclusion_params = _conclusion_param_versions(config)
    if CONCLUSION_NODE in dirty:
        conclusion_payload = build_conclusion(
            window_start=window.start_frame,
            window_end=window.end_frame,
            gop_summaries=gop_summaries,
            cross_summaries=cross_summaries,
            threshold=config.suspicion_threshold,
        )
        conclusion = GraphNode(
            node_id=CONCLUSION_NODE,
            node_type="integrity_conclusion",
            payload=conclusion_payload,
            param_versions=expected_conclusion_params,
            input_hashes=conclusion_inputs,
            produced_in_version=target_version,
        ).seal()
        recomputed.append(CONCLUSION_NODE)
    else:
        conclusion = _reuse_checked(
            previous_nodes,
            CONCLUSION_NODE,
            expected_conclusion_params,
            conclusion_inputs,
            base_snapshot,
        )
        _record_reuse(reused, base_snapshot, conclusion)
    nodes[CONCLUSION_NODE] = conclusion

    graph = DependencyGraph(
        version=target_version,
        param_versions=current_param_versions,
        nodes=nodes,
        edges=edges,
    ).seal()
    previous_ids = set() if previous_graph is None else set(previous_graph.nodes)
    deactivated = sorted(previous_ids - set(nodes))
    recomputed_sorted = sorted(set(recomputed))
    reused_sorted = sorted(reused, key=lambda item: item["node_id"])
    reuse_counts: dict[str, int] = {}
    for item in reused_sorted:
        reuse_counts[item["node_type"]] = reuse_counts.get(item["node_type"], 0) + 1
    audit = {
        "format": "forensic-incremental-run-v1",
        "run_id": run_id,
        "target_version": target_version,
        "base_snapshot": None
        if base_snapshot is None
        else {
            "version": base_snapshot.version,
            "snapshot_hash": base_snapshot.snapshot_hash,
        },
        "changed_parameters": changed_keys,
        "active_window": {"start_frame": window.start_frame, "end_frame": window.end_frame},
        "active_gops": active_gops,
        "dirty_seed_nodes": sorted(dirty_seed),
        "dirty_reasons": dirty_reason,
        "invalidated_closure": recomputed_sorted,
        "recomputed_nodes": recomputed_sorted,
        "reused_nodes": reused_sorted,
        "reuse_counts": reuse_counts,
        "deactivated_nodes": deactivated,
        "forced_full_recompute": force_full,
        "conclusion_hash": conclusion.content_hash,
        "integrity_status": conclusion.payload["integrity_status"],
        "overall_suspicion": conclusion.payload["overall_suspicion"],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    if previous_graph is not None and not force_full:
        expected_reused = len(nodes) - len(recomputed_sorted)
        if len(reused_sorted) != expected_reused:
            raise closure_mismatch(
                "closure accounting did not match the executed node plan",
                run_id=run_id,
                expected_reused=expected_reused,
                actual_reused=len(reused_sorted),
                recomputed_nodes=recomputed_sorted,
            )
    graph.audit_history = list(
        ([] if previous_graph is None else previous_graph.audit_history)
    )[-4:] + [audit]
    return BuildResult(graph=graph, audit=audit)


def _record_reuse(
    records: list[dict[str, Any]], base_snapshot: Snapshot | None, node: GraphNode
) -> None:
    if base_snapshot is None:
        raise dependency_stale("reuse was requested without a base snapshot", node_id=node.node_id)
    records.append(
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "source_snapshot_version": base_snapshot.version,
            "source_snapshot_hash": base_snapshot.snapshot_hash,
            "produced_in_version": node.produced_in_version,
            "content_hash": node.content_hash,
            "param_versions": node.param_versions,
        }
    )


def _reuse_checked(
    previous_nodes: dict[str, GraphNode],
    node_id: str,
    expected_params: dict[str, str],
    expected_inputs: dict[str, str],
    base_snapshot: Snapshot | None,
) -> GraphNode:
    if node_id not in previous_nodes:
        raise dependency_stale(
            "node selected for reuse is missing from the committed graph",
            node_id=node_id,
            snapshot_version=None if base_snapshot is None else base_snapshot.version,
        )
    node = previous_nodes[node_id]
    if node.param_versions != expected_params or node.input_hashes != expected_inputs:
        raise closure_mismatch(
            "reused node has stale dependencies outside the reported closure",
            node_id=node_id,
            node_type=node.node_type,
            previous_param_versions=node.param_versions,
            current_param_versions=expected_params,
            previous_input_hashes=node.input_hashes,
            current_input_hashes=expected_inputs,
        )
    return node


def _crash_hook(stage: str, requested_stage: str | None) -> None:
    if requested_stage == stage:
        os._exit(17)


def execute(
    workspace: Path,
    config: PipelineConfig,
    *,
    expected_base_version: int | None = None,
    expected_base_hash: str | None = None,
    verify_full: bool = True,
    full_recompute: bool = False,
    verify_source_hashes: bool = True,
    run_dir: Path | None = None,
    run_id: str | None = None,
    crash_after_stage: str | None = None,
) -> RunResult:
    workspace = Path(workspace)
    store = SnapshotStore(workspace)
    base_snapshot = store.load()
    current_version = 0 if base_snapshot is None else base_snapshot.version
    current_hash = None if base_snapshot is None else base_snapshot.snapshot_hash
    if expected_base_version is None:
        expected_base_version = current_version
    if expected_base_hash is None and base_snapshot is not None:
        expected_base_hash = current_hash
    if expected_base_version != current_version or expected_base_hash != current_hash:
        raise snapshot_version(
            "requested base version is stale",
            expected_base_version=expected_base_version,
            expected_base_hash=expected_base_hash,
            current_version=current_version,
            current_hash=current_hash,
        )
    source = load_source(workspace / "source", verify_hashes=verify_source_hashes)
    config.validate(source.spec.width, source.spec.height, source.frame_count)
    target_version = (0 if base_snapshot is None else base_snapshot.version) + 1
    selected_run_id = run_id or f"run-{target_version:06d}"
    if run_dir is None:
        run_dir = workspace / "runs" / selected_run_id
    session = RunSession.start(
        run_dir,
        run_id=selected_run_id,
        workspace=workspace,
        base_version=None if base_snapshot is None else base_snapshot.version,
        base_hash=None if base_snapshot is None else base_snapshot.snapshot_hash,
    )
    build = build_graph(
        source,
        config,
        base_snapshot=base_snapshot,
        target_version=target_version,
        force_full=full_recompute,
        run_id=selected_run_id,
    )
    full_verified = False
    if verify_full:
        if full_recompute:
            full_verified = True
        else:
            full_build = build_graph(
                source,
                config,
                base_snapshot=None,
                target_version=target_version,
                force_full=True,
                run_id=f"{selected_run_id}-full-check",
            )
            compare_graphs(
                build.graph,
                full_build.graph,
                context={"run_id": selected_run_id, "method": "incremental_vs_full"},
            )
            full_verified = True
    pending_path, _ = store.write_pending(
        selected_run_id,
        {
            "format": "forensic-pending-run-v1",
            "run_id": selected_run_id,
            "base_version": None if base_snapshot is None else base_snapshot.version,
            "base_snapshot_hash": None if base_snapshot is None else base_snapshot.snapshot_hash,
            "target_version": target_version,
            "graph_hash": build.graph.graph_hash(),
            "conclusion_hash": build.graph.nodes[CONCLUSION_NODE].content_hash,
            "changed_parameters": build.audit["changed_parameters"],
            "recomputed_count": len(build.audit["recomputed_nodes"]),
        },
    )
    commit = store.commit(
        build.graph,
        expected_base_version=expected_base_version,
        expected_base_hash=expected_base_hash,
        crash_hook=lambda stage: _crash_hook(stage, crash_after_stage),
    )
    store.clear_pending(pending_path)
    snapshot = store.load()
    if snapshot is None:
        raise dependency_stale("snapshot disappeared immediately after commit")
    if session is not None:
        session.close(
            snapshot_version=commit.version,
            snapshot_hash=commit.snapshot_hash,
            node_hashes=snapshot.graph.node_hashes(),
            conclusion_hash=snapshot.graph.nodes[CONCLUSION_NODE].content_hash,
            run_summary={
                "changed_parameters": build.audit["changed_parameters"],
                "recomputed_count": len(build.audit["recomputed_nodes"]),
                "reused_count": len(build.audit["reused_nodes"]),
                "full_verified": full_verified,
            },
        )
    return RunResult(
        snapshot=snapshot,
        commit=commit,
        build=build,
        full_verified=full_verified,
        run_session=session,
    )
