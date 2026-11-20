from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .camera import NUMERIC_POLICY, generate_trajectory, project_bounds
from .canonical import (
    canonical_bytes,
    canonical_line,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .culling import CullingPolicy, cull_blocks, projection_trigger_blocks
from .diagnostics import error_distribution
from .errors import ErrorCode, fail
from .heightfield import (
    DEFAULT_SEED,
    TerrainConfig,
    generate_heightfield,
    load_terrain,
    save_generated_terrain,
    validate_height_range,
)
from .hiz import (
    HIZ_BASE_RESOLUTION,
    build_hiz,
    render_depth_and_owner,
    save_hiz,
    validate_depth_bias,
)
from .quadtree import (
    QuadtreeManifest,
    build_quadtree,
    load_quadtree_manifest,
    manifest_path,
    save_quadtree_manifest,
    total_leaf_bytes,
)
from .streaming import StreamingBudgets, StreamingManager
from .workspace import MINIMUM_FREE_BYTES, Workspace


@dataclass(frozen=True)
class RunSettings:
    size: int = 1025
    seed: int = DEFAULT_SEED
    frame_count: int = 24
    track_variant: str = "canonical"
    hiz_resolution: int = HIZ_BASE_RESOLUTION
    projection_error_budget_px: float = 1.5
    depth_error_budget: float = 0.02
    base_world_expand: float = 0.05
    depth_bias: float = 1.0e-5
    memory_budget_bytes: int | None = None
    vram_budget_bytes: int | None = None
    budget_scale: float = 1.0
    minimum_free_bytes: int = MINIMUM_FREE_BYTES
    enforce_disk: bool = True

    def __post_init__(self) -> None:
        TerrainConfig(self.size, self.seed)
        if int(self.frame_count) < 1:
            raise ValueError("frame_count must be positive")
        if int(self.hiz_resolution) < 8 or int(self.hiz_resolution) & (int(self.hiz_resolution) - 1):
            raise ValueError("hiz_resolution must be a power of two and at least 8")
        if float(self.projection_error_budget_px) <= 0:
            raise ValueError("projection_error_budget_px must be positive")
        if float(self.depth_error_budget) <= 0:
            raise ValueError("depth_error_budget must be positive")
        if float(self.base_world_expand) < 0:
            raise ValueError("base_world_expand must be non-negative")
        if float(self.budget_scale) <= 0:
            raise ValueError("budget_scale must be positive")
        validate_depth_bias(self.depth_bias)


@dataclass(frozen=True)
class PreparedAssets:
    heights: np.ndarray
    terrain_metadata: dict[str, Any]
    terrain_sha256: str
    quadtree: QuadtreeManifest
    terrain_raw_relative: str
    terrain_meta_relative: str
    quadtree_relative: str


@dataclass(frozen=True)
class RunResult:
    workspace: Workspace
    assets: PreparedAssets
    summary: dict[str, Any]
    report_path: Path
    run_manifest_path: Path
    report_sha256: str
    run_manifest_sha256: str


def prepare_assets(workspace: Workspace, settings: RunSettings) -> PreparedAssets:
    config = TerrainConfig(settings.size, settings.seed)
    raw_path = workspace.terrain_dir / f"terrain_{settings.size}_{int(settings.seed):016x}.u16"
    meta_path = workspace.manifest_dir / f"terrain_{settings.size}_{int(settings.seed):016x}.terrain.json"
    quad_path = manifest_path(workspace.root, settings.size, settings.seed)

    if raw_path.is_file() and meta_path.is_file():
        heights, metadata = load_terrain(workspace.root, config)
        terrain_sha = str(metadata["content_sha256"])
    else:
        generated = generate_heightfield(config)
        raw_path, meta_path, terrain_sha, _ = save_generated_terrain(workspace.root, generated)
        heights = generated.heights
        metadata = generated.metadata()
        metadata["raw_file"] = raw_path.name

    validate_height_range(heights)
    quadtree = build_quadtree(heights, settings.seed, terrain_sha)
    if quad_path.is_file():
        persisted = load_quadtree_manifest(workspace.root, settings.size, settings.seed)
        if persisted.get("content_sha256") != quadtree.content_sha256 or persisted.get("terrain_sha256") != terrain_sha:
            fail(
                ErrorCode.E_BLOCK_MISSING,
                "persisted quadtree manifest does not match regenerated deterministic manifest",
                expected_sha256=quadtree.content_sha256,
                actual_sha256=persisted.get("content_sha256"),
            )
    else:
        save_quadtree_manifest(workspace.root, quadtree)

    return PreparedAssets(
        heights=heights,
        terrain_metadata=metadata,
        terrain_sha256=terrain_sha,
        quadtree=quadtree,
        terrain_raw_relative=raw_path.relative_to(workspace.root).as_posix(),
        terrain_meta_relative=meta_path.relative_to(workspace.root).as_posix(),
        quadtree_relative=quad_path.relative_to(workspace.root).as_posix(),
    )


def default_budgets(manifest: QuadtreeManifest, scale: float = 1.0) -> StreamingBudgets:
    total_memory, total_vram = total_leaf_bytes(manifest)
    memory = max(512 * 1024, int(total_memory * 0.48 * scale))
    vram = max(384 * 1024, int(total_vram * 0.32 * scale))
    return StreamingBudgets(memory_bytes=memory, vram_bytes=vram, prefetch_radius=1)


def _parameter_stamp(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(payload, newline=False))


def _prefetch_candidates(
    manifest: QuadtreeManifest,
    visible_ids: tuple[str, ...],
    eye: tuple[float, float, float],
    radius: int,
) -> tuple[str, ...]:
    candidates: set[str] = set()
    for block_id in visible_ids:
        candidates.update(manifest.neighbors(block_id, radius=radius))
    candidates.difference_update(visible_ids)
    eye_array = np.asarray(eye, dtype=np.float64)

    def key(block_id: str) -> tuple[float, str]:
        node = manifest.get_block(block_id)
        center = np.array(
            [
                0.5 * (node.bounds.xmin + node.bounds.xmax),
                0.5 * (node.bounds.ymin + node.bounds.ymax),
                0.5 * (node.bounds.zmin + node.bounds.zmax),
            ],
            dtype=np.float64,
        )
        return float(np.sum((center - eye_array) ** 2)), block_id

    return tuple(sorted(candidates, key=key))


def _degradation(
    priority: int,
    action: str,
    reason: str,
    parameter_stamp: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "priority": int(priority),
        "action": action,
        "reason": reason,
        "parameter_stamp": parameter_stamp,
        "details": details,
    }


def _hash_tree(root: Path, exclude: set[str]) -> tuple[list[dict[str, Any]], str, int]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        entries.append({"path": relative, "bytes": size, "sha256": digest})
    tree_hash = sha256_bytes(canonical_bytes(entries, newline=False))
    return entries, tree_hash, total_bytes


def run_workbench(
    root: str | Path,
    settings: RunSettings,
) -> RunResult:
    workspace = Workspace.create(
        root,
        minimum_free_bytes=settings.minimum_free_bytes,
        enforce_disk=settings.enforce_disk,
    )
    assets = prepare_assets(workspace, settings)
    manifest = assets.quadtree
    budgets = default_budgets(manifest, settings.budget_scale)
    if settings.memory_budget_bytes is not None or settings.vram_budget_bytes is not None:
        memory_budget = settings.memory_budget_bytes
        vram_budget = settings.vram_budget_bytes
        if memory_budget is None:
            memory_budget = int(vram_budget * (default_budgets(manifest).memory_bytes / float(default_budgets(manifest).vram_bytes)))
        if vram_budget is None:
            vram_budget = int(memory_budget * (default_budgets(manifest).vram_bytes / float(default_budgets(manifest).memory_bytes)))
        budgets = StreamingBudgets(
            memory_bytes=int(memory_budget),
            vram_bytes=int(vram_budget),
            prefetch_radius=budgets.prefetch_radius,
        )

    trajectory = generate_trajectory(assets.heights, settings.frame_count, settings.track_variant)
    manager = StreamingManager(assets.heights, manifest, budgets)
    report_rows: list[dict[str, Any]] = []
    stem = f"terrain_{settings.size}_{int(settings.seed):016x}"
    report_path = workspace.report_dir / f"{stem}.frame_reports.jsonl"
    hiz_prefix_dir = workspace.hiz_dir / stem
    hiz_manifest_rows: list[dict[str, Any]] = []
    last_stream_stamp = ""
    total_loads = 0
    total_unloads = 0
    total_prefetch_skipped = 0
    degradation_counts: dict[str, int] = {"LOWER_HIZ": 0, "EXPAND_BBOX": 0, "REDUCE_RESIDENT": 0}

    for pose in trajectory:
        projections = {
            node.block_id: project_bounds(
                node.block_id or "",
                node.bounds.expanded(settings.base_world_expand),
                pose,
                settings.hiz_resolution,
                settings.hiz_resolution,
            )
            for node in manifest.leaves
            if node.block_id is not None
        }
        depth_render = render_depth_and_owner(
            assets.heights,
            manifest,
            pose,
            projections,
            resolution=settings.hiz_resolution,
        )
        pyramid = build_hiz(depth_render.depth)
        hiz_prefix = hiz_prefix_dir / f"frame_{pose.frame:04d}"
        hiz_raw, hiz_json, hiz_content_sha = save_hiz(
            hiz_prefix,
            pyramid,
            extra={
                "frame": pose.frame,
                "size": settings.size,
                "seed": int(settings.seed),
                "track_id": pose.track_id,
                "base_depth_sha256": depth_render.base_sha256,
            },
        )
        hiz_manifest_rows.append(
            {
                "frame": pose.frame,
                "content_sha256": hiz_content_sha,
                "raw": hiz_raw.relative_to(workspace.root).as_posix(),
                "metadata": hiz_json.relative_to(workspace.root).as_posix(),
                "levels": pyramid.metadata()["levels"],
            }
        )

        trigger_blocks = projection_trigger_blocks(projections, settings.projection_error_budget_px)
        policy = CullingPolicy(depth_bias=settings.depth_bias)
        degradations: list[dict[str, Any]] = []
        projection_errors = [
            result.projection_error_px
            for result in projections.values()
            if result.intersects_frustum
        ]
        max_projection_error = max(projection_errors) if projection_errors else 0.0
        if trigger_blocks:
            policy = replace(policy, hiz_level_bias=min(pyramid.max_level, 1))
            degradations.append(
                _degradation(
                    0,
                    "LOWER_HIZ",
                    "projection_error_budget_exceeded",
                    "",
                    trigger_blocks=list(trigger_blocks),
                    trigger_count=len(trigger_blocks),
                    max_projection_error_px=max_projection_error,
                    projection_budget_px=settings.projection_error_budget_px,
                    new_hiz_level_bias=policy.hiz_level_bias,
                )
            )
            extra_px = min(8, max(1, int(math.ceil(max(0.0, max_projection_error - settings.projection_error_budget_px)))))
            policy = replace(policy, extra_screen_expand_px=extra_px)
            degradations.append(
                _degradation(
                    1,
                    "EXPAND_BBOX",
                    "projection_error_budget_exceeded",
                    "",
                    trigger_blocks=list(trigger_blocks),
                    extra_screen_expand_px=extra_px,
                )
            )

        def stamp_for(current_policy: CullingPolicy) -> str:
            return _parameter_stamp(
                {
                    "schema_version": 1,
                    "size": settings.size,
                    "seed": int(settings.seed),
                    "frame": pose.frame,
                    "track_id": pose.track_id,
                    "terrain_sha256": assets.terrain_sha256,
                    "quadtree_sha256": manifest.content_sha256,
                    "hiz_content_sha256": hiz_content_sha,
                    "hiz_resolution": settings.hiz_resolution,
                    "projection_error_budget_px": settings.projection_error_budget_px,
                    "depth_error_budget": settings.depth_error_budget,
                    "base_world_expand": settings.base_world_expand,
                    "policy": current_policy.to_dict(),
                    "numeric_policy": NUMERIC_POLICY,
                    "camera": {
                        "eye": list(pose.eye),
                        "target": list(pose.target),
                        "projection_matrix": list(pose.projection_matrix),
                        "view_projection_matrix": list(pose.view_projection_matrix),
                    },
                }
            )

        parameter_stamp = stamp_for(policy)
        cull_result = cull_blocks(
            manifest,
            projections,
            depth_render,
            pyramid,
            policy,
            parameter_stamp,
            settings.base_world_expand,
        )

        if cull_result.max_error_bound > settings.depth_error_budget and policy.hiz_level_bias < min(4, pyramid.max_level):
            policy = replace(policy, hiz_level_bias=policy.hiz_level_bias + 1)
            degradations.append(
                _degradation(
                    0,
                    "LOWER_HIZ",
                    "depth_error_budget_exceeded",
                    "",
                    observed_max_error_bound=cull_result.max_error_bound,
                    depth_error_budget=settings.depth_error_budget,
                    new_hiz_level_bias=policy.hiz_level_bias,
                )
            )
            parameter_stamp = stamp_for(policy)
            cull_result = cull_blocks(
                manifest,
                projections,
                depth_render,
                pyramid,
                policy,
                parameter_stamp,
                settings.base_world_expand,
            )
            if cull_result.max_error_bound > settings.depth_error_budget and policy.extra_screen_expand_px < 8:
                policy = replace(policy, extra_screen_expand_px=min(8, policy.extra_screen_expand_px + 1))
                degradations.append(
                    _degradation(
                        1,
                        "EXPAND_BBOX",
                        "depth_error_budget_exceeded",
                        "",
                        observed_max_error_bound=cull_result.max_error_bound,
                        depth_error_budget=settings.depth_error_budget,
                        extra_screen_expand_px=policy.extra_screen_expand_px,
                    )
                )
                parameter_stamp = stamp_for(policy)
                cull_result = cull_blocks(
                    manifest,
                    projections,
                    depth_render,
                    pyramid,
                    policy,
                    parameter_stamp,
                    settings.base_world_expand,
                )

        for degradation in degradations:
            degradation["parameter_stamp"] = parameter_stamp

        prefetch = _prefetch_candidates(
            manifest,
            cull_result.visible_block_ids,
            pose.eye,
            budgets.prefetch_radius,
        )
        stream_stamp = _parameter_stamp(
            {
                "culling_parameter_stamp": parameter_stamp,
                "memory_budget_bytes": budgets.memory_bytes,
                "vram_budget_bytes": budgets.vram_bytes,
                "prefetch_radius": budgets.prefetch_radius,
                "prefetch_candidates": len(prefetch),
            }
        )
        last_stream_stamp = stream_stamp
        stream_result = manager.step(pose.frame, cull_result.visible_block_ids, prefetch, stream_stamp)
        if stream_result.prefetch_skipped:
            degradations.append(
                _degradation(
                    2,
                    "REDUCE_RESIDENT",
                    "memory_or_vram_budget_pressure",
                    stream_stamp,
                    skipped_prefetch_blocks=list(stream_result.prefetch_skipped),
                    skipped_count=len(stream_result.prefetch_skipped),
                    memory_used_bytes=stream_result.memory_used_bytes,
                    vram_used_bytes=stream_result.vram_used_bytes,
                )
            )
        for degradation in degradations:
            degradation_counts[degradation["action"]] = degradation_counts.get(degradation["action"], 0) + 1
        manager.audit()
        total_loads += len(stream_result.completed_loads)
        total_unloads += len(stream_result.unloads)
        total_prefetch_skipped += len(stream_result.prefetch_skipped)

        error_dist = error_distribution(cull_result.error_values)
        depth_exhausted = cull_result.max_error_bound > settings.depth_error_budget
        report = {
            "schema_version": 1,
            "size": settings.size,
            "seed": int(settings.seed),
            "track_id": pose.track_id,
            "frame": pose.frame,
            "terrain_sha256": assets.terrain_sha256,
            "quadtree_sha256": manifest.content_sha256,
            "parameter_stamp": parameter_stamp,
            "stream_parameter_stamp": stream_stamp,
            "numeric_policy": NUMERIC_POLICY,
            "camera": pose.to_report(),
            "policy": policy.to_dict(),
            "depth_map": {
                "width": depth_render.width,
                "height": depth_render.height,
                "dtype": "float32",
                "content_sha256": depth_render.base_sha256,
                "projected_samples": depth_render.projected_samples,
                "accepted_samples": depth_render.accepted_samples,
                "covered_pixels": int(np.count_nonzero(depth_render.owner >= 0)),
            },
            "hiz": {
                "content_sha256": hiz_content_sha,
                "raw_file": hiz_raw.relative_to(workspace.root).as_posix(),
                "metadata_file": hiz_json.relative_to(workspace.root).as_posix(),
                "levels": pyramid.metadata()["levels"],
            },
            "degradations": degradations,
            "degradation_count": len(degradations),
            "stats": cull_result.to_stats(),
            "error_distribution": error_dist,
            "error_budget": {
                "budget": settings.depth_error_budget,
                "max_error_bound": cull_result.max_error_bound,
                "mean_error_bound": cull_result.mean_error_bound,
                "remaining": settings.depth_error_budget - cull_result.max_error_bound,
                "exhausted": bool(depth_exhausted),
            },
            "projection_budget": {
                "budget_px": settings.projection_error_budget_px,
                "max_projection_error_px": max_projection_error,
                "trigger_blocks": list(trigger_blocks),
                "exhausted": bool(trigger_blocks),
            },
            "streaming": {
                **stream_result.to_report(),
                "resident": manager.resident_report(pose.frame),
                "memory_budget_bytes": budgets.memory_bytes,
                "vram_budget_bytes": budgets.vram_bytes,
            },
            "draw_digest": stream_result.draw_digest,
            "decisions": [decision.to_report() for decision in cull_result.decisions],
        }
        report_without_hash = dict(report)
        frame_hash = sha256_bytes(canonical_bytes(report_without_hash, newline=False))
        report["frame_report_sha256"] = frame_hash
        report_rows.append(report)

    pre_close_audit = manager.audit()
    closed_blocks = manager.close(settings.frame_count - 1, last_stream_stamp)
    post_close_audit = manager.audit()
    ledger_path, ledger_sha = manager.write_ledger(workspace.root, f"{stem}.residency_ledger.jsonl")
    history_path, history_sha = manager.write_history(workspace.root, f"{stem}.residency_history.json")
    audit_path = workspace.ledger_dir / f"{stem}.ledger_audit.json"
    audit_sha = write_json(
        audit_path,
        {
            "pre_close": pre_close_audit,
            "closed_blocks": list(closed_blocks),
            "post_close": post_close_audit,
            "ledger_sha256": ledger_sha,
            "history_sha256": history_sha,
        },
    )

    report_bytes = b"".join(canonical_line(row).encode("utf-8") for row in report_rows)
    report_path.write_bytes(report_bytes)
    report_sha = sha256_bytes(report_bytes)
    hiz_manifest_path = hiz_prefix_dir / "hiz_manifest.json"
    hiz_manifest_sha = write_json(
        hiz_manifest_path,
        {
            "size": settings.size,
            "seed": int(settings.seed),
            "resolution": settings.hiz_resolution,
            "frames": hiz_manifest_rows,
        },
    )

    visible_total = sum(row["stats"]["visible_blocks"] for row in report_rows)
    occluded_total = sum(row["stats"]["occlusion_culled_blocks"] for row in report_rows)
    culled_total = sum(row["stats"]["culled_blocks"] for row in report_rows)
    uncertain_total = sum(row["stats"]["uncertain_blocks"] for row in report_rows)
    summary = {
        "schema_version": 1,
        "size": settings.size,
        "seed": int(settings.seed),
        "track_id": settings.track_variant,
        "frames": settings.frame_count,
        "terrain_sha256": assets.terrain_sha256,
        "quadtree_sha256": manifest.content_sha256,
        "report_sha256": report_sha,
        "hiz_manifest_sha256": hiz_manifest_sha,
        "ledger_sha256": ledger_sha,
        "history_sha256": history_sha,
        "ledger_audit_sha256": audit_sha,
        "projection_budget_px": settings.projection_error_budget_px,
        "depth_error_budget": settings.depth_error_budget,
        "streaming_budgets": budgets.to_dict(),
        "degradation_counts": degradation_counts,
        "total_visible_block_frames": visible_total,
        "total_occlusion_culled_block_frames": occluded_total,
        "total_culled_block_frames": culled_total,
        "total_uncertain_block_frames": uncertain_total,
        "total_loads": total_loads,
        "total_unloads": total_unloads,
        "total_prefetch_skipped": total_prefetch_skipped,
        "max_memory_utilization": max(row["streaming"]["memory_utilization"] for row in report_rows),
        "max_vram_utilization": max(row["streaming"]["vram_utilization"] for row in report_rows),
        "draw_digest_sequence_sha256": sha256_bytes(
            canonical_bytes([row["draw_digest"] for row in report_rows], newline=False)
        ),
        "workspace_layout": {
            "terrain": "terrain/",
            "manifests": "manifests/",
            "hiz": "hiz/",
            "reports": "reports/",
            "ledgers": "ledgers/",
            "metadata": "meta/",
        },
    }
    summary_path = workspace.meta_dir / f"{stem}.run_summary.json"
    summary_sha = write_json(summary_path, summary)

    run_manifest_path = workspace.meta_dir / "run_manifest.json"
    entries, tree_hash, total_bytes = _hash_tree(
        workspace.root,
        exclude={"meta/run_manifest.json"},
    )
    run_manifest = {
        "schema_version": 1,
        "size": settings.size,
        "seed": int(settings.seed),
        "track_id": settings.track_variant,
        "summary_sha256": summary_sha,
        "tree_sha256": tree_hash,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "files": entries,
    }
    run_manifest_sha = write_json(run_manifest_path, run_manifest)
    return RunResult(
        workspace=workspace,
        assets=assets,
        summary=summary,
        report_path=report_path,
        run_manifest_path=run_manifest_path,
        report_sha256=report_sha,
        run_manifest_sha256=run_manifest_sha,
    )
