from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cache import (
    BlockRecord,
    CacheIssue,
    atomic_replace_bundle,
    atomic_write_npy,
    block_path,
    clear_interrupt_marker,
    write_interrupt_marker,
    write_block_record,
    write_json_temp,
    read_block_record,
    read_manifest,
)
from .canonical import array_digest, sha256_json
from .errors import PipelineError
from .geometry import (
    BlockPlan,
    CameraRays,
    changed_support_voxels_in_bounds,
    changed_voxels_in_bounds,
    count_rays_hitting_changed,
    build_camera_rays,
    make_block_plans,
)
from .renderer import render_block
from .scheduler import execute_blocks
from .spec import TaskSpec, load_task, task_render_key
from .volume import (
    apply_edits,
    apply_normalized_edits,
    load_volume,
    volume_content_digest,
)

CAMERA_FIELDS = ("position", "target", "up", "fov_y_deg", "aspect")
RENDER_FIELDS = ("sampling_step_mm", "background_gray")


@dataclass(frozen=True)
class RenderResult:
    gray: np.ndarray
    depth: np.ndarray
    report: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class BlockWork:
    record: BlockRecord
    sampled_ray_count: int
    digest_retries: int


def _previous_job_usable(manifest: dict[str, Any] | None) -> bool:
    if not manifest or manifest.get("completed") is not True:
        return False
    job = manifest.get("job")
    if not isinstance(job, dict):
        return False
    required = (
        "base_digest",
        "content_digest",
        "volume",
        "edits",
        "transfer_function",
        "camera",
        "render",
        "version_id",
    )
    return all(key in job for key in required)


def _camera_fields_changed(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for field in CAMERA_FIELDS:
        if previous.get(field) != current.get(field):
            changed.append(field)
    return changed


def _render_fields_changed(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for field in RENDER_FIELDS:
        if previous.get(field) != current.get(field):
            changed.append(field)
    for field in ("dims", "spacing_mm", "scalar_domain"):
        if previous.get(field) != current.get(field):
            changed.append(field)
    return changed


def _layout_fields_changed(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for field in ("width", "height", "block_width", "block_height"):
        if previous.get(field) != current.get(field):
            changed.append(field)
    return changed


def _intersects_plan(edit: Any, plan: BlockPlan) -> bool:
    if plan.voxel_bounds_min is None or plan.voxel_bounds_max is None:
        return False
    start = edit.start
    stop = edit.stop_exclusive
    return all(
        start[axis] < plan.voxel_bounds_max[axis] + 1
        and stop[axis] > plan.voxel_bounds_min[axis]
        for axis in range(3)
    )


def _block_edit_digest(task: TaskSpec, plan: BlockPlan) -> str:
    dependencies = [edit.normalized() for edit in task.edits if _intersects_plan(edit, plan)]
    return sha256_json(dependencies)


def _reconstruct_previous_volume(task: TaskSpec, base: np.ndarray, job: dict[str, Any]) -> np.ndarray | None:
    if job.get("base_digest") is None:
        return None
    edits = job.get("edits")
    if not isinstance(edits, list):
        return None
    try:
        previous = apply_normalized_edits(base, edits)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    expected = job.get("content_digest")
    if not isinstance(expected, str):
        return None
    if volume_content_digest(previous, task.volume) != expected:
        return None
    return previous


def _write_interrupt(
    task: TaskSpec,
    run_id: str,
    version_id: str,
    completed_indices: list[int],
    pending_indices: list[int],
) -> None:
    write_interrupt_marker(
        task.state_dir,
        run_id=run_id,
        version_id=version_id,
        completed_indices=sorted(set(completed_indices)),
        pending_indices=sorted(set(pending_indices)),
    )


def _raise_interrupted(
    task: TaskSpec,
    run_id: str,
    version_id: str,
    completed_indices: list[int],
    pending_indices: list[int],
) -> "NoReturn":  # type: ignore[name-defined]
    _write_interrupt(task, run_id, version_id, completed_indices, pending_indices)
    raise PipelineError(
        "E_INTERRUPTED",
        "$",
        "render interrupted at a block boundary",
        details={"exit_code": 130, "run_id": run_id},
    )


def _make_report(
    *,
    task: TaskSpec,
    run_id: str,
    version_id: str,
    content_digest: str,
    base_digest: str,
    tf_digest: str,
    camera_digest: str,
    render_digest: str,
    rays: CameraRays,
    plans: list[BlockPlan],
    final_records: dict[int, BlockRecord],
    change_details: dict[str, Any],
    block_rules: dict[int, set[str]],
    statuses: dict[int, str],
    block_codes: dict[int, set[str]],
    volume_changed_in_bounds: dict[int, int],
    volume_changed_in_extended_bounds: dict[int, int],
    rays_hit_changed: dict[int, int],
    sample_buffer_peak: int,
    invalid_by_change_criteria: int,
    recovered_persisted: int,
    cache_missing_without_change_rule: int,
    missing_before_render: int,
    gray_digest: str,
    depth_digest: str,
) -> dict[str, Any]:
    block_reports: list[dict[str, Any]] = []
    reused_count = 0
    rerendered_count = 0
    code_counts: dict[str, int] = {}
    for plan in plans:
        record = final_records[plan.index]
        status = statuses[plan.index]
        categories = set(block_rules[plan.index])
        if change_details["transfer_function_changed"] and record.has_nonzero_samples:
            categories.add("transfer_function")
        if status in {"rerendered", "digest_retry"} and not categories:
            categories.add("cache_missing")
        if status in {"reused", "reused_recovered"}:
            reused_count += 1
        else:
            rerendered_count += 1
        codes = sorted(block_codes.get(plan.index, set()))
        if codes:
            for code in codes:
                code_counts[code] = code_counts.get(code, 0) + 1
        else:
            code_counts["E_NONE"] = code_counts.get("E_NONE", 0) + 1
        block_reports.append(
            {
                "index": plan.index,
                "row": plan.row,
                "col": plan.col,
                "pixel_bounds": plan.pixel_bounds,
                "pixel_count": plan.pixel_count,
                "voxel_index_bounds": plan.voxel_bounds,
                "ray_hit_count": plan.ray_hit_count,
                "has_nonzero_samples": record.has_nonzero_samples,
                "invalid": bool(categories),
                "invalid_categories": sorted(categories),
                "volume_edit": {
                    "changed_voxels_in_bounds": int(volume_changed_in_bounds.get(plan.index, 0)),
                    "changed_voxels_in_extended_bounds": int(
                        volume_changed_in_extended_bounds.get(plan.index, 0)
                    ),
                    "rays_hitting_changed_voxels": int(rays_hit_changed.get(plan.index, 0)),
                    "affected": bool(rays_hit_changed.get(plan.index, 0) > 0),
                },
                "cache": {
                    "status": status,
                    "reused": status in {"reused", "reused_recovered"},
                    "rerendered": status not in {"reused", "reused_recovered"},
                    "recovered_persisted": status == "reused_recovered",
                    "digest_retry": status == "digest_retry" or "E_DIGEST" in codes,
                    "current_version_id": version_id,
                    "cached_version_id": record.version_id,
                    "edit_digest": record.edit_digest,
                    "current_edit_digest": _block_edit_digest(task, plan),
                    "payload_digest": record.meta["payload_digest"],
                    "pixel_count_verified": int(record.meta["pixel_count"]),
                },
                "business_codes": codes,
            }
        )

    return {
        "schema_version": 1,
        "ok": True,
        "run_id": run_id,
        "task_path": str(task.task_path),
        "render": {
            "output_size": [task.width, task.height],
            "block_size": [task.block_width, task.block_height],
            "block_grid": [task.block_cols, task.block_rows],
            "total_blocks": len(plans),
            "sampling_step_mm": task.sampling_step_mm,
            "background_gray": task.background_gray,
            "background_depth_mm": rays.max_depth_mm,
            "concurrency": task.concurrency,
            "memory_budget_bytes": task.memory_budget_bytes,
            "peak_concurrent_sample_bytes": sample_buffer_peak,
        },
        "version": {
            "volume_content_digest": content_digest,
            "volume_base_digest": base_digest,
            "transfer_function_digest": tf_digest,
            "camera_digest": camera_digest,
            "render_settings_digest": render_digest,
            "version_id": version_id,
            "edits_digest": task.edits_digest,
        },
        "changes": change_details,
        "summary": {
            "reused_blocks": reused_count,
            "rerendered_blocks": rerendered_count,
            "invalid_blocks_by_change_criteria": invalid_by_change_criteria,
            "persisted_recovery_blocks": recovered_persisted,
            "cache_missing_without_change_rule": cache_missing_without_change_rule,
            "missing_before_render": missing_before_render,
            "business_code_distribution": dict(sorted(code_counts.items())),
        },
        "blocks": block_reports,
        "outputs": {
            "gray": {
                "path": str(task.gray_path),
                "shape": [task.height, task.width],
                "dtype": "<f8",
                "digest": gray_digest,
            },
            "depth": {
                "path": str(task.depth_path),
                "shape": [task.height, task.width],
                "dtype": "<f8",
                "digest": depth_digest,
            },
            "report": {"path": str(task.report_path)},
        },
        "criteria": {
            "volume_edit": (
                "A block is invalid only when at least one pixel-center ray traverses a "
                "voxel cell whose edited scalar actually differs from the previous job."
            ),
            "transfer_function": (
                "A transfer-function change invalidates every block that samples at least "
                "one non-zero scalar; all-zero-scalar blocks remain reusable."
            ),
            "camera": "Any camera position/target/up/FOV/aspect change invalidates all blocks.",
            "union": "Per-block categories are the union of all applicable change classes.",
            "depth": (
                "Depth is the first contributing (non-zero scalar and opacity>0) sample in "
                "millimeters. Missing/empty rays use the maximum camera-to-volume-corner distance."
            ),
        },
    }


def run_task(
    task: TaskSpec,
    *,
    force_full: bool = False,
    interrupt_event: threading.Event | None = None,
) -> RenderResult:
    interrupt_event = interrupt_event or threading.Event()
    run_id = uuid.uuid4().hex

    # All validation that can fail without producing outputs happens before the
    # state or output directories are created.
    volume_data = load_volume(task.volume)
    current_volume = apply_edits(volume_data.array, task.edits)
    content_digest = volume_content_digest(current_volume, task.volume)
    rays = build_camera_rays(task.camera, task.width, task.height, task.volume)
    plans = make_block_plans(task, rays)
    max_block_bytes = max((plan.sample_buffer_bytes for plan in plans), default=0)
    if max_block_bytes > task.memory_budget_bytes:
        raise PipelineError(
            "E_MEMORY",
            "memory_budget_bytes",
            "one block's deterministic sampling buffer exceeds the memory budget",
            {
                "required_bytes": max_block_bytes,
                "memory_budget_bytes": task.memory_budget_bytes,
            },
        )

    render_key = task_render_key(task)
    render_digest = sha256_json(render_key)
    tf_digest = task.transfer_function.digest
    camera_digest = task.camera.digest
    version_basis = {
        "volume_content_digest": content_digest,
        "transfer_function_digest": tf_digest,
        "camera_digest": camera_digest,
        "render_settings_digest": render_digest,
    }
    version_id = sha256_json(version_basis)

    if interrupt_event.is_set():
        _raise_interrupted(task, run_id, version_id, [], [plan.index for plan in plans])

    previous_manifest = None if force_full else read_manifest(task.state_dir)
    previous_usable = _previous_job_usable(previous_manifest)
    previous_job = previous_manifest["job"] if previous_usable else None

    change_details: dict[str, Any] = {
        "initial_render": not previous_usable,
        "full_render_requested": force_full,
        "camera_changed": False,
        "camera_fields_changed": [],
        "transfer_function_changed": False,
        "layout_changed": False,
        "layout_fields_changed": [],
        "render_settings_changed": False,
        "render_settings_fields_changed": [],
        "volume_base_changed": False,
        "volume_delta_available": False,
        "volume_changed_voxels": 0,
    }
    changed_mask: np.ndarray | None = None

    if previous_job is not None:
        previous_camera = previous_job.get("camera", {})
        camera_fields = _camera_fields_changed(previous_camera, task.camera.normalized())
        change_details["camera_fields_changed"] = camera_fields
        change_details["camera_changed"] = bool(camera_fields)

        previous_tf = previous_job.get("transfer_function", {})
        previous_tf_digest = previous_tf.get("digest")
        if isinstance(previous_tf_digest, str):
            change_details["transfer_function_changed"] = previous_tf_digest != tf_digest
        else:
            change_details["transfer_function_changed"] = (
                previous_tf.get("points") != task.transfer_function.normalized()
                or previous_tf.get("options", {}) != task.transfer_function.options_dict
            )

        current_render = dict(render_key)
        current_render.update(
            {
                "dims": list(task.volume.dims),
                "spacing_mm": list(task.volume.spacing_mm),
                "scalar_domain": [task.volume.scalar_min, task.volume.scalar_max],
            }
        )
        previous_render = previous_job.get("render", {})
        render_fields = _render_fields_changed(previous_render, current_render)
        change_details["render_settings_fields_changed"] = render_fields
        change_details["render_settings_changed"] = bool(render_fields)
        layout_fields = _layout_fields_changed(previous_render, current_render)
        change_details["layout_fields_changed"] = layout_fields
        change_details["layout_changed"] = bool(layout_fields)

        if previous_job.get("base_digest") != volume_data.base_digest:
            change_details["volume_base_changed"] = True
        else:
            previous_volume = _reconstruct_previous_volume(task, volume_data.array, previous_job)
            if previous_volume is None:
                change_details["volume_base_changed"] = True
            else:
                changed_mask = current_volume != previous_volume
                change_details["volume_delta_available"] = True
                change_details["volume_changed_voxels"] = int(np.count_nonzero(changed_mask))

    if previous_usable and previous_job is not None and previous_job.get("content_digest") == content_digest:
        change_details["volume_changed_voxels"] = 0
        changed_mask = None

    base_rules: dict[int, set[str]] = {plan.index: set() for plan in plans}
    if change_details["initial_render"]:
        for plan in plans:
            base_rules[plan.index].add("initial_render")
    if force_full:
        for plan in plans:
            base_rules[plan.index].add("full_render")
    if change_details["camera_changed"]:
        for plan in plans:
            base_rules[plan.index].add("camera")
    if change_details["layout_changed"]:
        for plan in plans:
            base_rules[plan.index].add("layout")
    if change_details["render_settings_changed"]:
        for plan in plans:
            base_rules[plan.index].add("render_settings")
    if change_details["volume_base_changed"]:
        for plan in plans:
            base_rules[plan.index].add("volume_base_changed")

    volume_changed_in_bounds: dict[int, int] = {plan.index: 0 for plan in plans}
    volume_changed_in_extended_bounds: dict[int, int] = {plan.index: 0 for plan in plans}
    rays_hit_changed: dict[int, int] = {plan.index: 0 for plan in plans}
    if changed_mask is not None and bool(np.any(changed_mask)):
        for plan in plans:
            count_in_bounds = changed_voxels_in_bounds(plan, changed_mask)
            volume_changed_in_bounds[plan.index] = count_in_bounds
            volume_changed_in_extended_bounds[plan.index] = changed_support_voxels_in_bounds(
                plan, changed_mask, task.volume.dims
            )
            if volume_changed_in_extended_bounds[plan.index]:
                hit_count = count_rays_hitting_changed(task, rays, plan, changed_mask)
                rays_hit_changed[plan.index] = hit_count
                if hit_count:
                    base_rules[plan.index].add("volume_edit")

    records: dict[int, BlockRecord] = {}
    record_issues: dict[int, CacheIssue] = {}
    if not force_full:
        for plan in plans:
            record, issue = read_block_record(block_path(task.state_dir, plan), plan)
            if record is not None:
                records[plan.index] = record
            if issue is not None:
                record_issues[plan.index] = issue

    # A TF change can reuse blocks that provably sample only zero scalars. The
    # prior record's has_nonzero flag is exact whenever prior volume geometry on
    # this block did not change.
    if change_details["transfer_function_changed"]:
        for plan in plans:
            if "volume_edit" in base_rules[plan.index]:
                continue
            if any(
                rule in base_rules[plan.index]
                for rule in ("camera", "layout", "render_settings", "volume_base_changed", "initial_render", "full_render")
            ):
                continue
            record = records.get(plan.index)
            if record is not None and record.has_nonzero_samples:
                base_rules[plan.index].add("transfer_function")

    reuse: dict[int, bool] = {}
    recovered: set[int] = set()
    cache_missing_only: set[int] = set()
    previous_version = previous_job.get("version_id") if previous_job else None
    for plan in plans:
        record = records.get(plan.index)
        rules = base_rules[plan.index]
        if force_full:
            reuse[plan.index] = False
            continue
        if record is not None:
            if record.version_id == version_id:
                reuse[plan.index] = True
                if rules:
                    recovered.add(plan.index)
            elif previous_usable and not rules:
                reuse[plan.index] = True
            else:
                reuse[plan.index] = False
        else:
            reuse[plan.index] = False
            if not rules:
                cache_missing_only.add(plan.index)

    required = [plan for plan in plans if not reuse[plan.index]]
    invalid_by_change_criteria = sum(1 for plan in plans if base_rules[plan.index])
    recovered_persisted = len(recovered)
    missing_before_render = len(required)
    expected_missing = invalid_by_change_criteria - recovered_persisted + len(cache_missing_only)
    if expected_missing != missing_before_render:
        raise PipelineError(
            "E_STATE",
            "state_dir",
            "cache recovery accounting is inconsistent",
            {
                "invalid_by_criteria": invalid_by_change_criteria,
                "recovered_persisted": recovered_persisted,
                "cache_missing_only": len(cache_missing_only),
                "missing_before_render": missing_before_render,
            },
        )

    completed_indices = {plan.index for plan in plans if reuse[plan.index]}
    block_codes: dict[int, set[str]] = {}
    for index in record_issues:
        block_codes.setdefault(index, set()).add("E_DIGEST")

    def worker(plan: BlockPlan) -> BlockWork:
        edit_digest = _block_edit_digest(task, plan)
        digest_retries = 0
        for attempt in range(2):
            rendered = render_block(task, current_volume, task.transfer_function, rays, plan)
            record = write_block_record(
                task.state_dir,
                plan,
                version_id=version_id,
                edit_digest=edit_digest,
                has_nonzero_samples=rendered.has_nonzero_samples,
                gray=rendered.gray,
                depth=rendered.depth,
                run_id=run_id,
            )
            verified, issue = read_block_record(record.path, plan)
            if issue is None and verified is not None and verified.version_id == version_id:
                return BlockWork(
                    record=verified,
                    sampled_ray_count=rendered.sampled_ray_count,
                    digest_retries=digest_retries,
                )
            digest_retries += 1
            if attempt == 0:
                block_codes.setdefault(plan.index, set()).add("E_DIGEST")
        raise PipelineError(
            "E_DIGEST",
            "state_dir",
            "block digest verification failed after a fallback rerender",
            {"block_index": plan.index, "block_path": str(block_path(task.state_dir, plan))},
        )

    execution = execute_blocks(
        required,
        worker,
        concurrency=task.concurrency,
        memory_budget_bytes=task.memory_budget_bytes,
        interrupt_event=interrupt_event,
    )
    if execution.errors:
        index, error = execution.first_error() or (0, RuntimeError("unknown render error"))
        if isinstance(error, PipelineError):
            raise error
        raise PipelineError(
            "E_RENDER",
            f"blocks[{index}]",
            f"block render failed: {error}",
            {"block_index": index},
        )
    if execution.interrupted:
        completed_indices.update(execution.results.keys())
        _raise_interrupted(
            task,
            run_id,
            version_id,
            sorted(completed_indices),
            execution.not_started_indices,
        )

    result_work: dict[int, BlockWork] = execution.results
    completed_indices.update(result_work.keys())

    # Final block-level digest and declared-count verification, including blocks
    # that came from the cache. A bad record is rerendered once.
    final_records: dict[int, BlockRecord] = {}
    statuses: dict[int, str] = {}
    for plan in plans:
        if interrupt_event.is_set():
            _raise_interrupted(
                task,
                run_id,
                version_id,
                sorted(completed_indices),
                [item.index for item in plans if item.index not in completed_indices],
            )
        record, issue = read_block_record(block_path(task.state_dir, plan), plan)
        if issue is not None or record is None:
            block_codes.setdefault(plan.index, set()).add("E_DIGEST")
            try:
                work = worker(plan)
            except PipelineError as exc:
                if exc.code == "E_DIGEST":
                    raise
                raise
            record = work.record
            if work.digest_retries:
                block_codes.setdefault(plan.index, set()).add("E_DIGEST")
            statuses[plan.index] = "digest_retry"
            completed_indices.add(plan.index)
        elif reuse[plan.index]:
            statuses[plan.index] = "reused_recovered" if plan.index in recovered else "reused"
        elif plan.index in result_work and result_work[plan.index].digest_retries:
            statuses[plan.index] = "digest_retry"
            block_codes.setdefault(plan.index, set()).add("E_DIGEST")
        else:
            statuses[plan.index] = "rerendered"
        final_records[plan.index] = record

    if interrupt_event.is_set():
        _raise_interrupted(
            task,
            run_id,
            version_id,
            sorted(completed_indices),
            [item.index for item in plans if item.index not in completed_indices],
        )

    # Assemble strictly in row-major block order.
    gray_rows: list[np.ndarray] = []
    depth_rows: list[np.ndarray] = []
    for row in range(task.block_rows):
        gray_cols: list[np.ndarray] = []
        depth_cols: list[np.ndarray] = []
        for col in range(task.block_cols):
            index = row * task.block_cols + col
            record = final_records[index]
            if record.gray.shape[1] != task.block_width or record.gray.shape[0] != task.block_height:
                raise PipelineError(
                    "E_DIGEST",
                    "state_dir",
                    "block dimensions do not match the output grid",
                    {"block_index": index},
                )
            gray_cols.append(record.gray)
            depth_cols.append(record.depth)
        gray_rows.append(np.concatenate(gray_cols, axis=1))
        depth_rows.append(np.concatenate(depth_cols, axis=1))
    gray = np.ascontiguousarray(np.concatenate(gray_rows, axis=0), dtype=np.float64)
    depth = np.ascontiguousarray(np.concatenate(depth_rows, axis=0), dtype=np.float64)
    if gray.shape != (task.height, task.width) or depth.shape != (task.height, task.width):
        raise PipelineError("E_DIGEST", "output", "assembled image dimensions are inconsistent")
    if not np.all(np.isfinite(gray)) or not np.all(np.isfinite(depth)):
        raise PipelineError("E_DIGEST", "output", "assembled output contains non-finite values")

    gray_digest = array_digest(gray)
    depth_digest = array_digest(depth)
    report = _make_report(
        task=task,
        run_id=run_id,
        version_id=version_id,
        content_digest=content_digest,
        base_digest=volume_data.base_digest,
        tf_digest=tf_digest,
        camera_digest=camera_digest,
        render_digest=render_digest,
        rays=rays,
        plans=plans,
        final_records=final_records,
        change_details=change_details,
        block_rules=base_rules,
        statuses=statuses,
        block_codes=block_codes,
        volume_changed_in_bounds=volume_changed_in_bounds,
        volume_changed_in_extended_bounds=volume_changed_in_extended_bounds,
        rays_hit_changed=rays_hit_changed,
        sample_buffer_peak=execution.peak_sample_bytes,
        invalid_by_change_criteria=invalid_by_change_criteria,
        recovered_persisted=recovered_persisted,
        cache_missing_without_change_rule=len(cache_missing_only),
        missing_before_render=missing_before_render,
        gray_digest=gray_digest,
        depth_digest=depth_digest,
    )

    block_manifest = []
    for plan in plans:
        record = final_records[plan.index]
        block_manifest.append(
            {
                "index": plan.index,
                "row": plan.row,
                "col": plan.col,
                "version_id": record.version_id,
                "edit_digest": record.edit_digest,
                "payload_digest": record.meta["payload_digest"],
                "pixel_count": record.meta["pixel_count"],
                "has_nonzero_samples": record.has_nonzero_samples,
            }
        )
    manifest = {
        "schema_version": 1,
        "completed": True,
        "run_id": run_id,
        "job": {
            "task_path": str(task.task_path),
            "base_digest": volume_data.base_digest,
            "content_digest": content_digest,
            "volume": {
                "dims": list(task.volume.dims),
                "spacing_mm": list(task.volume.spacing_mm),
                "scalar_domain": [task.volume.scalar_min, task.volume.scalar_max],
            },
            "edits": [edit.normalized() for edit in task.edits],
            "edits_digest": task.edits_digest,
            "transfer_function": {
                "points": task.transfer_function.normalized(),
                "options": task.transfer_function.options_dict,
                "digest": tf_digest,
            },
            "camera": {**task.camera.normalized(), "digest": camera_digest},
            "render": {
                **render_key,
                "dims": list(task.volume.dims),
                "spacing_mm": list(task.volume.spacing_mm),
                "scalar_domain": [task.volume.scalar_min, task.volume.scalar_max],
                "block_cols": task.block_cols,
                "block_rows": task.block_rows,
            },
            "render_digest": render_digest,
            "version_id": version_id,
        },
        "blocks": block_manifest,
        "outputs": {
            "gray": {"path": str(task.gray_path), "digest": gray_digest},
            "depth": {"path": str(task.depth_path), "digest": depth_digest},
            "report": {"path": str(task.report_path)},
        },
    }

    temporary_gray = atomic_write_npy(task.gray_path, gray)
    temporary_depth = atomic_write_npy(task.depth_path, depth)
    temporary_report: Path | None = None
    temporary_manifest: Path | None = None
    try:
        temporary_report = write_json_temp(task.report_path, report)
        temporary_manifest = write_json_temp(task.state_dir / "manifest.json", manifest)
        targets = {
            task.gray_path: temporary_gray,
            task.depth_path: temporary_depth,
            task.report_path: temporary_report,
            task.state_dir / "manifest.json": temporary_manifest,
        }
        atomic_replace_bundle(targets)
        temporary_gray = temporary_depth = temporary_report = temporary_manifest = None  # type: ignore[assignment]
    finally:
        for temporary in (temporary_gray, temporary_depth, temporary_report, temporary_manifest):
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass
    clear_interrupt_marker(task.state_dir)
    return RenderResult(gray=gray, depth=depth, report=report, manifest=manifest)


def render_task(
    task_path: str | Path,
    *,
    state_dir_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
    force_full: bool = False,
    interrupt_event: threading.Event | None = None,
) -> RenderResult:
    task = load_task(
        task_path,
        state_dir_override=state_dir_override,
        output_dir_override=output_dir_override,
    )
    return run_task(task, force_full=force_full, interrupt_event=interrupt_event)
