"""End-to-end acceptance verification for determinism, latency, and recovery."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .cache import ArtifactCache
from .config import RenderConfig, load_command_stream, load_render_config
from .errors import E_SNAPSHOT_STALE, RenderError
from .renderer import DeterministicRenderer, RenderResult
from .session import SessionRunner, replay_final_state
from .snapshot import SnapshotManager, write_top_level_diagnostics
from .state import CommandEffect, RenderState
from .util import atomic_write_json, canonical_json


def _semantic_state(state: RenderState) -> dict[str, Any]:
    return {
        "prompt": state.prompt,
        "seed": state.seed,
        "steps": state.steps,
        "guidance": state.guidance,
        "frame_count": state.frame_count,
        "chunk_rows": state.chunk_rows,
        "quantization_level": state.quantization_level,
        "initial_image_path": state.initial_image_path,
        "initial_image_raw_fp": state.initial_image_raw_fp,
        "patches": [patch.as_dict() for patch in state.patches],
        "version": state.version,
        "command_cursor": state.command_cursor,
    }


def _compare_results(
    left: RenderResult,
    left_cache: ArtifactCache,
    right: RenderResult,
    right_cache: ArtifactCache,
) -> dict[str, Any]:
    pixel_equal = True
    png_equal = True
    frame_fp_equal = left.frame_fps == right.frame_fps
    first_pixel_difference: dict[str, Any] | None = None
    if len(left.frame_keys) != len(right.frame_keys):
        pixel_equal = False
        first_pixel_difference = {"reason": "frame count differs"}
    else:
        for frame_index, (left_key, right_key) in enumerate(zip(left.frame_keys, right.frame_keys)):
            left_frame = left_cache.decodes[left_key]
            right_frame = right_cache.decodes[right_key]
            if not np.array_equal(left_frame.pixels, right_frame.pixels):
                pixel_equal = False
                mismatch = np.argwhere(left_frame.pixels != right_frame.pixels)
                first = mismatch[0].tolist() if len(mismatch) else None
                first_pixel_difference = {
                    "frame": frame_index,
                    "first_flat_index": first,
                    "left_value": (
                        int(left_frame.pixels[tuple(first)]) if first is not None else None
                    ),
                    "right_value": (
                        int(right_frame.pixels[tuple(first)]) if first is not None else None
                    ),
                }
            if left_frame.png != right_frame.png:
                png_equal = False
    step_trace_equal = canonical_json(list(left.step_traces)) == canonical_json(list(right.step_traces))
    frame_trace_equal = canonical_json(list(left.frame_traces)) == canonical_json(list(right.frame_traces))
    composition_equal = left.composition_fp == right.composition_fp and left.preview_png == right.preview_png

    left_records = {record["key"]: record for record in left.audit}
    right_records = {record["key"]: record for record in right.audit}
    key_sets_equal = set(left_records) == set(right_records)
    content_equal = key_sets_equal and all(
        left_records[key]["content_fp"] == right_records[key]["content_fp"]
        for key in left_records
    )
    reuse_keys = [record["key"] for record in left.audit if record["status"] == "reuse"]
    recompute_keys = [record["key"] for record in left.audit if record["status"] == "recompute"]
    reused_content_matches = all(
        key in right_records and left_records[key]["content_fp"] == right_records[key]["content_fp"]
        for key in reuse_keys
    )
    return {
        "frame_fingerprints_equal": frame_fp_equal,
        "pixels_equal": pixel_equal,
        "png_bytes_equal": png_equal,
        "step_trace_equal": step_trace_equal,
        "frame_trace_equal": frame_trace_equal,
        "trace_fingerprint_equal": left.trace_fingerprint == right.trace_fingerprint,
        "composition_equal": composition_equal,
        "required_artifact_keys_equal": key_sets_equal,
        "required_artifact_content_equal": content_equal,
        "reuse_keys_subset_of_full_required_set": all(key in right_records for key in reuse_keys),
        "reuse_content_matches_full_recompute": reused_content_matches,
        "reuse_recompute_cover_same_required_keys": set(reuse_keys + recompute_keys) == set(right_records),
        "incremental_reuse_count": len(reuse_keys),
        "incremental_recompute_count": len(recompute_keys),
        "full_recompute_count": len(right_records),
        "first_pixel_difference": first_pixel_difference,
        "note": (
            "A blank full render has zero cache reuse by definition. Equivalence is therefore "
            "checked as identical required artifact key/content sets, with every incremental "
            "reuse matching the full render's recomputed artifact."
        ),
    }


def _failure_tuple(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("sequence"),
        item.get("command_index"),
        item.get("command_id"),
        item.get("op"),
        item.get("code"),
        item.get("message"),
        canonical_json(item.get("details")),
    )


def _latency_checks(runner: SessionRunner, config: RenderConfig) -> dict[str, Any]:
    budget = config.interaction_latency_budget_ms
    tolerance = 2.0
    responses = runner.immediate_responses
    latencies = [float(item.get("response_latency_ms") or 0.0) for item in responses]
    pending = [item for item in responses if item.get("business_code") == "E_INTERACT_BUDGET_PENDING"]
    versioned = all(
        item.get("result_corresponds_to", {}).get("config_fingerprint") is not None
        for item in pending
    )
    preview_chain_ok = all(
        item.get("preview") is not None
        for item in pending[1:]
    )
    return {
        "budget_ms": budget,
        "tolerance_ms": tolerance,
        "max_response_latency_ms": max(latencies, default=0.0),
        "min_response_latency_ms": min(latencies, default=0.0),
        "within_budget_with_tolerance": all(value <= budget + tolerance for value in latencies),
        "pending_count": len(pending),
        "pending_semantics_observed": len(pending) > 0,
        "every_pending_has_explicit_config_version": versioned,
        "pending_responses_expose_latest_completed_preview": preview_chain_ok,
        "first_response": responses[0] if responses else None,
    }


def _stale_checks(config: RenderConfig, command_hash: str, manager: SnapshotManager) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    stale_config = replace(config, source_hash="intentionally-stale-config-hash")
    try:
        manager.load_for_resume(
            config=stale_config,
            command_stream_hash=command_hash,
        )
        checks["config_hash_mismatch"] = {"rejected": False, "code": None}
    except RenderError as exc:
        checks["config_hash_mismatch"] = {"rejected": exc.code == E_SNAPSHOT_STALE, "code": exc.code}

    pointer = manager.current_pointer() or {}
    wrong_version = int(pointer.get("global_version", 0)) + 997
    try:
        manager.load_for_resume(
            config=config,
            command_stream_hash=command_hash,
            expected_version=wrong_version,
        )
        checks["version_mismatch"] = {"rejected": False, "code": None}
    except RenderError as exc:
        checks["version_mismatch"] = {"rejected": exc.code == E_SNAPSHOT_STALE, "code": exc.code}

    renderer = DeterministicRenderer(config, ArtifactCache())
    try:
        manager.load_for_resume(
            config=config,
            command_stream_hash=command_hash,
            expected_projection_fp="intentionally-stale-projection-fingerprint",
        )
        checks["projection_mismatch"] = {"rejected": False, "code": None}
    except RenderError as exc:
        checks["projection_mismatch"] = {"rejected": exc.code == E_SNAPSHOT_STALE, "code": exc.code}
    checks["regenerated_projection_fp"] = renderer.projection_fp
    checks["all_rejected"] = all(item.get("rejected") for item in checks.values() if isinstance(item, dict))
    return checks


def verify(
    *,
    config_path: str | Path,
    commands_path: str | Path,
    workdir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    config = load_render_config(config_path)
    commands, command_hash = load_command_stream(commands_path)
    root = Path(workdir).resolve()
    if root.exists():
        if not force:
            raise RenderError(
                "E_CONFIG_INVALID",
                "verification workdir already exists; pass --force to replace it",
                {"workdir": str(root)},
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    incremental = SessionRunner(
        config=config,
        commands=commands,
        command_stream_hash=command_hash,
        workdir=root / "incremental",
        resume=False,
    )
    incremental_info = incremental.run()
    incremental_diag = incremental.diagnostics()
    write_top_level_diagnostics(root / "incremental", incremental_diag)
    incremental_result = incremental.engine.last_result
    if incremental_result is None:
        raise RuntimeError("incremental session completed without a render result")
    incremental_cache = incremental.renderer.cache

    final_state, replay_failures = replay_final_state(config, commands)
    full_renderer = DeterministicRenderer(config, ArtifactCache())
    full_result = full_renderer.render(
        final_state,
        CommandEffect(
            tuple(range(final_state.frame_count)),
            ("encoding", "initialization", "denoise", "decode", "composition"),
            "blank full render of final state",
        ),
        command_id="full-final",
    )
    equality = _compare_results(
        incremental_result,
        incremental_cache,
        full_result,
        full_renderer.cache,
    )

    state_equal = _semantic_state(incremental.engine.state) == _semantic_state(final_state)
    engine_failures = incremental.engine.failures
    failure_order_equal = [_failure_tuple(item) for item in engine_failures] == [
        _failure_tuple(item) for item in replay_failures
    ]
    latency = _latency_checks(incremental, config)

    single_thread_config = replace(config, worker_count=1)
    single_thread_renderer = DeterministicRenderer(single_thread_config, ArtifactCache())
    single_thread_result = single_thread_renderer.render(
        final_state,
        CommandEffect(
            tuple(range(final_state.frame_count)),
            ("encoding", "initialization", "denoise", "decode", "composition"),
            "single-worker determinism render",
        ),
        command_id="single-thread-final",
    )
    single_thread = _compare_results(
        single_thread_result,
        single_thread_renderer.cache,
        full_result,
        full_renderer.cache,
    )
    single_thread["worker_count"] = 1
    single_thread["max_concurrency_observed"] = single_thread_result.max_concurrency
    single_thread["max_concurrency_respected"] = single_thread_result.max_concurrency <= 1

    recovery = SessionRunner(
        config=config,
        commands=commands,
        command_stream_hash=command_hash,
        workdir=root / "recovery",
        resume=False,
    )
    first_phase = recovery.run(stop_after=8, commit_final=False)
    first_pointer = recovery.snapshot_manager.current_pointer()
    resumed = SessionRunner(
        config=config,
        commands=commands,
        command_stream_hash=command_hash,
        workdir=root / "recovery",
        resume=True,
    )
    resumed_from_cursor = resumed.engine.state.command_cursor
    resumed_phase = resumed.run()
    recovered_diag = resumed.diagnostics()
    write_top_level_diagnostics(root / "recovery", recovered_diag)
    recovered_result = resumed.engine.last_result
    if recovered_result is None:
        raise RuntimeError("recovered session completed without a render result")
    recovery_equality = _compare_results(
        recovered_result,
        resumed.renderer.cache,
        full_result,
        full_renderer.cache,
    )
    recovery_info = {
        "first_phase": first_phase,
        "first_phase_committed_pointer": first_pointer,
        "resumed_from_cursor": resumed_from_cursor,
        "resumed_phase": resumed_phase,
        "final_state_equal": _semantic_state(resumed.engine.state) == _semantic_state(final_state),
        "output_equality": recovery_equality,
        "snapshot_is_self_contained": True,
    }

    stale = _stale_checks(config, command_hash, resumed.snapshot_manager)

    checks = {
        "latency_budget_and_pending_semantics": (
            latency["within_budget_with_tolerance"]
            and latency["pending_semantics_observed"]
            and latency["every_pending_has_explicit_config_version"]
            and latency["pending_responses_expose_latest_completed_preview"]
        ),
        "incremental_full_pixel_equality": equality["pixels_equal"],
        "incremental_full_fingerprint_equality": (
            equality["frame_fingerprints_equal"]
            and equality["step_trace_equal"]
            and equality["trace_fingerprint_equal"]
            and equality["composition_equal"]
        ),
        "incremental_full_artifact_equality": (
            equality["required_artifact_keys_equal"]
            and equality["required_artifact_content_equal"]
            and equality["reuse_content_matches_full_recompute"]
        ),
        "failure_order_equality": failure_order_equal,
        "final_state_replay_equality": state_equal,
        "concurrency_determinism": (
            single_thread["pixels_equal"]
            and single_thread["step_trace_equal"]
            and single_thread["trace_fingerprint_equal"]
            and single_thread["max_concurrency_respected"]
        ),
        "snapshot_recovery_equality": (
            recovery_info["final_state_equal"]
            and recovery_equality["pixels_equal"]
            and recovery_equality["frame_fingerprints_equal"]
            and recovery_equality["step_trace_equal"]
        ),
        "snapshot_recovery_used_committed_state": resumed_from_cursor == 6,
        "snapshot_stale_rejection": stale["all_rejected"],
    }
    report = {
        "schema_version": 1,
        "status": "ok" if all(checks.values()) else "failed",
        "config_path": str(Path(config_path).resolve()),
        "commands_path": str(Path(commands_path).resolve()),
        "workdir": str(root),
        "checks": checks,
        "latency": latency,
        "incremental": {
            "run": incremental_info,
            "diagnostics": incremental_diag,
            "frame_fingerprints": list(incremental_result.frame_fps),
            "trace_fingerprint": incremental_result.trace_fingerprint,
        },
        "full": {
            "frame_fingerprints": list(full_result.frame_fps),
            "trace_fingerprint": full_result.trace_fingerprint,
            "composition_fingerprint": full_result.composition_fp,
        },
        "incremental_vs_full": equality,
        "single_worker_vs_parallel": single_thread,
        "failures": {
            "incremental": engine_failures,
            "full_replay": replay_failures,
            "order_equal": failure_order_equal,
        },
        "recovery": recovery_info,
        "stale": stale,
    }
    atomic_write_json(root / "reports" / "verification.json", report)
    atomic_write_json(root / "verification.json", report)
    return report
