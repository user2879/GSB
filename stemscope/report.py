from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .cache import CacheStore
from .config import RemixConfig
from .conversion import convert_package
from .dashboard import generate_dashboard
from .hashing import sha256_json
from .injection import load_injection_manifest, run_injection_case
from .model import FORMAT_VERSIONS
from .package import parse_package
from .render import RenderResult, render, write_frame_details
from .snapshot import SnapshotStore
from .state import ClipRuntime, WorkbenchState
from .wavio import write_pcm16_wav
from .workbench import Workbench, compare_render_results


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _waveform(pcm: np.ndarray, points: int = 180) -> list[list[int]]:
    array = np.asarray(pcm, dtype=np.int32).reshape(-1)
    if array.size == 0:
        return [[0, 0] for _ in range(points)]
    if array.size <= points:
        return [[int(value), int(value)] for value in array]
    edges = np.linspace(0, array.size, points + 1, dtype=np.int64)
    result: list[list[int]] = []
    for start, end in zip(edges, edges[1:]):
        block = array[start:end]
        if block.size == 0:
            result.append([0, 0])
        else:
            result.append([int(block.min()), int(block.max())])
    return result


def validate_three_formats(data_root: Path, clip_ids: list[str]) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_pcm_equal = True
    all_repeat_equal = True
    for clip_id in clip_ids:
        segment_dir = data_root / "segments" / clip_id
        parsed = {
            version: parse_package(segment_dir, version, clip_id=clip_id)
            for version in FORMAT_VERSIONS
        }
        entry: dict[str, Any] = {"clip_id": clip_id, "versions": {}, "track_parity": True}
        for version, result in parsed.items():
            if result.ok and result.package is not None:
                entry["versions"][version] = result.package.to_package_info()
            else:
                entry["track_parity"] = False
                all_pcm_equal = False
                failures.extend(error.to_dict() for error in result.failures)
        if all(result.ok for result in parsed.values()):
            packages = [parsed[version].package for version in FORMAT_VERSIONS]
            assert all(package is not None for package in packages)
            base = packages[0]
            for package in packages[1:]:
                if sorted(package.tracks) != sorted(base.tracks):
                    entry["track_parity"] = False
                    continue
                for name in sorted(base.tracks):
                    if not np.array_equal(base.tracks[name].pcm, package.tracks[name].pcm):
                        entry["track_parity"] = False
            repeat = parse_package(segment_dir, "v1", clip_id=clip_id)
            repeat_equal = bool(
                repeat.ok
                and repeat.package is not None
                and repeat.package.package_fingerprint == parsed["v1"].package.package_fingerprint
                and all(
                    np.array_equal(repeat.package.tracks[name].pcm, parsed["v1"].package.tracks[name].pcm)
                    for name in parsed["v1"].package.tracks
                )
            )
            entry["repeat_parse_equal"] = repeat_equal
            all_repeat_equal = all_repeat_equal and repeat_equal
            entry["track_parity"] = bool(entry["track_parity"])
            all_pcm_equal = all_pcm_equal and entry["track_parity"]
        else:
            entry["repeat_parse_equal"] = False
            all_repeat_equal = False
        clips.append(entry)
    return {
        "versions": list(FORMAT_VERSIONS),
        "clip_count": len(clip_ids),
        "pcm_equality": all_pcm_equal,
        "repeat_parse_determinism": all_repeat_equal,
        "clips": clips,
        "failures": sorted(failures, key=lambda item: tuple(item.get("sort_key", []))),
    }


def validate_conversion_matrix(data_root: Path, working_root: Path, clip_id: str = "seg_000") -> dict[str, Any]:
    _rmtree(working_root)
    working_root.mkdir(parents=True, exist_ok=True)
    packages = {
        version: parse_package(data_root / "segments" / clip_id, version, clip_id=clip_id).package
        for version in FORMAT_VERSIONS
    }
    conversions: list[dict[str, Any]] = []
    loss_fields: list[dict[str, Any]] = []
    all_ok = True
    for source_version in FORMAT_VERSIONS:
        source = packages[source_version]
        assert source is not None
        for target_version in FORMAT_VERSIONS:
            if source_version == target_version:
                continue
            destination = working_root / f"{source_version}_to_{target_version}"
            result = convert_package(source, target_version, destination, clip_id=clip_id)
            frame_equal = False
            if result.ok and result.target_package is not None:
                frame_equal = all(
                    np.array_equal(source.tracks[name].pcm, result.target_package.tracks[name].pcm)
                    for name in source.tracks
                )
            all_ok = all_ok and result.ok and frame_equal
            record = dict(result.record)
            record["frame_equal"] = frame_equal
            record["source_format"] = source_version
            record["target_format"] = target_version
            conversions.append(record)
            for loss in result.record.get("losses", []):
                loss_fields.append(
                    {
                        **loss,
                        "target_format_version": target_version,
                        "path": source_version + "_to_" + target_version,
                    }
                )
    loss_fields.sort(
        key=lambda item: (
            item.get("field", ""),
            item.get("source_track", ""),
            item.get("source_format_version", ""),
            item.get("target_format_version", ""),
        )
    )
    return {
        "clip_id": clip_id,
        "all_ok": all_ok,
        "conversions": conversions,
        "loss_fields": loss_fields,
        "conversion_count": len(conversions),
    }


def validate_quantization(data_root: Path, base_config: RemixConfig) -> dict[str, Any]:
    package = parse_package(data_root / "segments" / "seg_000", "v1", clip_id="seg_000").package
    assert package is not None
    clip = ClipRuntime.from_package("seg_000", package)
    results: dict[int, RenderResult] = {}
    for steps in (1000, 2000):
        config = base_config.clone()
        config.gain_quantization_steps = steps
        config.workers = min(config.workers, 2)
        state = WorkbenchState(config=config, clips={"seg_000": clip})
        results[steps] = render(state, CacheStore(), previous_state=None, retain_frame_details=False)
    quantized: dict[str, dict[str, int]] = {}
    for steps in (1000, 2000):
        config = base_config.clone()
        config.gain_quantization_steps = steps
        quantized[str(steps)] = {
            name: config.quantized_gain_int(name) for name in sorted(config.tracks)
        }
    changed_bytes = int(np.count_nonzero(results[1000].pcm != results[2000].pcm))
    return {
        "steps": [1000, 2000],
        "quantized_gain_ints": quantized,
        "pcm_sha256": {
            "1000": results[1000].pcm_sha256,
            "2000": results[2000].pcm_sha256,
        },
        "pcm_changed": results[1000].pcm_sha256 != results[2000].pcm_sha256,
        "changed_frames": changed_bytes,
        "passed": changed_bytes > 0
        and quantized["1000"] != quantized["2000"]
        and results[1000].pcm_sha256 != results[2000].pcm_sha256,
    }


def verify_sequence(
    data_root: Path,
    runtime_root: Path,
    config: RemixConfig,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    _rmtree(runtime_root)
    verifier = Workbench(
        data_root,
        runtime_root,
        config,
        resume=False,
        start_worker=False,
    )
    state = verifier.state
    cache = verifier.cache
    steps: list[dict[str, Any]] = []
    final_result: RenderResult = verifier.latest_result
    try:
        for index, command in enumerate(commands):
            previous_state = state
            new_state = previous_state.clone()
            param_version = previous_state.param_version + 1
            new_state = verifier.apply_command_to_state(
                previous_state, new_state, command, param_version
            )
            retain = index == len(commands) - 1
            incremental = render(
                new_state,
                cache,
                previous_state=previous_state,
                retain_frame_details=retain,
                simulated_compute_ms=0.0,
            )
            full = render(
                new_state,
                CacheStore(),
                previous_state=None,
                retain_frame_details=False,
                simulated_compute_ms=0.0,
            )
            comparison = compare_render_results(incremental, full)
            steps.append(
                {
                    "step": index,
                    "command_id": command.get("id", f"cmd_{index + 1:03d}"),
                    "command_type": command.get("type"),
                    "param_version": param_version,
                    "incremental": incremental.to_report_payload(),
                    "full": {
                        "pcm_sha256": full.pcm_sha256,
                        "aggregate_peak": full.aggregate_peak,
                        "segment_count": len(full.segments),
                        "frame_detail_sha256": full.frame_detail_sha256,
                        "recomputed_tasks": len(full.recomputed),
                        "failures": [error.to_dict() for error in full.failures],
                    },
                    "parity": comparison,
                }
            )
            cache = verifier.cache
            state = new_state
            final_result = incremental
    finally:
        verifier.close()
    return {
        "steps": steps,
        "final_result": final_result,
        "final_state": state,
        "all_parity": all(step["parity"]["canonical_equal"] for step in steps),
        "all_task_coverage": all(step["parity"]["task_coverage_equal"] for step in steps),
    }


def _with_cache_actions(result: RenderResult) -> dict[str, Any]:
    payload = result.to_report_payload()
    reused_mix = {
        item["segment_index"]
        for item in result.reused
        if item.get("stage") == "mix"
    }
    recomputed_mix = {
        item["segment_index"]
        for item in result.recomputed
        if item.get("stage") == "mix"
    }
    for segment in payload["segments"]:
        index = segment["segment_index"]
        if index in reused_mix:
            segment["cache_action"] = "reused"
        elif index in recomputed_mix:
            segment["cache_action"] = "recomputed"
        else:
            segment["cache_action"] = "unknown"
    return payload


def _snapshot_details(store: SnapshotStore) -> dict[str, Any]:
    manifest = store.read_manifest() or {}
    return {
        "global_version": manifest.get("global_version"),
        "config_hash": manifest.get("config_hash"),
        "format_versions": manifest.get("format_versions", []),
        "completed_segments": manifest.get("completed_segments", []),
        "stage_versions": manifest.get("stage_versions", {}),
        "stage_entry_counts": manifest.get("stage_entry_counts", {}),
        "temp_dirs_remaining": store._temp_dirs_remaining(),
    }


def run_acceptance(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    interactions_path: str | Path | None = None,
    resume: bool = False,
    expected_global_version: str | None = None,
    workers_override: int | None = None,
    budget_override_ms: float | None = None,
    format_override: str | None = None,
) -> dict[str, Any]:
    data = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path).resolve() if config_path else data / "remix_config.json"
    interactions_file = (
        Path(interactions_path).resolve() if interactions_path else data / "interactions.json"
    )
    initial_config = RemixConfig.from_path(config_file)
    if workers_override is not None:
        initial_config.workers = int(workers_override)
    if budget_override_ms is not None:
        initial_config.interaction_budget_ms = float(budget_override_ms)
    if format_override is not None:
        initial_config.format_version = format_override
    initial_config.validate()
    with open(interactions_file, "r", encoding="utf-8") as handle:
        interactions = json.load(handle)
    commands = list(interactions.get("commands", []))
    sample_manifest = json.loads((data / "generator_manifest.json").read_text(encoding="utf-8"))
    clip_ids = list(sample_manifest.get("clip_ids", []))

    temp_check_path = output / ".temp-write-check"
    temp_check_path.write_bytes(b"stemscope-temp-ok\n")
    temp_writable = temp_check_path.read_bytes() == b"stemscope-temp-ok\n"
    temp_check_path.unlink()

    format_validation = validate_three_formats(data, clip_ids)
    conversion_matrix = validate_conversion_matrix(data, output / "conversion_matrix")
    quantization = validate_quantization(data, initial_config)

    runtime = output / "runtime"
    if not resume:
        _rmtree(runtime)
        _rmtree(output / "verification_runtime")
    workbench = Workbench(
        data,
        runtime,
        initial_config,
        resume=resume,
        expected_global_version=expected_global_version,
        start_worker=True,
    )
    initial_result = workbench.latest_result
    initial_audit = dict(workbench.snapshot_audit)
    parse_stats = dict(workbench.parse_stats)
    responses: list[dict[str, Any]] = []
    try:
        for command in commands:
            budget = float(command.get("budget_ms", workbench.state.config.interaction_budget_ms))
            response = workbench.submit(command, budget_ms=budget)
            response = dict(response)
            response["type"] = command.get("type", "")
            response["command"] = command
            responses.append(response)
        workbench.wait_for_idle()
        final_result = workbench.latest_result
        final_state = workbench.state.clone()
        final_full = workbench.render_full_from_blank(final_state)
        final_comparison = compare_render_results(final_result, final_full)
        snapshot_post = dict(workbench.snapshot_audit)
        snapshot_details = _snapshot_details(workbench.snapshot)
        snapshot_global_version = workbench.global_version
        snapshot_config_hash = workbench.snapshot_config_hash
        budget_hits = workbench.budget_hits
        completed_jobs = {
            job.command_id: job for job in workbench.completed_jobs if job.result is not None
        }
        interactive_checkpoint_commits = sum(
            int(job.result.checkpoint_commits)
            for job in workbench.completed_jobs
            if job.result is not None
        )
        job_errors = {
            job.command_id: job.error for job in workbench.completed_jobs if job.error is not None
        }
        final_cache_stats = dict(final_result.cache_stats)
    finally:
        workbench.close()

    resume_workbench = Workbench(
        data,
        runtime,
        initial_config,
        resume=True,
        start_worker=False,
    )
    try:
        resume_info = {
            "status": resume_workbench.snapshot_audit.get("status"),
            "valid": bool(resume_workbench.snapshot_audit.get("reusable")),
            "parse_hits": resume_workbench.parse_stats.get("cache_hits", 0),
            "parsed": resume_workbench.parse_stats.get("parsed", 0),
            "gain_hits": resume_workbench.latest_result.cache_stats.get("gain_hits", 0),
            "mix_hits": resume_workbench.latest_result.cache_stats.get("mix_hits", 0),
            "pcm_matches_initial": resume_workbench.latest_result.pcm_sha256 == initial_result.pcm_sha256,
        }
        resume_workbench.close()
    except Exception:
        resume_workbench.close()
        raise

    stale_version_runtime = output / "stale_version_runtime"
    _rmtree(stale_version_runtime)
    shutil.copytree(runtime, stale_version_runtime)
    stale_version_workbench = Workbench(
        data,
        stale_version_runtime,
        initial_config,
        resume=True,
        expected_global_version="definitely-stale-version",
        start_worker=False,
    )
    stale_version = {
        "status": stale_version_workbench.snapshot_audit.get("status"),
        "rejected": stale_version_workbench.snapshot_audit.get("status") == "stale",
        "codes": [error.code for error in stale_version_workbench.startup_failures],
        "errors": list(stale_version_workbench.snapshot_audit.get("errors", [])),
    }
    stale_version_workbench.close()
    _rmtree(stale_version_runtime)

    stale_config_runtime = output / "stale_config_runtime"
    _rmtree(stale_config_runtime)
    shutil.copytree(runtime, stale_config_runtime)
    stale_store = SnapshotStore(stale_config_runtime / "snapshot")
    stale_config_audit = stale_store.audit(
        expected_global_version=snapshot_global_version,
        expected_config_hash="definitely-stale-config-hash",
    )
    stale_config = {
        "status": stale_config_audit.get("status"),
        "rejected": stale_config_audit.get("status") == "stale",
        "codes": [item["code"] for item in stale_config_audit.get("errors", [])],
        "errors": list(stale_config_audit.get("errors", [])),
    }
    _rmtree(stale_config_runtime)

    verification = verify_sequence(
        data,
        output / "verification_runtime",
        initial_config,
        commands,
    )
    verification_final: RenderResult = verification["final_result"]
    verification_state: WorkbenchState = verification["final_state"]
    interactive_vs_verification = compare_render_results(final_result, verification_final)
    step_interactive_parity: list[dict[str, Any]] = []
    for step in verification["steps"]:
        job = completed_jobs.get(step["command_id"])
        if job is None or job.result is None:
            step_interactive_parity.append(
                {"command_id": step["command_id"], "result_available": False, "canonical_equal": False}
            )
        else:
            step_interactive_parity.append(
                {
                    "command_id": step["command_id"],
                    "result_available": True,
                    "canonical_equal": job.result.pcm_sha256
                    == step["incremental"]["pcm_sha256"]
                    and job.result.frame_detail_sha256
                    == step["incremental"]["frame_detail_sha256"],
                    "interactive_pcm_sha256": job.result.pcm_sha256,
                    "verification_pcm_sha256": step["incremental"]["pcm_sha256"],
                }
            )

    serial_config = verification_state.config.clone()
    serial_config.workers = 1
    serial_state = verification_state.clone()
    serial_state.config = serial_config
    serial_result = render(
        serial_state,
        CacheStore(),
        previous_state=None,
        retain_frame_details=False,
    )
    concurrency_comparison = compare_render_results(verification_final, serial_result)
    configured_workers = int(initial_config.workers)
    concurrency_limit_ok = (
        verification_final.max_observed_workers <= configured_workers
        and serial_result.max_observed_workers <= 1
        and parse_stats.get("max_observed_workers", 1) <= configured_workers
    )

    injection_manifest = load_injection_manifest(data / "injection_manifest.json")
    injection_results = []
    injection_root = output / "injection_runs"
    _rmtree(injection_root)
    for case in injection_manifest.get("cases", []):
        injection_results.append(
            run_injection_case(data, injection_root / str(case["id"]), case)
        )

    frame_detail_path = output / "frame_details.jsonl"
    if verification_final.frame_detail_lines:
        write_frame_details(verification_final, frame_detail_path)
        detail_written = True
    else:
        detail_written = False
    mix_path = output / "mix_final.wav"
    write_pcm16_wav(mix_path, verification_final.pcm, initial_config.sample_rate)
    conversion_records_path = output / "conversion_records.json"
    _write_json(
        conversion_records_path,
        {
            "interactive": final_state.conversion_records,
            "matrix": conversion_matrix["conversions"],
        },
    )

    final_report_payload = _with_cache_actions(verification_final)
    final_state_info = verification_state.to_info()
    track_cards = []
    for name in sorted(verification_state.clips["seg_000"].tracks):
        track = verification_state.clips["seg_000"].tracks[name]
        settings = verification_state.config.tracks[name]
        track_cards.append(
            {
                "name": name,
                "clip_id": "seg_000",
                "frames": track.frames,
                "max_abs_amplitude": track.max_abs_amplitude,
                "source_format_version": track.source_format,
                "gain": settings.gain,
                "quantized_gain_int": verification_state.config.quantized_gain_int(name),
                "muted": settings.muted,
                "start_ms": settings.start_ms,
                "end_ms": settings.end_ms,
                "waveform": _waveform(track.pcm),
                "content_fingerprint": track.content_fingerprint(),
            }
        )
    visualization = {
        "clip_id": "seg_000",
        "tracks": track_cards,
        "mix_waveform": _waveform(verification_final.pcm),
    }

    failed_budget_responses = [
        response for response in responses if response.get("status") == "budget_exceeded"
    ]
    completed_within_budget = [
        response for response in responses if response.get("status") == "completed"
    ]
    budget_semantics = {
        "budget_hits": budget_hits,
        "completed_within_budget": len(completed_within_budget),
        "failed_responses": len(failed_budget_responses),
        "all_failed_are_stale": all(
            response.get("is_latest") is False
            and response.get("code") == "E_INTERACT_BUDGET_EXCEEDED"
            and response.get("summary", {}).get("param_version")
            != response.get("requested_param_version")
            for response in failed_budget_responses
        ),
        "all_elapsed_within_budget": all(
            float(response.get("elapsed_ms", 0.0)) <= float(response.get("budget_ms", 0.0)) + 0.5
            for response in failed_budget_responses
        ),
        "semantics_note": "超预算响应只携带最近一次已完成摘要，param_version 与 requested_param_version 分离，is_latest=false。",
    }
    budget_ok = (
        bool(failed_budget_responses)
        and budget_semantics["all_failed_are_stale"]
        and budget_semantics["all_elapsed_within_budget"]
        and budget_hits == len(failed_budget_responses)
    )

    diagnostics_failure_dicts = sorted(
        {
            tuple(item.get("sort_key", [])): item
            for item in (
                final_state.failures
                + [error.to_dict() for error in verification_final.failures]
                + [error.to_dict() for error in final_result.failures]
            )
        }.values(),
        key=lambda item: tuple(item.get("sort_key", [])),
    )

    stage_timings = {
        "parse_ms": round(float(parse_stats.get("elapsed_ms", 0.0)), 3),
        "convert_ms": round(
            sum(float(record.get("elapsed_ms", 0.0)) for record in final_state.conversion_records), 3
        ),
        "gain_ms": verification_final.stage_times_ms.get("gain", 0.0),
        "mix_ms": verification_final.stage_times_ms.get("mix", 0.0),
        "render_total_ms": verification_final.stage_times_ms.get("render_total", 0.0),
    }

    acceptance_checks = [
        {
            "id": "temp_directory_writable",
            "passed": temp_writable,
        },
        {
            "id": "three_format_parse_and_frame_parity",
            "passed": format_validation["pcm_equality"],
        },
        {
            "id": "repeat_parse_determinism",
            "passed": format_validation["repeat_parse_determinism"],
        },
        {
            "id": "six_direction_conversion_frame_parity",
            "passed": conversion_matrix["all_ok"],
        },
        {
            "id": "v3_to_v2_checksum_loss_ledger",
            "passed": any(
                loss["field"] == "block_checksum_crc32"
                and loss["source_format_version"] == "v3"
                and loss["target_format_version"] == "v2"
                for loss in conversion_matrix["loss_fields"]
            ),
        },
        {"id": "quantization_changes_fixed_int_and_pcm", "passed": quantization["passed"]},
        {"id": "incremental_full_parity_all_edits", "passed": verification["all_parity"]},
        {
            "id": "task_coverage_parity_all_edits",
            "passed": verification["all_task_coverage"]
            and final_comparison["task_coverage_equal"]
            and interactive_vs_verification["task_coverage_equal"],
        },
        {
            "id": "interactive_vs_verification_parity",
            "passed": interactive_vs_verification["canonical_equal"]
            and all(item["canonical_equal"] for item in step_interactive_parity),
        },
        {
            "id": "concurrent_vs_serial_parity",
            "passed": concurrency_comparison["canonical_equal"]
            and concurrency_limit_ok,
        },
        {"id": "budget_response_semantics", "passed": budget_ok},
        {
            "id": "snapshot_resume",
            "passed": resume_info["valid"]
            and resume_info["pcm_matches_initial"]
            and resume_info["mix_hits"] > 0,
        },
        {
            "id": "snapshot_stale_version_rejected",
            "passed": stale_version["rejected"]
            and "E_SNAPSHOT_STALE" in stale_version["codes"],
        },
        {
            "id": "snapshot_stale_config_rejected",
            "passed": stale_config["rejected"]
            and "E_SNAPSHOT_STALE" in stale_config["codes"],
        },
        {
            "id": "injection_failure_paths",
            "passed": bool(injection_results) and all(item["passed"] for item in injection_results),
        },
    ]
    acceptance_passed = all(item["passed"] for item in acceptance_checks)

    report: dict[str, Any] = {
        "schema": "stemscope/diagnostic-report/v1",
        "workbench": {"name": "StemScope", "version": "1.0.0"},
        "environment": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "platform": sys.platform,
        },
        "input": {
            "data_root": str(data),
            "seed": sample_manifest.get("seed"),
            "sample_manifest": sample_manifest,
            "interactions_path": str(interactions_file),
            "config_path": str(config_file),
        },
        "config": {
            "initial": initial_config.to_dict(),
            "final": final_state.config.to_dict(),
            "initial_hash": initial_config.config_hash(),
            "final_hash": final_state.config.config_hash(),
        },
        "format_validation": format_validation,
        "conversions": {
            "matrix": conversion_matrix,
            "interactive_records": final_state.conversion_records,
            "loss_fields": conversion_matrix["loss_fields"],
        },
        "quantization": quantization,
        "initial_render": initial_result.to_report_payload(),
        "final_state": final_state_info,
        "final_render": final_report_payload,
        "full_parity": {
            "final_interactive_vs_full": final_comparison,
            "interactive_vs_verification": interactive_vs_verification,
            "step_parity": [step["parity"] for step in verification["steps"]],
        },
        "incremental_sequence": verification["steps"],
        "interaction": {
            "budget_ms": final_state.config.interaction_budget_ms,
            "responses": responses,
            "budget_hits": budget_hits,
            "completed_within_budget": len(completed_within_budget),
            "semantics": budget_semantics,
            "semantics_note": budget_semantics["semantics_note"],
            "job_errors": {
                command_id: error.to_dict() for command_id, error in job_errors.items()
            },
        },
        "concurrency": {
            "configured_workers": configured_workers,
            "verified_workers": [1, configured_workers],
            "serial_vs_concurrent": concurrency_comparison,
            "limit_respected": concurrency_limit_ok,
            "observed": {
                "configured": verification_final.max_observed_workers,
                "serial": serial_result.max_observed_workers,
                "parse": parse_stats.get("max_observed_workers", 1),
            },
            "last_incremental_cache_stats": final_cache_stats,
        },
        "snapshot": {
            "initial_status": initial_audit.get("status"),
            "initial_audit": initial_audit,
            "post_run_status": snapshot_post.get("status"),
            "global_version": snapshot_global_version,
            "config_hash": snapshot_config_hash,
            "manifest": snapshot_details,
            "interactive_checkpoint_commits": interactive_checkpoint_commits,
            "resume": resume_info,
            "stale_version": stale_version,
            "stale_config": stale_config,
            "temp_dirs_remaining": snapshot_details["temp_dirs_remaining"],
        },
        "injections": injection_results,
        "diagnostics": {
            "failures": diagnostics_failure_dicts,
            "empty_expected_for_success_path": not diagnostics_failure_dicts,
        },
        "stage_timings_ms": stage_timings,
        "artifacts": {
            "mix_final_wav": str(mix_path),
            "frame_details_jsonl": str(frame_detail_path) if detail_written else None,
            "frame_detail_line_count": len(verification_final.frame_detail_lines),
            "conversion_records_json": str(conversion_records_path),
            "dashboard_html": str(output / "dashboard.html"),
            "report_json": str(output / "report.json"),
        },
        "visualization": visualization,
        "acceptance": {
            "checks": acceptance_checks,
            "passed": acceptance_passed,
            "passed_count": sum(1 for item in acceptance_checks if item["passed"]),
            "total_count": len(acceptance_checks),
        },
    }

    fingerprint_payload = {
        "schema": report["schema"],
        "input": {
            "seed": report["input"]["seed"],
            "sample_content_sha256": sample_manifest.get("content_sha256"),
            "clip_ids": clip_ids,
        },
        "config": report["config"],
        "format_validation": format_validation,
        "conversion_matrix": {
            "all_ok": conversion_matrix["all_ok"],
            "conversions": [
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"elapsed_ms", "target_digest"}
                }
                for record in conversion_matrix["conversions"]
            ],
            "loss_fields": conversion_matrix["loss_fields"],
        },
        "quantization": quantization,
        "final_render": {
            key: final_report_payload[key]
            for key in (
                "pcm_sha256",
                "aggregate_peak",
                "segments",
                "frame_detail_sha256",
                "failures",
            )
        },
        "parity": report["full_parity"],
        "concurrency": {
            "serial_vs_concurrent": concurrency_comparison,
            "limit_respected": concurrency_limit_ok,
        },
        "budget": {
            "statuses": [response.get("status") for response in responses],
            "semantics": budget_semantics,
        },
        "snapshot": {
            "resume": resume_info,
            "stale_version": stale_version,
            "stale_config": stale_config,
            "stage_entry_counts": snapshot_details.get("stage_entry_counts", {}),
        },
        "injections": [
            {
                "case_id": item["case_id"],
                "expected_code": item["expected_code"],
                "observed_codes": item["observed_codes"],
                "passed": item["passed"],
            }
            for item in injection_results
        ],
        "acceptance": acceptance_checks,
    }
    report_fingerprint = sha256_json(fingerprint_payload)
    report["report_fingerprint"] = report_fingerprint
    report["report_id"] = f"stemscope-{report_fingerprint[:16]}"

    dashboard_path = generate_dashboard(report, output / "dashboard.html")
    report["artifacts"]["dashboard_html"] = str(dashboard_path)
    _write_json(output / "report.json", report)
    return report
