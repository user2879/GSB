from __future__ import annotations

import io
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .codec import parse_v1_package, parse_v2_package
from .config import PipelineConfig
from .conversion import v1_to_v2, v2_to_v1
from .fusion import fuse_instances
from .interactive import InteractiveEngine
from .io_utils import atomic_write_bytes, atomic_write_json, read_json, sha256_bytes
from .pipeline import SampleLayout, build_session
from .png_codec import decode_panorama_png
from .snapshot import SnapshotStore


def _grid_digest(grid: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(grid, dtype=np.int32).tobytes())


def _bundle_digest(bundle) -> dict[str, Any]:
    return {
        "instances": [
            {
                "key": item.key,
                "class_id": item.class_id,
                "cache_key": item.cache_key,
                "area": item.area,
            }
            for item in bundle.sorted_instances()
        ],
        "failures": [failure.to_dict() for failure in bundle.sorted_failures()],
        "dimensions": [bundle.width, bundle.height],
    }


def _compare_parse_bundles(parallel, serial) -> dict[str, Any]:
    left = _bundle_digest(parallel)
    right = _bundle_digest(serial)
    return {
        "parallel_equal_serial": left == right,
        "parallel_instance_order": [item["key"] for item in left["instances"]],
        "serial_instance_order": [item["key"] for item in right["instances"]],
        "parallel_failure_order": [
            item["instance_key"] for item in left["failures"]
        ],
        "serial_failure_order": [item["instance_key"] for item in right["failures"]],
        "instance_count": len(left["instances"]),
        "failure_count": len(left["failures"]),
    }


def _expected_frame(v1_bundle) -> tuple[np.ndarray, np.ndarray]:
    instance_ids = np.zeros((v1_bundle.height, v1_bundle.width), dtype=np.uint32)
    class_ids = np.zeros((v1_bundle.height, v1_bundle.width), dtype=np.uint32)
    for item in sorted(v1_bundle.instances, key=lambda value: value.instance_id):
        instance_ids[item.mask] = item.instance_id
        class_ids[item.mask] = item.class_id
    return instance_ids, class_ids


def _validate_png_frames(sample_dir: Path, v1_bundle) -> dict[str, Any]:
    images = sorted((sample_dir / "images").glob("*.png"))
    frames: list[dict[str, Any]] = []
    expected_ids, expected_classes = _expected_frame(v1_bundle)
    frame_zero_exact = False
    for path in images:
        instance_ids, class_ids = decode_panorama_png(path)
        exact = False
        if path.stem == "panorama_000":
            exact = bool(
                np.array_equal(instance_ids, expected_ids)
                and np.array_equal(class_ids, expected_classes)
            )
            frame_zero_exact = exact
        frames.append(
            {
                "file": path.name,
                "shape": list(instance_ids.shape),
                "instance_ids": sorted(int(value) for value in np.unique(instance_ids)),
                "class_ids": sorted(int(value) for value in np.unique(class_ids)),
                "frame_zero_exact_v1_match": exact if path.stem == "panorama_000" else None,
            }
        )
    sizes = {(frame["shape"][0], frame["shape"][1]) for frame in frames}
    return {
        "frame_count": len(frames),
        "fixed_resolution": len(sizes) == 1,
        "resolution": (
            [next(iter(sizes))[1], next(iter(sizes))[0]] if sizes else None
        ),
        "frame_zero_exact_v1_match": frame_zero_exact,
        "frames": frames,
    }


def _replace_metadata_in_copy(
    sample_dir: Path,
    destination: Path,
    replacement: dict[str, Any],
) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(sample_dir, destination)
    metadata_path = destination / "instances.json"
    metadata = read_json(metadata_path)
    entries = metadata["instances"]
    found = False
    for entry in entries:
        if (
            int(entry.get("instance_id", -1)) == int(replacement["instance_id"])
            and entry.get("source_format", "v1") == replacement.get("source_format", "v1")
        ):
            entry["confidence"] = replacement["confidence"]
            entry["crowd"] = replacement["crowd"]
            if "palette" in replacement:
                entry["palette"] = replacement["palette"]
            found = True
            break
    if not found:
        raise RuntimeError("verification metadata replacement target was not found")
    atomic_write_json(metadata_path, metadata)
    return destination


def _colorize(grid: np.ndarray, palette: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    for class_id, color in palette.items():
        rgba[grid == class_id] = color
    return rgba


def _save_grid_artifacts(
    grid: np.ndarray,
    palette: dict[int, tuple[int, int, int, int]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "final_panorama.png"
    buffer = io.BytesIO()
    Image.fromarray(_colorize(grid, palette), mode="RGBA").save(buffer, format="PNG")
    atomic_write_bytes(png_path, buffer.getvalue())
    grid_path = output_dir / "final_category_grid.npz"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, category_grid=grid.astype(np.int32))
    atomic_write_bytes(grid_path, buffer.getvalue())
    return {
        "png": str(png_path),
        "grid": str(grid_path),
        "grid_sha256": _grid_digest(grid),
    }


def _run_forced_kill(
    sample_dir: Path,
    snapshot_root: Path,
    expected_generation: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "tools" / "forced_kill_worker.py"),
        "--sample",
        str(sample_dir),
        "--snapshots",
        str(snapshot_root),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    store = SnapshotStore(snapshot_root)
    recovered = store.recover()
    return {
        "worker_exit_code": completed.returncode,
        "forced_sigkill": completed.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL),
        "recovered_generation": recovered.generation,
        "recovered_previous_published_generation": recovered.generation
        == expected_generation,
        "unpublished_generation_ignored": recovered.generation == expected_generation,
        "stderr": completed.stderr.strip(),
    }


def _run_graceful_signal(sample_dir: Path, snapshot_root: Path) -> dict[str, Any]:
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "tools" / "signal_worker.py"),
        "--sample",
        str(sample_dir),
        "--snapshots",
        str(snapshot_root),
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready_line = process.stdout.readline() if process.stdout else ""
    process.send_signal(signal.SIGTERM)
    stdout_tail, stderr = process.communicate(timeout=30)
    store = SnapshotStore(snapshot_root)
    recovered = store.recover()
    return {
        "ready": ready_line.strip() == "READY",
        "signal": "SIGTERM",
        "exit_code": process.returncode,
        "zero_exit": process.returncode == 0,
        "post_signal_commit": recovered.generation is not None,
        "generation": recovered.generation,
        "stdout_tail": stdout_tail.strip(),
        "stderr": stderr.strip(),
    }


def run_verification(sample_dir: str | Path, work_dir: str | Path) -> dict[str, Any]:
    sample_path = Path(sample_dir).resolve()
    work_path = Path(work_dir).resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    conversion_dir = work_path / "conversion"
    artifact_dir = work_path / "artifacts"
    snapshot_root = work_path / "snapshots"
    for path in (conversion_dir, artifact_dir, snapshot_root):
        if path.exists():
            shutil.rmtree(path)
    conversion_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    layout = SampleLayout.discover(sample_path)
    config = PipelineConfig.load(layout.config_path)
    sample_manifest = read_json(sample_path / "sample_manifest.json")

    session = build_session(sample_path, config=config)
    v1_parallel = parse_v1_package(
        layout.v1_path,
        config=config,
        metadata=session.metadata_document,
        max_workers=config.max_workers,
    )
    v1_serial = parse_v1_package(
        layout.v1_path,
        config=config,
        metadata=session.metadata_document,
        max_workers=1,
    )
    v2_parallel = parse_v2_package(
        layout.v2_path,
        config=config,
        max_workers=config.max_workers,
    )
    v2_serial = parse_v2_package(
        layout.v2_path,
        config=config,
        max_workers=1,
    )
    parse_concurrency = {
        "v1": _compare_parse_bundles(v1_parallel, v1_serial),
        "v2": _compare_parse_bundles(v2_parallel, v2_serial),
    }
    png_validation = _validate_png_frames(sample_path, v1_parallel)

    v1_to_v2_report = v1_to_v2(
        v1_parallel,
        conversion_dir / "roundtrip_v2.json",
        config=config,
    )
    v2_to_v1_report = v2_to_v1(
        v2_parallel,
        conversion_dir / "roundtrip_v1.bin",
        conversion_dir / "roundtrip_v1_sidecar.json",
        config=config,
    )
    conversion_validation = {
        "v1_to_v2": v1_to_v2_report,
        "v2_to_v1": v2_to_v1_report,
        "both_pixel_exact": bool(
            v1_to_v2_report["comparison"]["pixel_equal"]
            and v2_to_v1_report["comparison"]["pixel_equal"]
        ),
        "degradation_loss_count": len(v2_to_v1_report["degradation_losses"]),
    }

    serial_fusion = fuse_instances(
        session.sorted_instances,
        config=config,
        max_workers=1,
    )
    fusion_concurrency = {
        "parallel_equal_serial": bool(
            _grid_digest(session.fusion_result.category_grid)
            == _grid_digest(serial_fusion.category_grid)
            and session.fusion_result.matches == serial_fusion.matches
            and session.fusion_result.class_distribution
            == serial_fusion.class_distribution
        ),
        "parallel_grid_sha256": _grid_digest(session.fusion_result.category_grid),
        "serial_grid_sha256": _grid_digest(serial_fusion.category_grid),
        "parallel_match_order": [
            item["pair_key"] for item in session.fusion_result.matches
        ],
        "serial_match_order": [item["pair_key"] for item in serial_fusion.matches],
    }

    commands = [
        {
            "type": "set_params",
            "iou_match_threshold": 0.25,
            "retention_mode": "prefer_non_crowd",
            "min_coverage": 0.25,
        },
        {
            "type": "set_params",
            "retention_mode": "prefer_non_crowd",
            "min_coverage": 0.5,
        },
        {
            "type": "set_params",
            "iou_match_threshold": 0.333333333,
            "retention_mode": "confidence",
            "min_coverage": 0.333333333,
        },
        {
            "type": "replace_metadata",
            "instance_key": "v1:2",
            "confidence": 0.97,
            "crowd": False,
        },
        {"type": "set_formats", "formats": ["v2"]},
        {"type": "set_formats", "formats": ["v1", "v2"]},
        {
            "type": "set_params",
            "iou_match_threshold": 0.4,
            "retention_mode": "prefer_non_crowd",
            "min_coverage": 0.25,
        },
        {
            "type": "set_params",
            "iou_match_threshold": 0.333333333,
            "retention_mode": "prefer_non_crowd",
            "min_coverage": 0.333333333,
        },
    ]
    store = SnapshotStore(snapshot_root)
    engine = InteractiveEngine(session)
    interaction_responses: list[dict[str, Any]] = []
    immediate_statuses: list[str] = []
    snapshots: list[dict[str, Any]] = []
    try:
        for command in commands:
            immediate = engine.submit(command)
            immediate_statuses.append(immediate["status"])
            command_seq = immediate["command_seq"]
            if immediate["status"] == "pending":
                final_response = engine.wait_command(command_seq)
            else:
                final_response = engine.response_for(command_seq) or immediate
            interaction_responses.append(
                {
                    "immediate": immediate,
                    "final_status": final_response["status"],
                    "command_seq": command_seq,
                }
            )
            if command_seq % config.snapshot_commit_interval == 0:
                manifest = store.commit(
                    session,
                    report_payload={"command_seq": command_seq},
                )
                snapshots.append(manifest)
    finally:
        engine.close()
    probe_session = build_session(sample_path, config=config)

    def slow_probe_worker(work):
        time.sleep(0.08)
        return work()

    probe_engine = InteractiveEngine(
        probe_session,
        budget_ms=5,
        worker_wrapper=slow_probe_worker,
    )
    try:
        probe_started = time.perf_counter()
        probe_pending = probe_engine.submit(
            {"type": "set_params", "iou_match_threshold": 0.4}
        )
        probe_elapsed_ms = (time.perf_counter() - probe_started) * 1000.0
        probe_completed = probe_engine.wait_command(
            probe_pending["command_seq"],
            timeout=2,
        )
    finally:
        probe_engine.close()
    budget_probe = {
        "budget_ms": 5,
        "injected_work_ms": 80,
        "elapsed_ms": probe_elapsed_ms,
        "status": probe_pending["status"],
        "is_latest": probe_pending["is_latest"],
        "business_code": probe_pending["business_code"],
        "preview_is_latest": probe_pending["preview"]["is_latest"],
        "preview_source_command_seq": probe_pending["preview_source_command_seq"],
        "requested_params_snapshot_version": probe_pending[
            "requested_params_snapshot_version"
        ],
        "preview_params_snapshot_version": probe_pending[
            "params_snapshot_version"
        ],
        "eventual_status": probe_completed["status"],
        "eventual_is_latest": probe_completed["is_latest"],
        "pending_within_budget": probe_elapsed_ms < 60,
    }
    final_seq = len(commands)
    if not snapshots or final_seq % config.snapshot_commit_interval != 0:
        manifest = store.commit(
            session,
            report_payload={"command_seq": final_seq, "final": True},
        )
        snapshots.append(manifest)
    recovered_before_kill = store.recover()
    recovery_pixel_equal = bool(
        np.array_equal(
            recovered_before_kill.category_grid,
            session.fusion_result.category_grid,
        )
    )
    expected_generation = recovered_before_kill.generation
    forced_kill = _run_forced_kill(sample_path, snapshot_root, expected_generation)
    recovery_after_forced_kill = store.recover()
    manifest_instance_versions = recovery_after_forced_kill.manifest[
        "instance_result_versions"
    ]
    session_instance_versions = {
        item.key: item.cache_key for item in session.sorted_instances
    }
    manifest_versions_consistent = bool(
        manifest_instance_versions == session_instance_versions
        and recovery_after_forced_kill.manifest["fusion_result_version"]
        == session.fusion_result.fusion_result_version
    )
    forced_kill["pixels_still_match"] = bool(
        np.array_equal(
            recovery_after_forced_kill.category_grid,
            session.fusion_result.category_grid,
        )
    )
    restarted_manifest = store.commit(
        session,
        report_payload={"phase": "restart_after_forced_kill"},
    )
    recovered_after_restart = store.recover()
    restart_recovery = {
        "new_global_version": restarted_manifest["global_version"],
        "recovered_generation": recovered_after_restart.generation,
        "pixel_equal": bool(
            np.array_equal(
                recovered_after_restart.category_grid,
                session.fusion_result.category_grid,
            )
        ),
    }
    graceful = _run_graceful_signal(sample_path, work_path / "snapshots_signal")

    final_config = session.config
    oracle_sample = _replace_metadata_in_copy(
        sample_path,
        work_path / "oracle_sample",
        {
            "instance_id": 2,
            "source_format": "v1",
            "confidence": 0.97,
            "crowd": False,
        },
    )
    oracle = build_session(oracle_sample, config=final_config)
    final_rebuild = {
        "pixel_equal": bool(
            np.array_equal(
                oracle.fusion_result.category_grid,
                session.fusion_result.category_grid,
            )
        ),
        "incremental_grid_sha256": _grid_digest(session.fusion_result.category_grid),
        "full_rebuild_grid_sha256": _grid_digest(oracle.fusion_result.category_grid),
        "instance_versions_equal": {
            item.key: item.cache_key for item in session.sorted_instances
        }
        == {item.key: item.cache_key for item in oracle.sorted_instances},
        "incremental_fusion_version": session.fusion_result.fusion_result_version,
        "full_rebuild_fusion_version": oracle.fusion_result.fusion_result_version,
        "incremental_instance_count": len(session.instances),
        "full_rebuild_instance_count": len(oracle.instances),
    }
    artifacts = _save_grid_artifacts(
        session.fusion_result.category_grid,
        final_config.palette,
        artifact_dir,
    )
    instance_manifest = {
        "class_distribution": session.fusion_result.class_distribution,
        "fusion_result_version": session.fusion_result.fusion_result_version,
        "instances": [
            item.summary() for item in session.fusion_result.fused_instances
        ],
    }
    instance_manifest_path = artifact_dir / "panorama_instances.json"
    atomic_write_json(instance_manifest_path, instance_manifest)
    artifacts["instances"] = str(instance_manifest_path)

    event_by_trigger: dict[str, list[dict[str, Any]]] = {}
    for event in session.events:
        event_by_trigger.setdefault(event["trigger"], []).append(event)
    boundary_sensitive = session.fusion_result.boundary_sensitive
    pending_count = immediate_statuses.count("pending")
    pending_demonstrated = bool(
        budget_probe["status"] == "pending"
        and budget_probe["business_code"] == "E_INTERACT_BUDGET_PENDING"
        and budget_probe["is_latest"] is False
        and budget_probe["preview_is_latest"] is False
        and budget_probe["eventual_status"] == "complete"
    )
    failure_codes = sorted(
        {
            failure.code for failure in session.failures
        }.union(session.fusion_result.failure_codes)
    )
    report = {
        "schema_version": 1,
        "sample": {
            "path": str(sample_path),
            "seed": sample_manifest["seed"],
            "width": sample_manifest["width"],
            "height": sample_manifest["height"],
            "pixel_encoding": sample_manifest["pixel_encoding"],
            "files": sample_manifest["files"],
        },
        "configuration": final_config.to_dict(),
        "double_format_parsing": {
            "v1_instance_count": len(v1_parallel.instances),
            "v2_instance_count": len(v2_parallel.instances),
            "combined_instance_count": len(session.instances),
            "v1_failures": [failure.to_dict() for failure in v1_parallel.failures],
            "v2_failures": [failure.to_dict() for failure in v2_parallel.failures],
            "v1_dimensions": [v1_parallel.width, v1_parallel.height],
            "v2_dimensions": [v2_parallel.width, v2_parallel.height],
        },
        "png_validation": png_validation,
        "conversion_validation": conversion_validation,
        "fusion_validation": {
            "class_distribution": session.fusion_result.class_distribution,
            "fused_instance_count": len(session.fusion_result.fused_instances),
            "matched_pair_count": len(
                [item for item in session.fusion_result.matches if item["accepted"]]
            ),
            "match_decisions": session.fusion_result.matches,
            "instance_list": instance_manifest["instances"],
            "conflict_pixels": session.fusion_result.conflict_pixels,
            "fusion_result_version": session.fusion_result.fusion_result_version,
            "boundary_diagnostics": session.fusion_result.boundary_diagnostics,
            "boundary_sensitive": boundary_sensitive,
        },
        "incremental_reuse": {
            "events": session.events,
            "events_by_trigger": {
                trigger: len(events) for trigger, events in sorted(event_by_trigger.items())
            },
            "three_required_triggers_present": all(
                trigger in event_by_trigger
                for trigger in (
                    "metadata_replacement",
                    "retention_policy_change",
                    "format_version_switch",
                )
            ),
            "decode_counts": dict(sorted(session.decode_counts.items())),
            "metadata_replacement_decoded_instances": [
                event["decoded_instances"]
                for event in session.events
                if event["trigger"] == "metadata_replacement"
            ],
            "unaffected_instances_not_decoded": all(
                set(event["decoded_instances"]).issubset(
                    set(event["invalidated_instances"])
                )
                for event in session.events
                if event["trigger"]
                in {
                    "metadata_replacement",
                    "retention_policy_change",
                    "fusion_threshold_change",
                }
            ),
        },
        "interactive_budget": {
            "budget_ms": config.interaction_budget_ms,
            "command_count": len(commands),
            "immediate_statuses": immediate_statuses,
            "pending_count": pending_count,
            "budget_probe": budget_probe,
            "pending_semantics_demonstrated": pending_demonstrated,
            "responses": interaction_responses,
            "pending_response_semantics_code": "E_INTERACT_BUDGET_PENDING",
            "stale_result_never_labeled_latest": all(
                response["immediate"].get("is_latest") is False
                for response in interaction_responses
                if response["immediate"]["status"] == "pending"
            ),
        },
        "incremental_vs_full_rebuild": final_rebuild,
        "concurrency_comparison": {
            "max_workers": config.max_workers,
            "parse": parse_concurrency,
            "fusion": fusion_concurrency,
            "parse_equal": all(
                value["parallel_equal_serial"]
                for value in parse_concurrency.values()
            ),
            "fusion_equal": fusion_concurrency["parallel_equal_serial"],
        },
        "snapshot_recovery": {
            "snapshots_committed": len(snapshots),
            "generations": [manifest["generation"] for manifest in snapshots],
            "version_changes": [manifest["version_changes"] for manifest in snapshots],
            "recovered_generation": recovered_before_kill.generation,
            "recovered_pixel_equal": recovery_pixel_equal,
            "instance_versions_manifest_consistent": manifest_versions_consistent,
            "forced_kill": forced_kill,
            "restart_after_kill": restart_recovery,
            "graceful_sigterm": graceful,
        },
        "failure_codes": failure_codes,
        "business_codes": sorted(
            set(failure_codes).union(
                {"E_INTERACT_BUDGET_PENDING"}
                if pending_count or pending_demonstrated
                else set()
            )
        ),
        "artifacts": artifacts,
        "overall_pass": bool(
            png_validation["fixed_resolution"]
            and png_validation["frame_zero_exact_v1_match"]
            and conversion_validation["both_pixel_exact"]
            and parse_concurrency["v1"]["parallel_equal_serial"]
            and parse_concurrency["v2"]["parallel_equal_serial"]
            and fusion_concurrency["parallel_equal_serial"]
            and final_rebuild["pixel_equal"]
            and final_rebuild["instance_versions_equal"]
            and recovery_pixel_equal
            and manifest_versions_consistent
            and forced_kill["forced_sigkill"]
            and forced_kill["pixels_still_match"]
            and restart_recovery["pixel_equal"]
            and graceful["zero_exit"]
            and graceful["post_signal_commit"]
            and len(boundary_sensitive) > 0
            and pending_demonstrated
        ),
    }
    return report
