from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_json, sha256_file
from .config import BudgetConfig, FingerprintConfig
from .generator import GeneratorConfig, generate_materials
from .search import compare_retrieval_candidates, find_first_candidate_divergence, search
from .workbench import Workbench


TRUTH_ALIGNMENT_TOLERANCE_SAMPLES = 4_410  # 100 ms


def _truth_check(report: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    expected_start = int(query["expected_start_sample"])
    for rank, candidate in enumerate(report["candidates"], start=1):
        if candidate["track_id"] != query["expected_track_id"]:
            continue
        distance = abs(int(candidate["alignment_start_sample"]) - expected_start)
        if distance <= TRUTH_ALIGNMENT_TOLERANCE_SAMPLES:
            return {
                "query_id": query["query_id"],
                "variant_kind": query["variant_kind"],
                "expected_track_id": query["expected_track_id"],
                "expected_start_sample": expected_start,
                "matched_rank": rank,
                "matched_track_id": candidate["track_id"],
                "matched_start_sample": candidate["alignment_start_sample"],
                "alignment_error_samples": distance,
                "score": candidate["score"],
                "hit": True,
            }
    return {
        "query_id": query["query_id"],
        "variant_kind": query["variant_kind"],
        "expected_track_id": query["expected_track_id"],
        "expected_start_sample": expected_start,
        "candidates_considered": [
            {
                "track_id": candidate["track_id"],
                "alignment_start_sample": candidate["alignment_start_sample"],
                "score": candidate["score"],
            }
            for candidate in report["candidates"]
        ],
        "hit": False,
    }


def _recovery_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "clean_previous_exit": report["clean_previous_exit"],
        "recovered_version": report["recovered_version"],
        "manifest_hash": report["manifest_hash"],
        "orphan_temp_file_count": len(report["orphan_temp_files"]),
        "orphan_partition_count": len(report["orphan_partitions"]),
        "orphan_manifest_count": len(report["orphan_manifests"]),
        "orphan_operation_count": len(report["orphan_operations"]),
        "orphan_fingerprint_count": len(report["orphan_fingerprints"]),
        "orphan_audio_count": len(report["orphan_audio"]),
        "orphan_temp_files": report["orphan_temp_files"],
        "orphan_partitions": report["orphan_partitions"],
        "orphan_fingerprints": report["orphan_fingerprints"],
        "orphan_audio": report["orphan_audio"],
        "action": report["action"],
    }


def _version_hashes(workbench: Workbench) -> list[dict[str, Any]]:
    return [
        {
            "version": item["version"],
            "parent_version": item["parent_version"],
            "operation_type": item["operation"].get("type"),
            "state_digest": item["state_digest"],
            "index_hash": item["manifest_hash"],
        }
        for item in workbench.version_chain()
    ]


def run_demo(root: str | Path, *, seed: int = 20261119) -> dict[str, Any]:
    demo_root = Path(root).resolve()
    if demo_root.exists():
        shutil.rmtree(demo_root)
    material_root = demo_root / "materials"
    workbench_root = demo_root / "workbench"
    control_root = demo_root / "control"

    generated = generate_materials(
        material_root,
        GeneratorConfig(seed=seed, track_count=7),
    )
    manifest = generated["manifest"]
    track_records = {record["track_id"]: record for record in manifest["tracks"]}
    query_records = {record["query_id"]: record for record in manifest["queries"]}
    query_paths = generated["queries"]

    workbench = Workbench(workbench_root)
    for track_id in [f"track_{index:02d}" for index in range(1, 7)]:
        workbench.import_track(track_id, generated["tracks"][track_id])
    baseline_version = workbench.current.version
    baseline_index_hash = workbench.current.manifest_hash
    baseline_state_digest = workbench.current.state_digest

    baseline_query_ids = [
        "q_01_exact_cross",
        "q_02_pitch_up",
        "q_03_time_fast",
        "q_04_exact_same",
    ]
    baseline_reports = {
        query_id: search(
            workbench,
            query_paths[query_id],
            query_id=query_id,
        )
        for query_id in baseline_query_ids
    }
    baseline_truth = [
        _truth_check(baseline_reports[query_id], query_records[query_id])
        for query_id in baseline_query_ids
    ]
    if not all(item["hit"] for item in baseline_truth):
        raise AssertionError("baseline retrieval missed a ground-truth duplicate")

    degraded_budget = BudgetConfig(
        query_threads=4,
        partition_memory_mb=0.80,
        query_time_budget_ms=2,
        index_partition_memory_mb=8.0,
        max_candidates=50,
    )
    degraded_report = search(
        workbench,
        query_paths["q_03_time_fast"],
        budget=degraded_budget,
        query_id="q_03_time_fast",
    )
    degraded_truth = _truth_check(degraded_report, query_records["q_03_time_fast"])
    if not degraded_truth["hit"]:
        raise AssertionError("degraded retrieval omitted a ground-truth duplicate")
    degradation_actions = [item["action"] for item in degraded_report["degradation_decisions"]]
    for required_action in ("serial_query", "evict_partition_cache", "skip_low_score_candidates"):
        if required_action not in degradation_actions:
            raise AssertionError(f"expected degradation action was not recorded: {required_action}")

    changed_config = FingerprintConfig(
        window_samples=1024,
        hop_samples=256,
        peak_density=18.0,
        band_edges_hz=(90.0, 300.0, 900.0, 2200.0, 5200.0, 10000.0),
    )
    changed_state = workbench.update_config(
        fingerprint_config=changed_config,
        similarity_threshold=0.15,
    )
    changed_report = search(
        workbench,
        query_paths["q_01_exact_cross"],
        query_id="q_01_exact_cross",
    )
    restored_state = workbench.revert(baseline_version)
    restored_reports = {
        query_id: search(
            workbench,
            query_paths[query_id],
            query_id=query_id,
        )
        for query_id in baseline_query_ids
    }
    for query_id in baseline_query_ids:
        compare_retrieval_candidates(baseline_reports[query_id], restored_reports[query_id])
    divergence = find_first_candidate_divergence(
        baseline_reports["q_01_exact_cross"],
        restored_reports["q_01_exact_cross"],
    )
    if divergence is not None:
        raise AssertionError(f"rollback candidate divergence: {divergence}")
    workbench.close()

    shutil.copytree(workbench_root, control_root)
    environment = os.environ.copy()
    project_root = str(Path(__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = project_root + os.pathsep + environment.get("PYTHONPATH", "")
    crash_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_workbench",
            "_crash-import",
            "--root",
            str(workbench_root),
            "--track-id",
            "track_07",
            "--audio",
            generated["tracks"]["track_07"],
            "--crash-after-shards",
            "2",
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if crash_result.returncode == 0:
        raise AssertionError("crash helper exited cleanly instead of being force-terminated")
    force_signal = getattr(signal, "SIGKILL", signal.SIGTERM)

    recovered = Workbench(workbench_root)
    recovery_report = recovered.recovery_report
    if recovery_report["status"] != "recovered":
        raise AssertionError("workbench did not detect the unclean previous exit")
    if recovered.current.version != restored_state.version:
        raise AssertionError("recovery advanced beyond the last committed state")
    if recovered.current.manifest_hash != baseline_index_hash:
        raise AssertionError("recovery did not retain the last committed index hash")
    if not (
        recovery_report["orphan_temp_files"]
        or recovery_report["orphan_partitions"]
        or recovery_report["orphan_fingerprints"]
    ):
        raise AssertionError("crash left no orphan evidence to recover")

    resumed_state = recovered.import_track("track_07", generated["tracks"]["track_07"])
    resumed_report = search(
        recovered,
        query_paths["q_05_pitch_down_cross"],
        budget=degraded_budget,
        query_id="q_05_pitch_down_cross",
    )
    resumed_truth = _truth_check(resumed_report, query_records["q_05_pitch_down_cross"])
    if not resumed_truth["hit"]:
        raise AssertionError("post-recovery retrieval missed its ground-truth duplicate")
    historical_reconstruction = [
        recovered.reconstruct_version(baseline_version),
        recovered.reconstruct_version(changed_state.version),
    ]
    if not all(
        item["index_hash_match"] and item["all_shards_match"]
        for item in historical_reconstruction
    ):
        raise AssertionError("historical index reconstruction hash check failed")
    recovered_version_hashes = _version_hashes(recovered)
    atomic_write_json(demo_root / "crash_recovery.json", recovery_report)
    atomic_write_json(
        demo_root / "version_hashes.json",
        {"current_version": recovered.current.version, "versions": recovered_version_hashes},
    )
    recovered.close()

    control = Workbench(control_root)
    control_state = control.import_track("track_07", generated["tracks"]["track_07"])
    control_report = search(
        control,
        query_paths["q_05_pitch_down_cross"],
        budget=degraded_budget,
        query_id="q_05_pitch_down_cross",
    )
    compare_retrieval_candidates(resumed_report, control_report)
    control_version_hashes = _version_hashes(control)
    control.close()
    if resumed_state.state_digest != control_state.state_digest:
        raise AssertionError("resumed state differs from uninterrupted control state")
    if resumed_state.manifest_hash != control_state.manifest_hash:
        raise AssertionError("resumed index hash differs from uninterrupted control index hash")
    if recovered_version_hashes != control_version_hashes:
        raise AssertionError("resumed version chain differs from uninterrupted control chain")

    report: dict[str, Any] = {
        "schema": 1,
        "demo": {
            "seed": seed,
            "material_manifest_sha256": generated["manifest_sha256"],
            "sample_rate": manifest["sample_rate"],
            "bit_depth": manifest["bit_depth"],
            "channels": manifest["channels"],
            "track_count": len(manifest["tracks"]),
            "query_count": len(manifest["queries"]),
            "truth_interval_convention": manifest["sample_range_convention"],
        },
        "track_wav_sha256": {
            track_id: track_records[track_id]["wav_sha256"]
            for track_id in sorted(track_records)
        },
        "query_wav_sha256": {
            query_id: query_records[query_id]["wav_sha256"]
            for query_id in sorted(query_records)
        },
        "version_hashes": recovered_version_hashes,
        "baseline": {
            "version": baseline_version,
            "index_hash": baseline_index_hash,
            "truth_checks": baseline_truth,
            "retrieval_reports": baseline_reports,
        },
        "degraded": {
            "budget": degraded_budget.to_dict(),
            "actions": degradation_actions,
            "truth_check": degraded_truth,
            "retrieval_report": degraded_report,
        },
        "config_change": {
            "from_version": baseline_version,
            "to_version": changed_state.version,
            "from_index_hash": baseline_index_hash,
            "to_index_hash": changed_state.manifest_hash,
            "config_digest": changed_config.digest,
            "threshold": changed_state.similarity_threshold,
            "retrieval_report": changed_report,
        },
        "rollback": {
            "target_version": baseline_version,
            "new_version": restored_state.version,
            "target_index_hash": baseline_index_hash,
            "restored_index_hash": restored_state.manifest_hash,
            "index_hash_exact": restored_state.manifest_hash == baseline_index_hash,
            "state_digest_exact": restored_state.state_digest == baseline_state_digest,
            "candidate_reports_equal": True,
            "first_candidate_divergence": None,
            "shards_rewritten": 0,
        },
        "historical_reconstruction": {
            "checks": historical_reconstruction,
            "all_match": all(
                item["index_hash_match"] and item["all_shards_match"]
                for item in historical_reconstruction
            ),
            "mode": "read_only_in_memory_audit",
        },
        "crash_recovery": {
            "exit_code": crash_result.returncode,
            "termination": (
                "SIGKILL" if crash_result.returncode == -force_signal else "non-zero_exit"
            ),
            "recovery": _recovery_summary(recovery_report),
            "recovered_version_before_resume": restored_state.version,
            "recovered_index_hash_before_resume": baseline_index_hash,
            "resumed_version_after_import": resumed_state.version,
            "resumed_index_hash_after_import": resumed_state.manifest_hash,
        },
        "resume_equivalence": {
            "control_version": control_state.version,
            "resumed_state_digest": resumed_state.state_digest,
            "control_state_digest": control_state.state_digest,
            "resumed_index_hash": resumed_state.manifest_hash,
            "control_index_hash": control_state.manifest_hash,
            "index_hash_equal": resumed_state.manifest_hash == control_state.manifest_hash,
            "version_chains_equal": recovered_version_hashes == control_version_hashes,
            "candidates_equal": True,
            "truth_check": resumed_truth,
            "resumed_retrieval_report": resumed_report,
            "control_retrieval_report": control_report,
        },
        "clean_shutdown_marker": {
            "written_by_normal_close": True,
            "forced_child_did_not_write_it": True,
        },
    }
    report_path = demo_root / "report.json"
    atomic_write_json(report_path, report)
    report_hash = sha256_file(report_path)
    (demo_root / "report.sha256").write_text(report_hash + "\n", encoding="ascii")
    summary = {
        "schema": 1,
        "report_sha256": report_hash,
        "material_manifest_sha256": generated["manifest_sha256"],
        "baseline_version": baseline_version,
        "baseline_index_hash": baseline_index_hash,
        "changed_version": changed_state.version,
        "changed_index_hash": changed_state.manifest_hash,
        "rollback_version": restored_state.version,
        "rollback_index_hash": restored_state.manifest_hash,
        "rollback_index_exact": restored_state.manifest_hash == baseline_index_hash,
        "crash_exit_code": crash_result.returncode,
        "recovery_status": recovery_report["status"],
        "orphan_temp_files": len(recovery_report["orphan_temp_files"]),
        "orphan_partitions": len(recovery_report["orphan_partitions"]),
        "resumed_version": resumed_state.version,
        "resumed_index_hash": resumed_state.manifest_hash,
        "control_index_hash": control_state.manifest_hash,
        "resume_index_exact": resumed_state.manifest_hash == control_state.manifest_hash,
        "truth_hits": len(baseline_truth) + int(degraded_truth["hit"]) + int(resumed_truth["hit"]),
        "truth_checks": len(baseline_truth) + 2,
        "degradation_actions": degradation_actions,
    }
    print(canonical_json(summary))
    return report
