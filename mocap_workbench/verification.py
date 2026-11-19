"""End-to-end local acceptance verification for all three frame tiers."""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import export_snapshot
from .constants import DEFAULT_SEED, FRAME_COUNTS, SCHEMA_CONSISTENCY
from .detection import compute_evidence, detect, detection_payload, thresholds_from_mapping
from .generator import generate_bundle, load_bundle
from .io import atomic_write_json, canonical_json_bytes, sha256_bytes
from .model import Snapshot, Thresholds
from .persistence import StateStore, simulate_incomplete_write
from .pipeline import (
    build_with_evidence,
    check_consistency,
    rebuild_full_for_consistency,
)
from .repair import repair_payload
from .preview import PreviewEngine


def _active_auto_frames(snapshot: Snapshot) -> set[int]:
    frames: set[int] = set()
    for anchors in snapshot.anchors.joint_auto:
        frames.update(anchors)
    frames.update(snapshot.anchors.root_auto)
    frames.difference_update(snapshot.suppressed_auto)
    return frames


def _choose_anchor_edit(snapshot: Snapshot) -> tuple[frozenset[int], frozenset[int], int, int]:
    active_auto = _active_auto_frames(snapshot)
    defective_frames = {entry.frame for entry in snapshot.detection.entries}
    for old in sorted(active_auto):
        if old in {0, snapshot.frame_count - 1}:
            continue
        if old in snapshot.keyframes:
            continue
        for delta in (-5, 5, -9, 9, -13, 13, -21, 21):
            new = old + delta
            if new <= 0 or new >= snapshot.frame_count - 1:
                continue
            if new in active_auto or new in snapshot.keyframes or new in defective_frames:
                continue
            return (
                frozenset(set(snapshot.keyframes) | {new}),
                frozenset(set(snapshot.suppressed_auto) | {old}),
                old,
                new,
            )
    raise RuntimeError("verification data has no editable automatic anchor")


def _outside_pose_hashes_preserved(before: Snapshot, after: Snapshot, dirty_frames: set[int]) -> bool:
    for frame in range(before.frame_count):
        if frame not in dirty_frames and before.repair.frame_hashes[frame] != after.repair.frame_hashes[frame]:
            return False
    return True


def _outside_metric_hashes_preserved(before: Snapshot, after: Snapshot, evaluation_intervals: tuple[str, ...]) -> bool:
    affected = set(evaluation_intervals)
    for key, metrics in before.evaluation.interval_metrics.items():
        if key in affected:
            continue
        if key in after.evaluation.interval_metrics:
            if metrics.get("metric_sha256") != after.evaluation.interval_metrics[key].get("metric_sha256"):
                return False
    return True


def _repair_payload_bytes(snapshot: Snapshot) -> bytes:
    return canonical_json_bytes(repair_payload(snapshot.to_meta_dict(), snapshot.record, snapshot.repair))


def _detection_bytes(snapshot: Snapshot) -> bytes:
    return canonical_json_bytes(detection_payload(snapshot.detection, snapshot.evidence, snapshot.record))


def verify_tier(
    workspace: Path,
    frame_count: int,
    seed: int,
    keep_duplicate: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    tier_root = workspace / "runtime" / "verification" / str(frame_count)
    if tier_root.exists():
        shutil.rmtree(tier_root)
    tier_root.mkdir(parents=True, exist_ok=True)
    tier_started = time.perf_counter()
    result: dict[str, Any] = {
        "frame_count": frame_count,
        "seed": seed,
        "ok": False,
        "checks": {},
        "timings_ms": {},
    }

    started = time.perf_counter()
    bundle_a = generate_bundle(tier_root / "generated_a", frame_count, seed)
    generate_a_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    bundle_b = generate_bundle(tier_root / "generated_b", frame_count, seed)
    generate_b_ms = (time.perf_counter() - started) * 1000.0
    result["timings_ms"].update({"generate_a": generate_a_ms, "generate_b": generate_b_ms})
    files = ("record.json", "truth.json", "defects.json")
    generation_identical = all(
        (tier_root / "generated_a" / name).read_bytes() == (tier_root / "generated_b" / name).read_bytes()
        for name in files
    )
    result["checks"]["generation_byte_identical"] = generation_identical
    result["artifact_hashes"] = {
        name: sha256_bytes((tier_root / "generated_a" / name).read_bytes()) for name in files
    }
    result["defect_count"] = len(bundle_a.defects)

    started = time.perf_counter()
    evidence_a = compute_evidence(bundle_a.record)
    evidence_a_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    evidence_b = compute_evidence(bundle_a.record)
    evidence_b_ms = (time.perf_counter() - started) * 1000.0
    result["timings_ms"].update({"evidence_a": evidence_a_ms, "evidence_b": evidence_b_ms})
    thresholds = thresholds_from_mapping()
    detection_a = detect(bundle_a.record, bundle_a.defects, thresholds, evidence=evidence_a)
    detection_b = detect(bundle_a.record, bundle_a.defects, thresholds_from_mapping(), evidence=evidence_b)
    detection_a_bytes = canonical_json_bytes(detection_payload(detection_a, evidence_a, bundle_a.record))
    detection_b_bytes = canonical_json_bytes(detection_payload(detection_b, evidence_b, bundle_a.record))
    result["checks"]["detection_byte_identical"] = detection_a_bytes == detection_b_bytes
    result["checks"]["evidence_repeat_identical"] = evidence_a.evidence_hash == evidence_b.evidence_hash
    result["detection"] = detection_a.comparison
    (tier_root / "detection_default.json").write_bytes(detection_a_bytes)

    started = time.perf_counter()
    baseline = build_with_evidence(
        bundle_a.record,
        bundle_a.truth,
        bundle_a.defects,
        evidence_a,
        thresholds,
        frozenset(),
        0,
        committed=True,
    )
    baseline_ms = (time.perf_counter() - started) * 1000.0
    baseline_repair_bytes = _repair_payload_bytes(baseline)
    started = time.perf_counter()
    second_baseline = rebuild_full_for_consistency(baseline, bundle_a.defects)
    baseline_repeat_ms = (time.perf_counter() - started) * 1000.0
    second_baseline_bytes = _repair_payload_bytes(second_baseline)
    result["timings_ms"].update({"baseline_build": baseline_ms, "baseline_full_repeat": baseline_repeat_ms})
    result["checks"]["repair_byte_identical_same_inputs"] = baseline_repair_bytes == second_baseline_bytes
    result["baseline"] = {
        "trusted": baseline.evaluation.trusted,
        "summary": baseline.evaluation.summary,
        "version": baseline.version.to_dict(),
    }

    changed_thresholds = replace(thresholds, joint_jump_deg=5.0)
    threshold_preview = PreviewEngine().generate(
        baseline,
        bundle_a.defects,
        thresholds=changed_thresholds,
        operation="verification-threshold",
    )
    threshold_incremental = threshold_preview.candidate
    threshold_consistency = threshold_incremental.consistency
    threshold_inc_ms = threshold_preview.elapsed_ms
    threshold_dirty_frames = {
        frame for frame, _ in threshold_incremental.closure.dirty_joint_coords
    } | set(threshold_incremental.closure.dirty_root_frames)
    result["timings_ms"]["threshold_incremental"] = threshold_inc_ms
    result["threshold_increment"] = {
        "changed_keys": list(threshold_incremental.closure.changed_keys),
        "preview_elapsed_ms": threshold_preview.elapsed_ms,
        "preview_budget_ms": threshold_preview.budget_ms,
        "preview_within_budget": not threshold_preview.over_budget,
        "dirty_joint_coords": len(threshold_incremental.closure.dirty_joint_coords),
        "dirty_root_frames": len(threshold_incremental.closure.dirty_root_frames),
        "evaluation_intervals": list(threshold_incremental.closure.evaluation_intervals),
        "closure_exact": set(threshold_incremental.closure.dirty_joint_coords)
        == set(threshold_incremental.closure.actual_recomputed_joint_coords)
        and set(threshold_incremental.closure.dirty_root_frames)
        == set(threshold_incremental.closure.actual_recomputed_root_frames),
        "outside_pose_hashes_unchanged": _outside_pose_hashes_preserved(
            baseline, threshold_incremental, threshold_dirty_frames
        ),
        "outside_metric_hashes_unchanged": _outside_metric_hashes_preserved(
            baseline,
            threshold_incremental,
            threshold_incremental.closure.evaluation_intervals,
        ),
        "consistency": threshold_consistency.to_dict(),
        "evaluation_trusted": threshold_incremental.evaluation.trusted,
    }

    # Exercise the keyframe closure independently from the threshold stress test
    # so the committed final state remains a normal, trusted baseline plus one
    # manually moved anchor.
    new_keyframes, new_suppressed, old_anchor, new_anchor = _choose_anchor_edit(baseline)
    keyframe_preview = PreviewEngine().generate(
        baseline,
        bundle_a.defects,
        keyframes=new_keyframes,
        suppressed_auto=new_suppressed,
        operation="verification-keyframe",
    )
    keyframe_incremental = replace(keyframe_preview.candidate, committed=True)
    keyframe_consistency = keyframe_incremental.consistency
    keyframe_inc_ms = keyframe_preview.elapsed_ms
    keyframe_dirty_frames = {
        frame for frame, _ in keyframe_incremental.closure.dirty_joint_coords
    } | set(keyframe_incremental.closure.dirty_root_frames)
    result["timings_ms"]["keyframe_incremental"] = keyframe_inc_ms
    result["keyframe_increment"] = {
        "moved_auto_anchor_from": old_anchor,
        "added_user_keyframe_at": new_anchor,
        "changed_keys": list(keyframe_incremental.closure.changed_keys),
        "preview_elapsed_ms": keyframe_preview.elapsed_ms,
        "preview_budget_ms": keyframe_preview.budget_ms,
        "preview_within_budget": not keyframe_preview.over_budget,
        "dirty_joint_coords": len(keyframe_incremental.closure.dirty_joint_coords),
        "dirty_root_frames": len(keyframe_incremental.closure.dirty_root_frames),
        "evaluation_intervals": list(keyframe_incremental.closure.evaluation_intervals),
        "closure_exact": set(keyframe_incremental.closure.dirty_joint_coords)
        == set(keyframe_incremental.closure.actual_recomputed_joint_coords)
        and set(keyframe_incremental.closure.dirty_root_frames)
        == set(keyframe_incremental.closure.actual_recomputed_root_frames),
        "closure_nonempty": bool(keyframe_incremental.closure.dirty_joint_coords or keyframe_incremental.closure.dirty_root_frames),
        "outside_pose_hashes_unchanged": _outside_pose_hashes_preserved(
            baseline, keyframe_incremental, keyframe_dirty_frames
        ),
        "outside_metric_hashes_unchanged": _outside_metric_hashes_preserved(
            baseline,
            keyframe_incremental,
            keyframe_incremental.closure.evaluation_intervals,
        ),
        "consistency": keyframe_consistency.to_dict(),
        "evaluation_trusted": keyframe_incremental.evaluation.trusted,
        "evaluation_summary": keyframe_incremental.evaluation.summary,
    }

    export_dir = tier_root / "exports"
    export_paths = export_snapshot(
        keyframe_incremental,
        export_dir,
        extra={
            "verification_generation_identical": generation_identical,
            "verification_detection_identical": result["checks"]["detection_byte_identical"],
        },
    )
    result["exports"] = {key: str(Path(value).relative_to(workspace)) if value else "" for key, value in export_paths.items()}

    namespace = f"verify-{frame_count}"
    state_root = workspace / "runtime" / "state" / namespace
    if state_root.exists():
        shutil.rmtree(state_root)
    store = StateStore(workspace, namespace)
    first_recovery = store.begin_session()
    commit_info = store.commit(keyframe_incremental, bundle_a.defects)
    close_marker = store.close_cleanly()
    clean_store = StateStore(workspace, namespace)
    clean_recovery = clean_store.begin_session()
    simulate_incomplete_write(clean_store)
    crash_store = StateStore(workspace, namespace)
    crash_recovery = crash_store.begin_session()
    restored = crash_store.restore_committed_snapshot(bundle_a.record, bundle_a.truth, bundle_a.defects)
    crash_store.close_cleanly()
    result["persistence"] = {
        "commit_id": commit_info.commit_id,
        "commit_content_hash": commit_info.content_hash,
        "clean_exit_marker": str(close_marker.relative_to(workspace)),
        "first_session_recovery": first_recovery.__dict__,
        "normal_restart_previous_exit_clean": clean_recovery.previous_exit_clean,
        "forced_restart_detected": bool(
            crash_recovery.previous_session_present and not crash_recovery.previous_exit_clean
        ),
        "orphan_staging_removed": crash_recovery.orphan_staging_files_removed,
        "orphan_blobs_removed": crash_recovery.orphan_blobs_removed,
        "restored_content_hash_matches": bool(restored and restored.content_hash == keyframe_incremental.content_hash),
        "restored_repair_hash_matches": bool(restored and restored.repair.repair_hash == keyframe_incremental.repair.repair_hash),
        "recovery_log": crash_recovery.recovery_log_path,
    }

    all_checks = {
        **result["checks"],
        "detection_exact_match": detection_a.comparison["exact_match"],
        "threshold_closure_exact": result["threshold_increment"]["closure_exact"],
        "threshold_outside_pose_hashes_unchanged": result["threshold_increment"]["outside_pose_hashes_unchanged"],
        "threshold_outside_metric_hashes_unchanged": result["threshold_increment"]["outside_metric_hashes_unchanged"],
        "threshold_incremental_full_consistent": threshold_consistency.consistent,
        "threshold_preview_within_budget": result["threshold_increment"]["preview_within_budget"],
        "keyframe_closure_exact": result["keyframe_increment"]["closure_exact"],
        "keyframe_closure_nonempty": result["keyframe_increment"]["closure_nonempty"],
        "keyframe_outside_pose_hashes_unchanged": result["keyframe_increment"]["outside_pose_hashes_unchanged"],
        "keyframe_outside_metric_hashes_unchanged": result["keyframe_increment"]["outside_metric_hashes_unchanged"],
        "keyframe_incremental_full_consistent": keyframe_consistency.consistent,
        "keyframe_preview_within_budget": result["keyframe_increment"]["preview_within_budget"],
        "normal_restart_clean": clean_recovery.previous_exit_clean,
        "forced_restart_detected": result["persistence"]["forced_restart_detected"],
        "orphans_recovered": crash_recovery.orphan_staging_files_removed >= 1
        and crash_recovery.orphan_blobs_removed >= 1,
        "restored_snapshot_matches": result["persistence"]["restored_content_hash_matches"]
        and result["persistence"]["restored_repair_hash_matches"],
    }
    result["checks"] = all_checks
    result["ok"] = all(bool(value) for value in all_checks.values())
    result["timings_ms"]["tier_total"] = (time.perf_counter() - tier_started) * 1000.0
    if not keep_duplicate:
        shutil.rmtree(tier_root / "generated_b", ignore_errors=True)
    return result


def run_verification(
    workspace: Path,
    frame_counts: tuple[int, ...] = FRAME_COUNTS,
    seed: int = DEFAULT_SEED,
    keep_duplicate: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    started = time.perf_counter()
    tiers = []
    for frame_count in frame_counts:
        print(f"[verify] tier {frame_count}: generating, detecting, repairing, recovering...", flush=True)
        tier = verify_tier(workspace, frame_count, seed, keep_duplicate=keep_duplicate)
        tiers.append(tier)
        print(
            f"[verify] tier {frame_count}: {'PASS' if tier['ok'] else 'FAIL'} "
            f"in {tier['timings_ms']['tier_total'] / 1000.0:.2f}s",
            flush=True,
        )
    report = {
        "schema": "mocap-verification/v1",
        "seed": seed,
        "frame_counts": list(frame_counts),
        "ok": all(tier["ok"] for tier in tiers),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "tiers": tiers,
        "environment": {
            "python": __import__("sys").version,
            "numpy": np.__version__,
            "platform": __import__("platform").platform(),
        },
    }
    atomic_write_json(workspace / "artifacts" / "verification_report.json", report)
    return report
