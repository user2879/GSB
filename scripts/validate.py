"""Run the full local acceptance matrix and write artifacts/validation_report.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evalworkbench.errors import BusinessError
from evalworkbench.closure import build_context, compute_frame_closure, compute_segment_closure
from evalworkbench.constants import DEFAULT_INJECTION, ENCODING_PARAMS, METRIC_POLICY
from evalworkbench.formats import migrate_metric_record, parse_metric_binary, parse_project_json
from evalworkbench.generator import generate_assets
from evalworkbench.ledger import LedgerStore
from evalworkbench.pipeline import (
    build_carriers,
    evaluate_workspace,
    preview_workspace,
    recover_workspace,
)
from evalworkbench.quality import validate_shots

INJECTIONS = ROOT / "injections"
MUTATIONS = ROOT / "mutations"


class Validation:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.observations: dict[str, Any] = {}

    def check(self, name: str, condition: bool, details: Any = None) -> None:
        record = {"name": name, "passed": bool(condition), "details": details}
        self.checks.append(record)
        print(("PASS " if condition else "FAIL ") + name)
        if not condition:
            raise AssertionError(json.dumps(record, ensure_ascii=False, default=str))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check["passed"] for check in self.checks)


def _latest(workspace: Path) -> dict[str, Any]:
    snapshot = LedgerStore(workspace).load_latest_committed()
    if snapshot is None:
        raise AssertionError("no committed ledger")
    return snapshot


def _codes(result: dict[str, Any]) -> set[str]:
    return {error["code"] for error in result.get("business_errors", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, help="workspace to retain for inspection")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "validation_report.json")
    args = parser.parse_args()
    started = time.time()
    workspace = (args.workspace or Path(tempfile.mkdtemp(prefix="evalworkbench-validation-"))).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    second_workspace = Path(tempfile.mkdtemp(prefix="evalworkbench-determinism-"))
    validation = Validation()
    report: dict[str, Any] = {
        "schema_version": "validation-report/1.0",
        "workspace": str(workspace),
        "python": sys.version,
        "entrypoint": "python -m evalworkbench.cli demo --workspace <workspace>",
    }
    try:
        generation_a = generate_assets(workspace)
        generation_b = generate_assets(second_workspace)
        same_files = all(
            generation_a["files"][name]["sha256"] == generation_b["files"][name]["sha256"]
            for name in generation_a["files"]
        )
        validation.check(
            "deterministic generation is byte-identical",
            generation_a["content_hash"] == generation_b["content_hash"] and same_files,
            {
                "content_hash": generation_a["content_hash"],
                "second_content_hash": generation_b["content_hash"],
                "files_equal": same_files,
            },
        )
        validation.check(
            "fixture dimensions, duration, frame/shot/audio counts",
            generation_a["frame_count"] == 300
            and generation_a["shot_count"] == 4
            and generation_a["audio_sample_count_per_channel"] == 480_000
            and generation_a["audio_total_samples"] == 960_000,
            {
                "frame_count": generation_a["frame_count"],
                "shot_count": generation_a["shot_count"],
                "samples_per_channel": generation_a["audio_sample_count_per_channel"],
                "total_samples": generation_a["audio_total_samples"],
            },
        )
        carriers = build_carriers(workspace)
        validation.check(
            "three primary carriers plus v1 metric fixture built",
            set(carriers["files"]) == {"project_json", "metric_binary", "legacy_json", "metric_binary_v1"},
            carriers["files"],
        )
        v1 = parse_metric_binary(workspace / "carriers" / "metrics_v1.emf")
        v2 = parse_metric_binary(workspace / "carriers" / "metrics_v2.emf")
        migrated, changed = migrate_metric_record(v1.declared_frames[0], 1)
        migrated_again, changed_again = migrate_metric_record(migrated, 2)
        core_equal = all(
            before[field] == after[field]
            for before, after in zip(v1.declared_frames, v2.declared_frames, strict=True)
            for field in ("frame_index", "shot_id", "timestamp_ms", "psnr_db", "ssim", "flags")
        )
        validation.check(
            "metric v1 migration is lossless on core fields and idempotent",
            core_equal and changed and not changed_again and migrated == migrated_again,
            {"core_equal": core_equal, "first_migration_changed": changed, "second_migration_changed": changed_again},
        )

        baseline = evaluate_workspace(workspace)
        validation.check(
            "baseline evaluation passes with no business errors",
            baseline["status"] == "ok",
            {"status": baseline["status"], "business_errors": baseline["business_errors"]},
        )
        statuses = {(pair["left"], pair["right"]): pair["status"] for pair in baseline["equivalence"]["pairs"]}
        statuses_for_report = {f"{left}__{right}": status for (left, right), status in statuses.items()}
        validation.check(
            "carrier equivalence is explicit and assumptions are classified",
            statuses[("project_json", "metric_binary")] == "equivalent"
            and statuses[("project_json", "legacy_json")] == "equivalent_with_assumptions"
            and statuses[("metric_binary", "legacy_json")] == "equivalent_with_assumptions"
            and baseline["equivalence"]["overall"]["equivalent"],
            {
                "pairs": statuses_for_report,
                "overall": baseline["equivalence"]["overall"],
                "first_mismatch": baseline["equivalence"]["first_mismatch"],
            },
        )
        validation.check(
            "carrier-declared metrics equal decoded actual metrics",
            all(claim["status"] == "equivalent" for claim in baseline["carrier_claims"]),
            baseline["carrier_claims"],
        )
        validation.check(
            "segment boundaries match truth and all segments are complete",
            [(segment["start_frame"], segment["end_frame"]) for segment in baseline["segment_summaries"]]
            == [(0, 89), (90, 179), (180, 239), (240, 299)]
            and all(segment["complete"] for segment in baseline["segment_summaries"]),
            baseline["segment_summaries"],
        )
        validation.check(
            "audio alignment is within tolerance under the shared timebase",
            baseline["audio"]["overall"]["status"] == "ok"
            and baseline["audio"]["timebase"]["master"] == "global_content_start",
            {"overall": baseline["audio"]["overall"], "timebase": baseline["audio"]["timebase"]},
        )
        first_snapshot = _latest(workspace)
        second = evaluate_workspace(workspace)
        second_snapshot = _latest(workspace)
        metric_shards = ("actual_frames.json", "actual_segments.json", "audio.json", "equivalence.json")
        byte_stable = all(
            first_snapshot["manifest"]["shard_hashes"][shard] == second_snapshot["manifest"]["shard_hashes"][shard]
            for shard in metric_shards
        )
        validation.check(
            "repeat decode produces byte-identical metric ledger and zero-frame closure",
            second["status"] == "ok" and byte_stable and second["closure"]["reason"] == "no_changes",
            {
                "shard_hashes": {shard: second_snapshot["manifest"]["shard_hashes"][shard] for shard in metric_shards},
                "closure": second["closure"],
                "metric_ledger_digest": second["metric_ledger_digest"],
            },
        )
        build_carriers(second_workspace)
        cross_workspace = evaluate_workspace(second_workspace)
        validation.check(
            "equivalence digest depends on content hash, not workspace location",
            baseline["content_hash"] == cross_workspace["content_hash"]
            and baseline["equivalence"]["report_digest"] == cross_workspace["equivalence"]["report_digest"]
            and baseline["metric_ledger_digest"] == cross_workspace["metric_ledger_digest"],
            {
                "content_hash": baseline["content_hash"],
                "report_digest": baseline["equivalence"]["report_digest"],
                "cross_workspace_report_digest": cross_workspace["equivalence"]["report_digest"],
                "cross_workspace_status": cross_workspace["status"],
            },
        )
        bad_schema_path = workspace / "carriers" / "invalid_schema_for_validation.json"
        bad_schema_document = json.loads((workspace / "carriers" / "project_v2.json").read_text(encoding="utf-8"))
        bad_schema_document["schema_version"] = "eval-project/9.9"
        bad_schema_path.write_text(json.dumps(bad_schema_document), encoding="utf-8")
        try:
            try:
                parse_project_json(bad_schema_path)
            except BusinessError as exc:
                schema_code = exc.code
            else:
                schema_code = None
        finally:
            bad_schema_path.unlink(missing_ok=True)
        validation.check(
            "project schema-version mismatch is rejected as E_SCHEMA_VERSION",
            schema_code == "E_SCHEMA_VERSION",
            {"code": schema_code},
        )
        closure_base = _latest(workspace)
        prior_frames = closure_base["shards"]["actual_frames.json"]["records"]
        prior_segments = closure_base["shards"]["actual_segments.json"]["segments"]
        closure_shots = validate_shots(json.loads((workspace / "assets" / "truth.json").read_text())["shots"], 300)
        closure_injection = dict(DEFAULT_INJECTION)
        encoding_context = build_context(
            closure_base["manifest"]["content_hash"],
            {**ENCODING_PARAMS, "qstep": 9},
            METRIC_POLICY,
            closure_shots,
            closure_injection,
            300,
        )
        policy_context = build_context(
            closure_base["manifest"]["content_hash"],
            ENCODING_PARAMS,
            {**METRIC_POLICY, "ssim_window": 4},
            closure_shots,
            closure_injection,
            300,
        )
        encoding_frames = compute_frame_closure(prior_frames, encoding_context)
        policy_frames = compute_frame_closure(prior_frames, policy_context)
        encoding_segments = compute_segment_closure(prior_segments, closure_shots, prior_frames, encoding_context)
        policy_segments = compute_segment_closure(prior_segments, closure_shots, prior_frames, policy_context)
        validation.check(
            "encoding and policy snapshots invalidate exactly all frames and segments",
            encoding_frames == list(range(300))
            and policy_frames == list(range(300))
            and encoding_segments == [0, 1, 2, 3]
            and policy_segments == [0, 1, 2, 3],
            {
                "encoding_frames": len(encoding_frames),
                "encoding_segments": encoding_segments,
                "policy_frames": len(policy_frames),
                "policy_segments": policy_segments,
            },
        )

        fault_expectations = {
            "binary_field_bitmap.json": "E_FIELD_BITMAP",
            "binary_metric_version.json": "E_METRIC_VERSION",
            "binary_endian.json": "E_ENDIAN_MISMATCH",
            "binary_index_width.json": "E_INDEX_OVERFLOW",
        }
        fault_results: dict[str, Any] = {}
        for filename, expected_code in fault_expectations.items():
            result = evaluate_workspace(workspace, injection_path=INJECTIONS / filename)
            observed = _codes(result)
            validation.check(
                f"injection {filename} is isolated as {expected_code}",
                expected_code in observed
                and result["equivalence"]["overall"]["invalid_carrier"]
                and not result["equivalence"]["overall"]["equivalent"]
                and "project_json" in {claim["carrier"] for claim in result["carrier_claims"]},
                {"codes": sorted(observed), "equivalence": result["equivalence"]["overall"]},
            )
            fault_results[filename] = sorted(observed)

        drift = evaluate_workspace(workspace, injection_path=INJECTIONS / "audio_drift.json")
        validation.check(
            "audio drift injection yields E_AUDIO_DRIFT with first offending segment",
            "E_AUDIO_DRIFT" in _codes(drift)
            and drift["audio"]["overall"]["first_drift_segment"] == 0
            and abs(drift["audio"]["segments"][0]["offset_samples"] - 96) <= 1
            and abs(drift["audio"]["segments"][0]["gain_ratio"] - 1.05) < 1e-3,
            {"overall": drift["audio"]["overall"], "segments": drift["audio"]["segments"]},
        )
        rate_lie = evaluate_workspace(workspace, injection_path=INJECTIONS / "audio_rate_lie.json")
        validation.check(
            "audio sample-rate lie is isolated as E_AUDIO_SAMPLE_RATE",
            rate_lie["audio"]["error"]["code"] == "E_AUDIO_SAMPLE_RATE" and "E_AUDIO_SAMPLE_RATE" in _codes(rate_lie),
            rate_lie["audio"]["error"],
        )
        dropped = evaluate_workspace(workspace, injection_path=INJECTIONS / "drop_and_timestamp.json")
        validation.check(
            "drop and timestamp-jump faults isolate exact units and never pass silently",
            dropped["missing_units"]["missing_frames"] == [7, 150]
            and {"E_DROPPED_FRAME", "E_TIMESTAMP_JUMP", "E_EQUIV_MISMATCH"} <= _codes(dropped)
            and all(claim["status"] == "mismatch" for claim in dropped["carrier_claims"]),
            {
                "missing_units": dropped["missing_units"],
                "codes": sorted(_codes(dropped)),
                "claims": dropped["carrier_claims"],
                "joint_timebase": dropped["audio"]["overall"],
            },
        )
        reset = evaluate_workspace(workspace)
        validation.check("reset to baseline after video-fault injection", reset["status"] == "ok", reset["status"])
        closure_fault = evaluate_workspace(workspace, injection_path=INJECTIONS / "closure_mismatch.json")
        validation.check(
            "closure executor mismatch is detected as E_CLOSURE_MISMATCH",
            "E_CLOSURE_MISMATCH" in _codes(closure_fault) and not closure_fault["closure"]["match"],
            closure_fault["closure"],
        )
        reset = evaluate_workspace(workspace)
        validation.check("reset to baseline before shot mutation", reset["status"] == "ok", reset["status"])
        before_shot = _latest(workspace)
        before_hashes = {
            int(record["frame_index"]): record["record_hash"]
            for record in before_shot["shards"]["actual_frames.json"]["records"]
        }
        shot_mutation = evaluate_workspace(workspace, mutation_path=MUTATIONS / "shots_merge_1_2.json")
        after_shot = _latest(workspace)
        after_hashes = {
            int(record["frame_index"]): record["record_hash"]
            for record in after_shot["shards"]["actual_frames.json"]["records"]
        }
        changed_frames = {index for index in before_hashes if before_hashes[index] != after_hashes[index]}
        expected_frames = list(range(180, 300))
        validation.check(
            "shot mutation recomputes exactly the dependency closure",
            shot_mutation["closure"]["changed_inputs"] == ["shots"]
            and shot_mutation["closure"]["frames"] == expected_frames
            and shot_mutation["closure"]["segments"] == [1, 2, 3]
            and shot_mutation["closure"]["match"]
            and sorted(changed_frames) == expected_frames,
            {
                "closure": shot_mutation["closure"],
                "changed_record_hashes": len(changed_frames),
                "outside_closure_unchanged": all(
                    before_hashes[index] == after_hashes[index]
                    for index in before_hashes
                    if index not in set(expected_frames)
                ),
                "segment_summaries": shot_mutation["segment_summaries"],
            },
        )
        reset = evaluate_workspace(workspace)
        validation.check("reset to baseline before crash-recovery checks", reset["status"] == "ok", reset["status"])

        previous = _latest(workspace)
        crash = subprocess.run(
            [
                sys.executable,
                "-m",
                "evalworkbench.cli",
                "evaluate",
                "--workspace",
                str(workspace),
                "--crash-point",
                "before_manifest",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        recovery = recover_workspace(workspace)
        validation.check(
            "forced termination before manifest leaves a recoverable prior snapshot and reclaims orphans",
            crash.returncode == 92
            and recovery["latest_revision"] == previous["manifest"]["revision"]
            and recovery["orphan_count"] >= 1
            and not recovery["normal_exit_confirmed"],
            {"crash_returncode": crash.returncode, "recovery": recovery},
        )
        crash_committed = subprocess.run(
            [
                sys.executable,
                "-m",
                "evalworkbench.cli",
                "evaluate",
                "--workspace",
                str(workspace),
                "--crash-point",
                "after_manifest",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        recovery_committed = recover_workspace(workspace)
        validation.check(
            "forced termination after manifest recovers the newest committed snapshot without faking closure",
            crash_committed.returncode == 93
            and recovery_committed["latest_revision"] == previous["manifest"]["revision"] + 1
            and not recovery_committed["normal_exit_confirmed"],
            {"crash_returncode": crash_committed.returncode, "recovery": recovery_committed},
        )
        latest = _latest(workspace)
        try:
            evaluate_workspace(workspace, expected_revision=latest["manifest"]["revision"] + 100)
        except BusinessError as exc:
            stale_code = exc.code
        else:
            stale_code = None
        validation.check(
            "stale snapshot commit is rejected as E_SNAPSHOT_STALE",
            stale_code == "E_SNAPSHOT_STALE",
            {"code": stale_code, "current_revision": latest["manifest"]["revision"]},
        )
        preview_path = workspace / "preview_frame_000.ppm"
        preview = preview_workspace(workspace, 0, preview_path)
        validation.check(
            "frame decode playback preview is valid 640x360 RGB PPM",
            preview_path.read_bytes().startswith(b"P6\n640 360\n255\n") and preview["width"] == 640,
            preview,
        )
        report["observations"] = {
            "generation": generation_a,
            "carriers": carriers,
            "baseline": {
                "status": baseline["status"],
                "content_hash": baseline["content_hash"],
                "metric_ledger_digest": baseline["metric_ledger_digest"],
                "segment_summaries": baseline["segment_summaries"],
                "audio": baseline["audio"]["overall"],
                "equivalence": baseline["equivalence"],
                "carrier_claims": baseline["carrier_claims"],
            },
            "repeat": second["closure"],
            "cross_workspace_equivalence_digest": baseline["equivalence"]["report_digest"],
            "encoding_policy_closure": {
                "encoding_frames": len(encoding_frames),
                "encoding_segments": encoding_segments,
                "policy_frames": len(policy_frames),
                "policy_segments": policy_segments,
            },
            "binary_faults": fault_results,
            "audio_drift": {"overall": drift["audio"]["overall"], "segments": drift["audio"]["segments"]},
            "drop_and_timestamp": {
                "missing_units": dropped["missing_units"],
                "business_codes": sorted(_codes(dropped)),
                "audio_joint_timebase": dropped["audio"]["overall"],
            },
            "shot_closure": shot_mutation["closure"],
            "recovery_before_manifest": recovery,
            "recovery_after_manifest": recovery_committed,
            "preview": preview,
        }
        report["observed_business_codes"] = sorted(
            {
                *fault_expectations.values(),
                "E_AUDIO_DRIFT",
                "E_AUDIO_SAMPLE_RATE",
                "E_SCHEMA_VERSION",
                "E_DROPPED_FRAME",
                "E_TIMESTAMP_JUMP",
                "E_EQUIV_MISMATCH",
                "E_CLOSURE_MISMATCH",
                "E_SNAPSHOT_STALE",
            }
        )
    except Exception as exc:
        report["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["checks"] = validation.checks
        report["passed"] = validation.passed and "fatal_error" not in report
        report["duration_seconds"] = round(time.time() - started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        second_workspace_path = str(second_workspace)
        if not os.environ.get("KEEP_VALIDATION_ASSETS"):
            import shutil

            shutil.rmtree(second_workspace, ignore_errors=True)
    print(json.dumps({"passed": report["passed"], "checks": len(validation.checks), "report": str(args.report)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
