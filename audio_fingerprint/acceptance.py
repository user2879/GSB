from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG, parse_config
from .demo import create_demo_workspace
from .engine import FingerprintEngine, deep_diff
from .manifest import load_json_file, validate_commands, validate_manifest
from .retrieval import rank_candidates, vote_candidate
from .store import PersistentStore
from .synth import generate_clip_file
from .workspace import apply_command_to_inputs


ROOT = Path(__file__).resolve().parent.parent


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _add_check(report: dict[str, Any], name: str, passed: bool, **detail: Any) -> bool:
    report["checks"].append({"name": name, "passed": bool(passed), **detail})
    return bool(passed)


def _count(step: dict[str, Any], product_type: str, status: str) -> int:
    return int(step.get("counts", {}).get(f"{product_type}:{status}", 0))


def make_tiny_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    seeds = [("tiny-alpha", "trk-tiny-alpha", 1, 3101, 22050, 16), ("tiny-beta", "trk-tiny-beta", 1, 3202, 44100, 8), ("tiny-gamma", "trk-tiny-gamma", 1, 3303, 11025, 24)]
    for clip_id, track_id, version, seed, rate, depth in seeds:
        spec = {
            "clip_id": clip_id,
            "track_id": track_id,
            "version": version,
            "role": "library",
            "seed": seed,
            "path": f"clips/{clip_id}.wav",
            "internal_sample_rate": 11025,
            "internal_sample_count": 4096,
            "source_sample_rate": rate,
            "bit_depth": depth,
            "declared_sample_rates": [11025, 22050, 44100],
            "declared_bit_depths": [8, 16, 24, 32],
        }
        clips.append(generate_clip_file(spec, data_dir))
    query = {
        "clip_id": "q-tiny-alpha",
        "track_id": "qtrk-tiny-alpha",
        "version": 1,
        "role": "query",
        "expected_track_id": "trk-tiny-alpha",
        "seed": 3101,
        "start_sample": 256,
        "excerpt_sample_count": 2048,
        "path": "queries/q-tiny-alpha.wav",
        "internal_sample_rate": 11025,
        "internal_sample_count": 4096,
        "source_sample_rate": 22050,
        "bit_depth": 16,
        "declared_sample_rates": [11025, 22050, 44100],
        "declared_bit_depths": [8, 16, 24, 32],
    }
    clips.append(generate_clip_file(query, data_dir))
    manifest = {"manifest_version": 1, "clips": clips}
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update({"candidate_limit": 3, "bucket_capacity": 5, "max_concurrency": 2})
    return manifest, config, data_dir


def run_acceptance(write_report: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "project_version": __version__,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
        },
        "checks": [],
    }
    try:
        workspace = ROOT / "sample_workspace"
        create_demo_workspace(workspace)
        data_dir = workspace / "data"
        state_root = _reset_dir(ROOT / "state" / "acceptance")

        manifest, manifest_issues_load = load_json_file(workspace / "manifest.json")
        raw_config, config_load_issues = load_json_file(workspace / "config.json")
        config, config_issues = parse_config(raw_config)
        valid_clips, manifest_issues = validate_manifest(manifest, data_dir)
        issue_codes = [item["code"] for item in config_issues + config_load_issues + manifest_issues_load + manifest_issues]
        expected_codes = ["E1001", "E1002", "E1003", "E1004", "E1005", "E2002"]
        _add_check(
            report,
            "validation_business_codes",
            issue_codes == expected_codes and len(valid_clips) == 9,
            actual_codes=issue_codes,
            expected_codes=expected_codes,
            valid_clip_count=len(valid_clips),
        )

        invalid_payload, _ = load_json_file(workspace / "invalid_commands.json")
        invalid_commands, command_issues = validate_commands(invalid_payload)
        command_codes = [item["code"] for item in command_issues]
        _add_check(
            report,
            "invalid_command_stream_continues",
            len(invalid_commands) == 1
            and command_codes == ["E3001", "E3002", "E3003", "E2003", "E2003"],
            command_codes=command_codes,
            valid_command_count=len(invalid_commands),
        )

        main_engine = FingerprintEngine(
            copy.deepcopy(manifest),
            copy.deepcopy(raw_config),
            data_dir,
            state_root / "main",
        )
        main_engine.start_run(run_id="acceptance-baseline")
        baseline_run = main_engine.run_until_complete()
        baseline_canonical = copy.deepcopy(main_engine.export_canonical())
        _add_check(
            report,
            "baseline_build_completed",
            baseline_run["completed"] is True and len(baseline_canonical["frames"]) == 657,
            status=baseline_run["status"],
            frames=len(baseline_canonical["frames"]),
            pairs=len(baseline_canonical["pairs"]),
            buckets=len(baseline_canonical["buckets"]),
        )

        norms = {item["clip_id"]: item for item in baseline_canonical["normalizations"]}
        frame_map = main_engine.snapshot["frames"]
        format_mismatches: list[dict[str, Any]] = []
        for index in range(95):
            left = frame_map[f"frame:c-alpha-v1:{index:06d}"]
            right = frame_map[f"frame:c-alpha-v2-fmt:{index:06d}"]
            for field in ("energy_q", "bands_q"):
                if left[field] != right[field]:
                    format_mismatches.append({"frame": index, "field": field})
        format_ok = (
            not format_mismatches
            and norms["c-alpha-v1"]["normalized_content_fingerprint"]
            == norms["c-alpha-v2-fmt"]["normalized_content_fingerprint"]
            and norms["c-alpha-v1"]["original_format_fingerprint"]
            != norms["c-alpha-v2-fmt"]["original_format_fingerprint"]
        )
        _add_check(
            report,
            "format_normalization_framewise_identity",
            format_ok,
            mismatches=format_mismatches[:5],
            left_format={k: norms["c-alpha-v1"][k] for k in ("original_sample_rate", "original_bit_depth", "resample_ratio")},
            right_format={k: norms["c-alpha-v2-fmt"][k] for k in ("original_sample_rate", "original_bit_depth", "resample_ratio")},
            normalized_fingerprint=norms["c-alpha-v1"]["normalized_content_fingerprint"],
        )

        query_checks: dict[str, Any] = {}
        query_ok = True
        for query_id, result in main_engine.snapshot["query_results"].items():
            expected = next(c.get("expected_track_id") for c in manifest["clips"] if c["clip_id"] == query_id)
            final = result["final_conclusion"]
            boundary = next((c for c in result["candidates"] if c["boundary_candidate"]), None)
            query_checks[query_id] = {
                "expected_track_id": expected,
                "actual_track_id": None if final is None else final["track_id"],
                "offset": None if final is None else final["authoritative_offset_frames"],
                "boundary": None if boundary is None else boundary["boundary_detail"],
            }
            query_ok = query_ok and final is not None and final["track_id"] == expected and boundary is not None
        alpha_arbitration = next(
            item for item in main_engine.snapshot["query_results"]["q-alpha-offset2"]["version_arbitrations"]
            if item["track_id"] == "trk-alpha"
        )
        _add_check(
            report,
            "voting_candidates_boundary_and_arbitration",
            query_ok
            and alpha_arbitration["authoritative_version"] in (1, 2)
            and alpha_arbitration["authoritative_offset_frames"] == 2
            and len(alpha_arbitration["covered_versions"]) == 1,
            queries=query_checks,
            alpha_arbitration=alpha_arbitration,
        )

        tie_candidate = {
            "hits": [
                {"offset_frames": 2, "strength_q": 100},
                {"offset_frames": -2, "strength_q": 100},
            ]
        }
        tied_vote = vote_candidate(tie_candidate, {"vote_fraction_bits": 8})
        synthetic_hits = [
            {"track_id": "trk-z", "version": 1, "clip_id": "z", "offset_frames": 0, "strength_q": 1},
            {"track_id": "trk-a", "version": 1, "clip_id": "a", "offset_frames": 0, "strength_q": 1},
        ] * 3
        ranked = rank_candidates(synthetic_hits, {"candidate_limit": 1})
        _add_check(
            report,
            "deterministic_tie_rules",
            tied_vote["main_offset_frames"] == -2
            and ranked[0]["track_id"] == "trk-a"
            and ranked[0]["boundary_candidate"] is True
            and ranked[0]["boundary_detail"]["hit_count_delta"] == 0,
            offset_tie_main=tied_vote["main_offset_frames"],
            candidate_tie_first=ranked[0]["track_id"],
            boundary=ranked[0]["boundary_detail"],
        )

        edit_payload, _ = load_json_file(workspace / "edit_commands.json")
        edit_commands, edit_validation_issues = validate_commands(edit_payload)
        step_reports: list[dict[str, Any]] = []
        for command in edit_commands:
            apply_issues = apply_command_to_inputs(main_engine.manifest, main_engine.config, command, data_dir)
            if apply_issues:
                raise RuntimeError(f"command apply failed: {apply_issues}")
            main_engine._context = None
            main_engine.start_run(command)
            main_engine.run_until_complete()
            receipt = main_engine.snapshot["last_receipt"]
            counts: dict[str, int] = {}
            for entry in receipt["ledger"]:
                key = f"{entry['product_type']}:{entry['status']}"
                counts[key] = counts.get(key, 0) + 1
            step_reports.append(
                {
                    "op": command["op"],
                    "command_id": command["command_id"],
                    "batch_count": receipt["batch_count"],
                    "hit": receipt["reuse_hit_count"],
                    "rebuilt": receipt["rebuilt_count"],
                    "counts": counts,
                    "invalidation_count": receipt["invalidation_count"],
                }
            )

        frame_threshold_step = step_reports[3]
        capacity_step = step_reports[4]
        gear_step = step_reports[5]
        precise_ok = (
            _count(step_reports[2], "spec_frame", "rebuilt") == 0
            and _count(frame_threshold_step, "peak_set", "rebuilt") == 1
            and _count(frame_threshold_step, "fingerprint_pair", "rebuilt") > 0
            and _count(frame_threshold_step, "inverted_bucket", "rebuilt") > 0
            and _count(frame_threshold_step, "inverted_bucket", "reuse_hit") > 0
            and _count(capacity_step, "fingerprint_pair", "rebuilt") == 0
            and _count(capacity_step, "inverted_bucket", "rebuilt") > 0
            and _count(gear_step, "spec_frame", "rebuilt") == 0
            and _count(gear_step, "peak_set", "rebuilt") == 0
            and _count(gear_step, "fingerprint_pair", "reuse_hit") == 0
            and _count(gear_step, "fingerprint_pair", "rebuilt") > 0
        )
        _add_check(report, "precise_invalidation_and_reuse", precise_ok, steps=step_reports)

        incremental_canonical = main_engine.export_canonical()
        full_engine = FingerprintEngine(
            copy.deepcopy(main_engine.manifest),
            copy.deepcopy(main_engine.config),
            data_dir,
            state_root / "full-rebuild",
        )
        full_engine.start_run(run_id="blank-full-rebuild")
        full_engine.run_until_complete()
        full_canonical = full_engine.export_canonical()
        differences = deep_diff(incremental_canonical, full_canonical)
        _add_check(
            report,
            "incremental_equals_blank_full_rebuild",
            not differences,
            incremental_fingerprint=main_engine.canonical_fingerprint(),
            full_fingerprint=full_engine.canonical_fingerprint(),
            difference_count=len(differences),
            first_differences=differences[:20],
        )

        budget_manifest = copy.deepcopy(manifest)
        budget_config = copy.deepcopy(raw_config)
        create_demo_workspace(workspace)
        budget_config.update(
            {
                "batch_sample_budget": 2560,
                "clip_sample_quota": 768,
                "max_concurrency": 1,
            }
        )
        budget_engine = FingerprintEngine(
            budget_manifest,
            budget_config,
            data_dir,
            state_root / "budget",
            serial=True,
        )
        budget_engine.start_run(run_id="budget-recovery")
        first_batch = budget_engine.run_next_batch()
        first_selected = first_batch["batch"]["selected_product_ids"]
        frame_todos = [todo for todo in budget_engine.snapshot["todos"] if todo.get("kind") == "frame_processing"]
        first_pending = min(frame_todos, key=lambda todo: (todo["track_id"], todo["version"], todo["clip_id"], todo["frame_index"]))
        second_batch = budget_engine.run_next_batch()
        second_selected = second_batch["batch"]["selected_product_ids"]
        budget_final = budget_engine.run_until_complete()
        budget_canonical = budget_engine.export_canonical()
        budget_differences = deep_diff(baseline_canonical, budget_canonical)
        event_codes = {event["code"] for event in first_batch["batch"]["events"]}
        budget_ok = (
            first_batch["completed"] is False
            and len(first_selected) == 10
            and first_selected[0] == "frame:c-alpha-v1:000000"
            and first_pending["product_id"] == "frame:c-alpha-v1:000004"
            and second_selected[0] == "frame:c-alpha-v1:000004"
            and "B4001" in event_codes
            and "B4002" in event_codes
            and budget_final["completed"] is True
            and not budget_differences
        )
        _add_check(
            report,
            "budget_quota_todo_priority_and_recovery",
            budget_ok,
            first_selected_count=len(first_selected),
            first_selected_first_four=first_selected[:4],
            first_pending=first_pending,
            second_first_selected=second_selected[0],
            event_codes=sorted(event_codes),
            batch_count=budget_engine.snapshot["active_run"]["batch_count"],
            difference_count=len(budget_differences),
            first_differences=budget_differences[:10],
        )

        tiny_root = state_root / "tiny"
        tiny_manifest, tiny_config, tiny_data = make_tiny_inputs(tiny_root / "audio")
        concurrency_canonicals: dict[str, str] = {}
        concurrency_diffs: list[str] = []
        for mode, workers, serial in [("serial", 1, True), ("workers2", 2, False), ("workers4", 4, False)]:
            mode_config = copy.deepcopy(tiny_config)
            mode_config["max_concurrency"] = workers
            mode_config["batch_sample_budget"] = 768
            mode_config["clip_sample_quota"] = 100000
            engine = FingerprintEngine(
                copy.deepcopy(tiny_manifest),
                mode_config,
                tiny_data,
                state_root / f"concurrency-{mode}",
                serial=serial,
            )
            engine.start_run(run_id=f"concurrency-{mode}")
            engine.run_until_complete()
            concurrency_canonicals[mode] = engine.canonical_fingerprint()
            if not concurrency_canonicals.get("serial"):
                concurrency_canonicals["serial"] = concurrency_canonicals[mode]
        concurrency_ok = len(set(concurrency_canonicals.values())) == 1
        _add_check(
            report,
            "threadpool_serial_field_equivalence",
            concurrency_ok,
            fingerprints=concurrency_canonicals,
            differences=concurrency_diffs,
            max_configured_workers=4,
        )

        signal_engine = FingerprintEngine(
            copy.deepcopy(tiny_manifest),
            copy.deepcopy(tiny_config) | {"batch_sample_budget": 512, "max_concurrency": 2},
            tiny_data,
            state_root / "signal",
        )
        signal_engine.install_signal_handlers()
        os.kill(os.getpid(), signal.SIGTERM)
        signal_handler_ok = signal_engine.stop_requested and signal_engine.signal_received == int(signal.SIGTERM)
        signal_engine.stop_requested = False
        signal_engine.signal_received = None

        import audio_fingerprint.engine as engine_module

        original_ordered_map = engine_module.ordered_map

        def stop_after_allocation(items: list[Any], worker: Any, max_concurrency: int, serial: bool = False) -> list[Any]:
            results = original_ordered_map(items, worker, max_concurrency, serial)
            signal_engine.request_stop(int(signal.SIGTERM))
            return results

        engine_module.ordered_map = stop_after_allocation
        try:
            signal_engine.start_run(run_id="signal-batch")
            signal_result = signal_engine.run_next_batch()
        finally:
            engine_module.ordered_map = original_ordered_map
        graceful_ok = signal_result["status"] == "graceful_stop_after_committed_batch"
        recovered = FingerprintEngine(
            copy.deepcopy(tiny_manifest),
            copy.deepcopy(tiny_config),
            tiny_data,
            state_root / "signal",
        )
        recovery_run = recovered.run_until_complete()
        signal_canonical = recovered.canonical_fingerprint()
        serial_engine = FingerprintEngine(
            copy.deepcopy(tiny_manifest),
            copy.deepcopy(tiny_config),
            tiny_data,
            state_root / "concurrency-serial",
            serial=True,
        )
        # Existing committed serial state is reused; export remains canonical.
        serial_canonical = serial_engine.canonical_fingerprint()
        _add_check(
            report,
            "signal_completes_batch_and_snapshot_recovery",
            signal_handler_ok
            and graceful_ok
            and recovery_run["completed"] is True
            and signal_canonical == serial_canonical,
            signal_handler_ok=signal_handler_ok,
            signal_result=signal_result.get("status"),
            recovery_status=recovery_run.get("status"),
            recovered_fingerprint=signal_canonical,
            serial_fingerprint=serial_canonical,
        )

        crash_dir = state_root / "crash-store"
        crash_store = PersistentStore(crash_dir)
        crash_store.commit_snapshot({"global_version": 1, "ok": True})
        (crash_dir / "snapshot.json.tmp").write_text("{partial-json", encoding="utf-8")
        reloaded = PersistentStore(crash_dir).load_snapshot()
        _add_check(
            report,
            "forced_termination_committed_snapshot_only",
            reloaded is not None and reloaded.get("global_version") == 1 and reloaded.get("ok") is True,
            reloaded_version=None if reloaded is None else reloaded.get("global_version"),
            stale_temp_ignored=True,
        )

        bucket_eviction_count = sum(len(bucket.get("evicted_entries", [])) for bucket in baseline_canonical["buckets"])
        _add_check(
            report,
            "bucket_eviction_records_present",
            bucket_eviction_count > 0,
            evicted_entries=bucket_eviction_count,
            bucket_count=len(baseline_canonical["buckets"]),
        )

    except Exception:
        report["fatal_error"] = traceback.format_exc()

    report["acceptance_passed"] = all(check["passed"] for check in report["checks"]) and "fatal_error" not in report
    if write_report:
        artifacts = ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        report_path = artifacts / "verification_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_path = artifacts / "verification_summary.txt"
        lines = [
            f"acceptance_passed={report['acceptance_passed']}",
            f"python={report['environment']['python'].split()[0]} numpy={report['environment']['numpy']}",
        ]
        for check in report["checks"]:
            lines.append(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}")
        if "fatal_error" in report:
            lines.append(report["fatal_error"])
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run_acceptance()
    print(json.dumps({"acceptance_passed": report["acceptance_passed"], "checks": [(c["name"], c["passed"]) for c in report["checks"]]}, ensure_ascii=False, indent=2))
    if "fatal_error" in report:
        print(report["fatal_error"], file=sys.stderr)
    return 0 if report["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
