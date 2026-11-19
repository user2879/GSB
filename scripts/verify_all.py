from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gsb.errors import GSBError
from gsb.generation import corrupt_sample_file, ensure_table_sample
from gsb.logging_store import StructuredLogger
from gsb.model import load_scenario
from gsb.session import Session, restore_snapshot_into
from gsb.snapshot import restore_snapshot
from gsb.util import atomic_write_text, canonical_json, sha256_file


def reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_tests() -> dict:
    process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    text = process.stdout + process.stderr
    tests_run = None
    for line in text.splitlines():
        if line.startswith("Ran ") and " tests " in line:
            tests_run = int(line.split()[1])
            break
    return {
        "command": f"{sys.executable} -m unittest discover -s tests -v",
        "returncode": process.returncode,
        "tests_run": tests_run,
        "output_tail": text.splitlines()[-12:],
    }


def scenario_result(spec_path: Path, workdir: Path, logger: StructuredLogger) -> tuple[Session, dict]:
    session = Session.from_spec_file(spec_path, workdir, logger=logger)
    result = session.run()
    return session, {
        "status": result.decision["outcome"],
        "parameter_version": session.parameter_version,
        "histogram_estimate": result.histogram_estimate["estimate_from_fixed"]
        if "estimate_from_fixed" in result.histogram_estimate
        else result.histogram_estimate["estimate_fixed"] / 1_000_000,
        "histogram_fixed": result.histogram_estimate["estimate_fixed"],
        "sampling_estimate": result.sampling_estimate["estimate_from_fixed"],
        "sampling_fixed": result.sampling_estimate["estimate_fixed"],
        "truth_rows": result.truth["row_count"],
        "interval": result.decision["interval"],
        "within_interval": result.decision["within_interval"],
        "selected_path": result.decision["selected_path"],
        "path_reason": result.decision["path_selection"]["reason"],
        "overlap_pair_count": result.histogram_estimate["overlap_pair_count"],
        "sample_hit_pairs": result.sampling_estimate["sample_hit_pairs"],
        "sample_hit_rate": result.sampling_estimate["sample_hit_rate"],
        "attribution_top": result.attribution["top_k"][:3],
        "analysis_digest": result.analysis_digest,
        "sample_files": {
            name: {
                "path": sample.path,
                "file_digest": sample.file_digest,
                "content_digest": sample.content_digest,
            }
            for name, sample in sorted(session.samples.items())
        },
    }


def main() -> int:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    work = ROOT / "var" / "verification"
    reset(work)
    logger = StructuredLogger(work / "logs" / "events.sqlite3", work / "logs" / "events.jsonl")
    tests = run_tests()
    checks: list[dict] = []

    skewed, skewed_result = scenario_result(ROOT / "examples" / "demo_skewed.json", work / "skewed", logger)
    float_session, float_result = scenario_result(ROOT / "examples" / "demo_float.json", work / "float", logger)
    miss_session, miss_result = scenario_result(ROOT / "examples" / "demo_estimate_miss.json", work / "miss", logger)
    checks.append({"name": "skewed_demo_both_paths_within_interval", "passed": skewed_result["within_interval"]})
    checks.append({"name": "float_demo_both_paths_within_interval", "passed": float_result["within_interval"]})
    miss_diagnostics = bool(
        miss_result["status"] == "E_EST"
        and (
            miss_result["attribution_top"]
            or miss_session.decision.get("dominant_deviation_bucket") is not None
        )
    )
    checks.append({"name": "E_EST_diagnostics_present", "passed": miss_diagnostics})
    checks.append({"name": "attribution_top_k_present", "passed": bool(skewed_result["attribution_top"])})

    deterministic = ensure_table_sample(load_scenario(ROOT / "examples" / "demo_skewed.json"), "orders", work / "det-a")
    deterministic_again = ensure_table_sample(
        load_scenario(ROOT / "examples" / "demo_skewed.json"), "orders", work / "det-b"
    )
    byte_equal = Path(deterministic.path).read_bytes() == Path(deterministic_again.path).read_bytes()
    deterministic_result = {
        "first_path": deterministic.path,
        "second_path": deterministic_again.path,
        "first_file_digest": deterministic.file_digest,
        "second_file_digest": deterministic_again.file_digest,
        "first_content_digest": deterministic.content_digest,
        "second_content_digest": deterministic_again.content_digest,
        "byte_equal": byte_equal,
        "sha256_recheck": sha256_file(deterministic.path) == deterministic.file_digest,
    }
    checks.append({"name": "npz_byte_identical_regeneration", "passed": byte_equal})
    checks.append(
        {
            "name": "npz_content_digest_identical_regeneration",
            "passed": deterministic.content_digest == deterministic_again.content_digest,
        }
    )

    snapshot_path = work / "snapshots" / "skewed_v1.json"
    snapshot_meta = skewed.save_snapshot(snapshot_path)
    evolution = skewed.evolve(buckets=20, sample_size=480, sample_seed=778899)
    evolved_result = {
        "old_parameter_version": evolution.transition["old_parameter_version"],
        "new_parameter_version": evolution.transition["new_parameter_version"],
        "affected_pair_count": evolution.transition["affected_pair_count"],
        "parameter_diff": evolution.transition["parameter_diff"],
        "stale_versions": skewed.stale_summary(),
        "new_analysis_digest": skewed.current.analysis_digest,
    }
    restored_session, replay = restore_snapshot(snapshot_path, work / "restore", logger=logger)
    replay_match = restored_session.current.analysis_digest == skewed_result["analysis_digest"]
    replace_ok = False
    post_restore_version = None
    if replay_match:
        replaced, transition = restore_snapshot_into(snapshot_path, work / "restore-into", current=skewed)
        replace_ok = (
            transition.get("version_changed") is True
            and replaced.parameter_version == 1
            and replaced.current.analysis_digest == skewed_result["analysis_digest"]
            and transition.get("stale_artifacts_excluded") is True
        )
        post_restore_version = replaced.evolve(buckets=20).transition["new_parameter_version"]
    checks.append({"name": "snapshot_replay_matches_original", "passed": replay_match})
    checks.append({"name": "snapshot_restore_replaces_current_state", "passed": replace_ok})
    checks.append(
        {
            "name": "parameter_version_moves_from_1_to_2",
            "passed": evolution.transition["new_parameter_version"] == 2,
        }
    )
    checks.append(
        {
            "name": "post_restore_evolution_does_not_reuse_old_version",
            "passed": post_restore_version == 3,
        }
    )

    anomalies: dict[str, dict] = {}
    anomaly_specs = {
        "missing_parameter_version": ROOT / "examples" / "anomalies" / "missing_parameter_version.json",
        "zero_buckets": ROOT / "examples" / "anomalies" / "zero_buckets.json",
        "invalid_spec": ROOT / "examples" / "anomalies" / "invalid_spec.json",
    }
    for name, path in anomaly_specs.items():
        try:
            load_scenario(path)
        except GSBError as error:
            anomalies[name] = error.to_dict()
    try:
        Session.from_spec_file(
            ROOT / "examples" / "anomalies" / "sample_size_exceeded.json",
            work / "anomaly-sample",
            logger=logger,
        ).run()
    except GSBError as error:
        anomalies["sample_size_exceeded"] = error.to_dict()
    checks.append(
        {
            "name": "invalid_spec_returns_E_SPEC",
            "passed": anomalies.get("invalid_spec", {}).get("code") == "E_SPEC",
        }
    )
    checks.append(
        {
            "name": "missing_version_returns_E_VERSION",
            "passed": anomalies.get("missing_parameter_version", {}).get("code") == "E_VERSION",
        }
    )
    checks.append(
        {
            "name": "zero_buckets_returns_E_BUCKET",
            "passed": anomalies.get("zero_buckets", {}).get("code") == "E_BUCKET",
        }
    )
    checks.append(
        {
            "name": "oversized_sample_returns_E_SAMPLE",
            "passed": anomalies.get("sample_size_exceeded", {}).get("code") == "E_SAMPLE",
        }
    )

    corruption_work = work / "corruption"
    corruption_session = Session.from_spec_file(
        ROOT / "examples" / "demo_skewed.json", corruption_work, logger=logger
    )
    corruption_target = corruption_session.samples[sorted(corruption_session.samples)[0]].path
    corruption_details = corrupt_sample_file(corruption_target)
    try:
        corruption_session.verify_sample_files()
        corruption_error = None
    except GSBError as error:
        corruption_error = error.to_dict()
        logger.error(error, "verification_corruption_injection")
    checks.append(
        {
            "name": "corrupted_npz_returns_E_SAMPLE",
            "passed": corruption_error is not None and corruption_error.get("code") == "E_SAMPLE",
        }
    )

    events = logger.read_events(100)
    checks.append(
        {
            "name": "structured_sqlite_log_has_events",
            "passed": bool(events) and (work / "logs" / "events.sqlite3").exists(),
        }
    )
    checks.append({"name": "unittest_suite_passed", "passed": tests["returncode"] == 0})

    all_passed = all(check["passed"] for check in checks)
    verification = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "all_passed": all_passed,
        "checks": checks,
        "unit_tests": tests,
        "scenarios": {
            "skewed": skewed_result,
            "float": float_result,
            "estimate_miss": miss_result,
        },
        "determinism": deterministic_result,
        "parameter_evolution": evolved_result,
        "snapshot": {
            "save": snapshot_meta,
            "replay": replay,
            "replay_matches_original": replay_match,
            "state_replacement_passed": replace_ok,
            "post_restore_evolution_version": post_restore_version,
        },
        "anomalies": anomalies,
        "corruption_injection": {
            "details": corruption_details,
            "error": corruption_error,
        },
        "structured_log_event_count": len(events),
        "structured_log_recent": events[:5],
    }
    atomic_write_text(reports / "verification.json", canonical_json(verification, pretty=True))
    atomic_write_text(reports / "demo_report.json", canonical_json(skewed_result, pretty=True))
    atomic_write_text(reports / "float_report.json", canonical_json(float_result, pretty=True))
    atomic_write_text(reports / "estimate_miss_report.json", canonical_json(miss_result, pretty=True))
    print(json.dumps({"all_passed": all_passed, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
