#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from streamparse.app import IncrementalApp
from streamparse.engine import full_decode, semantic_payload
from streamparse.errors import E_DECODE_PARAM_INVALID, E_INPUT_INVALID, E_RESPONSE_BUDGET_PENDING
from streamparse.loader import load_config, load_dataset, load_extra_scores
from streamparse.reporting import write_report_atomic
from streamparse.snapshot import SnapshotManager


def make_check(name: str, passed: bool, details: Any = None) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def main() -> int:
    started = time.monotonic()
    sample_dir = ROOT / "sample_data"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, Any]] = []

    dataset, config, injections, load_failures = load_dataset(sample_dir)
    initial_manager = SnapshotManager(artifacts / "snapshots_initial", version_provider=lambda: 0)
    initial_report = full_decode(
        dataset,
        config,
        injections=injections,
        version=0,
        initial_failures=load_failures,
        snapshot_manager=initial_manager,
    )
    write_report_atomic(artifacts / "diagnostic_report.json", initial_report)

    injected = {item["id"]: item for item in initial_report["injection_report"]["items"]}
    expected = {
        "INJ-001-NONFINITE": E_INPUT_INVALID,
        "INJ-002-FRAME-COUNT": E_INPUT_INVALID,
        "INJ-003-OVERLAP": E_INPUT_INVALID,
        "INJ-004-ZERO-LOOKAHEAD": E_DECODE_PARAM_INVALID,
    }
    checks.append(
        make_check(
            "four_injections_hit_and_map_codes",
            all(injected.get(key, {}).get("hit_count") == 1 and injected[key].get("observed_business_code") == code for key, code in expected.items()),
            {key: injected.get(key) for key in expected},
        )
    )
    sessions = {item["session_id"]: item for item in initial_report["sessions"]}
    checks.append(
        make_check(
            "failed_shard_isolated_other_shards_continue",
            sessions["S-INJ-SHARD"]["status"] == "partial" and sessions["S-INJ-SHARD"]["frame_count"] > 0 and sessions["S01"]["status"] == "ok",
            {
                "S-INJ-SHARD": {
                    "status": sessions["S-INJ-SHARD"]["status"],
                    "frame_count": sessions["S-INJ-SHARD"]["frame_count"],
                },
                "S01": {"status": sessions["S01"]["status"]},
            },
        )
    )
    s01 = sessions["S01"]
    checks.append(
        make_check(
            "overlap_ownership_later_shard_wins",
            [(item["global_frame"], item["winning_shard_seq"], item["overwritten_shard_seq"]) for item in s01["ownership_conflicts"]]
            == [(3, 1, 0), (4, 1, 0), (5, 2, 1), (6, 2, 1)],
            s01["ownership_conflicts"],
        )
    )
    checks.append(
        make_check(
            "smoothing_frames_marked",
            sessions["S02"]["smoothed_frames"] == [4],
            {"smoothed_frames": sessions["S02"]["smoothed_frames"]},
        )
    )

    serial_config = replace(config, thread_pool_concurrency=1)
    parallel_config = replace(config, thread_pool_concurrency=4)
    serial_report = full_decode(dataset, serial_config, injections=injections, initial_failures=load_failures)
    parallel_report = full_decode(dataset, parallel_config, injections=injections, initial_failures=load_failures)
    checks.append(
        make_check(
            "serial_parallel_semantic_equality",
            semantic_payload(serial_report) == semantic_payload(parallel_report)
            and serial_report["failures"] == parallel_report["failures"],
            {
                "serial_workers": serial_report["thread_pool"]["max_workers"],
                "parallel_workers": parallel_report["thread_pool"]["max_workers"],
            },
        )
    )

    with tempfile.TemporaryDirectory(prefix="streamparse-acceptance-") as temp_raw:
        temp = Path(temp_raw)
        app_config = replace(
            config,
            response_budget_ms=5000,
            snapshot_root=str(temp / "snapshots"),
            report_path=str(temp / "diagnostic_report.json"),
        )
        app = IncrementalApp(sample_dir, workspace_root=temp, config=app_config)
        try:
            append_rows = load_extra_scores(sample_dir / "extras" / "append_scores.jsonl", app.dataset.vocab)
            replace_rows = load_extra_scores(sample_dir / "extras" / "replace_scores.jsonl", app.dataset.vocab)
            new_rows = load_extra_scores(sample_dir / "extras" / "new_session_scores.jsonl", app.dataset.vocab)
            commands = [
                {"cmd": "set_strategy", "session_id": "S01", "strategy": "greedy"},
                {"cmd": "set_strategy", "session_id": "S01", "strategy": "lookahead"},
                {"cmd": "set_param", "session_id": "S01", "key": "lookahead_frames", "value": 2},
                {"cmd": "set_param", "session_id": "S01", "key": "overlap_frames", "value": 1},
                {
                    "cmd": "append_shard",
                    "session_id": "S01",
                    "shard_seq": 3,
                    "start_frame": 8,
                    "end_frame": 11,
                    "scores": append_rows,
                },
                {
                    "cmd": "replace_shard",
                    "session_id": "S01",
                    "shard_seq": 1,
                    "start_frame": 3,
                    "end_frame": 6,
                    "scores": replace_rows,
                },
                {
                    "cmd": "add_session",
                    "session_id": "S05",
                    "shards": [
                        {
                            "shard_seq": 0,
                            "start_frame": 0,
                            "end_frame": 4,
                            "scores": new_rows[:5],
                        }
                    ],
                },
                {"cmd": "delete_session", "session_id": "S03"},
                {"cmd": "set_param", "key": "lm_weight", "value": 0.7},
                {"cmd": "set_strategy", "strategy": "greedy"},
            ]
            edit_results = []
            all_equal = True
            for command in commands:
                response = app.execute(command)
                app.wait()
                canonical = full_decode(
                    app.dataset,
                    app.config,
                    overrides=app.overrides,
                    injections=app.injections,
                    version=app.version,
                    initial_failures=app.load_failures,
                )
                equal = semantic_payload(app.latest_report) == semantic_payload(canonical)
                all_equal = all_equal and equal
                edit_results.append(
                    {
                        "command": command["cmd"],
                        "status": response["status"],
                        "version": app.version,
                        "incremental_equals_full": equal,
                    }
                )
            checks.append(
                make_check(
                    "incremental_equals_full_after_every_edit",
                    all_equal,
                    edit_results,
                )
            )
            write_report_atomic(artifacts / "diagnostic_report_incremental.json", app.latest_report)
        finally:
            app.close()

        budget_config = replace(
            config,
            response_budget_ms=20,
            test_compute_delay_ms=180,
            snapshot_root=str(temp / "budget_snapshots"),
            report_path=str(temp / "budget_report.json"),
        )
        budget_app = IncrementalApp(sample_dir, workspace_root=temp, config=budget_config)
        try:
            started_budget = time.monotonic()
            pending = budget_app.execute(
                {"cmd": "set_strategy", "session_id": "S01", "strategy": "greedy"}
            )
            elapsed_ms = (time.monotonic() - started_budget) * 1000.0
            latest_summary_available = bool(pending["summary"]["sessions"])
            completed = budget_app.execute({"cmd": "wait"})
            checks.append(
                make_check(
                    "budget_pending_returns_latest_summary_then_completes",
                    pending["status"] == "pending"
                    and E_RESPONSE_BUDGET_PENDING in pending["business_codes"]
                    and elapsed_ms < 80.0
                    and latest_summary_available
                    and completed["status"] == "ok",
                    {
                        "elapsed_ms": round(elapsed_ms, 3),
                        "budget_ms": budget_config.response_budget_ms,
                        "pending_version": pending["pending_version"],
                        "latest_completed_version_at_return": pending["latest_completed_version"],
                        "final_status": completed["status"],
                    },
                )
            )
        finally:
            budget_app.close()

    snapshot_manifest_path = artifacts / "snapshots_initial" / "current" / "manifest.json"
    manifest = (
        json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
        if snapshot_manifest_path.exists()
        else {}
    )
    checks.append(
        make_check(
            "atomic_snapshot_manifest_has_version_hash_and_cursor",
            manifest.get("stage") == "final"
            and manifest.get("global_version") == 0
            and bool(manifest.get("config_hash"))
            and isinstance(manifest.get("completed_frame_cursor"), dict)
            and (artifacts / "snapshots_initial" / "current" / "decode.json").exists(),
            {
                "manifest_path": str(snapshot_manifest_path.relative_to(ROOT)),
                "global_version": manifest.get("global_version"),
                "config_hash": manifest.get("config_hash"),
                "cursor_sessions": sorted((manifest.get("completed_frame_cursor") or {}).keys()),
            },
        )
    )

    acceptance = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.system(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "numpy": np.__version__,
        },
        "sample": json.loads((sample_dir / "sample_manifest.json").read_text(encoding="utf-8")),
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "test_count": len(checks),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "artifacts": {
            "diagnostic_report": "artifacts/diagnostic_report.json",
            "diagnostic_report_incremental": "artifacts/diagnostic_report_incremental.json",
            "snapshot_manifest": "artifacts/snapshots_initial/current/manifest.json",
            "acceptance_report": "artifacts/acceptance_report.json",
        },
    }
    write_report_atomic(artifacts / "acceptance_report.json", acceptance)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
