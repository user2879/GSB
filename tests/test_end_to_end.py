from __future__ import annotations

import copy
import json
import signal
from pathlib import Path

import pytest

from inference_attribution.errors import DiagnosticError
from inference_attribution.invalidation import assert_replay_consistent
from inference_attribution.ledger import Ledger
from inference_attribution.planner import build_plan
from inference_attribution.spec import load_merge_rules, load_requests, load_workers
from helpers import SAMPLES, cli_json, dynamic_workers, run_cli


def test_forced_termination_recovery_equals_full_run(tmp_path: Path):
    workers_path = tmp_path / "workers.json"
    dynamic_workers(workers_path)
    workers = load_workers(workers_path)
    requests = load_requests(SAMPLES / "requests.json", workers)
    rules = load_merge_rules(SAMPLES / "merge_rules.json")
    plan = build_plan(workers, requests)

    common = (
        "--requests",
        str(SAMPLES / "requests.json"),
        "--workers",
        str(workers_path),
        "--rules",
        str(SAMPLES / "merge_rules.json"),
    )
    full_path = tmp_path / "full.sqlite"
    full_result = run_cli("run", *common, "--ledger", str(full_path))
    assert full_result.returncode == 0, full_result.stderr
    full_json = cli_json(full_result)
    assert full_json["progress_cursor"] == 3
    assert full_json["max_global_running"] <= rules.global_concurrency
    assert all(
        full_json["max_worker_running"][worker_id] <= worker.concurrency_quota
        for worker_id, worker in workers.by_id().items()
    )

    crash_path = tmp_path / "crash.sqlite"
    crash_result = run_cli(
        "run", *common, "--ledger", str(crash_path), "--crash-after", "1"
    )
    assert crash_result.returncode in {137, -signal.SIGKILL}
    assert crash_result.stdout == ""

    with Ledger(crash_path) as partial:
        recovery = partial.recovery_report()
        assert len(recovery.completed_batch_ids) == 1
        assert len(recovery.unknown_batch_ids) == 2
        assert recovery.progress_cursor == 1
        assert recovery.requires_rerun is True
        assert recovery.interrupted_process_exit_report is None
        assert "exit report" in recovery.to_dict()["note"]

    resume_result = run_cli("run", *common, "--ledger", str(crash_path))
    assert resume_result.returncode == 0, resume_result.stderr
    resume_json = cli_json(resume_result)
    assert resume_json["progress_cursor"] == 3
    assert set(resume_json["skipped_batch_ids"]) == set(
        recovery.completed_batch_ids
    )

    with Ledger(full_path) as full, Ledger(crash_path) as recovered:
        full_rows = full.rows()
        recovered_rows = recovered.rows()
        assert len(full_rows) == len(recovered_rows) == 3
        assert_replay_consistent(plan, full_rows, recovered_rows)
        assert {
            row["batch_id"]: row["receipt_digest"] for row in full_rows
        } == {
            row["batch_id"]: row["receipt_digest"] for row in recovered_rows
        }
        assert recovered.recovery_report().unknown_batch_ids == ()

    expected_with_bad_digest = copy.deepcopy(full_rows)
    expected_with_bad_digest[0]["receipt_digest"] = "0" * 64
    with pytest.raises(DiagnosticError) as caught:
        assert_replay_consistent(plan, expected_with_bad_digest, recovered_rows)
    assert caught.value.code == "E_REPLAY"
    assert caught.value.object == full_rows[0]["batch_id"]
    assert caught.value.field == "receipt_digest"
