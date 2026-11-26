from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hopledger.errors import HopLedgerError, ReplayError
from hopledger.ledger import Ledger
from hopledger.proc import ProcessManager
from hopledger.replay import compare_result_maps
from hopledger.runner import Runner
from hopledger.sample import make_sample_configs


def test_killed_hop_recovers_from_committed_batches_and_replays(tmp_path: Path):
    root = tmp_path
    full_bundle = make_sample_configs(root / "full" / "config")
    full_ledger = Ledger(root / "full" / "ledger.sqlite3")
    full_manager = ProcessManager(full_bundle, root / "full" / "logs")
    full_runner = Runner(full_bundle, full_ledger, manager=full_manager)
    try:
        full_manager.start_all()
        full_runner.run_all()
        full_results = full_ledger.all_results()
    finally:
        full_manager.stop_all()
        full_ledger.close()

    recovery_bundle = make_sample_configs(root / "recovery" / "config")
    recovery_ledger = Ledger(root / "recovery" / "ledger.sqlite3")
    recovery_manager = ProcessManager(recovery_bundle, root / "recovery" / "logs")
    recovery_runner = Runner(recovery_bundle, recovery_ledger, manager=recovery_manager)
    request_ids = recovery_bundle.request_ids
    try:
        recovery_manager.start_all()
        recovery_runner.run_batch("batch-0001", request_ids[:2], seq=1)

        recovery_manager.kill_node("rev2")
        with pytest.raises(HopLedgerError):
            recovery_runner.run_batch("batch-0002", request_ids[2:], seq=2)

        assert recovery_ledger.committed_request_ids() == request_ids[:2]
        unknown_batch_ids = recovery_ledger.mark_unknown_batches()
        assert unknown_batch_ids == ["batch-0002"]
        unknown = recovery_ledger.unknown_batches()
        assert unknown[0]["status"] == "unknown"
        assert unknown[0]["request_ids"] == request_ids[2:]

        recovery_manager.restart_node("rev2")
        reports = recovery_runner.rerun_unknown_batches()
        assert [report["batch_id"] for report in reports] == ["batch-0002"]
        assert recovery_ledger.committed_request_ids() == request_ids
        assert recovery_ledger.verify_content_digest()

        comparison = compare_result_maps(full_results, recovery_ledger.all_results())
        assert comparison["ok"] is True
        assert comparison["request_count"] == 4
        assert comparison["receipt_count"] == 12
    finally:
        recovery_manager.stop_all()
        recovery_ledger.close()


def test_replay_difference_is_located_with_e_replay(tmp_path: Path):
    bundle = make_sample_configs(tmp_path / "config")
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    manager = ProcessManager(bundle, tmp_path / "logs")
    runner = Runner(bundle, ledger, manager=manager)
    try:
        manager.start_all()
        runner.run_all()
        results = ledger.all_results()
        mutated = copy.deepcopy(results)
        mutated["req-0001"]["conclusion"]["nodes"][0]["scale_factor"] = 999

        with pytest.raises(ReplayError) as exc_info:
            compare_result_maps(results, mutated)
        error = exc_info.value
        assert error.code == "E_REPLAY"
        assert error.details["request_id"] == "req-0001"
        assert error.details["hop_id"] == "fwd1"
        assert error.details["field"] == "scale_factor"
    finally:
        manager.stop_all()
        ledger.close()
