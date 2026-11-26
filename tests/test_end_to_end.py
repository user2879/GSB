import pytest

from cacheflow.acceptance import prepare_acceptance_files, run_acceptance
from cacheflow.errors import VerifyError
from cacheflow.ledger import Ledger
from cacheflow.replay import verify_ledger
from cacheflow.runner import Runner


def test_real_tcp_pipeline_and_ledger_replay(tmp_path):
    bundle = prepare_acceptance_files(tmp_path / "spec")
    work_dir = tmp_path / "work"
    with Runner(bundle, work_dir, rpc_timeout_seconds=10.0) as runner:
        result = runner.run()
    assert result["peaks"]["global"] <= bundle.global_concurrency
    assert result["peaks"]["nodes"]["node-a"] <= bundle.node("node-a").max_leases
    assert result["conclusion"]["uniqueness"]["unique_holding_ok"] is True

    verified = verify_ledger(bundle, work_dir / "cacheflow.sqlite3")
    assert verified["ok"] is True
    assert verified["unique_holding_ok"] is True
    assert verified["canonical_bytes"] == result["snapshot_bytes"]


def test_force_kill_unknown_request_recovery_and_bytewise_replay(tmp_path):
    report = run_acceptance(tmp_path / "acceptance")
    assert report["ok"] is True
    assert report["bytewise_repeat"] is True
    assert report["unique_holding_ok"] is True
    kill_step = report["steps"][1]
    assert kill_step["forced_returncode"] == -9
    assert kill_step["node_exit_report_promised"] is False
    recovery_step = report["steps"][2]
    assert recovery_step["status"] == "unknown_rerun_required"
    assert recovery_step["hold_ids"]
    assert report["capacity"]["node-c"]["used_slots"] == 3
    assert report["capacity"]["node-a"]["used_slots"] == 1


def test_verify_detects_snapshot_tampering(tmp_path):
    bundle = prepare_acceptance_files(tmp_path / "spec")
    work_dir = tmp_path / "work"
    with Runner(bundle, work_dir, rpc_timeout_seconds=10.0) as runner:
        runner.run()
    ledger_path = work_dir / "cacheflow.sqlite3"
    ledger = Ledger(ledger_path)
    with ledger.conn:
        ledger.conn.execute("UPDATE snapshots SET digest='0' WHERE snapshot_key='latest'")
    ledger.close()

    with pytest.raises(VerifyError) as exc:
        verify_ledger(bundle, ledger_path)
    assert exc.value.code == "E_VERIFY"
    assert exc.value.field == "digest"
