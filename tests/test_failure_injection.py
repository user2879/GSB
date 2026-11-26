from __future__ import annotations

from pathlib import Path

from hopledger.ledger import Ledger
from hopledger.proc import ProcessManager
from hopledger.runner import Runner
from hopledger.sample import make_sample_configs


def test_deterministic_forward_failure_is_a_completed_error_record(tmp_path: Path):
    bundle = make_sample_configs(tmp_path / "config", failure_ratio=1.0)
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    manager = ProcessManager(bundle, tmp_path / "logs")
    runner = Runner(bundle, ledger, manager=manager)
    try:
        manager.start_all()
        report = runner.run_batch("batch-0001", ["req-0001"], seq=1)
        summary = report["summaries"][0]
        assert summary["ok"] is False
        assert summary["primary_cause"] == "forward_failure_injected"

        conclusion = ledger.get_conclusion("req-0001")
        assert conclusion["ok"] is False
        assert conclusion["error"]["code"] == "E_FORWARD_INJECTED"
        assert conclusion["receipts"][0]["node_id"] == "fwd1"
        assert conclusion["receipts"][0]["failure_injected"] is True
        assert ledger.verify_content_digest()
    finally:
        manager.stop_all()
        ledger.close()
