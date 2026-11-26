from __future__ import annotations

import json
import stat
from pathlib import Path

from hopledger.ledger import Ledger
from hopledger.proc import ProcessManager
from hopledger.runner import Runner
from hopledger.sample import make_sample_configs
from hopledger.snapshot import open_or_replace_snapshot


def test_real_loopback_chain_ledger_and_snapshot(tmp_path: Path):
    bundle = make_sample_configs(tmp_path / "config")
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    manager = ProcessManager(bundle, tmp_path / "logs")
    runner = Runner(bundle, ledger, manager=manager)
    try:
        manager.start_all()
        report = runner.run_all()
        assert report["ok"] is True
        assert len(report["request_ids"]) == 4
        assert report["max_observed_global"] == 2
        assert report["max_observed_hop"]["fwd1"] == 1
        assert report["max_observed_hop"]["rev2"] == 1
        assert report["max_observed_hop"]["origin"] == 1

        assert ledger.verify_content_digest()
        conclusion = ledger.get_conclusion("req-0001")
        assert conclusion["e2e_total_units"] == 6000
        attribution = ledger.get_attribution("req-0001")
        assert attribution["dominant_hop"] == "fwd1"

        snapshot_path = tmp_path / "session-snapshot.json"
        snapshot, replacement = open_or_replace_snapshot(
            snapshot_path, ledger, bundle.chain, bundle.diagnostic
        )
        assert replacement["replaced"] is False
        assert len(snapshot["requests"]) == 4

        snapshot_path.write_text(json.dumps({
            "snapshot_format_version": 1,
            "hop_set_version": 999,
            "scale_caliber_version": 999,
            "content_digest": "stale",
        }), encoding="utf-8")
        new_snapshot, replaced = open_or_replace_snapshot(
            snapshot_path, ledger, bundle.chain, bundle.diagnostic
        )
        assert replaced["replaced"] is True
        assert "hop_set_version_mismatch" in replaced["reasons"]
        backup = Path(replaced["backup"])
        assert backup.exists()
        assert stat.S_IMODE(backup.stat().st_mode) == 0o444
        assert len(new_snapshot["requests"]) == 4
    finally:
        manager.stop_all()
        ledger.close()
