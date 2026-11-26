from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from hopledger.errors import HopLedgerError
from hopledger.ledger import Ledger
from hopledger.proc import ProcessManager
from hopledger.replay import compare_result_maps
from hopledger.runner import Runner
from hopledger.sample import make_sample_configs
from hopledger.snapshot import open_or_replace_snapshot


def _full_run(root: Path) -> dict:
    config_dir = root / "full" / "config"
    bundle = make_sample_configs(config_dir)
    ledger = Ledger(root / "full" / "ledger.sqlite3")
    manager = ProcessManager(bundle, root / "full" / "logs")
    runner = Runner(bundle, ledger, manager=manager)
    try:
        manager.start_all()
        report = runner.run_all()
        return {"report": report, "results": ledger.all_results()}
    finally:
        manager.stop_all()
        ledger.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(tempfile.mkdtemp(prefix="hopledger-acceptance-", dir="artifacts")))
    parser.add_argument("--report", type=Path, default=Path("artifacts/acceptance-report.json"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    full = _full_run(args.root)
    bundle = make_sample_configs(args.root / "recovered" / "config")
    ledger = Ledger(args.root / "recovered" / "ledger.sqlite3")
    manager = ProcessManager(bundle, args.root / "recovered" / "logs")
    runner = Runner(bundle, ledger, manager=manager)
    request_ids = bundle.request_ids
    recovery_report: dict = {}
    try:
        manager.start_all()
        runner.run_batch("batch-0001", request_ids[:2], seq=1)
        manager.kill_node("rev2")
        interrupted_code = None
        try:
            runner.run_batch("batch-0002", request_ids[2:], seq=2)
        except HopLedgerError as exc:
            interrupted_code = exc.code
        marked_unknown = ledger.mark_unknown_batches()
        manager.restart_node("rev2")
        replayed = runner.rerun_unknown_batches()
        replay = compare_result_maps(full["results"], ledger.all_results())

        chain_path = bundle.config_dir / "chain.json"
        chain_data = json.loads(chain_path.read_text(encoding="utf-8"))
        chain_data["hops"][0]["config_version"] = 2
        chain_path.write_text(json.dumps(chain_data), encoding="utf-8")
        runner.reload_bundle()
        manager.restart_node("fwd1")
        plan = runner.invalidation_plan()
        evolution = runner.apply_invalidation(plan)
        snapshot, replacement = open_or_replace_snapshot(
            args.root / "snapshot.json", ledger, bundle.chain, bundle.diagnostic
        )
        recovery_report = {
            "ok": True,
            "root": str(args.root),
            "interrupted_error_code": interrupted_code,
            "marked_unknown_batches": marked_unknown,
            "replayed_batches": [item["batch_id"] for item in replayed],
            "replay": replay,
            "evolution": evolution,
            "snapshot_replacement": replacement,
            "ledger_digest_valid": ledger.verify_content_digest(),
            "max_observed_global": runner.max_observed_global,
            "max_observed_hop": runner.max_observed_hop,
            "requests": [
                {
                    "request_id": request_id,
                    "e2e_total_units": ledger.get_conclusion(request_id)["e2e_total_units"],
                    "dominant_hop": ledger.get_attribution(request_id)["dominant_hop"],
                }
                for request_id in ledger.committed_request_ids()
            ],
        }
    finally:
        manager.stop_all()
        ledger.close()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(recovery_report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(recovery_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
