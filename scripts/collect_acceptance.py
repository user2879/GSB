from __future__ import annotations

import json
import platform
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sharddiag.ledger import Ledger  # noqa: E402
from sharddiag.spec import load_spec  # noqa: E402


def load_json(relative: str):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    spec = load_spec(ROOT / "config/spec.sample.json")
    run = load_json("artifacts/reports/run-report.json")
    invalidation = load_json("artifacts/reports/invalidation.json")
    evolution = load_json("artifacts/reports/evolution.json")
    cancel = load_json("artifacts/reports/cancel-report.json")
    recovered = load_json("artifacts/crash-demo/recovered-report.json")

    fresh = Ledger.open(ROOT / "artifacts/ledger/sample.sqlite3", expected_spec_digest=spec.spec_digest)
    fresh_digest = fresh.canonical_accounting_summary()["ledger_summary_digest"]
    fresh.close()
    crashed = Ledger.open(
        ROOT / "artifacts/crash-demo/crash.sqlite3", expected_spec_digest=spec.spec_digest
    )
    crash_digest = crashed.canonical_accounting_summary()["ledger_summary_digest"]
    recovery_batches = [
        batch["payload"] for batch in crashed.batches() if batch["kind"] == "recovery_reconcile"
    ]
    crashed.close()

    test_text = (ROOT / "artifacts/reports/test-results.txt").read_text(encoding="utf-8")
    summary_line = next((line for line in test_text.splitlines() if line.startswith("Ran ")), "")
    tests_ok = test_text.rstrip().endswith("OK")

    try:
        import tkinter

        root = tkinter.Tk()
        tk_version = root.tk.call("info", "patchlevel")
        root.destroy()
        tk_status = f"ok {tk_version}"
    except Exception as exc:  # noqa: BLE001
        tk_status = f"unavailable: {type(exc).__name__}: {exc}"

    summary = {
        "ok": bool(
            tests_ok
            and run
            and run.get("ok")
            and invalidation
            and invalidation.get("ok")
            and evolution
            and evolution.get("ok")
            and recovered
            and recovered.get("ok")
            and fresh_digest == crash_digest
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sqlite": sqlite3.sqlite_version,
            "tkinter": tk_status,
        },
        "tests": {"summary": summary_line, "ok": tests_ok},
        "spec_digest": spec.spec_digest,
        "fresh_ledger_summary_digest": fresh_digest,
        "crash_recovered_ledger_summary_digest": crash_digest,
        "digests_match": fresh_digest == crash_digest,
        "crash_recovery_batches": recovery_batches,
        "water_error_attribution": run["water_report"]["error_attribution"] if run else None,
        "global_budget_error": run["water_report"]["global_budget_aggregation"] if run else None,
        "invalid_shard_ids": invalidation["propagation"]["invalid_shard_ids"] if invalidation else None,
        "unaffected_shard_ids": invalidation["propagation"]["unaffected_shard_ids"] if invalidation else None,
        "invalidation_verification": invalidation["verification"] if invalidation else None,
        "water_invalidation_verification": invalidation.get("water_verification") if invalidation else None,
        "evolution": {
            "ok": evolution["ok"] if evolution else False,
            "initial_used_units": evolution["initial_used_units"] if evolution else None,
            "final_active_used_units": evolution["final_active_used_units"] if evolution else None,
            "cross_generation_code": (
                evolution["cross_generation_renewal_error"] or {}
            ).get("code")
            if evolution
            else None,
            "ownership_identity_unchanged": evolution["ownership_identity_unchanged"] if evolution else None,
        },
        "cancel_freeze": {
            "ok_expected_false": cancel is not None and cancel.get("ok") is False,
            "s06_status": cancel["shards"]["s06_c_aggregate"]["status"] if cancel else None,
            "s07_missing": cancel["shards"]["s07_sink"]["missing_upstream_ids"] if cancel else None,
        },
    }
    output = ROOT / "artifacts/reports/acceptance-summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
