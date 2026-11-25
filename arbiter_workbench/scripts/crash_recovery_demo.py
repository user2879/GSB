"""End-to-end forced-termination and recovery replay demonstration.

Scenario:
  1. Full arbitration commits all views to the SQLite ledger.
  2. A simulated forced kill leaves only an UNKNOWN marker for one view
     (its real commit transaction is never finalized).
  3. Restart recovery trusts only committed rows, flags the UNKNOWN view for a
     full rerun, and replays committed views with E_REPLAY-safe key/field checks.
  4. A snapshot is exported, tampered, and rejected for whole replacement.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbiter.engine import ArbitrationEngine
from arbiter.ledger import Ledger, UNKNOWN
from arbiter.protocol import pick_free_port
from arbiter.recovery import assert_no_fabricated_exit_report, recover
from arbiter.sample import DEFAULT_KEYS
from arbiter.service_manager import ServiceManager
from arbiter.spec import EnvSpec, load_declarations

DECLS = ROOT / "decls"
WORK = ROOT / "run" / "crash_demo"
EXIT_REPORT = WORK / "exit_report.json"


def fresh_manager(decls) -> ServiceManager:
    envs = []
    for env in decls.environments.values():
        envs.append(EnvSpec(env.env_id, env.host, pick_free_port(),
                            env.lease_capacity, env.seed, env.weight))
    return ServiceManager(envs)


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    db_path = WORK / "ledger.sqlite"

    decls = load_declarations(DECLS, DEFAULT_KEYS)
    ledger = Ledger(db_path, session_scale=2)
    engine = ArbitrationEngine(decls, 2, ledger)
    manager = fresh_manager(decls)
    try:
        manager.start_all()
        engine.start_services_and_ingest(manager)
        outputs = engine.run_full()
        print("[1] committed views:", sorted(outputs))
    finally:
        manager.shutdown_all()
        ledger.close()

    # simulate SIGKILL mid-transaction: only an UNKNOWN marker survives
    ledger = Ledger(db_path, session_scale=2)
    ledger.begin_unknown("view-edge-policy", version=99)
    ledger.close()
    # explicitly do NOT write an exit report for the forced termination
    print("[2] forced termination simulated; UNKNOWN marker left for view-edge-policy v99")

    def replay_fn(view_id: str, version: int):
        ledger_ro = Ledger(WORK / "replay.sqlite", session_scale=2)
        mgr = fresh_manager(decls)
        eng = ArbitrationEngine(decls, 2, ledger_ro)
        try:
            mgr.start_all()
            eng.start_services_and_ingest(mgr)
            return eng.arbitrate_view(view_id, version)
        finally:
            mgr.shutdown_all()
            ledger_ro.close()

    report = recover(db_path, session_scale=2, replay_fn=replay_fn)
    print("[3] recovery report:")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    assert "view-edge-policy" in report.unknown_views
    assert report.recovered_views

    assert_no_fabricated_exit_report(EXIT_REPORT)
    print("[4] no fabricated forced-termination exit report (as required)")

    ledger = Ledger(db_path, session_scale=2)
    snapshot = WORK / "snapshot.json"
    digest = ledger.export_snapshot(snapshot)
    ledger.close()
    print(f"[5] snapshot exported digest={digest[:16]}...")

    tampered = WORK / "tampered.json"
    data = json.loads(snapshot.read_text())
    data["content_digest"] = "0" * 64
    tampered.write_text(json.dumps(data))
    ledger = Ledger(WORK / "restored.sqlite", session_scale=2)
    try:
        ledger.replace_with_snapshot(tampered)
        raise SystemExit("expected E_STATE on tampered snapshot")
    except Exception as exc:
        print("[6] tampered snapshot rejected:", exc)
    finally:
        ledger.close()
    print("CRASH_RECOVERY_DEMO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
