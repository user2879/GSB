from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import os
import subprocess
import time

from driftwatch.config import load_inputs
from driftwatch.engine import DriftEngine, recover_ledger
from driftwatch.errors import DriftError
from driftwatch.ledger import Ledger
from driftwatch.processes import port_is_free

ROOT = Path(__file__).resolve().parent


def _reset_database(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _wait(predicate, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError("condition did not become true before timeout")


def _worker(args: argparse.Namespace) -> int:
    engine = DriftEngine.from_files(args.snapshots, args.agents, args.criteria, args.db)
    marker = Path(args.marker)

    def hook(event: str, payload: dict) -> None:
        if event == "batch_begun" and payload["snapshot_seq"] == 2:
            marker.write_text(str(os.getpid()), encoding="utf-8")
            time.sleep(120)

    try:
        engine.run(lifecycle_hook=hook)
    finally:
        engine.close()
    return 0


def _parent(args: argparse.Namespace) -> int:
    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    _reset_database(db)
    marker = Path(str(db) + ".pause-marker")
    if marker.exists():
        marker.unlink()
    chain, agents, criteria = load_inputs(args.snapshots, args.agents, args.criteria)
    occupied = [agent.port for agent in agents if not port_is_free(agent.host, agent.port)]
    if occupied:
        raise DriftError.port("example ports are already occupied", occupied=occupied)

    env = os.environ.copy()
    packages = str(ROOT.parent)
    env["PYTHONPATH"] = packages if not env.get("PYTHONPATH") else packages + os.pathsep + env["PYTHONPATH"]
    worker = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--snapshots",
            args.snapshots,
            "--agents",
            args.agents,
            "--criteria",
            args.criteria,
            "--db",
            str(db),
            "--marker",
            str(marker),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait(marker.exists, 15.0)
        if worker.poll() is not None:
            out, err = worker.communicate(timeout=2)
            raise RuntimeError(f"worker exited before the kill marker: {err or out}")
        worker.kill()
        worker.communicate(timeout=8)
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=5)
    _wait(lambda: all(port_is_free(agent.host, agent.port) for agent in agents), 10.0)

    with Ledger(str(db)) as ledger:
        before = [
            {"snapshot_seq": row["snapshot_seq"], "status": row["status"]}
            for row in ledger.connection.execute("SELECT snapshot_seq, status FROM batches ORDER BY started_ms")
        ]
        recovered = ledger.recover_incomplete()
        generation = ledger.active_generation() or 0
        unknown = ledger.required_recollection(generation)
        aborted = [
            {"snapshot_seq": row["snapshot_seq"], "status": row["status"]}
            for row in ledger.connection.execute("SELECT snapshot_seq, status FROM batches ORDER BY started_ms")
        ]

    recovered_report = recover_ledger(str(db))
    report = recovered_report["report"]
    with Ledger(str(db)) as ledger:
        view = ledger.session_snapshot(report["generation"])

    payload = {
        "ok": True,
        "killed_pid": int(marker.read_text(encoding="utf-8")),
        "batches_before_recovery": before,
        "batches_after_recovery": aborted,
        "recovered_unknown": recovered,
        "replayed_committed_batches": report["replay"],
        "recollected_and_committed": report["committed_batches"],
        "final_view": report["view_summary"],
        "recovery_source": recovered_report["source"],
        "final_snapshots": sorted({row["snapshot_seq"] for row in view["drift_view"]}),
        "exit_report_claimed": False,
        "note": "A forced kill has no exit report. Recovery uses only atomically committed batches, marks interrupted host metrics unknown, recollects them, and verifies replay.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demonstrate force-kill recovery and ledger replay")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshots", default=str(ROOT / "snapshots.json"))
    parser.add_argument("--agents", default=str(ROOT / "agents_v1.json"))
    parser.add_argument("--criteria", default=str(ROOT / "criteria.json"))
    parser.add_argument("--db", default=str(ROOT / "demo-data" / "crash.sqlite3"))
    parser.add_argument("--marker", default="")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
