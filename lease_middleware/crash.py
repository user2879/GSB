"""Force-kill recovery demonstration."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .cluster import PROJECT_ROOT, NodeCluster
from .ledger import Ledger, recover_snapshot, write_snapshot
from .service import prepare_dir
from .timeutil import monotonic_ms


CRASH_PORT = 48191


def crash_victim(directory: str | Path, *, port: int = CRASH_PORT) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(directory / "ledger.sqlite")
    node = {
        "node_id": "crash-node",
        "host": "127.0.0.1",
        "port": port,
        "lease_limit": 1,
        "in_flight_limit": 1,
        "node_generation": 1,
    }
    cluster = NodeCluster([node], log_dir=directory / "node-logs")
    try:
        cluster.start_all(["crash-node"])
        client = cluster.clients["crash-node"]
        client.grant("lease-crash", "equity-crash", 60_000)
        grant_records = [
            {
                "entry_kind": "lease_transition",
                "event_seq": 1,
                "lease_id": "lease-crash",
                "equity_id": "equity-crash",
                "node_id": "crash-node",
                "node_generation": 1,
                "generation_before": None,
                "generation_after": 1,
                "matched_tag": None,
                "version_tag": 1,
                "from_state": None,
                "to_state": "granted",
                "trigger": "grant",
                "reason": "initial_grant",
                "at_ms": 0,
                "expires_at_ms": 60_000,
            },
            {
                "entry_kind": "ownership",
                "event_seq": 1,
                "equity_id": "equity-crash",
                "lease_id": "lease-crash",
                "node_id": "crash-node",
                "generation": 1,
                "node_generation": 1,
                "change": "acquired",
                "reason": "initial_grant",
                "effective": True,
                "counted_reclamation": False,
                "owner_after_lease": "lease-crash",
                "owner_after_node": "crash-node",
                "at_ms": 0,
            },
        ]
        ledger.append_records(grant_records)
        ledger.record_intent(
            intent_id="intent-renew-lease-crash",
            event_seq=2,
            lease_id="lease-crash",
            op="renew",
            payload={"version_tag": 1, "at_ms": 100},
            monotonic_ms=monotonic_ms(),
        )
        # The node accepts and advances its tag, but the center is killed
        # before the outcome transaction is committed.
        reply = client.renew("lease-crash", 1)
        (directory / "pre-kill-reply.json").write_bytes(canonical_bytes(reply))
        ledger.close()
        os.kill(os.getpid(), signal.SIGKILL)
    finally:
        cluster.stop_all()
        ledger.close()


def run_crash_demo(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or Path.cwd())
    destination = prepare_dir(root / "artifacts" / "crash")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log_path = destination / "victim.log"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lease_middleware.cli",
                "crash-victim",
                "--dir",
                str(destination),
            ],
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return_code = process.wait(timeout=20)

    time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    if return_code != -signal.SIGKILL:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"crash victim exited with {return_code}: {log_text[-1000:]}")

    ledger = Ledger.open(destination / "ledger.sqlite")
    try:
        recovery = ledger.recovery_report()
        snapshot_path = destination / "session-snapshot.json"
        write_snapshot(snapshot_path, {"committed_records": ledger.records()})
        envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
        envelope["format_version"] = 999
        snapshot_path.write_text(canonical_bytes(envelope).decode("utf-8"), encoding="utf-8")
        _, snapshot_recovery = recover_snapshot(snapshot_path, ledger)
        report = {
            "ok": True,
            "victim_returncode": return_code,
            "forced_signal": "SIGKILL",
            "recovery": recovery,
            "snapshot_recovery": snapshot_recovery,
        }
        (destination / "recovery_report.json").write_bytes(canonical_bytes(report))
        return report
    finally:
        ledger.close()
