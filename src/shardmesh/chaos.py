"""SIGKILL a node subprocess and recover solely from committed ledger state."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

from .engine import Engine
from .generator import demo_spec_dict, generate_samples
from .ledger import Ledger
from .replay import replay_ledger
from .spec import parse_spec


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def run_kill_recovery(root: str | Path, *, target: str = "s01", companion: str = "s05") -> dict[str, Any]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ports = [free_port(), free_port(), free_port()]
    raw_spec = demo_spec_dict(ports=ports, lease_ttl_ms=2000)
    spec = parse_spec(raw_spec, root / "chaos-spec.json")
    samples = generate_samples(raw_spec)
    (root / "chaos-spec.json").write_text(json.dumps(raw_spec, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    ledger_path = root / "chaos-ledger.sqlite3"
    for path in [ledger_path, Path(str(ledger_path) + "-wal"), Path(str(ledger_path) + "-shm")]:
        path.unlink(missing_ok=True)

    ledger = Ledger(ledger_path)
    ledger.initialize_spec(spec.to_dict())
    engine = Engine(
        spec,
        samples,
        ledger,
        work_dir=root / "first-runtime",
        delay_overrides={target: 2000},
    )
    kill_returncode: int | None = None
    first_failure: dict[str, Any] | None = None
    try:
        engine.start()
        for shard_id in (target, companion):
            engine.arrived.add(shard_id)
            engine.mark_wait_start(shard_id)
        engine.pump()
        deadline = time.monotonic() + 3.0
        while target not in engine.futures and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.45)
        victim_node = spec.shard_map[target].node_id
        kill_returncode = engine.cluster.get(victim_node).kill_hard()
        deadline = time.monotonic() + 5.0
        while target in engine.futures and time.monotonic() < deadline:
            engine._consume_finished()
            time.sleep(0.03)
        engine.settle()
        if target in engine.failed:
            error = engine.failed[target]
            first_failure = {"code": error.code, "message": str(error)}
    finally:
        engine.close()
        ledger.close()

    recovered_ledger = Ledger(ledger_path)
    recovery_engine = Engine(spec, samples, recovered_ledger, work_dir=root / "recovery-runtime")
    recovery = recovery_engine.recover_unknown_shards()
    pre_rerun_replay = replay_ledger(recovered_ledger, samples)
    try:
        with recovery_engine:
            records = recovery_engine.run_pending()
        post_rerun_replay = replay_ledger(recovered_ledger, samples)
    finally:
        recovered_ledger.close()

    report = {
        "consistent": post_rerun_replay["consistent"],
        "target_shard": target,
        "victim_node": spec.shard_map[target].node_id,
        "sigkill_returncode": kill_returncode,
        "in_flight_failure": first_failure,
        "recovery": recovery,
        "pre_rerun_replay_count": pre_rerun_replay["checked_count"],
        "post_rerun_replay": post_rerun_replay,
        "rerun_record_count": len(records),
        "exit_report_promised": False,
        "note": "SIGKILL provides no application exit report; only committed rows are trusted.",
    }
    (root / "chaos-report.json").write_text(json.dumps(report, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    return report
