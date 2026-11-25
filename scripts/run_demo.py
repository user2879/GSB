#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lease_arbiter.canonical import canonical_bytes, canonical_json, digest
from lease_arbiter.recovery import replay_against_ledger
from lease_arbiter.runtime import run_live
from lease_arbiter.spec import load_spec
from lease_arbiter.verifier import verify_spec


NODES = ROOT / "config" / "nodes.json"
LEASES = ROOT / "config" / "leases.json"
EVENTS = ROOT / "samples" / "events.jsonl"
LEDGERS = ROOT / "ledgers"
RESULTS = ROOT / "results"


def _write_json(path: Path, payload: dict[str, Any]) -> bytes:
    data = canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def _clear_sqlite(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    LEDGERS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    spec = load_spec(NODES, LEASES, EVENTS)

    verify = verify_spec(spec)
    verify_bytes = _write_json(RESULTS / "verify.json", verify)
    projection_bytes = canonical_bytes(verify["comparison"]["sequential_projection"]) + b"\n"
    (RESULTS / "projection.json").write_bytes(projection_bytes)

    live_runs: dict[str, Any] = {}
    live_output_bytes: dict[str, bytes] = {}
    live_projection_bytes: dict[str, bytes] = {}
    for path in ("sequential", "batch"):
        ledger = LEDGERS / f"live-{path}.sqlite3"
        _clear_sqlite(ledger)
        result = run_live(spec, ledger_path=ledger, requested_path=path, fresh=True)
        report = {"ok": True, "command": "run", **result}
        data = _write_json(RESULTS / f"live-{path}.json", report)
        live_output_bytes[path] = data
        live_projection_bytes[path] = canonical_bytes(result["comparison_projection"]) + b"\n"
        (RESULTS / f"projection-{path}.json").write_bytes(live_projection_bytes[path])
        live_runs[path] = result

    if live_projection_bytes["sequential"] != live_projection_bytes["batch"]:
        raise RuntimeError("live sequential and batch projections are not byte-identical")
    if live_projection_bytes["sequential"] != projection_bytes:
        raise RuntimeError("live projection differs from independent verifier projection")

    crash_ledger = LEDGERS / "crash-recovery.sqlite3"
    _clear_sqlite(crash_ledger)
    command = [
        sys.executable,
        "-m",
        "lease_arbiter",
        "run",
        "--nodes",
        str(NODES),
        "--leases",
        str(LEASES),
        "--events",
        str(EVENTS),
        "--ledger",
        str(crash_ledger),
        "--path",
        "sequential",
        "--fresh",
        "--crash-after-event",
        "5",
    ]
    crash_process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    crash_exit = crash_process.wait(timeout=15)
    _kill_process_group(crash_process)
    # Give the kernel a moment to release the loopback listeners before replay.
    time.sleep(0.25)
    replay = replay_against_ledger(spec, crash_ledger)
    replay_report = {"ok": True, "command": "replay", **replay}
    _write_json(RESULTS / "crash-replay.json", replay_report)

    if crash_exit != 137:
        raise RuntimeError(f"crash run exited with {crash_exit}, expected 137")
    if replay["unknown_request_count"] != 1:
        raise RuntimeError(f"expected one unknown pending request, got {replay['unknown_request_count']}")

    comparison = {
        "ok": True,
        "command": "compare",
        "byte_identical": True,
        "independent_vs_live_sequential": projection_bytes == live_projection_bytes["sequential"],
        "independent_vs_live_batch": projection_bytes == live_projection_bytes["batch"],
        "live_sequential_vs_live_batch": live_projection_bytes["sequential"] == live_projection_bytes["batch"],
        "projection_file_bytes": len(projection_bytes),
        "canonical_projection_bytes": len(projection_bytes.rstrip(b"\n")),
        "projection_sha256": digest(verify["comparison"]["sequential_projection"]),
    }
    _write_json(RESULTS / "byte-consistency.json", comparison)

    summary = {
        "ok": True,
        "format_version": 1,
        "spec_digest": spec.digest,
        "declaration_version": spec.declaration_version,
        "verify_sha256": digest(verify),
        "verify_file_bytes": len(verify_bytes),
        "live": {
            path: {
                "summary": live_runs[path]["summary"],
                "projection_sha256": live_runs[path]["comparison_sha256"],
                "ledger_snapshot_version": live_runs[path]["runtime"]["ledger"]["snapshot_version"],
                "node_requests": live_runs[path]["runtime"]["scheduler"]["request_count"],
                "max_inflight_total": live_runs[path]["runtime"]["scheduler"]["max_inflight_total"],
            }
            for path in ("sequential", "batch")
        },
        "crash_recovery": {
            "exit_code": crash_exit,
            "committed_events": replay["committed_events"],
            "unknown_request_count": replay["unknown_request_count"],
            "snapshot_version": replay["snapshot_version"],
            "replay_ok": replay["status"] == "replay_ok",
        },
        "byte_consistency": comparison,
        "artifacts": {
            "verify": "results/verify.json",
            "live_sequential": "results/live-sequential.json",
            "live_batch": "results/live-batch.json",
            "crash_replay": "results/crash-replay.json",
            "byte_consistency": "results/byte-consistency.json",
            "projection": "results/projection.json",
        },
    }
    _write_json(RESULTS / "demo-report.json", summary)
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
