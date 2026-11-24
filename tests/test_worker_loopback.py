from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"


def _free_port(start: int, end: int, *, excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    for port in range(start, end + 1):
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free loopback port in configured range")


def _one_call(port: int, batch_id: str) -> dict:
    request = {
        "batch_id": batch_id,
        "request_ids": ["r-001", "r-004"],
        "receipt_version": "r3",
        "duration_ms": 6,
        "covered_refs": {},
    }
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
        connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
        buffer = bytearray()
        while b"\n" not in buffer:
            chunk = connection.recv(65536)
            if not chunk:
                break
            buffer.extend(chunk)
    return json.loads(bytes(buffer).split(b"\n", 1)[0].decode("utf-8"))


def test_worker_binds_loopback_and_repeats_deterministically(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(SAMPLE, project, ignore=shutil.ignore_patterns(".state"))
    config_path = project / "config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    other_ports = {
        int(worker["port"])
        for worker in raw["workers"]
        if worker["worker_id"] != "w-alpha"
    }
    port = _free_port(18300, 18399, excluded=other_ports)
    for worker in raw["workers"]:
        if worker["worker_id"] == "w-alpha":
            worker["port"] = port
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    environment = dict(os.environ)
    path_entries = [str(ROOT / "src")]
    if (ROOT / ".deps").is_dir():
        path_entries.insert(0, str(ROOT / ".deps"))
    environment["PYTHONPATH"] = os.pathsep.join(
        path_entries + [environment.get("PYTHONPATH", "")]
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gsb_ledger.worker",
            "--project",
            str(project),
            "--worker-id",
            "w-alpha",
        ],
        cwd=str(ROOT),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read().decode("utf-8", errors="replace")
                raise AssertionError(f"worker exited early: {stderr}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("worker did not bind within timeout")
        first = _one_call(port, "b-smoke-001")
        second = _one_call(port, "b-smoke-001")
        assert first["ok"] is True
        assert first["worker_id"] == "w-alpha"
        assert first["receipts"] == second["receipts"]
        assert all(row["param_version"] == 4 for row in first["receipts"])
        assert all(len(row["output_digest"]) == 71 for row in first["receipts"])
        assert all("param_digest" in row for row in first["receipts"])
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
