import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ckpt_diag.errors import LeaseError
from ckpt_diag.protocol import request
from ckpt_diag.samplegen import build_spec_dict
from ckpt_diag.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def shard_process(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(build_spec_dict()))
    spec = load_spec(spec_path)
    shard = spec.shard("shard-alpha")
    proc = subprocess.Popen(
        [sys.executable, "-m", "ckpt_diag.shard_process",
         "--spec", str(spec_path), "--shard-id", shard.shard_id,
         "--ttl-ms", "5000"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(ROOT), env={**__import__("os").environ,
                            "PYTHONPATH": str(ROOT / "src")})
    ready = json.loads(proc.stdout.readline())
    assert ready["ready"]
    yield shard
    proc.terminate()
    proc.wait(timeout=5)


def test_ping_issue_run_renew_release(shard_process):
    port = shard_process.port
    assert request("127.0.0.1", port, {"action": "ping"})["ok"]
    issue = request("127.0.0.1", port,
                    {"action": "issue_lease", "batch_id": "int-batch"})
    assert issue["ok"]
    lease = issue["lease"]
    run = request("127.0.0.1", port, {
        "action": "run_batch", "lease_id": lease["lease_id"],
        "batch_id": "int-batch", "param_version": 1,
        "sample_ids": ["s-0", "s-1"], "scope_layers": ["L0"]})
    assert run["ok"] and run["elapsed_ms"] >= 0
    assert run["result"]["total_loss"] >= 0
    renewed = request("127.0.0.1", port, {
        "action": "renew_lease", "lease_id": lease["lease_id"],
        "expected_version": 1})
    assert renewed["lease"]["lease_version"] == 2
    stale = request("127.0.0.1", port, {
        "action": "renew_lease", "lease_id": lease["lease_id"],
        "expected_version": 1})
    assert stale["ok"] is False and stale["error"]["code"] == "E_LEASE"
    released = request("127.0.0.1", port, {
        "action": "release_lease", "lease_id": lease["lease_id"],
        "expected_version": 2})
    assert released["ok"]


def test_run_batch_rejects_expired_or_missing_lease(shard_process):
    port = shard_process.port
    response = request("127.0.0.1", port, {
        "action": "run_batch", "lease_id": "lease-nope",
        "batch_id": "x", "param_version": 1, "sample_ids": []})
    assert response["error"]["code"] == "E_LEASE"
