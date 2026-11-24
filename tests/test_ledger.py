from __future__ import annotations

import json
import os
import socket
import signal
from pathlib import Path

import pytest

from receipt_ledger.config import validate_workers
from receipt_ledger.engine import LedgerEngine
from receipt_ledger.generator import init_workspace, invalid_validation_sample
from receipt_ledger.normalizer import attribute_requests, build_receipt_stage
from receipt_ledger.ports import check_loopback_port
from receipt_ledger.receipts import downgrade_to_r1, parse_receipt_text, upgrade_to_r3
from receipt_ledger.runner import run_pipeline
from receipt_ledger.snapshot import SnapshotStore, diff_objects
from receipt_ledger.runtime import SignalGate, SignalInterrupt
from receipt_ledger.utils import read_json
from receipt_ledger.workspace import load_workspace


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    init_workspace(tmp_path, seed=234)
    return tmp_path


def test_validation_failures_are_isolated() -> None:
    payload = invalid_validation_sample()["workers.json"]
    workers, failures = validate_workers(payload, port_range=[43000, 43999])
    codes = [item["code"] for item in failures]
    assert codes.count("E_SCHEMA_INVALID") >= 5
    messages = " ".join(item["message"] for item in failures)
    assert "unknown field" in messages
    assert "duplicate worker identifier" in messages
    assert "port" in messages
    assert "finite" in messages
    assert any(worker["worker_id"] == "W-OK" for worker in workers)


def test_request_stream_strict_seq_and_negative_batch(workspace: Path) -> None:
    import receipt_ledger.config as config

    payload = json.loads((workspace / "data" / "invalid_cases" / "request_stream.json").read_text())
    _requests, batches, failures = config.validate_request_stream(payload, {"W-OK"})
    messages = [item["message"] for item in failures]
    assert any("strictly increasing" in message for message in messages)
    assert any("arrived_ms" in message for message in messages)
    assert any("batch_size" in message for message in messages)
    assert [batch["batch_id"] for batch in batches] == ["B-OK"]


def test_rules_unknown_workers_fail() -> None:
    from receipt_ledger.config import validate_rules

    payload = invalid_validation_sample()["rules.json"]
    rules, failures = validate_rules(payload, {"W-OK"})
    assert rules is None
    assert len(failures) == 2
    assert all(item["code"] == "E_SCHEMA_INVALID" for item in failures)


def test_three_version_counts_ranks_and_fallback(workspace: Path) -> None:
    inputs, texts, _hash, _injections = load_workspace(workspace / "data")
    stage = build_receipt_stage(texts, inputs["workers"], inputs["batches"])
    assert stage["receipt_counts"] == {"r1": 3, "r2": 9, "r3": 10, "total": 22}

    b301 = [r for r in stage["receipts"] if r["batch_id"] == "B301"]
    from receipt_ledger.receipts import rank_batch_receipts

    ordered_ids = [r["request_id"] for r in rank_batch_receipts(b301)]
    assert ordered_ids == ["REQ-0012", "REQ-0014", "REQ-0011", "REQ-0013"]

    r1 = next(r for r in stage["receipts"] if r["record_version"] == "r1")
    assert r1["rank_source"] == "lexicographic_request_id"
    assert "lexicographic" in r1["rank_basis"]


def test_conversion_labels_sidecar_and_truncation(workspace: Path) -> None:
    _inputs, texts, _hash, _inj = load_workspace(workspace / "data")
    r1_receipts, _ = parse_receipt_text(texts["receipts-r1.log"], source="r1")
    upgraded = upgrade_to_r3(r1_receipts[0])
    assert upgraded["latency_ms"] is None
    assert upgraded["description"] == "__MISSING__"
    assert any("explicit_missing" in label for label in upgraded["precision_labels"])

    r3_receipts, _ = parse_receipt_text(texts["receipts-r3.log"], source="r3")
    line, sidecar, truncated = downgrade_to_r1(r3_receipts[0])
    assert line.startswith("R1|")
    assert sidecar["stable_rank"] is not None
    assert "rank" in truncated
    assert "param_digest" in truncated


def test_attribution_decisions(workspace: Path) -> None:
    inputs, texts, _hash, _inj = load_workspace(workspace / "data")
    stage = build_receipt_stage(texts, inputs["workers"], inputs["batches"])
    result = attribute_requests(stage["receipts"])
    by_id = {item["request_id"]: item for item in result["attributions"]}

    assert by_id["REQ-0004"]["authoritative_batch_id"] == "B102"
    assert "batch_id_asc=B102" in by_id["REQ-0004"]["decision_basis"][-1]
    assert by_id["REQ-0015"]["covered_receipts"][0]["reason"] == "r3_override_reference"
    assert by_id["REQ-0016"]["status"] == "conflict"
    assert len(by_id["REQ-0016"]["candidate_receipt_uids"]) == 2
    assert by_id["REQ-0017"]["status"] == "undecidable"


def test_full_and_restart_incremental_are_field_equal(workspace: Path) -> None:
    full = run_pipeline(workspace / "data", workspace / "runs" / "full", reset=True, jobs=1)
    incremental = run_pipeline(
        workspace / "data",
        workspace / "runs" / "incremental",
        reset=True,
        restart_each_command=True,
        jobs=4,
    )
    assert diff_objects(full["ledger"], incremental["ledger"]) == []


def test_budget_pending_then_continuation_matches_full(workspace: Path) -> None:
    pending = run_pipeline(
        workspace / "data", workspace / "runs" / "budget", reset=True, max_commands=1, jobs=1
    )
    assert pending["ledger"]["summary"]["pending_count"] > 0
    assert "E_NORMALIZE_BUDGET_PENDING" in pending["ledger"]["business_code_counts"]

    completed = run_pipeline(workspace / "data", workspace / "runs" / "budget", jobs=4)
    full = run_pipeline(workspace / "data", workspace / "runs" / "full-reference", reset=True, jobs=1)
    assert completed["ledger"]["summary"]["pending_count"] == 0
    assert diff_objects(full["ledger"], completed["ledger"]) == []


def test_injections_are_isolated_and_mapped(workspace: Path) -> None:
    injected = run_pipeline(
        workspace / "data",
        workspace / "runs" / "injected",
        reset=True,
        injection_config=workspace / "data" / "injections.json",
        jobs=1,
    )
    results = injected["ledger"]["injection_results"]
    assert len(results) == 4
    assert all(item["observed"] for item in results)
    codes = injected["ledger"]["business_code_counts"]
    assert codes["E_RECEIPT_TRUNCATED"] >= 1
    assert codes["E_VERSION_UNKNOWN"] >= 1
    assert codes["E_RECEIPT_NONFINITE_LATENCY"] >= 1
    assert codes["E_SCHEMA_INVALID"] >= 1
    # The non-finite line is isolated; B102 still has the other two parsed receipts.
    b102 = next(item for item in injected["ledger"]["batch_metrics"] if item["batch_id"] == "B102")
    assert b102["receipt_count"] == 2
    b201 = next(item for item in injected["ledger"]["batch_metrics"] if item["batch_id"] == "B201")
    assert b201["receipt_count"] == 4


def test_snapshot_atomic_pointer_and_config_mismatch(workspace: Path) -> None:
    run = workspace / "runs" / "snapshot"
    result = run_pipeline(workspace / "data", run, reset=True, max_commands=2, jobs=1)
    store = SnapshotStore(run / "snapshots")
    pointer = store.load_pointer()
    assert pointer is not None
    manifest = read_json(store.releases / pointer["release"] / "manifest.json")
    assert manifest["completed_command_cursor"] == 2
    assert manifest["artifact_versions"]["01_parse"] == 1

    state, _manifest, reason = store.load_compatible_state("different-hash")
    assert state is None
    assert reason == "config_hash_changed"


def test_failed_commit_keeps_current_snapshot(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = workspace / "runs" / "atomic-failure"
    result = run_pipeline(workspace / "data", run, reset=True, max_commands=1, jobs=1)
    store = SnapshotStore(run / "snapshots")
    before = store.load_pointer()
    assert before is not None

    def boom(*args, **kwargs):
        raise RuntimeError("simulated stage failure")

    monkeypatch.setattr("receipt_ledger.utils.write_json", boom)
    engine = LedgerEngine.from_state(read_json(store.releases / before["release"] / "state.json"))
    with pytest.raises(RuntimeError, match="simulated"):
        store.commit(
            state=engine.to_state(),
            ledger=engine.export_ledger(),
            config_hash=result["ledger"]["config_hash"],
            command_cursor=engine.command_cursor,
            global_version=99,
        )
    after = store.load_pointer()
    assert after == before


def test_port_occupancy_detection() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    try:
        result = check_loopback_port("127.0.0.1", port)
        assert result["free"] is False
        assert result["code"] == "E_PORT_IN_USE"
    finally:
        sock.close()


def test_signal_gate_handles_sigterm() -> None:
    gate = SignalGate()
    gate.install()
    try:
        with pytest.raises(SignalInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
        assert gate.stop_requested is True
    finally:
        gate.restore()


def test_resource_metrics_are_integer_fixed_point(workspace: Path) -> None:
    ledger = run_pipeline(workspace / "data", workspace / "runs" / "metrics", reset=True)["ledger"]
    for metric in ledger["batch_metrics"]:
        assert isinstance(metric["mac"], int)
        assert isinstance(metric["batch_utilization_q"], int)
        assert 0 <= metric["batch_utilization_q"] <= 2_000_000
    assert ledger["worker_version_metrics"]


def test_arbitrary_command_sequences_survive_snapshot_roundtrip(workspace: Path) -> None:
    import random

    inputs, texts, config_hash, _inj = load_workspace(workspace / "data")
    base_rules = json.loads((workspace / "data" / "rules.json").read_text())
    request_ids = [item["request_id"] for item in inputs["requests"]]

    for seed in range(24):
        rng = random.Random(9000 + seed)
        commands = []
        for index in range(12):
            command_id = f"R{seed}-{index:02d}"
            op = rng.choice([
                "normalize", "normalize", "checkpoint", "grant_budget",
                "set_param_version", "replace_request", "switch_rules",
            ])
            if op == "normalize":
                commands.append({"command_id": command_id, "op": "normalize"})
            elif op == "checkpoint":
                commands.append({"command_id": command_id, "op": "checkpoint"})
            elif op == "grant_budget":
                commands.append({
                    "command_id": command_id, "op": "grant_budget",
                    "batch_budget": rng.choice([1, 2, 8]),
                    "worker_quota": rng.choice([2, 4, 16]),
                })
            elif op == "set_param_version":
                worker_id = rng.choice(["W1", "W3"])
                version = rng.choice([1, 2] if worker_id == "W1" else [1, 2, 3])
                commands.append({
                    "command_id": command_id, "op": "set_param_version",
                    "worker_id": worker_id, "param_version": version,
                    "param_digest": f"digest-{seed}-{index}",
                })
            elif op == "replace_request":
                commands.append({
                    "command_id": command_id, "op": "replace_request",
                    "request_id": rng.choice(request_ids),
                    "request": {"output_digest": f"replacement-{seed}-{index}"},
                })
            else:
                rules = dict(base_rules)
                rules["rules_id"] = f"random-rules-{seed}-{index}"
                commands.append({"command_id": command_id, "op": "switch_rules", "rules": rules})

        direct = LedgerEngine(inputs, texts, config_hash, jobs=1)
        direct.apply_commands(commands)
        direct_ledger = direct.export_ledger()

        restored = LedgerEngine(inputs, texts, config_hash, jobs=1)
        for command in commands:
            restored.apply_command(command)
            state = json.loads(json.dumps(restored.to_state(), allow_nan=False))
            restored = LedgerEngine.from_state(state, jobs=1)
        restored_ledger = restored.export_ledger()
        assert diff_objects(direct_ledger, restored_ledger) == []
