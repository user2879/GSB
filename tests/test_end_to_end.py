from __future__ import annotations

import sqlite3

import pytest

from ckptsim.errors import CkptError
from ckptsim.impact import expected_batch_records, selective_recompute
from ckptsim.plan import build_event_plan
from ckptsim.runner import Runner, RunnerOptions
from ckptsim.spec import load_bundle, load_layers
from ckptsim.verify import verify_ledger


def _bundle(files):
    return load_bundle(files.shards, files.layers, files.checkpoints, files.events)


def test_end_to_end_replay_and_checkpoint_rollback(spec_files, tmp_path) -> None:
    bundle = _bundle(spec_files)
    ledger_path = tmp_path / "run" / "ledger.sqlite"
    result = Runner(
        bundle,
        RunnerOptions(ledger_path, lease_ttl_ms=10_000, client_timeout=5),
    ).run()
    assert result["ok"] is True
    assert len(result["batches"]) == 7
    assert result["state"]["current_versions"] == {"shard-a": 3, "shard-b": 3}
    assert result["state"]["generation_counter"] == 3
    assert result["rollbacks"][0]["revoked_versions"] == [
        {"parameter_version": 2, "generation": 2}
    ]
    assert result["forced_exit_report"] is None
    verified = verify_ledger(bundle, ledger_path, raise_on_error=False)
    assert verified.verified is True
    assert verified.comparisons > 100
    reopened = Runner(
        bundle,
        RunnerOptions(ledger_path, lease_ttl_ms=10_000, client_timeout=5),
    ).run()
    assert reopened["ok"] is True
    assert [
        (item["shard_id"], item["batch_id"], item["loss_digest"])
        for item in reopened["batches"]
    ] == [
        (item["shard_id"], item["batch_id"], item["loss_digest"])
        for item in result["batches"]
    ]
    assert verify_ledger(bundle, ledger_path, raise_on_error=False).verified is True


def test_forced_worker_kill_leaves_unknown_then_reruns(spec_files, tmp_path) -> None:
    bundle = _bundle(spec_files)
    killed: list[str] = []

    def kill_once(batch, manager) -> None:
        if batch.shard_id == "shard-a" and not killed:
            killed.append(batch.batch_id)
            manager.kill("shard-a")

    ledger_path = tmp_path / "crash" / "ledger.sqlite"
    result = Runner(
        bundle,
        RunnerOptions(
            ledger_path,
            lease_ttl_ms=10_000,
            client_timeout=3,
            max_recovery_attempts=2,
        ),
        on_batch_started=kill_once,
    ).run()
    assert result["ok"] is True
    assert killed == ["b-a1"]
    assert result["unknown_attempts"]
    assert any(event["status"] == "worker_restarted" for event in result["recovery_events"])
    assert verify_ledger(bundle, ledger_path, raise_on_error=False).verified is True


def test_verify_locates_tampered_loss_digest(spec_files, tmp_path) -> None:
    bundle = _bundle(spec_files)
    ledger_path = tmp_path / "tamper" / "ledger.sqlite"
    Runner(
        bundle,
        RunnerOptions(ledger_path, lease_ttl_ms=10_000, client_timeout=5),
    ).run()
    connection = sqlite3.connect(ledger_path)
    connection.execute(
        "UPDATE batches SET loss_digest='tampered' WHERE shard_id='shard-a' AND batch_id='b-a1'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(CkptError) as caught:
        verify_ledger(bundle, ledger_path)
    error = caught.value
    assert error.code == "E_VERIFY"
    assert error.shard_id == "shard-a"
    assert error.batch_id == "b-a1"
    assert error.field == "loss_digest"
    assert error.extra["first_mismatch"]["field"] == "loss_digest"


def test_dependency_invalidation_recomputes_only_affected_subset(spec_files) -> None:
    bundle = _bundle(spec_files)
    plan = build_event_plan(bundle)
    baseline = expected_batch_records(bundle, plan)
    edited = load_layers(spec_files.edited_layers)
    result = selective_recompute(bundle, edited, plan, baseline)
    affected = {
        (item["shard_id"], item["batch_id"])
        for item in result["impact"]["affected_batches"]
    }
    unchanged = {
        (item["shard_id"], item["batch_id"])
        for item in result["impact"]["unchanged_batches"]
    }
    assert affected == {
        ("shard-a", "b-a1"),
        ("shard-a", "b-a3"),
        ("shard-b", "b-b1"),
        ("shard-b", "b-b3"),
    }
    assert unchanged == {
        ("shard-a", "b-a2"),
        ("shard-a", "b-a4"),
        ("shard-b", "b-b2"),
    }
    assert result["impact"]["affected_checkpoints"] == ["cp-base"]
    assert result["recomputed_checkpoint_count"] == 1
    checkpoint_change = result["recomputed_checkpoints"][0]
    assert checkpoint_change["checkpoint_id"] == "cp-base"
    assert checkpoint_change["baseline_digest_before"] != checkpoint_change["baseline_digest_after"]
    assert result["unchanged_verified"] is True
    assert result["unchanged_before_digest"] == result["unchanged_after_digest"]
