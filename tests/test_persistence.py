import json

import pytest

from gpu_diag.errors import E_LEDGER, E_REPLAY, E_PREVIEW, DiagError
from gpu_diag.session import (authoritative_recompute, recover_session,
                              restore_from_snapshot, run_session,
                              verify_ledger_digest)
from gpu_diag.persist.ledger import Ledger
from gpu_diag.digest import canonical_bytes


def _paths(sample_dir):
    return (sample_dir / "pools.json", sample_dir / "tasks.json",
            sample_dir / "params.json")


def test_ledger_rounds_committed_and_byte_identical(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    run_session(pools, tasks, params, state_dir)
    verify_ledger_digest(state_dir)
    db = state_dir / "ledger.sqlite"
    import sqlite3
    count = sqlite3.connect(str(db)).execute(
        "SELECT COUNT(*) FROM rounds").fetchone()[0]
    assert count > 0
    snapshot_bytes = (state_dir / "snapshot.json").read_bytes()
    assert snapshot_bytes == canonical_bytes(json.loads(snapshot_bytes))


def test_repeated_runs_produce_identical_snapshot(sample_dir, state_dir, tmp_path):
    pools, tasks, params = _paths(sample_dir)
    run_session(pools, tasks, params, state_dir)
    second = tmp_path / "state2"
    run_session(pools, tasks, params, second)
    assert (state_dir / "snapshot.json").read_bytes() == \
           (second / "snapshot.json").read_bytes()


def test_authoritative_recompute_matches(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    run_session(pools, tasks, params, state_dir)
    result = authoritative_recompute(pools, tasks, params, state_dir)
    assert result["byte_identical"] is True


def test_preview_divergence_detected(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    result = run_session(pools, tasks, params, state_dir)
    ledger = Ledger(state_dir / "ledger.sqlite", 1, result.declaration_digest)
    ledger.tamper_round_water(1)
    ledger.close()
    with pytest.raises(DiagError) as exc:
        authoritative_recompute(pools, tasks, params, state_dir)
    assert exc.value.code == E_PREVIEW
    assert exc.value.details[0]["round"] == 1


def test_ledger_digest_corruption_detected(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    result = run_session(pools, tasks, params, state_dir)
    ledger = Ledger(state_dir / "ledger.sqlite", 1, result.declaration_digest)
    ledger.tamper_round_water(1)
    ledger.close()
    with pytest.raises(DiagError) as exc:
        verify_ledger_digest(state_dir)
    assert exc.value.code == E_LEDGER


def test_forced_termination_marks_unknown(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    result = run_session(pools, tasks, params, state_dir,
                         crash_after_round=8)
    assert result.unknown_shards, "crash must leave uncommitted shards unknown"
    assert all(item["reason"].endswith("rerun required")
               for item in result.unknown_shards)
    recovered = recover_session(pools, tasks, params, state_dir)
    assert recovered["last_committed_round"] == 7
    assert {u["shard_id"] for u in recovered["unknown_shards"]} == \
           {u["shard_id"] for u in result.unknown_shards}


def test_snapshot_restore_replay_ok(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    run_session(pools, tasks, params, state_dir)
    outcome = restore_from_snapshot(pools, tasks, params, state_dir)
    assert outcome["replay_ok"] is True
    assert outcome["version_diffs"] == []


def test_snapshot_version_mismatch_reports_replacement(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    run_session(pools, tasks, params, state_dir)
    snapshot = json.loads((state_dir / "snapshot.json").read_text())
    snapshot["versions"]["task_version"] = "task-v999"
    (state_dir / "snapshot.json").write_text(json.dumps(snapshot))
    outcome = restore_from_snapshot(pools, tasks, params, state_dir)
    assert any(d["action"] == "whole-state replacement"
               for d in outcome["version_diffs"])


def test_replay_divergence_raises_e_replay(sample_dir, state_dir):
    pools, tasks, params = _paths(sample_dir)
    run_session(pools, tasks, params, state_dir)
    snapshot = json.loads((state_dir / "snapshot.json").read_text())
    snapshot["rounds"][0]["water_levels"]["__global__"]["gpu_used"] = 999
    (state_dir / "snapshot.json").write_text(json.dumps(snapshot))
    with pytest.raises(DiagError) as exc:
        restore_from_snapshot(pools, tasks, params, state_dir)
    assert exc.value.code == E_REPLAY
    assert exc.value.details[0]["round"] == 1
