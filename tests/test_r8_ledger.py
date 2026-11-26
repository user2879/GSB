import json

import pytest

from ckpt_diag.compute import compute_batch
from ckpt_diag.errors import ReplayError
from ckpt_diag.ledger import LEDGER_FORMAT_VERSION, Ledger
from ckpt_diag.samplegen import build_spec_dict
from ckpt_diag.spec import load_spec


def _batch(spec, batch_id="batch-1"):
    shard = spec.shard("shard-alpha")
    result = compute_batch(shard, batch_id, 1, ("s-0", "s-1"), ("L0",))
    return {
        "batch_id": batch_id, "shard_id": shard.shard_id, "cost_units": 1,
        "param_version": result.param_version,
        "param_summary": result.param_summary,
        "loss_summary": result.loss_summary,
        "total_loss": result.total_loss,
        "sample_ids": list(result.sample_ids),
        "scope_layers": list(result.scope_layers),
        "elapsed_ms": 3,
    }, result


def _checkpoints(spec):
    return [{"checkpoint_id": cp.checkpoint_id, "shard_id": cp.shard_id,
             "generation": cp.generation, "param_version": cp.param_version,
             "loss_baseline": cp.loss_baseline, "state": "frozen"}
            for cp in spec.checkpoints]


def test_atomic_commit_writes_all_tables(tmp_path):
    spec = load_spec(_write(tmp_path))
    ledger = Ledger(tmp_path / "ledger.db")
    batch, _ = _batch(spec)
    lease = {"lease_id": "lease-1", "shard_id": "shard-alpha",
             "lease_version": 1, "param_version": 1, "granted_ms": 0,
             "expires_ms": 10, "state": "released"}
    ownership = [{"unit_index": 0, "owner_kind": "active",
                  "shard_id": "shard-alpha", "lease_id": "lease-1",
                  "lease_version": 1, "param_version": 1, "event_seq": 1,
                  "basis": "grant_from_free_pool"}]
    ledger.commit_batch(seq=1, batch=batch, lease=lease,
                        ownership_records=ownership, decisions=[],
                        checkpoint_states=_checkpoints(spec))
    assert len(ledger.committed_batches()) == 1
    assert ledger.progress()["last_seq"] == 1
    assert ledger.conn.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 1
    assert ledger.conn.execute("SELECT COUNT(*) FROM ownership").fetchone()[0] == 1
    ledger.close()


def test_snapshot_version_mismatch_replaces_whole_db(tmp_path):
    spec = load_spec(_write(tmp_path))
    path = tmp_path / "ledger.db"
    ledger = Ledger(path)
    batch, _ = _batch(spec)
    ledger.commit_batch(seq=1, batch=batch, lease=None,
                        ownership_records=[], decisions=[],
                        checkpoint_states=_checkpoints(spec))
    ledger.close()
    reopened = Ledger(path).open_snapshot(expected_version=999,
                                          expected_digest=None)
    assert reopened.expected_version() == LEDGER_FORMAT_VERSION
    assert len(reopened.committed_batches()) == 0
    reason = reopened.conn.execute(
        "SELECT value FROM meta WHERE key='last_replace_reason'").fetchone()[0]
    assert reason == "format_version_mismatch"
    reopened.close()


def test_snapshot_digest_mismatch_replaces(tmp_path):
    spec = load_spec(_write(tmp_path))
    path = tmp_path / "ledger.db"
    ledger = Ledger(path)
    batch, _ = _batch(spec)
    ledger.commit_batch(seq=1, batch=batch, lease=None,
                        ownership_records=[], decisions=[],
                        checkpoint_states=_checkpoints(spec))
    reopened = ledger.open_snapshot(expected_version=1,
                                    expected_digest="deadbeef")
    assert len(reopened.committed_batches()) == 0
    reopened.close()


def test_crash_recovery_marks_pending_for_rerun(tmp_path):
    ledger = Ledger(tmp_path / "crash.db")
    ledger.mark_pending("batch-lost", seq=7, note="killed before commit")
    outcome = ledger.recover_after_crash()
    assert outcome.rerun_required == ["batch-lost"]
    assert outcome.verified_batches == 0
    ledger.close()


def test_replay_recomputes_and_detects_first_diff(tmp_path):
    spec = load_spec(_write(tmp_path))
    ledger = Ledger(tmp_path / "replay.db")
    batch, result = _batch(spec)
    ledger.commit_batch(seq=1, batch=batch, lease=None,
                        ownership_records=[], decisions=[],
                        checkpoint_states=_checkpoints(spec))

    def recompute(row):
        fresh = compute_batch(spec.shard(row["shard_id"]), row["batch_id"],
                              row["param_version"],
                              tuple(json.loads(row["sample_ids"])),
                              tuple(json.loads(row["scope_layers"])))
        return {"param_summary": fresh.param_summary,
                "loss_summary": fresh.loss_summary,
                "total_loss": fresh.total_loss,
                "param_version": fresh.param_version,
                "sample_ids": fresh.sample_ids}

    outcome = ledger.verify_replay(recompute)
    assert outcome.verified_batches == 1 and outcome.first_diff is None

    def corrupt_recompute(row):
        good = recompute(row)
        good["total_loss"] += 999
        return good

    with pytest.raises(ReplayError) as exc:
        ledger.verify_replay(corrupt_recompute)
    assert exc.value.code == "E_REPLAY"
    assert exc.value.field == "total_loss"
    ledger.close()


def _write(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(build_spec_dict()))
    return path
