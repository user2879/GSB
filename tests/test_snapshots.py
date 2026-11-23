import pytest

from kvdiag.errors import ChainError
from kvdiag.ledger import Ledger
from kvdiag.pipeline import replay_verify, run_pipeline
from kvdiag.log import MemoryLogger
from kvdiag.snapshots import (build_snapshot_rows, check_chain,
                              replay_from_snapshot)
from kvdiag.spec import load_spec


def _run(tree):
    logger = MemoryLogger()
    report = run_pipeline(tree["spec"], tree["db"],
                          renewals_path=tree["renewals"], logger=logger)
    assert report["replay_ok"] is True
    return report


def test_replay_matches_direct_rebuild(tree):
    _run(tree)
    spec = load_spec(tree["spec"])
    ledger = Ledger(tree["db"])
    assert replay_verify(ledger, spec) is True
    ledger.close()


def test_missing_parent_rejected(tree):
    _run(tree)
    ledger = Ledger(tree["db"])
    ledger.conn.execute(
        "UPDATE snapshots SET parent_id='snap-ghost' WHERE parent_id IS NULL")
    ledger.conn.commit()
    head = ledger.query(
        "SELECT snapshot_id FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
    )[0]["snapshot_id"]
    with pytest.raises(ChainError) as exc:
        check_chain(ledger, head)
    assert exc.value.code == "E_CHAIN"
    ledger.close()


def test_cycle_rejected(tree):
    _run(tree)
    ledger = Ledger(tree["db"])
    snaps = ledger.query("SELECT * FROM snapshots ORDER BY snapshot_id")
    first, last = snaps[0], snaps[-1]
    ledger.conn.execute("UPDATE snapshots SET parent_id=? WHERE snapshot_id=?",
                        (last["snapshot_id"], first["snapshot_id"]))
    ledger.conn.commit()
    with pytest.raises(ChainError) as exc:
        check_chain(ledger, last["snapshot_id"])
    assert "cycle" in str(exc.value)
    ledger.close()


def test_missing_shard_on_chain_reported(tree):
    _run(tree)
    spec = load_spec(tree["spec"])
    ledger = Ledger(tree["db"])
    victim = ledger.query(
        "SELECT snapshot_id FROM snapshots WHERE shard_id='shard-2'"
    )[0]["snapshot_id"]
    relay = ledger.query(
        "SELECT snapshot_id FROM snapshots WHERE shard_id='shard-1'"
    )[0]["snapshot_id"]
    ledger.conn.execute("DELETE FROM snapshots WHERE snapshot_id=?", (victim,))
    ledger.conn.execute("UPDATE snapshots SET parent_id=? WHERE shard_id='shard-3'",
                        (relay,))
    ledger.conn.commit()
    head = ledger.query(
        "SELECT snapshot_id FROM snapshots WHERE shard_id='shard-3'"
    )[0]["snapshot_id"]
    with pytest.raises(ChainError) as exc:
        replay_from_snapshot(ledger, spec, head)
    assert exc.value.code == "E_CHAIN"
    assert exc.value.shard_id == "shard-2"
    assert exc.value.details["missing_shard_id"] == "shard-2"
    ledger.close()
