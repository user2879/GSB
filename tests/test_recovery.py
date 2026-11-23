import json
import subprocess
import sys
from pathlib import Path

from kvdiag.ledger import Ledger
from kvdiag.log import MemoryLogger
from kvdiag.pipeline import replay_verify, run_pipeline
from kvdiag.recovery import recover, write_session_snapshot
from kvdiag.spec import load_spec


def test_digest_mismatch_triggers_wholesale_replace(tree):
    logger = MemoryLogger()
    run_pipeline(tree["spec"], tree["db"], renewals_path=tree["renewals"],
                 session_out=str(tree["root"] / "session.json"), logger=logger)
    # Tamper one committed row on disk, then recover from the session file.
    ledger = Ledger(tree["db"])
    ledger.conn.execute(
        "UPDATE conclusions SET diff=99 WHERE shard_id='shard-1'")
    ledger.conn.commit()
    ledger.close()
    spec = load_spec(tree["spec"])
    ledger = Ledger(tree["db"])
    report = recover(spec, ledger, tree["root"] / "session.json", logger=logger)
    assert report["replaced"] is True
    assert report["diffs"] == ["conclusions:shard-1:content-differs"]
    row = ledger.query("SELECT diff FROM conclusions WHERE shard_id='shard-1'")
    assert row[0]["diff"] == 0  # restored from session, not merged
    assert replay_verify(ledger, spec) is True
    ledger.close()
    codes = [r["code"] for r in logger.records]
    assert "E_SNAPSHOT" in codes


def test_clean_reopen_does_not_replace(tree):
    session = tree["root"] / "session.json"
    run_pipeline(tree["spec"], tree["db"], renewals_path=tree["renewals"],
                 session_out=str(session), logger=MemoryLogger())
    spec = load_spec(tree["spec"])
    ledger = Ledger(tree["db"])
    report = recover(spec, ledger, session, logger=MemoryLogger())
    assert report["replaced"] is False
    assert report["diffs"] == []
    assert report["unknown_shards"] == []
    ledger.close()


def test_forced_termination_recovers_committed_batches_only(tree):
    db = str(tree["db"])
    session = str(tree["root"] / "session.json")
    cmd = [sys.executable, "-m", "kvdiag", "run", "--spec", str(tree["spec"]),
           "--db", db, "--renewals", str(tree["renewals"]),
           "--session-out", session, "--crash-before-batch", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 3  # hard termination, batch 2 never committed

    spec = load_spec(tree["spec"])
    ledger = Ledger(db)
    logger = MemoryLogger()
    report = recover(spec, ledger, session, logger=logger)
    assert report["unknown_shards"] == ["shard-2", "shard-3"]
    states = {r["shard_id"]: r["state"]
              for r in ledger.query("SELECT * FROM progress")}
    assert states["shard-2"] == "unknown"
    ledger.close()

    # Rerun the unknown shards; the chain continues from committed batches.
    run_pipeline(tree["spec"], db, renewals_path=str(tree["renewals"]),
                 session_out=session, only_shards=report["unknown_shards"],
                 logger=logger)
    ledger = Ledger(db)
    assert set(ledger.committed_shards()) == set(spec.shard_ids())
    assert replay_verify(ledger, spec) is True
    ledger.close()


def test_batch_commit_is_atomic(tree):
    # A failing batch must leave no partial rows behind.
    spec = load_spec(tree["spec"])
    ledger = Ledger(tree["db"])
    try:
        ledger.conn.execute("BEGIN")
        ledger.conn.execute(
            "INSERT INTO progress VALUES ('shard-0', 1, 'committed')")
        raise RuntimeError("simulated crash before COMMIT")
    except RuntimeError:
        ledger.conn.execute("ROLLBACK")
    assert ledger.committed_shards() == []
    ledger.close()
