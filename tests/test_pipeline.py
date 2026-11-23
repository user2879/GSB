import json

from kvdiag.ledger import Ledger
from kvdiag.log import MemoryLogger
from kvdiag.pipeline import run_pipeline
from kvdiag.spec import load_spec


def test_full_pipeline_converges_and_commits(tree):
    logger = MemoryLogger()
    report = run_pipeline(tree["spec"], tree["db"],
                          renewals_path=tree["renewals"], logger=logger)
    assert report["replay_ok"] is True
    assert len(report["batches"]) == 2
    for conclusion in report["conclusions"].values():
        assert conclusion["status"] == "converged"
        assert conclusion["diff"] == 0

    ledger = Ledger(tree["db"])
    spec = load_spec(tree["spec"])
    assert set(ledger.committed_shards()) == set(spec.shard_ids())
    entries = ledger.query("SELECT * FROM index_entries")
    assert entries
    for e in entries:
        assert e["shard_id"] in spec.shard_ids()
        assert len(e["key_digest"]) == 16
        assert e["index_version"] >= 1
    conclusions = ledger.query("SELECT * FROM conclusions")
    assert len(conclusions) == len(spec.shards)
    assert ledger.format_version() == 1
    assert ledger.get_meta("content_digest") == ledger.compute_digest()
    ledger.close()


def test_pipeline_is_deterministic_across_runs(tree, tmp_path):
    logger = MemoryLogger()
    run_pipeline(tree["spec"], tree["db"], renewals_path=tree["renewals"],
                 logger=logger)
    ledger = Ledger(tree["db"])
    first = {t: ledger.query("SELECT * FROM %s" % t)
             for t in ("index_entries", "snapshots", "conclusions")}
    ledger.close()

    db2 = tmp_path / "second.sqlite"
    run_pipeline(tree["spec"], db2, renewals_path=tree["renewals"],
                 logger=MemoryLogger())
    ledger = Ledger(db2)
    second = {t: ledger.query("SELECT * FROM %s" % t)
              for t in ("index_entries", "snapshots", "conclusions")}
    ledger.close()
    # Positions, digests and conclusions are identical; wall-clock lease
    # timestamps are intentionally excluded from this comparison.
    assert first == second


def test_wait_events_persisted(tree):
    run_pipeline(tree["spec"], tree["db"], renewals_path=tree["renewals"],
                 logger=MemoryLogger())
    ledger = Ledger(tree["db"])
    events = ledger.query("SELECT * FROM wait_events")
    ledger.close()
    assert events
    for e in events:
        assert e["wait_end_mono_ms"] >= e["wait_start_mono_ms"]
