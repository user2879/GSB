"""Headless validation of the Tkinter view model (stable render order)."""

from gateway.engine import run
from gateway.ledger import Ledger
from gateway.ui import build_view_model


def test_view_model_stable_order(sample):
    ledger = Ledger(sample["db"])
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=2, forward_repeat=5)
    view = build_view_model(result, ledger)
    ledger.close()
    seqs = [row["request_seq"] for row in view["ledger"]]
    assert seqs == sorted(seqs)
    ids = [b["model_id"] for b in view["backends"]]
    assert ids == sorted(ids)
    starts = [(t["run_start"], t["batch_id"]) for t in view["timeline"]]
    assert starts == sorted(starts)
    assert view["config"]["version"] == 1
    assert view["strategy"]["tie_threshold"] == 1
    assert len(view["config"]["snapshots"]) == 1
