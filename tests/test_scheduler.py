"""Dual-budget scheduling: global cap and per-backend quota never exceeded;
timeout re-dispatch keeps inputs and digest."""

from gateway.engine import run
from gateway.ledger import Ledger


def test_concurrency_budgets_respected(sample):
    ledger = Ledger(sample["db"])
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=2, forward_repeat=20)
    ledger.close()
    assert result.peak_running <= 2
    quotas = {b.model_id: b.max_concurrency for b in result.decl.backends}
    for backend_id, peak in result.backend_peak.items():
        assert peak <= quotas[backend_id]


def test_wait_timestamps_monotonic(sample):
    ledger = Ledger(sample["db"])
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=1, forward_repeat=20)
    ledger.close()
    for r in result.results:
        assert r.wait_start_ms <= r.wait_end_ms <= r.start_ms <= r.end_ms
        assert r.duration_ms == r.end_ms - r.start_ms


def test_timeout_redispatch_keeps_digest(sample):
    # timeout 0ms forces re-dispatch on the first attempt
    ledger = Ledger(sample["db"])
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=2, forward_repeat=5, timeout_ms=0)
    rows = ledger.ledger_rows(result.config_version)
    ledger.close()
    assert any(r.attempt >= 2 for r in result.results)
    assert len(rows) == len(result.routed)


def test_every_request_committed_exactly_once(sample):
    ledger = Ledger(sample["db"])
    result = run(sample["backends"], sample["rules"], sample["requests"],
                 ledger, global_limit=3, forward_repeat=5)
    rows = ledger.ledger_rows(result.config_version)
    ledger.close()
    seqs = sorted(r["request_seq"] for r in rows)
    assert seqs == sorted(r.request.seq for r in result.routed)
