"""Requirements 5 + 7: budget semantics and concurrency determinism."""

from __future__ import annotations

from rag_app.engine import Engine
from rag_app.errors import E_ANSWER_BUDGET_PENDING
from rag_app.models import Command
from rag_app.replay import _comparable_turns, load_inputs
from rag_app.snapshot import SnapshotStore


def _build(sample_dir, snapshot_dir, faults=None, budget=120, concurrency=4):
    config, rules, default_faults, docs, turns, _cmds, failures = load_inputs(
        sample_dir
    )
    config = config.replace(
        first_response_budget_ms=budget, max_concurrency=concurrency
    )
    merged = dict(default_faults)
    if faults:
        merged.update(faults)
    return Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=SnapshotStore(snapshot_dir),
        base_failures=failures,
        faults=merged,
    )


def test_budget_pending_returns_intermediate_status_without_blocking(
    sample_dir, snapshot_dir
):
    faults = {
        "retrieval_delay_ms": 200,
        "retrieval_delay_targets": ["S-A#0001"],
    }
    engine = _build(
        sample_dir, snapshot_dir / "p", faults=faults, budget=50
    )
    command = Command(seq=1, op="set_bm25", args={"k1": 1.9})
    summary = engine.handle_command(command)
    assert summary["code"] == E_ANSWER_BUDGET_PENDING
    assert summary["status"] == "PENDING"
    assert summary["pending_turns"]
    assert summary["elapsed_ms"] <= 50 + 40  # scheduling slack only
    # The latest completed summary is still served immediately.
    assert "current_citations" in summary
    assert "progress" in summary
    drained = engine.drain_pending()
    assert drained["code"] == "OK"
    assert engine.completed_count() == len(engine.records)
    engine.shutdown()


def test_concurrency_levels_produce_identical_results(
    sample_dir, snapshot_dir
):
    payloads = {}
    for level in (1, 2, 4, 8):
        engine = _build(
            sample_dir,
            snapshot_dir / f"c{level}",
            budget=10**6,
            concurrency=level,
        )
        engine.run_full()
        payloads[level] = _comparable_turns(engine)
        order = [row["turn"] for row in engine.turn_results()]
        assert order == sorted(order)
        engine.shutdown()
    for level in (2, 4, 8):
        assert payloads[level] == payloads[1]


def test_max_concurrency_is_capped_at_configured_limit(sample_dir, snapshot_dir):
    engine = _build(sample_dir, snapshot_dir / "cap", concurrency=3)
    engine.run_full()
    assert engine.executor._max_workers == 3
    engine.shutdown()


def test_pending_state_eventually_converges_to_full_equivalence(
    sample_dir, snapshot_dir
):
    faults = {
        "retrieval_delay_ms": 60,
        "retrieval_delay_targets": ["S-A#0001", "S-B#0001"],
    }
    engine = _build(
        sample_dir, snapshot_dir / "p", faults=faults, budget=80
    )
    engine.run_full()
    pending_seen = False
    for seq, op, args in [
        (1, "set_bm25", {"k1": 1.7}),
        (2, "set_recall_top_k", {"recall_top_k": 5}),
        (3, "set_template", {"template": "concise"}),
    ]:
        summary = engine.handle_command(
            Command(seq=seq, op=op, args=args)
        )
        pending_seen = pending_seen or summary["code"] == E_ANSWER_BUDGET_PENDING
    engine.drain_pending()
    assert pending_seen
    assert engine.completed_count() == len(engine.records)
    engine.shutdown()
