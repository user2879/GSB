"""Requirements 4 + 6: fingerprint invalidation and field-level consistency."""

from __future__ import annotations

from rag_app.engine import Engine
from rag_app.engine_state import apply_command
from rag_app.replay import (
    _comparable_turns,
    _failure_codes,
    diff_results,
    load_inputs,
)
from rag_app.snapshot import SnapshotStore


def _build(sample_dir, snapshot_dir, faults=None):
    config, rules, default_faults, docs, turns, _cmds, failures = load_inputs(
        sample_dir
    )
    merged = dict(default_faults)
    if faults:
        merged.update(faults)
    engine = Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=SnapshotStore(snapshot_dir),
        base_failures=failures,
        faults=merged,
    )
    return engine


def test_incremental_matches_blank_full_run(sample_dir, snapshot_dir):
    config, rules, faults, docs, turns, commands, failures = load_inputs(
        sample_dir
    )
    engine = Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=SnapshotStore(snapshot_dir / "inc"),
        base_failures=failures,
        faults=faults,
    )
    engine.run_full()
    for command in commands:
        engine.handle_command(command)
    engine.drain_pending()
    incremental = _comparable_turns(engine)
    inc_failures = _failure_codes(engine)
    engine.shutdown()

    config2, rules2, faults2, docs2, turns2, _c, failures2 = load_inputs(
        sample_dir
    )
    reference = Engine(
        docs=docs2,
        turns=turns2,
        config=config2,
        rules=rules2,
        store=SnapshotStore(snapshot_dir / "ref"),
        base_failures=failures2,
        faults=faults2,
    )
    for command in commands:
        apply_command(reference.state, command)
    reference._rebuild_if_needed()
    for record in reference.sorted_records():
        record.effective_query = reference.state.effective_query(record.turn)
    reference.run_full()
    full = _comparable_turns(reference)
    ref_failures = _failure_codes(reference)
    reference.shutdown()

    assert diff_results(incremental, full) == []
    assert inc_failures == ref_failures


def test_template_switch_never_reruns_retrieval(sample_dir, snapshot_dir):
    engine = _build(sample_dir, snapshot_dir / "e")
    engine.run_full()
    retrievals_before = engine.stats.retrievals_executed
    from rag_app.models import Command

    command = Command(
        seq=1, op="set_template", args={"template": "concise"}
    )
    engine.handle_command(command)
    engine.drain_pending()
    assert engine.stats.retrievals_executed == retrievals_before
    assert engine.stats.stage_skips["retrieval"] > 0
    assert engine.stats.stage_skips["extraction"] > 0
    assert engine.stats.invalidations["retrieval"] == 0
    engine.shutdown()


def test_context_change_only_invalidates_turn_and_downstream(
    sample_dir, snapshot_dir
):
    engine = _build(sample_dir, snapshot_dir / "e")
    engine.run_full()
    from rag_app.models import Command

    command = Command(
        seq=1,
        op="set_context",
        args={
            "session_id": "S-A",
            "turn_no": 3,
            "is_followup": True,
            "context_ref": [1, 2],
        },
    )
    summary = engine.handle_command(command)
    affected = set(summary["affected_turns"])
    # S-A#3 and every S-A follow-up that transitively references it.
    assert "S-A#0003" in affected
    assert "S-A#0001" not in affected
    assert "S-B#0001" not in affected
    assert all(turn.startswith("S-A#") for turn in affected)
    engine.shutdown()


def test_cache_keys_differ_when_bm25_changes(sample_dir, snapshot_dir):
    engine = _build(sample_dir, snapshot_dir / "e")
    engine.run_full()
    record = engine.sorted_records()[0]
    old_key = record.artifacts.retrieval.key
    from rag_app.models import Command

    engine.handle_command(
        Command(seq=1, op="set_bm25", args={"k1": 2.2})
    )
    engine.drain_pending()
    new_key = record.artifacts.retrieval.key
    assert old_key != new_key
    engine.shutdown()


def test_deterministic_repeat_generation(sample_dir, tmp_path):
    from rag_app.generator import generate

    again = tmp_path / "sample-again"
    generate(again)
    for name in (
        "documents.jsonl",
        "queries.jsonl",
        "commands.jsonl",
        "retrieval_config.json",
        "domain_rules.json",
    ):
        assert (sample_dir / name).read_text(encoding="utf-8") == (
            again / name
        ).read_text(encoding="utf-8")
