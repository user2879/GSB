from __future__ import annotations

import copy

from mtcomp.cache import sentence_keys
from mtcomp.commands import apply_command
from mtcomp.confidence import (
    ACCEPT,
    REVIEW,
    ConfidenceResult,
    boundary_detail,
    classify_confidence,
)
from mtcomp.engine import Engine, EngineOptions, canonical_sentence_payload
from conftest import make_state, payloads, run_engine


def _events_by_sentence(engine: Engine, run_id: int) -> dict[str, dict]:
    return {
        event["sentence_id"]: event
        for event in engine.ledger_events
        if event["run_id"] == run_id
    }


def test_precise_invalidation_reuses_nondependent_sentence():
    state = make_state(
        [
            ("A", "the red cat", 0),
            ("B", "a b", 0),
            ("C", "the cat runs", 1),
        ]
    )
    engine = run_engine(state)
    before = {row["id"]: row for row in engine.sentence_rows()}
    assert before["B"]["accounting"]["recompute_count"] == 1

    outcome = apply_command(
        engine.state,
        {"op": "lex_frequency", "source": "cat", "target": "猫", "value": 25},
        0,
        engine.config,
    )
    assert outcome.failure is None
    report = engine.run_cycle()
    events = _events_by_sentence(engine, report.run_id)

    assert events["A"]["action"] == "RECOMPUTED"
    assert events["C"]["action"] == "RECOMPUTED"
    assert events["B"]["action"] == "REUSED"
    assert events["B"]["recompute_delta"] == 0
    assert events["B"]["charged_points"] == 0
    assert not report.pending_ids

    after = {row["id"]: row for row in engine.sentence_rows()}
    assert after["B"]["translation"] == before["B"]["translation"]
    assert after["B"]["cache"] == before["B"]["cache"]
    assert after["B"]["accounting"]["recompute_count"] == 1
    assert after["B"]["accounting"]["reuse_count"] == 1


def test_unreferenced_lexicon_addition_does_not_recompute_any_sentence():
    state = make_state(
        [
            ("A", "the red cat", 0),
            ("B", "a b", 0),
        ]
    )
    engine = run_engine(state)
    outcome = apply_command(
        engine.state,
        {
            "op": "lex_add",
            "source": "unused",
            "target": "未用",
            "frequency": 3,
            "alignment_confidence": 0.4,
        },
        0,
        engine.config,
    )
    assert outcome.failure is None
    report = engine.run_cycle()
    assert report.charged_points == 0
    assert all(event["action"] == "REUSED" for event in report.events)
    assert all(event["recompute_delta"] == 0 for event in report.events)


def test_budget_pending_queue_is_retained_and_processed_next_run():
    state = make_state(
        [
            ("A", "a b", 0),
            ("B", "c d", 0),
            ("C", "e f", 1),
        ],
        budget=2,
        ngram=1,
        concurrency=4,
    )
    engine = Engine(state, options=EngineOptions())
    first = engine.run_cycle()
    assert first.pending_ids == {"B"}
    assert first.exhausted
    assert first.exhausted[0]["first_pending_key"] == [0, "B"]
    assert any(
        failure.code == "E_DECODE_BUDGET_PENDING"
        and failure.sentence_id == "B"
        and failure.active
        for failure in engine.failures
    )
    first_rows = {row["id"]: row for row in engine.sentence_rows()}
    assert first_rows["A"]["status"] != "PENDING"
    assert first_rows["C"]["status"] != "PENDING"
    assert first_rows["B"]["status"] == "PENDING"
    first_events = _events_by_sentence(engine, first.run_id)
    assert first_events["A"]["action"] == "RECOMPUTED"
    assert first_events["B"]["action"] == "PENDING"
    assert first_events["C"]["action"] == "RECOMPUTED"

    stable_ac = {
        key: canonical_sentence_payload(first_rows[key]) for key in ("A", "C")
    }
    second = engine.run_cycle()
    assert not second.pending_ids
    second_rows = {row["id"]: row for row in engine.sentence_rows()}
    assert {
        key: canonical_sentence_payload(second_rows[key]) for key in ("A", "C")
    } == stable_ac
    second_events = _events_by_sentence(engine, second.run_id)
    assert second_events["A"]["action"] == "REUSED"
    assert second_events["B"]["action"] == "RECOMPUTED"
    assert second_events["C"]["action"] == "REUSED"
    assert second_events["B"]["remaining_budget_after"] == 0
    assert not any(
        failure.code == "E_DECODE_BUDGET_PENDING" and failure.active
        for failure in engine.failures
    )


def test_thread_pool_matches_serial_execution():
    specs = [
        ("S1", "a b c", 0),
        ("S2", "a c d", 0),
        ("S3", "b d e", 1),
        ("S4", "c e f", 1),
        ("S5", "a d f", 2),
        ("S6", "b c e", 2),
    ]
    serial = run_engine(make_state(specs, concurrency=1))
    parallel = run_engine(make_state(specs, concurrency=8))
    assert payloads(serial) == payloads(parallel)
    serial_static = [
        {
            key: row[key]
            for key in (
                "sentence_id",
                "stable_key",
                "decode_points_static",
                "cache_keys",
                "active_error_code",
            )
        }
        for row in serial.ledger_rows()
    ]
    parallel_static = [
        {
            key: row[key]
            for key in (
                "sentence_id",
                "stable_key",
                "decode_points_static",
                "cache_keys",
                "active_error_code",
            )
        }
        for row in parallel.ledger_rows()
    ]
    assert serial_static == parallel_static
    assert all(
        report.max_workers_used <= 8 for report in parallel.run_reports
    )
    assert max(report.max_workers_used for report in parallel.run_reports) > 1


def test_scheduling_config_does_not_enter_result_cache_keys():
    narrow = make_state([("A", "the red cat", 0)], concurrency=1, budget=5, commit_interval=9)
    wide = make_state([("A", "the red cat", 0)], concurrency=8, budget=500, commit_interval=1)
    sentence_a = narrow.sentences["A"]
    sentence_b = wide.sentences["A"]
    keys_a = sentence_keys(narrow, sentence_a)
    keys_b = sentence_keys(wide, sentence_b)
    assert keys_a.alignment_key == keys_b.alignment_key
    assert keys_a.decode_key == keys_b.decode_key
    assert keys_a.confidence_key == keys_b.confidence_key


def test_confidence_equality_rule_and_flip_causes():
    assert classify_confidence(100, 100) == REVIEW
    base = ConfidenceResult(
        confidence_key="base",
        confidence_fixed=100,
        confidence_permille=391,
        threshold_fixed=100,
        threshold_permille=391,
        status=REVIEW,
        boundary=True,
        fixed_point_bits=8,
        scale=256,
    )
    reference = ConfidenceResult(
        confidence_key="ref",
        confidence_fixed=120,
        confidence_permille=469,
        threshold_fixed=100,
        threshold_permille=391,
        status=ACCEPT,
        boundary=False,
        fixed_point_bits=9,
        scale=512,
    )
    aligned = boundary_detail(
        base, reference, base_path=((0, 1, ("甲",)),), reference_path=((0, 1, ("乙",)),)
    )
    rounded = boundary_detail(
        base, reference, base_path=((0, 1, ("甲",)),), reference_path=((0, 1, ("甲",)),)
    )
    assert aligned["decision_flip"] is True
    assert aligned["flip_cause"] == "ALIGNMENT_QUANTIZATION_RESIDUAL"
    assert aligned["delta_fixed"] == 20
    assert rounded["flip_cause"] == "DECODE_ROUNDING_ACCUMULATION"


def test_repeated_clean_runs_are_byte_stable_payloadwise():
    state = make_state(
        [
            ("S1", "the red cat runs", 0),
            ("S2", "a red cat", 1),
        ],
        lexicon_specs=[
            ("the", "这", 10, 0.9),
            ("red", "红", 9, 0.8),
            ("red", "红色", 9, 0.8),
            ("cat", "猫", 8, 0.95),
            ("runs", "跑", 7, 0.85),
            ("a", "一", 6, 0.9),
        ],
    )
    first = run_engine(copy.deepcopy(state))
    second = run_engine(copy.deepcopy(state))
    assert payloads(first) == payloads(second)
    assert first.ledger_rows() == second.ledger_rows()


def test_budget_tightening_does_not_change_completed_results():
    specs = [
        ("A", "a b", 0),
        ("B", "c d", 0),
        ("C", "e f", 1),
        ("D", "a c", 1),
    ]
    baseline = run_engine(make_state(specs, budget=1000, ngram=1))
    baseline_rows = {row["id"]: row for row in baseline.sentence_rows()}
    constrained = Engine(make_state(specs, budget=2, ngram=1))
    first = constrained.run_cycle()
    assert first.pending_ids
    for row in constrained.sentence_rows():
        if row["id"] in first.pending_ids:
            continue
        assert canonical_sentence_payload(row) == canonical_sentence_payload(
            baseline_rows[row["id"]]
        )
    constrained.run_to_idle()
    final_rows = {row["id"]: row for row in constrained.sentence_rows()}
    assert {
        key: canonical_sentence_payload(row) for key, row in final_rows.items()
    } == {
        key: canonical_sentence_payload(row) for key, row in baseline_rows.items()
    }


def test_batch_partition_does_not_change_translation_payload():
    split_a = run_engine(
        make_state(
            [
                ("S1", "the red cat", 0),
                ("S2", "a red cat", 1),
                ("S3", "the cat runs", 2),
            ]
        )
    )
    split_b = run_engine(
        make_state(
            [
                ("S1", "the red cat", 9),
                ("S2", "a red cat", 9),
                ("S3", "the cat runs", 9),
            ],
            max_batch=3,
        )
    )
    fields = (
        "translation",
        "fragments",
        "confidence_permille",
        "confidence_fixed",
        "status",
        "boundary",
        "boundary_detail",
        "references",
        "cache",
    )
    left = {
        row["id"]: {field: row[field] for field in fields}
        for row in split_a.sentence_rows()
    }
    right = {
        row["id"]: {field: row[field] for field in fields}
        for row in split_b.sentence_rows()
    }
    assert left == right


def test_commuting_edit_order_converges_to_same_results():
    specs = [
        ("S1", "the red cat runs", 0),
        ("S2", "a red cat", 1),
    ]
    commands = [
        {"op": "lex_frequency", "source": "red", "target": "红", "value": 33},
        {"op": "lex_confidence", "source": "cat", "target": "猫", "value": 0.97},
    ]
    left = Engine(make_state(specs))
    right = Engine(make_state(specs))
    for index, command in enumerate(commands):
        apply_command(left.state, command, index, left.config)
    for index, command in enumerate(reversed(commands)):
        apply_command(right.state, command, index, right.config)
    left.run_to_idle()
    right.run_to_idle()
    assert payloads(left) == payloads(right)
    assert left.ledger_rows() == right.ledger_rows()


def test_three_stage_keys_are_distinct_and_parameter_sensitive():
    state = make_state([("A", "the red cat", 0)])
    keys = sentence_keys(state, state.sentences["A"])
    assert len(
        {keys.alignment_key, keys.decode_key, keys.confidence_key}
    ) == 3
    changed = copy.deepcopy(state)
    changed.params = changed.params.with_field("order_weight", 1.5)
    changed_keys = sentence_keys(changed, changed.sentences["A"])
    assert changed_keys.alignment_key != keys.alignment_key
    assert changed_keys.decode_key != keys.decode_key
    assert changed_keys.confidence_key != keys.confidence_key


def test_missing_alignment_recovers_after_dictionary_addition():
    state = make_state(
        [("S1", "the red cat", 0)],
        lexicon_specs=[
            ("the", "这", 10, 0.9),
            ("cat", "猫", 9, 0.9),
        ],
    )
    engine = run_engine(state)
    row = engine.sentence_rows()[0]
    assert row["status"] == "ERROR"
    assert row["error"]["code"] == "E_NO_ALIGNMENT"
    assert any(
        failure.code == "E_NO_ALIGNMENT" and failure.active
        for failure in engine.failures
    )
    outcome = apply_command(
        engine.state,
        {
            "op": "lex_add",
            "source": "red",
            "target": "红",
            "frequency": 8,
            "alignment_confidence": 0.92,
        },
        0,
        engine.config,
    )
    assert outcome.failure is None
    engine.run_cycle()
    row = engine.sentence_rows()[0]
    assert row["status"] != "ERROR"
    assert row["translation"]
    assert not any(
        failure.code == "E_NO_ALIGNMENT" and failure.active
        for failure in engine.failures
    )
