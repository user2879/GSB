from dialog_state_pipeline.errors import PipelineError
from dialog_state_pipeline.slots import (
    apply_updates, effective_weighted_value, make_records, parse_value,
    validate_confidence,
)


def test_type_parsing_and_constraints(fsm):
    assert parse_value("city", "beijing", fsm) == "beijing"
    assert parse_value("score", 42, fsm) == 42
    assert parse_value("temperature", 21.5, fsm) == 21.5
    assert parse_value("bool", True, fsm) is True

    for bad in ("mars", 123, None):
        try:
            parse_value("city", bad, fsm)
        except PipelineError as exc:
            assert exc.code == "E_SLOT_INVALID"
        else:  # pragma: no cover
            raise AssertionError("expected E_SLOT_INVALID")

    try:
        parse_value("score", 101, fsm)
    except PipelineError as exc:
        assert exc.code == "E_SLOT_INVALID"
    else:  # pragma: no cover
        raise AssertionError("expected range failure")


def test_confidence_is_fixed_point_integer(config):
    validate_confidence(0, config.confidence_max)
    validate_confidence(10000, config.confidence_max)
    for bad in (0.5, True, -1, 10001, "9000"):
        try:
            validate_confidence(bad, config.confidence_max)
        except PipelineError as exc:
            assert exc.code == "E_SLOT_INVALID"
        else:  # pragma: no cover
            raise AssertionError(f"{bad!r} should be rejected")


def test_four_merge_strategies(fsm, config):
    records = make_records(fsm)
    apply_updates(records, [
        {"slot": "city", "value": "beijing", "confidence": 9000},
        {"slot": "visit_count", "value": 10, "confidence": 8000},
        {"slot": "comfort", "value": 20.0, "confidence": 5000},
        {"slot": "member_tier", "value": "gold", "confidence": 9000},
    ], 1, "B1", fsm, config)

    # overwrite
    apply_updates(records, [
        {"slot": "city", "value": "shanghai", "confidence": 7000}],
        2, "B1", fsm, config)
    assert records["city"].value == "shanghai"
    assert records["city"].confidence == 7000

    # larger: smaller value skipped, larger accepted
    apply_updates(records, [
        {"slot": "visit_count", "value": 5, "confidence": 9000}],
        2, "B1", fsm, config)
    assert records["visit_count"].value == 10
    apply_updates(records, [
        {"slot": "visit_count", "value": 25, "confidence": 9000}],
        3, "B1", fsm, config)
    assert records["visit_count"].value == 25

    # reject overwrite: first write wins
    apply_updates(records, [
        {"slot": "member_tier", "value": "platinum", "confidence": 9000}],
        2, "B1", fsm, config)
    assert records["member_tier"].value == "gold"
    assert records["member_tier"].history[-1]["action"] == "reject_overwrite"


def test_weighted_accumulation_fixed_order(fsm, config):
    records = make_records(fsm)
    updates = [
        (10.0, 2000),
        (20.0, 3000),
        (30.0, 5000),
    ]
    for idx, (value, conf) in enumerate(updates, start=1):
        apply_updates(records,
                      [{"slot": "comfort", "value": value, "confidence": conf}],
                      idx, "B1", fsm, config)
    record = records["comfort"]
    # (10*2000 + 20*3000 + 30*5000) / 10000 = 23.0
    assert record.weight_total == 10000
    assert record.value == 23.0
    # Fixed-point accumulator: scaled by 10**6.
    assert record.weighted_sum == 23_000_000 * 10_000
    # provenance records strategy and source turn for every merge
    assert [h["turn"] for h in record.history] == [1, 2, 3]
    assert all(h["strategy"] == "weighted" for h in record.history)


def test_invalid_slot_leaves_record_untouched(fsm, config):
    records = make_records(fsm)
    apply_updates(records,
                  [{"slot": "comfort", "value": 18.0, "confidence": 9000}],
                  1, "B1", fsm, config)
    failures = apply_updates(records, [
        {"slot": "comfort", "value": "garbage", "confidence": 9000},
        {"slot": "comfort", "value": 22.0, "confidence": 8000},
    ], 2, "B1", fsm, config)
    assert len(failures) == 1
    assert failures[0]["code"] == "E_SLOT_INVALID"
    # The valid second operator still applied; the bad one changed nothing.
    assert records["comfort"].weight_total == 9000 + 8000

