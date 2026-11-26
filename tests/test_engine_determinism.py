from pathlib import Path

from dispatch_tokens.engine import SchedulerEngine
from dispatch_tokens.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[1] / "samples" / "demo_spec.json"


def _serialized(result):
    import json
    from dataclasses import asdict
    payload = {
        "outcomes": {k: asdict(v) for k, v in result.outcomes.items()},
        "refill_audit": result.refill_audit,
        "decisions": result.decisions,
        "water_levels": result.water_levels,
        "invalidations": result.invalidations,
        "carryovers": result.revision_carryovers,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def test_two_runs_are_byte_identical():
    spec = load_spec(SPEC_PATH)
    first = SchedulerEngine(spec).run()
    second = SchedulerEngine(spec).run()
    assert _serialized(first) == _serialized(second)


def test_arbitrary_event_batching_keeps_balances():
    spec = load_spec(SPEC_PATH)
    full = SchedulerEngine(spec).run()
    # 把事件切成多个批次逐段推进，时钟仍由事件 ts 驱动，结论必须一致。
    engine = SchedulerEngine(spec)
    events = list(spec.events)
    mid = len(events) // 2
    engine.run(events[:mid])
    split_result = engine.run(events[mid:])
    assert split_result.water_levels == full.water_levels
    assert split_result.refill_audit == full.refill_audit
    assert split_result.decisions == full.decisions


def test_revision_carryover_present_and_balanced():
    spec = load_spec(SPEC_PATH)
    result = SchedulerEngine(spec).run()
    scale_changes = [c for c in result.revision_carryovers
                     if c["old_scale"] != c["new_scale"]]
    assert scale_changes
    for carry in scale_changes:
        assert carry["carried_balance"] >= 0
        assert carry["refill"]["after_clamp"] >= 0


def test_dispatch_plans_carry_error_metadata():
    spec = load_spec(SPEC_PATH)
    result = SchedulerEngine(spec).run()
    for plan in result.dispatch_plans:
        assert "error_influenced" in plan
        assert "comparison_error_bound" in plan
        assert isinstance(plan["gap"], int)
