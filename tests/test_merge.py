from receipt_ledger.merge import build_batch_plan, merge_receipts


def _receipt(merge_seq, param_version, fmt):
    return {
        "merge_seq": merge_seq, "param_version": param_version,
        "source_format": fmt, "batch_id": "b",
        "output_digest": "d", "filled_fields": [],
    }


def test_merge_prefers_param_then_format_then_seq():
    receipts = [
        _receipt(0, 1, "r3"),
        _receipt(1, 2, "r1"),
        _receipt(2, 2, "r3"),
    ]
    winner, rejected = merge_receipts(receipts)
    assert winner["merge_seq"] == 2
    reasons = {item["merge_seq"]: item["reason"] for item in rejected}
    assert reasons[0] == "superseded:lower-param-version"
    assert reasons[1] == "superseded:lower-format-rank"


def test_batch_plan_stable_and_complete(v1_spec):
    plan_a = build_batch_plan(v1_spec)
    plan_b = build_batch_plan(v1_spec)
    assert [t.batch_id for t in plan_a] == [t.batch_id for t in plan_b]
    for rule in v1_spec.rules:
        rule_tasks = [t for t in plan_a if t.rule_id == rule.rule_id]
        covered = sorted(seq for t in rule_tasks for seq in t.request_seqs)
        assert covered == sorted(r.seq for r in v1_spec.requests)
        assert all(len(t.requests) <= rule.batch_size for t in rule_tasks)
