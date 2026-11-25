from __future__ import annotations

import copy

from vscan.planner import compile_plan
from vscan.spec import parse_spec


def test_plan_summary_is_order_and_batch_size_invariant(spec_dict):
    first = compile_plan(parse_spec(copy.deepcopy(spec_dict)))
    reordered = copy.deepcopy(spec_dict)
    reordered["queries"][0]["project"].reverse()
    reordered["queries"][0]["predicate"]["args"].reverse()
    second = compile_plan(parse_spec(reordered))
    assert first.to_dict(include_batches=False)["plan_summary_digest"] == second.to_dict(
        include_batches=False
    )["plan_summary_digest"]

    summaries = set()
    for batch_rows in (1, 3, 4, 7, 100):
        plan = compile_plan(parse_spec(copy.deepcopy(spec_dict)), batch_rows=batch_rows)
        summaries.add(plan.to_dict(include_batches=False)["plan_summary_digest"])
    assert len(summaries) == 1


def test_batch_boundaries_cover_every_row_exactly(spec):
    plan = compile_plan(spec)
    batches = [batch for batch in plan.batches() if batch.segment_id == "s1"]
    assert [(batch.start_row, batch.end_row) for batch in batches] == [
        (0, 4),
        (4, 8),
        (8, 12),
    ]
    assert sum(batch.rows for batch in batches) == 12
    cursor = 0
    for batch in batches:
        assert batch.start_row == cursor
        cursor = batch.end_row
    assert cursor == 12


def test_column_pruning_and_pushdown_derivation(spec):
    query_plan = compile_plan(spec).queries[0]
    segment = query_plan.per_segment[0]
    assert segment["decode_columns"] == ["g", "r", "v"]
    assert segment["pruned_columns"] == ["unused"]
    assert query_plan.batches[0].predicate_columns == ("g", "v")
    assert query_plan.batches[0].output_columns == ("g", "r", "v")


def test_boundaries_digest_is_stable_for_same_batch_size(spec):
    first = compile_plan(spec)
    second = compile_plan(spec)
    assert first.queries[0].batch_boundaries_digest == second.queries[0].batch_boundaries_digest
