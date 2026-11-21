from __future__ import annotations

from trace_arbiter.adjudicate import authoritative_summary, batch_decision_summary


def test_summary_is_order_independent_and_flags_rule_difference(sample_dataset) -> None:
    nodes = sample_dataset.nodes_payload()["nodes"]
    spans = sample_dataset.spans_payload()
    first = authoritative_summary(nodes, spans, ruleset_version=1)
    second = authoritative_summary(list(reversed(nodes)), list(reversed(spans)), ruleset_version=1)
    assert first[0].encode("utf-8") == second[0].encode("utf-8")
    traces = {trace["trace_id"]: trace for trace in first[2]["trace_decisions"]}
    cross = traces["trace-cross"]
    assert cross["root_span_id"] == "tc1"
    assert cross["sampled_node_ids"] == ["api"]
    assert cross["rule_difference"]["detected"] is True
    assert cross["rule_difference"]["span_ids"] == ["tc1", "tc2", "tc3"]
    worker_span = next(item for item in cross["span_decisions"] if item["span_id"] == "tc2")
    assert worker_span["sampled"] is False
    assert worker_span["downstream_visible"] is False
    assert worker_span["reported_sampled_matches"] is True


def test_target_spans_disambiguate_duplicate_span_ids() -> None:
    nodes = [
        {
            "node_id": "api",
            "host": "127.0.0.1",
            "port": 40001,
            "propagation_rule_version": 1,
            "sample_ratio_bps": 10000,
            "random_seed": 1,
        }
    ]
    spans = [
        {
            "trace_id": "trace-a",
            "span_id": "same",
            "parent_span_id": None,
            "node_id": "api",
            "timestamp_ms": 1,
            "duration_us": 1,
            "sampled": True,
        },
        {
            "trace_id": "trace-b",
            "span_id": "same",
            "parent_span_id": None,
            "node_id": "api",
            "timestamp_ms": 1,
            "duration_us": 1,
            "sampled": True,
        },
    ]
    summary, _digest, payload = batch_decision_summary(
        nodes,
        spans,
        ruleset_version=1,
        target_node_id="api",
        target_spans=[{"trace_id": "trace-b", "span_id": "same"}],
    )
    assert b'"trace_id":"trace-b"' in summary.encode("utf-8")
    assert b'"trace_id":"trace-a"' not in summary.encode("utf-8")
    assert payload["target_spans"] == [{"span_id": "same", "trace_id": "trace-b"}]
