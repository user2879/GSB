from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_arbiter.errors import ArbiterError
from trace_arbiter.spec import load_dataset

from conftest import SAMPLE_SPANS, make_nodes, write_json, write_spans


def _valid_span() -> dict:
    return {
        "trace_id": "trace-a",
        "span_id": "s1",
        "parent_span_id": None,
        "node_id": "api",
        "timestamp_ms": 1000,
        "duration_us": 100,
        "sampled": True,
    }


def test_spec_node_rejections(tmp_path: Path) -> None:
    spans = write_spans(tmp_path / "spans.ndjson", [_valid_span()])

    duplicate_id = make_nodes()
    duplicate_id["nodes"][1]["node_id"] = duplicate_id["nodes"][0]["node_id"]
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(write_json(tmp_path / "duplicate.json", duplicate_id)), str(spans))
    assert exc.value.code == "E_SPEC"
    assert exc.value.path == "/nodes/1/node_id"

    bad_host = make_nodes()
    bad_host["nodes"][0]["host"] = "localhost"
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(write_json(tmp_path / "host.json", bad_host)), str(spans))
    assert exc.value.path.endswith("/host")

    duplicate_port = make_nodes()
    duplicate_port["nodes"][1]["port"] = duplicate_port["nodes"][0]["port"]
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(write_json(tmp_path / "port.json", duplicate_port)), str(spans))
    assert "duplicate loopback port" in exc.value.message

    decreasing = make_nodes()
    decreasing["nodes"][0]["propagation_rule_version"] = 3
    decreasing["nodes"][1]["propagation_rule_version"] = 2
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(write_json(tmp_path / "version.json", decreasing)), str(spans))
    assert exc.value.path == "/nodes/1/propagation_rule_version"

    bad_ratio = make_nodes()
    bad_ratio["nodes"][0]["sample_ratio_bps"] = 10_001
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(write_json(tmp_path / "ratio.json", bad_ratio)), str(spans))
    assert exc.value.path.endswith("/sample_ratio_bps")

    inconsistent = make_nodes()
    inconsistent["nodes"][1]["propagation_rule_version"] = 1
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(write_json(tmp_path / "cohort.json", inconsistent)), str(spans))
    assert "sharing a propagation_rule_version" in exc.value.message


def test_spec_span_errors_include_first_line(tmp_path: Path) -> None:
    nodes = write_json(tmp_path / "nodes.json", make_nodes())
    missing = _valid_span()
    missing["span_id"] = "s2"
    missing["parent_span_id"] = "missing"
    missing["timestamp_ms"] = 1010
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(nodes), str(write_spans(tmp_path / "missing.ndjson", [_valid_span(), missing])))
    assert exc.value.code == "E_SPEC"
    assert exc.value.line == 2
    assert exc.value.field == "parent_span_id"

    cycle_a = _valid_span()
    cycle_a["span_id"] = "a"
    cycle_a["parent_span_id"] = "b"
    cycle_b = _valid_span()
    cycle_b["span_id"] = "b"
    cycle_b["parent_span_id"] = "a"
    cycle_b["timestamp_ms"] = 1000
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(nodes), str(write_spans(tmp_path / "cycle.ndjson", [cycle_a, cycle_b])))
    assert exc.value.code == "E_SPEC"
    assert "cycle" in exc.value.message

    child_earlier = _valid_span()
    child_earlier["span_id"] = "s2"
    child_earlier["parent_span_id"] = "s1"
    child_earlier["timestamp_ms"] = 999
    with pytest.raises(ArbiterError) as exc:
        load_dataset(
            str(nodes),
            str(write_spans(tmp_path / "time.ndjson", [_valid_span(), child_earlier])),
        )
    assert exc.value.field == "timestamp_ms"
    assert exc.value.line == 2


def test_documented_error_injections(tmp_path: Path) -> None:
    nodes = write_json(tmp_path / "nodes.json", make_nodes())
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(nodes), str(SAMPLE_SPANS), inject="corrupt-message")
    assert exc.value.code == "E_SPEC"
    assert exc.value.line == 1
    with pytest.raises(ArbiterError) as exc:
        load_dataset(str(nodes), str(SAMPLE_SPANS), inject="parent-missing")
    assert exc.value.code == "E_CHAIN"
    assert exc.value.field == "parent_span_id"


def test_input_order_does_not_change_validation(tmp_path: Path) -> None:
    nodes = write_json(tmp_path / "nodes.json", make_nodes())
    lines = SAMPLE_SPANS.read_text(encoding="utf-8").splitlines()
    forward = load_dataset(str(nodes), str(write_spans(tmp_path / "f.ndjson", [json.loads(line) for line in lines])))
    reverse = load_dataset(
        str(nodes),
        str(write_spans(tmp_path / "r.ndjson", [json.loads(line) for line in reversed(lines)])),
    )
    assert forward.content_digest == reverse.content_digest
