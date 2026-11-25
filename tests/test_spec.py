from __future__ import annotations

import json
from pathlib import Path

import pytest

from lease_arbiter.errors import SpecError
from lease_arbiter.spec import load_spec


ROOT = Path(__file__).resolve().parent.parent


def _load_with_events(tmp_path: Path, events: str):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(events, encoding="utf-8")
    return load_spec(ROOT / "config" / "nodes.json", ROOT / "config" / "leases.json", events_path)


def test_sample_declaration_is_valid(sample_spec):
    assert len(sample_spec.nodes) == 2
    assert len(sample_spec.leases) == 4
    assert len(sample_spec.events) == 12
    assert sample_spec.declaration_version == 1_000_001


def test_duplicate_event_seq_reports_first_bad_line(tmp_path):
    events = (
        json.dumps({"seq": 1, "at_ms": 0, "op": "grant", "lease_id": "a-cache-1", "expected_label": 0}) + "\n"
        + json.dumps({"seq": 1, "at_ms": 0, "op": "grant", "lease_id": "a-cache-2", "expected_label": 0}) + "\n"
    )
    with pytest.raises(SpecError) as caught:
        _load_with_events(tmp_path, events)
    error = caught.value
    assert error.code == "E_SPEC"
    assert error.details["field"] == "seq"
    assert error.details["line"] == 2


def test_non_monotonic_at_ms_reports_line(tmp_path):
    events = (
        json.dumps({"seq": 1, "at_ms": 10, "op": "grant", "lease_id": "a-cache-1", "expected_label": 0}) + "\n"
        + json.dumps({"seq": 2, "at_ms": 9, "op": "release", "lease_id": "a-cache-1", "expected_label": 1}) + "\n"
    )
    with pytest.raises(SpecError) as caught:
        _load_with_events(tmp_path, events)
    assert caught.value.details["field"] == "at_ms"
    assert caught.value.details["line"] == 2


def test_unknown_event_field_reports_line(tmp_path):
    events = json.dumps(
        {
            "seq": 1,
            "at_ms": 0,
            "op": "grant",
            "lease_id": "a-cache-1",
            "expected_label": 0,
            "surprise": True,
        }
    ) + "\n"
    with pytest.raises(SpecError) as caught:
        _load_with_events(tmp_path, events)
    assert caught.value.details["field"] == "surprise"
    assert caught.value.details["line"] == 1


def test_host_must_be_literal_loopback(tmp_path):
    payload = json.loads((ROOT / "config" / "nodes.json").read_text(encoding="utf-8"))
    payload["nodes"][0]["host"] = "localhost"
    nodes_path = tmp_path / "nodes.json"
    nodes_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SpecError) as caught:
        load_spec(nodes_path, ROOT / "config" / "leases.json", ROOT / "samples" / "events.jsonl")
    assert caught.value.details["field"] == "nodes[0].host"


def test_duplicate_port_is_rejected(tmp_path):
    payload = json.loads((ROOT / "config" / "nodes.json").read_text(encoding="utf-8"))
    payload["nodes"][1]["port"] = payload["nodes"][0]["port"]
    nodes_path = tmp_path / "nodes.json"
    nodes_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SpecError) as caught:
        load_spec(nodes_path, ROOT / "config" / "leases.json", ROOT / "samples" / "events.jsonl")
    assert caught.value.details["field"] == "nodes[1].port"


def test_renew_interval_must_be_strictly_smaller_than_ttl(tmp_path):
    payload = json.loads((ROOT / "config" / "leases.json").read_text(encoding="utf-8"))
    payload["leases"][0]["renew_interval_ms"] = payload["leases"][0]["ttl_ms"]
    leases_path = tmp_path / "leases.json"
    leases_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SpecError) as caught:
        load_spec(ROOT / "config" / "nodes.json", leases_path, ROOT / "samples" / "events.jsonl")
    assert caught.value.details["field"] == "leases[0].renew_interval_ms"


def test_non_positive_global_quota_is_rejected(tmp_path):
    payload = json.loads((ROOT / "config" / "nodes.json").read_text(encoding="utf-8"))
    payload["global_concurrency"] = 0
    nodes_path = tmp_path / "nodes.json"
    nodes_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SpecError) as caught:
        load_spec(nodes_path, ROOT / "config" / "leases.json", ROOT / "samples" / "events.jsonl")
    assert caught.value.details["field"] == "global_concurrency"
