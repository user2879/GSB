from __future__ import annotations

import json
from pathlib import Path

import pytest

from cachelease.errors import SpecError
from cachelease.spec import load_config, load_events

from .helpers import one_node_config, write_events


def test_valid_config_and_events_load(tmp_path: Path) -> None:
    raw, path = one_node_config(tmp_path)
    config = load_config(path)
    events_path = write_events(
        tmp_path / "events.ndjson",
        [
            {"seq": 1, "batch": 1, "type": "write", "artifact": "core", "lease_ms": 1000},
            {"seq": 2, "batch": 1, "type": "read", "artifact": "core", "lease_ms": 1000},
        ],
    )
    events = load_events(events_path, config)
    assert len(events) == 2
    assert events[1].seq == 2


def test_duplicate_node_id_reports_first_location(tmp_path: Path) -> None:
    raw, path = one_node_config(tmp_path)
    raw["nodes"].append(dict(raw["nodes"][0], port=raw["nodes"][0]["port"]))
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_config(path)
    assert exc.value.code == "E_SPEC"
    assert "nodes[1].port" in (exc.value.location or "") or "nodes[1].node_id" in (exc.value.location or "")


def test_nonliteral_loopback_and_duplicate_ports_are_rejected(tmp_path: Path) -> None:
    raw, path = one_node_config(tmp_path)
    raw["nodes"][0]["host"] = "localhost"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_config(path)
    assert "nodes[0].host" in (exc.value.location or "")

    raw, path = one_node_config(tmp_path / "ports")
    second = dict(raw["nodes"][0])
    second["node_id"] = "node-b"
    raw["nodes"].append(second)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_config(path)
    assert "nodes[1].port" in (exc.value.location or "")


def test_artifact_cycle_is_rejected(tmp_path: Path) -> None:
    raw, path = one_node_config(tmp_path)
    raw["pipeline"]["artifacts"][0]["refs"] = ["api"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_config(path)
    assert exc.value.code == "E_SPEC"
    assert "cycle" in exc.value.message or "cycle" in (exc.value.location or "")


def test_event_sequence_decrease_is_rejected(tmp_path: Path) -> None:
    raw, path = one_node_config(tmp_path)
    config = load_config(path)
    events_path = write_events(
        tmp_path / "events.ndjson",
        [
            {"seq": 2, "batch": 1, "type": "read", "artifact": "core", "lease_ms": 1000},
            {"seq": 1, "batch": 1, "type": "read", "artifact": "core", "lease_ms": 1000},
        ],
    )
    with pytest.raises(SpecError) as exc:
        load_events(events_path, config)
    assert "events.ndjson:2.seq" in (exc.value.location or "")


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"format_version":1,"format_version":1}', encoding="utf-8")
    with pytest.raises(SpecError) as exc:
        load_config(path)
    assert "duplicate JSON key" in exc.value.message


def test_release_quota_batch_cannot_be_parallel(tmp_path: Path) -> None:
    raw, path = one_node_config(tmp_path)
    config = load_config(path)
    events_path = write_events(
        tmp_path / "events.ndjson",
        [
            {"seq": 1, "batch": 1, "type": "write", "artifact": "core", "lease_ms": 1000},
            {"seq": 2, "batch": 1, "type": "release", "artifact": "core"},
        ],
    )
    with pytest.raises(SpecError) as exc:
        load_events(events_path, config)
    assert "exactly one" in exc.value.message

