from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_ledger.errors import SpecError
from edge_ledger.spec import create_readonly_snapshot, load_spec_dir


def test_valid_sample_loads_and_snapshot_is_readonly(spec_dir: Path):
    bundle = load_spec_dir(spec_dir)
    assert len(bundle.nodes) == 3
    assert bundle.content.shard_count == 4
    assert bundle.content.total_bytes % bundle.content.shard_count == 0
    assert [event.seq for event in bundle.events] == list(range(1, 7))
    snapshot = spec_dir / "raw_snapshot"
    assert (snapshot / "nodes.json").exists()
    mode = (snapshot / "nodes.json").stat().st_mode & 0o777
    assert mode == 0o444


def test_duplicate_node_id_reports_efield_and_line(spec_dir: Path):
    path = spec_dir / "nodes.json"
    data = json.loads(path.read_text())
    data["nodes"][1]["node_id"] = data["nodes"][0]["node_id"]
    path.write_text(json.dumps(data, indent=2))
    with pytest.raises(SpecError) as exc:
        load_spec_dir(spec_dir)
    assert exc.value.code == "E_SPEC"
    assert "node_id" in (exc.value.field or "")
    assert exc.value.line and exc.value.line > 1


def test_non_literal_loopback_rejected(spec_dir: Path):
    path = spec_dir / "nodes.json"
    data = json.loads(path.read_text())
    data["nodes"][0]["host"] = "0.0.0.0"
    path.write_text(json.dumps(data, indent=2))
    with pytest.raises(SpecError) as exc:
        load_spec_dir(spec_dir)
    assert exc.value.field and exc.value.field.endswith("host")


def test_event_sequence_must_increase(spec_dir: Path):
    event_path = spec_dir / "event_stream.jsonl"
    records = [json.loads(line) for line in event_path.read_text().splitlines()]
    records.append({"seq": 1, "event_type": "out-of-order"})
    event_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    with pytest.raises(SpecError) as exc:
        load_spec_dir(spec_dir)
    assert exc.value.code == "E_SPEC"
    assert exc.value.line == len(records)


def test_scale_allowed_set(spec_dir: Path):
    path = spec_dir / "content_manifest.json"
    data = json.loads(path.read_text())
    data["contents"][0]["integer_scale"] = 7
    path.write_text(json.dumps(data, indent=2))
    with pytest.raises(SpecError) as exc:
        load_spec_dir(spec_dir)
    assert "integer_scale" in (exc.value.field or "")
