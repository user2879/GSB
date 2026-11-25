import json

import pytest

from lease_middleware.errors import SpecError
from lease_middleware.spec import load_bundle, validate_nodes_doc


def test_sample_specs_are_valid(bundle):
    assert len(bundle.nodes) == 3
    assert len(bundle.requests) == 6
    assert [event["seq"] for event in bundle.events] == list(range(1, 15))


def test_duplicate_node_id_reports_first_field():
    doc = {
        "quotas": {"global_in_flight": 2},
        "nodes": [
            {
                "node_id": "same",
                "host": "127.0.0.1",
                "port": 41001,
                "lease_limit": 1,
            },
            {
                "node_id": "same",
                "host": "127.0.0.1",
                "port": 41002,
                "lease_limit": 1,
            },
        ],
    }
    with pytest.raises(SpecError) as exc:
        validate_nodes_doc(doc)
    assert exc.value.code == "E_SPEC"
    assert exc.value.field == "nodes[1].node_id"


def test_host_must_be_literal_loopback():
    doc = {
        "quotas": {"global_in_flight": 2},
        "nodes": [
            {
                "node_id": "a",
                "host": "localhost",
                "port": 41001,
                "lease_limit": 1,
            }
        ],
    }
    with pytest.raises(SpecError) as exc:
        validate_nodes_doc(doc)
    assert exc.value.field == "nodes[0].host"


def test_duplicate_port_and_non_positive_quota(tmp_path):
    nodes = {
        "quotas": {"global_in_flight": 0},
        "nodes": [
            {
                "node_id": "a",
                "host": "127.0.0.1",
                "port": 41001,
                "lease_limit": 1,
            }
        ],
    }
    nodes_path = tmp_path / "nodes.json"
    requests_path = tmp_path / "requests.json"
    events_path = tmp_path / "events.json"
    nodes_path.write_text(json.dumps(nodes))
    requests_path.write_text('{"requests":[]}')
    events_path.write_text('{"events":[]}')
    with pytest.raises(SpecError) as exc:
        load_bundle(nodes_path, requests_path, events_path)
    assert exc.value.field == "quotas.global_in_flight"


def test_event_sequence_must_increase(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    import shutil

    shutil.copytree("samples", sample, dirs_exist_ok=True)
    events_path = sample / "events.json"
    doc = json.loads(events_path.read_text())
    doc["events"][1]["seq"] = doc["events"][0]["seq"]
    events_path.write_text(json.dumps(doc))
    with pytest.raises(SpecError) as exc:
        load_bundle(
            sample / "nodes.json",
            sample / "lease_requests.json",
            sample / "events.json",
        )
    assert exc.value.code == "E_SPEC"
    assert "events[1].seq" in exc.value.field
