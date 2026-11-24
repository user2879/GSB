from __future__ import annotations

import json

import pytest

from faultisolation.errors import SpecError
from faultisolation.spec import load_bundle, load_events, load_spec


def _write_spec(tmp_path, domains, instances, events_file="events.ndjson"):
    spec = {"domains": domains, "instances": instances, "events_file": events_file}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _write_events(tmp_path, rows, name="events.ndjson"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _dom(did="d1", port=19001, **over):
    base = {"domain_id": did, "host": "127.0.0.1", "port": port, "slots": 4,
            "lease_cap": 3, "lease_ms": 500, "renew_ms": 100, "seed": 1}
    base.update(over)
    return base


def test_valid_bundle_loads(small_bundle):
    assert small_bundle.events[0].seq == 1
    assert small_bundle.events[-1].seq == len(small_bundle.events)


def test_reject_non_loopback_host(tmp_path):
    spec = _write_spec(tmp_path, [_dom(host="10.0.0.1")],
                       [{"instance_id": "i1", "domain_id": "d1", "priority": 1, "demand": 1}])
    _write_events(tmp_path, [{"seq": 1, "arrival_ms": 0, "instance_id": "i1", "action": "acquire"}])
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.code == "E_SPEC" and exc.value.field == "host"


def test_reject_duplicate_port(tmp_path):
    spec = _write_spec(tmp_path, [_dom("d1", 19001), _dom("d2", 19001)],
                       [{"instance_id": "i1", "domain_id": "d1", "priority": 1, "demand": 1}])
    _write_events(tmp_path, [{"seq": 1, "arrival_ms": 0, "instance_id": "i1", "action": "acquire"}])
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.field == "port"


def test_reject_renew_ge_lease(tmp_path):
    spec = _write_spec(tmp_path, [_dom(lease_ms=100, renew_ms=100)],
                       [{"instance_id": "i1", "domain_id": "d1", "priority": 1, "demand": 1}])
    _write_events(tmp_path, [{"seq": 1, "arrival_ms": 0, "instance_id": "i1", "action": "acquire"}])
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.field == "renew_ms"


def test_reject_unknown_instance_domain(tmp_path):
    spec = _write_spec(tmp_path, [_dom()],
                       [{"instance_id": "i1", "domain_id": "ghost", "priority": 1, "demand": 1}])
    _write_events(tmp_path, [{"seq": 1, "arrival_ms": 0, "instance_id": "i1", "action": "acquire"}])
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.field == "domain_id"


def test_reject_non_monotonic_events(tmp_path):
    spec = _write_spec(tmp_path, [_dom()],
                       [{"instance_id": "i1", "domain_id": "d1", "priority": 1, "demand": 1}])
    _write_events(tmp_path, [
        {"seq": 1, "arrival_ms": 10, "instance_id": "i1", "action": "acquire"},
        {"seq": 2, "arrival_ms": 9, "instance_id": "i1", "action": "release"},
    ])
    with pytest.raises(SpecError) as exc:
        load_events(tmp_path / "events.ndjson", {"i1"})
    assert exc.value.field == "arrival_ms" and exc.value.line == 2


def test_reject_non_contiguous_seq(tmp_path):
    spec = _write_spec(tmp_path, [_dom()],
                       [{"instance_id": "i1", "domain_id": "d1", "priority": 1, "demand": 1}])
    _write_events(tmp_path, [
        {"seq": 1, "arrival_ms": 0, "instance_id": "i1", "action": "acquire"},
        {"seq": 3, "arrival_ms": 1, "instance_id": "i1", "action": "release"},
    ])
    with pytest.raises(SpecError) as exc:
        load_events(tmp_path / "events.ndjson", {"i1"})
    assert exc.value.field == "seq" and exc.value.line == 2


def test_reject_bool_disguised_as_int(tmp_path):
    spec = _write_spec(tmp_path, [_dom(slots=True)],
                       [{"instance_id": "i1", "domain_id": "d1", "priority": 1, "demand": 1}])
    _write_events(tmp_path, [{"seq": 1, "arrival_ms": 0, "instance_id": "i1", "action": "acquire"}])
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.field == "slots"


def test_reject_demand_exceeds_slots(tmp_path):
    spec = _write_spec(tmp_path, [_dom(slots=2)],
                       [{"instance_id": "i1", "domain_id": "d1", "priority": 1, "demand": 3}])
    _write_events(tmp_path, [{"seq": 1, "arrival_ms": 0, "instance_id": "i1", "action": "acquire"}])
    with pytest.raises(SpecError) as exc:
        load_spec(spec)
    assert exc.value.field == "demand"
