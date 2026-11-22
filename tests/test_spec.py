import json

import pytest

from faultdiag.errors import DiagError
from faultdiag.spec import (load_domains, load_events, load_instances)


def test_valid_sample_loads(spec_files):
    domains = load_domains(spec_files / "domains.json")
    instances = load_instances(spec_files / "instances.json", domains)
    events = load_events(spec_files / "events.ndjson", instances)
    assert len(domains) == 2 and len(instances) == 4 and len(events) == 10


def _expect_spec_error(path, kind, line, field, instances=None, domains=None):
    with pytest.raises(DiagError) as exc:
        if kind == "domains":
            load_domains(path)
        elif kind == "instances":
            load_instances(path, domains)
        else:
            load_events(path, instances)
    assert exc.value.code == "E_SPEC"
    assert exc.value.details["line"] == line
    assert exc.value.details["field"] == field


def test_duplicate_domain_id(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([
        {"id": "d1", "isolation_level": 1, "recovery_cooldown_ms": 0},
        {"id": "d1", "isolation_level": 1, "recovery_cooldown_ms": 0}]))
    _expect_spec_error(p, "domains", 2, "id")


def test_domain_level_out_of_range(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([
        {"id": "d1", "isolation_level": 7, "recovery_cooldown_ms": 0}]))
    _expect_spec_error(p, "domains", 1, "isolation_level")


def test_domain_negative_cooldown(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([
        {"id": "d1", "isolation_level": 1, "recovery_cooldown_ms": -5}]))
    _expect_spec_error(p, "domains", 1, "recovery_cooldown_ms")


def test_instance_unknown_domain(tmp_path, domains):
    p = tmp_path / "i.json"
    p.write_text(json.dumps([
        {"id": "x", "domain": "nope", "weight": 1, "replica": 1}]))
    _expect_spec_error(p, "instances", 1, "domain", domains=domains)


def test_instance_negative_weight(tmp_path, domains):
    p = tmp_path / "i.json"
    p.write_text(json.dumps([
        {"id": "x", "domain": "dom-a", "weight": -1, "replica": 1}]))
    _expect_spec_error(p, "instances", 1, "weight", domains=domains)


def test_instance_replica_zero(tmp_path, domains):
    p = tmp_path / "i.json"
    p.write_text(json.dumps([
        {"id": "x", "domain": "dom-a", "weight": 0, "replica": 0}]))
    _expect_spec_error(p, "instances", 1, "replica", domains=domains)


def test_event_seq_gap(tmp_path, instances):
    p = tmp_path / "e.ndjson"
    p.write_text('{"seq": 1, "ts_ms": 0, "instance": "s-a1", "fault": "crash"}\n'
                 '{"seq": 3, "ts_ms": 1, "instance": "s-a1", "fault": "crash"}\n')
    _expect_spec_error(p, "events", 2, "seq", instances=instances)


def test_event_ts_decreasing(tmp_path, instances):
    p = tmp_path / "e.ndjson"
    p.write_text('{"seq": 1, "ts_ms": 5, "instance": "s-a1", "fault": "crash"}\n'
                 '{"seq": 2, "ts_ms": 4, "instance": "s-a1", "fault": "crash"}\n')
    _expect_spec_error(p, "events", 2, "ts_ms", instances=instances)


def test_event_unknown_instance(tmp_path, instances):
    p = tmp_path / "e.ndjson"
    p.write_text('{"seq": 1, "ts_ms": 0, "instance": "ghost", "fault": "crash"}\n')
    _expect_spec_error(p, "events", 1, "instance", instances=instances)


def test_event_bad_fault(tmp_path, instances):
    p = tmp_path / "e.ndjson"
    p.write_text('{"seq": 1, "ts_ms": 0, "instance": "s-a1", "fault": "explode"}\n')
    _expect_spec_error(p, "events", 1, "fault", instances=instances)
