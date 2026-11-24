from __future__ import annotations

import pytest

from batch_diagnostic.errors import SpecError
from batch_diagnostic.spec import (
    SystemSpec,
    _validate_budget,
    _validate_requests,
    _validate_workers,
)


def test_request_unique_ids_and_seqs():
    raw = [
        {"request_id": "a", "arrival_seq": 0, "input_dim": 2, "seed": 1},
        {"request_id": "a", "arrival_seq": 1, "input_dim": 2, "seed": 1},
    ]
    with pytest.raises(SpecError) as exc:
        _validate_requests(raw)
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["field"] == "request_id"
    assert exc.value.context["entry"] == "requests[1]"


def test_request_negative_arrival_first_offender():
    raw = [
        {"request_id": "a", "arrival_seq": 0, "input_dim": 2, "seed": 1},
        {"request_id": "b", "arrival_seq": -3, "input_dim": 2, "seed": 1},
    ]
    with pytest.raises(SpecError) as exc:
        _validate_requests(raw)
    assert exc.value.context == {"entry": "requests[1]", "field": "arrival_seq", "value": -3}


def test_request_non_integer_bool_rejected():
    raw = [{"request_id": "a", "arrival_seq": True, "input_dim": 2, "seed": 1}]
    with pytest.raises(SpecError) as exc:
        _validate_requests(raw)
    assert exc.value.context["field"] == "arrival_seq"


def test_worker_loopback_literal_and_port_range():
    raw = [
        {
            "worker_id": "w",
            "host": "localhost",
            "port": 6100,
            "seed": 1,
            "param_version": 1,
            "input_dim": 2,
            "hidden_dim": 2,
            "batch_size": 1,
            "max_in_flight": 1,
            "retries": 1,
        }
    ]
    with pytest.raises(SpecError) as exc:
        _validate_workers(raw, 2)
    assert exc.value.context["field"] == "host"

    raw[0]["host"] = "127.0.0.1"
    raw[0]["port"] = 70000
    with pytest.raises(SpecError) as exc:
        _validate_workers(raw, 2)
    assert exc.value.context["field"] == "port"


def test_worker_duplicate_ports_and_dimension_match():
    base = {
        "host": "127.0.0.1",
        "seed": 1,
        "param_version": 1,
        "input_dim": 2,
        "hidden_dim": 2,
        "batch_size": 1,
        "max_in_flight": 1,
        "retries": 1,
    }
    raw = [
        {"worker_id": "w1", "port": 6100, **base},
        {"worker_id": "w2", "port": 6100, **base},
    ]
    with pytest.raises(SpecError) as exc:
        _validate_workers(raw, 2)
    assert exc.value.context["field"] == "port"

    raw[1]["port"] = 6101
    raw[1]["input_dim"] = 3
    with pytest.raises(SpecError) as exc:
        _validate_workers(raw, 2)
    assert exc.value.context["field"] == "input_dim"


def test_budget_positive_ints():
    with pytest.raises(SpecError) as exc:
        _validate_budget(
            {"global_max_in_flight": 0, "retry_limit": 1, "cost_budget_ms": 1}
        )
    assert exc.value.context["field"] == "global_max_in_flight"


def test_fingerprints_stable(system_spec):
    first = system_spec.content_fingerprint()
    second = SystemSpec(
        requests=system_spec.requests,
        workers=system_spec.workers,
        budget=system_spec.budget,
    ).content_fingerprint()
    assert first == second
