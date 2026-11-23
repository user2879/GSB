from __future__ import annotations

import copy
import json

import pytest

from leasebench.errors import LeasebenchError
from leasebench.validation import load_declaration, load_events, parse_declaration

from .conftest import ROOT, SAMPLES, dump_events, dump_json


def raw_v1() -> dict:
    return json.loads((SAMPLES / "declarations_v1.json").read_text(encoding="utf-8"))


def test_valid_declaration_and_events() -> None:
    declaration = load_declaration(SAMPLES / "declarations_v1.json")
    events = load_events(SAMPLES / "events.ndjson", declaration)
    assert declaration.declaration_version == 1
    assert list(sorted(declaration.functions)) == ["api", "batch", "report"]
    assert len(declaration.preemptions) == 2
    assert [event.event_seq for event in events] == list(range(1, 12))


def test_duplicate_function_reports_first_field(tmp_path) -> None:
    raw = raw_v1()
    raw["functions"][1]["function_id"] = "api"
    path = dump_json(tmp_path / "decl.json", raw)
    with pytest.raises(LeasebenchError) as caught:
        load_declaration(path)
    assert caught.value.code == "E_DECL"
    assert caught.value.field == "functions[1].function_id"
    assert "duplicate" in caught.value.message


def test_preemption_cycle_rejected() -> None:
    raw = raw_v1()
    raw["preemptions"].append(
        {"higher": "report", "lower": "api", "return_deadline_ms": 1}
    )
    with pytest.raises(LeasebenchError) as caught:
        parse_declaration(raw)
    assert caught.value.code == "E_DECL"
    assert caught.value.field == "preemptions"
    assert caught.value.details["cycle"]


def test_unknown_reference_and_self_edge() -> None:
    raw = raw_v1()
    raw["preemptions"][0]["lower"] = "missing"
    with pytest.raises(LeasebenchError) as caught:
        parse_declaration(raw)
    assert caught.value.field == "preemptions[0].lower"

    raw = raw_v1()
    raw["preemptions"][0]["lower"] = "api"
    with pytest.raises(LeasebenchError) as caught:
        parse_declaration(raw)
    assert caught.value.field == "preemptions[0].lower"


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("functions", 0, "max_concurrency"), True, "functions[0].max_concurrency"),
        (("functions", 0, "min_concurrency"), -1, "functions[0].min_concurrency"),
        (("functions", 0, "max_concurrency"), 1, "functions[0].min_concurrency"),
        (("global_concurrency",), -1, "global_concurrency"),
        (("declaration_version",), 0, "declaration_version"),
    ],
)
def test_declaration_integer_validation(path, value, field) -> None:
    raw = copy.deepcopy(raw_v1())
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(LeasebenchError) as caught:
        parse_declaration(raw)
    assert caught.value.code == "E_DECL"
    assert caught.value.field == field


def test_event_validation_monotonic_and_contiguous(tmp_path, declaration_v1) -> None:
    events = [
        {"event_seq": 1, "arrival_ms": 10, "function_id": "api", "expected_occupancy_ms": 1, "concurrency_demand": 1},
        {"event_seq": 2, "arrival_ms": 9, "function_id": "api", "expected_occupancy_ms": 1, "concurrency_demand": 1},
    ]
    with pytest.raises(LeasebenchError) as caught:
        load_events(dump_events(tmp_path / "bad.ndjson", events), declaration_v1)
    assert caught.value.field == "events[2].arrival_ms"

    events[1]["arrival_ms"] = 10
    events[1]["event_seq"] = 3
    with pytest.raises(LeasebenchError) as caught:
        load_events(dump_events(tmp_path / "gap.ndjson", events), declaration_v1)
    assert caught.value.field == "events[2].event_seq"


def test_event_unknown_function_and_negative_demand(tmp_path, declaration_v1) -> None:
    events = [
        {"event_seq": 1, "arrival_ms": 0, "function_id": "nope", "expected_occupancy_ms": 1, "concurrency_demand": 1}
    ]
    with pytest.raises(LeasebenchError) as caught:
        load_events(dump_events(tmp_path / "unknown.ndjson", events), declaration_v1)
    assert caught.value.field == "events[1].function_id"

    events[0]["function_id"] = "api"
    events[0]["concurrency_demand"] = -1
    with pytest.raises(LeasebenchError) as caught:
        load_events(dump_events(tmp_path / "negative.ndjson", events), declaration_v1)
    assert caught.value.field == "events[1].concurrency_demand"

