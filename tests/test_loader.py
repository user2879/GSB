"""要求1：强校验与 E_SPEC 行号定位。"""

import json

import pytest

from slo_burn.errors import SpecError
from slo_burn.loader import (
    load_batch_directory,
    load_batches,
    load_param_edits,
    load_service_specs,
)


def test_valid_workspace_loads(demo_workspace):
    specs = load_service_specs(demo_workspace / "services.json")
    assert [s.service_id for s in specs] == [
        "auth-service",
        "checkout-api",
        "search-edge",
    ]
    regular, late = load_batch_directory(demo_workspace / "batches", specs)
    assert len(regular) == 450
    assert len(late) == 15


def test_duplicate_service_id_rejected(tmp_path):
    payload = {
        "services": [
            {
                "service_id": "dup",
                "target_permyriad": 9990,
                "window_minutes": 10,
                "burn_threshold": 10000,
                "budget_floor": 0,
            },
            {
                "service_id": "dup",
                "target_permyriad": 9990,
                "window_minutes": 10,
                "burn_threshold": 10000,
                "budget_floor": 0,
            },
        ]
    }
    path = tmp_path / "services.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SpecError) as exc:
        load_service_specs(path)
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["line"] == 2


def test_invalid_target_bounds_with_location(tmp_path):
    payload = {
        "services": [
            {
                "service_id": "broken-svc",
                "target_permyriad": 12000,
                "window_minutes": 10,
                "burn_threshold": 10000,
                "budget_floor": 0,
            }
        ]
    }
    path = tmp_path / "services.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SpecError) as exc:
        load_service_specs(path)
    assert exc.value.context == {
        "service_id": "broken-svc",
        "line": 1,
        "field": "target_permyriad",
    }


def test_duplicate_minute_regular_stream(demo_workspace):
    specs = load_service_specs(demo_workspace / "services.json")
    with pytest.raises(SpecError) as exc:
        load_batches(demo_workspace / "invalid" / "bad_batches.ndjson", specs, "regular")
    assert exc.value.code == "E_SPEC"
    assert exc.value.context["line"] == 2
    assert exc.value.context["minute"] == 0


def test_unknown_service_in_batch(tmp_path, demo_workspace):
    specs = load_service_specs(demo_workspace / "services.json")
    path = tmp_path / "x.ndjson"
    path.write_text(
        json.dumps(
            {"service_id": "ghost", "minute": 0, "successes": 1, "failures": 0}
        )
        + "\n"
    )
    with pytest.raises(SpecError) as exc:
        load_batches(path, specs, "regular")
    assert exc.value.context["service_id"] == "ghost"


def test_non_negative_integer_enforcement(tmp_path, demo_workspace):
    specs = load_service_specs(demo_workspace / "services.json")
    path = tmp_path / "neg.ndjson"
    path.write_text(
        json.dumps(
            {"service_id": "checkout-api", "minute": 0, "successes": -1, "failures": 0}
        )
        + "\n"
    )
    with pytest.raises(SpecError) as exc:
        load_batches(path, specs, "regular")
    assert exc.value.context["field"] == "successes"


def test_param_edits_must_be_monotonic(demo_workspace):
    specs = load_service_specs(demo_workspace / "services.json")
    edits = load_param_edits(demo_workspace / "param_edits.json", specs)
    assert [edit.param_version for edit in edits if edit.service_id == "auth-service"] == [2, 3]
