"""Requirement 1: spec loading and strong validation with E_SPEC."""

import copy

import pytest

from obs_pipeline.errors import SpecError
from obs_pipeline.spec import topo_order, validate_spec


def test_valid_spec_loads(sample_spec):
    assert sample_spec.spec_version == 1
    assert [s.id for s in topo_order(sample_spec.stages)] == [
        "s_diff", "s_avg", "s_bucket", "s_norm",
    ]


def _expect(spec_dict, stage_id, field):
    with pytest.raises(SpecError) as excinfo:
        validate_spec(spec_dict)
    err = excinfo.value
    assert err.code == "E_SPEC"
    assert err.stage_id == stage_id
    assert err.field == field
    return err


def test_duplicate_stage_id(sample_spec_dict):
    spec = copy.deepcopy(sample_spec_dict)
    spec["stages"].append(dict(spec["stages"][0]))
    _expect(spec, "s_diff", "id")


def test_unknown_upstream(sample_spec_dict):
    spec = copy.deepcopy(sample_spec_dict)
    spec["stages"][1]["upstream"] = ["nope"]
    _expect(spec, "s_avg", "upstream")


def test_cycle_detected(sample_spec_dict):
    spec = copy.deepcopy(sample_spec_dict)
    spec["stages"][0]["upstream"] = ["s_norm"]
    err = _expect(spec, "s_diff", "upstream")
    assert "cycle" in err.message


def test_unsupported_scale(sample_spec_dict):
    spec = copy.deepcopy(sample_spec_dict)
    spec["stages"][0]["scale"] = 7
    _expect(spec, "s_diff", "scale")


def test_window_gap_and_overflow(sample_spec_dict):
    spec = copy.deepcopy(sample_spec_dict)
    spec["windows"][1]["start"] += 1
    _expect(spec, None, "windows")
    spec = copy.deepcopy(sample_spec_dict)
    spec["windows"][-1]["end"] = spec["task_interval"]["end"] + 5
    _expect(spec, None, "windows")


def test_null_bitmap_length(sample_spec_dict):
    spec = copy.deepcopy(sample_spec_dict)
    spec["sources"][0]["null_bitmap"] = spec["sources"][0]["null_bitmap"][:-1]
    _expect(spec, None, "sources.null_bitmap")


def test_non_integer_param(sample_spec_dict):
    spec = copy.deepcopy(sample_spec_dict)
    spec["stages"][1]["params"]["window"] = 2.5
    _expect(spec, "s_avg", "params.window")
