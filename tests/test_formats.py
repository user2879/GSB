from __future__ import annotations

from candidate_diagnostics.formats import (
    parse_v1_lines,
    result_to_records,
    v1_to_v2,
    v2_to_v1,
)
from candidate_diagnostics.pipeline import analyze_request


def test_v1_roundtrip_exact_permille_and_code_mapping(config, requests):
    result = analyze_request(requests["req-001"], config)
    records = result_to_records(result, config)
    lines, sidecar, failures = v2_to_v1(records, config)
    assert failures == []
    assert len(lines) == len(records)
    parsed, parse_failures = parse_v1_lines(lines, config)
    assert parse_failures == []
    by_id = {item["candidate_id"]: item for item in parsed}
    assert by_id[105]["selection_result"] == "SAMPLED"
    assert by_id[101]["selection_result"] == "SELECTED_NOT_SAMPLED"
    assert by_id[107]["selection_result"] == "TRUNCATED"
    assert by_id[102]["score"] == 0.96
    assert by_id[105]["score"] == 0.88
    assert by_id[102]["precision_label"] == "v1-permille-exact"
    assert by_id[105]["precision_label"] == "v1-permille-exact"
    assert sidecar["records"]["req-001:105"]["v2_fields"]["sampling_weight"] == 2.0


def test_sidecar_restores_v1_discarded_fields(config, requests):
    result = analyze_request(requests["req-001"], config)
    records = result_to_records(result, config)
    lines, sidecar, failures = v2_to_v1(records, config)
    assert failures == []
    restored, restore_failures = v1_to_v2(lines, config, sidecar)
    assert restore_failures == []
    record = next(item for item in restored if item["candidate_id"] == 105)
    assert record["sidecar_restored"] is True
    assert record["sampling_weight"] == 2.0
    assert record["sampling_scheme"] == config.raw["sampling"]["scheme"]
    assert "boundary" in record["restored_fields"]
    sidecar_fields = sidecar["records"]["req-001:105"]["v2_fields"]
    assert "truncation_residual" in sidecar_fields
    assert record["truncation_residual"] == sidecar_fields["truncation_residual"]


def test_unknown_code_and_field_width_failures_are_collected(config):
    lines = [
        "req-bad     |0001|0000000999|1234567|01",
        "req-bad     |0002|0000000998|   812|99",
    ]
    records, failures = parse_v1_lines(lines, config)
    assert records == []
    codes = [failure.code for failure in failures]
    assert "E_FIELD_TRUNCATED" in codes
    assert "E_SELECT_CODE_UNKNOWN" in codes


def test_half_up_score_conversion(config):
    records = [
        {
            "request_id": "req-x",
            "candidate_id": 1,
            "seq": 1,
            "score": 0.8125,
            "selection_result": "SAMPLED",
        }
    ]
    lines, sidecar, failures = v2_to_v1(records, config)
    assert failures == []
    assert "813" in lines[0]
    assert sidecar["records"]["req-x:1"]["v2_fields"]["truncation_residual"] == -0.5
