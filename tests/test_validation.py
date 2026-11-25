from __future__ import annotations

import copy
import math

from log_sampler.validation import load_batches, load_config, validate_config, validate_input


def _record(log_id: str, device: str = "edge-a", severity: str = "info") -> dict:
    return {
        "log_id": log_id,
        "timestamp": 1000,
        "device_id": device,
        "severity": severity,
        "template_id": "tmpl-01",
        "metric": 0.25,
    }


def test_sample_files_load_and_injection_isolation(sample_bundle):
    config, config_issues = load_config(sample_bundle["config_path"])
    batches, batch_issues = load_batches(sample_bundle["batches_path"])
    assert not config_issues
    assert not batch_issues
    assert not validate_config(config)
    result = validate_input(config, batches, sample_bundle["injections"])

    assert len(result.accepted_records) == 27
    assert result.rejected_batches == [0]
    assert {item.injection_id for item in result.failures} == {
        "INJ-001",
        "INJ-002",
        "INJ-003",
        "INJ-004",
    }
    assert all(item.code == "E_SAMPLE_FORMAT" for item in result.failures)
    accepted_ids = {item.log_id for item in result.accepted_records}
    assert "log-0001" in accepted_ids
    assert "log-0002" not in accepted_ids
    assert "log-0008" not in accepted_ids
    assert "log-0020" not in accepted_ids


def test_bad_records_are_quarantined_without_stopping_others(sample_bundle):
    records = [
        _record("log-good-1"),
        {**_record("log-bad-metric"), "metric": math.nan},
        _record("log-good-2", "edge-b", "error"),
        _record("log-good-1"),
        _record("log-bad-severity", "edge-a", "debug"),
        {**_record("log-negative"), "metric": -2},
        _record("log-good-3", "core-x", "warning"),
    ]
    validation = validate_input(
        sample_bundle["config"],
        [{"arrival_seq": 1, "records": records}],
        [],
    )
    assert {item.log_id for item in validation.accepted_records} == {
        "log-good-1",
        "log-good-2",
        "log-good-3",
    }
    assert len(validation.failures) == 4
    assert all(item.code == "E_SAMPLE_FORMAT" for item in validation.failures)
    assert {item.log_id for item in validation.failures} == {
        "log-bad-metric",
        "log-good-1",
        "log-bad-severity",
        "log-negative",
    }


def test_batch_cursor_regression_quarantines_only_bad_batch(sample_bundle):
    validation = validate_input(
        sample_bundle["config"],
        [
            {"arrival_seq": 1, "records": [_record("log-a")]},
            {"arrival_seq": 1, "records": [_record("log-b")]},
            {"arrival_seq": 2, "records": [_record("log-c")]},
        ],
        [],
    )
    assert [item.log_id for item in validation.accepted_records] == ["log-a", "log-c"]
    assert validation.accepted_batches == [1, 2]
    assert validation.rejected_batches == [1]
    assert len(validation.failures) == 1
    assert validation.failures[0].scope == "batch"


def test_invalid_config_fields_are_reported(sample_bundle):
    config = copy.deepcopy(sample_bundle["config"])
    config["seed"] = -1
    config["profiles"]["balanced"]["global_budget"] = 0
    config["profiles"]["balanced"]["bucket_capacity"] = -2
    config["profiles"]["balanced"]["quantization"] = {"decimals": 9, "tier": "ultra"}
    config["max_workers"] = 0
    config["evidence_format"] = "p9"
    config["devices"].append(copy.deepcopy(config["devices"][0]))
    issues = validate_config(config)
    fields = {item.field for item in issues}
    assert "seed" in fields
    assert "profiles.balanced.global_budget" in fields
    assert "profiles.balanced.bucket_capacity" in fields
    assert "profiles.balanced.quantization.decimals" in fields
    assert "profiles.balanced.quantization.tier" in fields
    assert any(item.field == "devices[3].device_id" for item in issues)
    assert "max_workers" in fields
    assert "evidence_format" in fields


def test_unknown_override_bucket_is_config_error(sample_bundle):
    config = copy.deepcopy(sample_bundle["config"])
    config["bucket_overrides"]["ghost|critical"] = {"capacity": 2}
    issues = validate_config(config)
    assert any("ghost|critical" in (item.message or "") for item in issues)

