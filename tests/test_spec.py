from __future__ import annotations

import json

import pytest

from trace_diag.errors import BucketError, SpecError
from trace_diag.spec import (
    load_policies_file,
    load_services_file,
    parse_policies_document,
    parse_services_document,
)


def test_valid_documents_are_normalized(service_raw, policy_raw):
    services = parse_services_document(service_raw)
    policies = parse_policies_document(policy_raw)
    assert services.version == 1
    assert services.services[0].fields["timestamp_ms"]["max_step_ms"] == 19
    assert policies.policies[0].quantum_us == 10
    assert policies.policies[0].version == 1


@pytest.mark.parametrize(
    ("mutate", "path"),
    [
        (lambda raw: raw["services"][0].update({"seed": 0}), "$.services[0].seed"),
        (lambda raw: raw["services"][0].update({"record_count": 0}), "$.services[0].record_count"),
        (
            lambda raw: raw["services"][0]["fields"].pop("trace_id"),
            "$.services[0].fields.trace_id",
        ),
        (
            lambda raw: raw["services"][0]["fields"]["duration_us"].update({"min_us": 0}),
            "$.services[0].fields.duration_us.min_us",
        ),
        (lambda raw: raw["services"][0].update({"unknown": 1}), "$.services[0].unknown"),
    ],
)
def test_service_validation_rejects_with_field_path(service_raw, clone, mutate, path):
    raw = clone(service_raw)
    mutate(raw)
    with pytest.raises(SpecError) as captured:
        parse_services_document(raw)
    assert captured.value.path == path


def test_duplicate_service_id_is_rejected(service_raw, clone):
    raw = clone(service_raw)
    duplicate = clone(raw["services"][0])
    raw["services"].append(duplicate)
    with pytest.raises(SpecError) as captured:
        parse_services_document(raw)
    assert captured.value.path == "$.services[1].id"


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("head_ratio_bp", 10_001, "$.policies[0].head_ratio_bp"),
        ("tail_ratio_bp", -1, "$.policies[0].tail_ratio_bp"),
        ("tail_duration_threshold_us", 0, "$.policies[0].tail_duration_threshold_us"),
        ("scale_power", 7, "$.policies[0].scale_power"),
        ("bucket_count", 1, "$.policies[0].bucket_count"),
        ("bucket_count", 257, "$.policies[0].bucket_count"),
    ],
)
def test_policy_validation_rejects_with_field_path(policy_raw, clone, field, value, path):
    raw = clone(policy_raw)
    raw["policies"][0][field] = value
    with pytest.raises(SpecError) as captured:
        parse_policies_document(raw)
    assert captured.value.path == path


def test_policy_ratio_injection_uses_bucket_error(tmp_path, policy_raw):
    path = tmp_path / "policies.json"
    raw = policy_raw
    raw["policies"][0]["head_ratio_bp"] = 10_001
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(BucketError) as captured:
        load_policies_file(path, inject_ratio_bucket_error=True)
    assert captured.value.path == "$.policies[0].head_ratio_bp"


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "services.json"
    path.write_text('{"version":1,"version":2,"services":[]}', encoding="utf-8")
    with pytest.raises(SpecError) as captured:
        load_services_file(path)
    assert captured.value.path == "$.version"


def test_loader_does_not_partially_return(service_raw, clone):
    raw = clone(service_raw)
    raw["services"][0]["seed"] = -1
    with pytest.raises(SpecError):
        parse_services_document(raw)
    valid = parse_services_document(service_raw)
    assert valid.services[0].seed == 101
