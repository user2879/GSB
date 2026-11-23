from __future__ import annotations

import json
from pathlib import Path

import pytest

from leaseaudit.errors import BusinessError, ErrorCode
from leaseaudit.scenario import load_scenario, validate_scenario


def test_valid_scenario_and_digests(scenario_doc):
    scenario = validate_scenario(scenario_doc)
    assert scenario.scenario_version == "test-v1"
    assert scenario.initial_retained_total == 4
    assert scenario.initial_global_allocatable == 8
    assert [source.source_id for source in scenario.ordered_sources()] == ["alpha", "beta", "gamma"]
    assert len(scenario.scenario_digest) == 64
    assert len(scenario.params_digest) == 64
    assert len(scenario.external_digest) == 64


@pytest.mark.parametrize(
    ("mutator", "field_path", "code"),
    [
        (lambda doc: doc["sources"][1].__setitem__("source_id", "alpha"), "sources[1].source_id", ErrorCode.E_SCENE),
        (lambda doc: doc["sources"][0].__setitem__("address", "localhost"), "sources[0].address", ErrorCode.E_SCENE),
        (lambda doc: doc["sources"][0].__setitem__("address", "0.0.0.0"), "sources[0].address", ErrorCode.E_SCENE),
        (lambda doc: doc["sources"][1].__setitem__("port", 55501), "sources[1].port", ErrorCode.E_SCENE),
        (lambda doc: doc["sources"][0].__setitem__("port", 0), "sources[0].port", ErrorCode.E_SCENE),
        (lambda doc: doc["sources"][0].__setitem__("port", 65536), "sources[0].port", ErrorCode.E_SCENE),
        (lambda doc: doc.__setitem__("renew_interval_ms", 1000), "renew_interval_ms", ErrorCode.E_SCENE),
        (lambda doc: doc["sources"][0].__setitem__("reserve_quota", -1), "sources[0].reserve_quota", ErrorCode.E_SCENE),
        (lambda doc: doc["sources"][0].__setitem__("lease_share", 2.5), "sources[0].lease_share", ErrorCode.E_SCENE),
        (lambda doc: doc.__setitem__("cluster_capacity", True), "cluster_capacity", ErrorCode.E_SCENE),
        (lambda doc: doc.__setitem__("reclaim_threshold", -1), "reclaim_threshold", ErrorCode.E_SCENE),
        (lambda doc: doc.__setitem__("unknown_root", 1), "unknown_root", ErrorCode.E_SCENE),
    ],
)
def test_validation_rejects_first_violation_without_partial_load(
    scenario_doc, mutator, field_path, code
):
    document = json.loads(json.dumps(scenario_doc))
    mutator(document)
    with pytest.raises(BusinessError) as caught:
        validate_scenario(document)
    assert caught.value.code == code
    assert caught.value.field_path == field_path


def test_reserve_quota_overflow_uses_equota(scenario_doc):
    document = json.loads(json.dumps(scenario_doc))
    document["cluster_capacity"] = 3
    document["sources"][0]["lease_share"] = 2
    document["sources"][0]["reserve_quota"] = 2
    document["sources"][1]["lease_share"] = 2
    document["sources"][1]["reserve_quota"] = 2
    with pytest.raises(BusinessError) as caught:
        validate_scenario(document)
    assert caught.value.code == ErrorCode.E_QUOTA
    assert caught.value.field_path == "sources[1].reserve_quota"
    assert caught.value.details["sum_reserve_quota"] == 4
    assert caught.value.details["excess"] == 1


def test_lease_share_sum_may_exceed_capacity_because_global_pool_caps_grants(scenario_doc):
    document = json.loads(json.dumps(scenario_doc))
    document["cluster_capacity"] = 4
    document["sources"] = document["sources"][:2]
    document["sources"][0].update({"lease_share": 5, "reserve_quota": 1})
    document["sources"][1].update({"lease_share": 5, "reserve_quota": 1})
    scenario = validate_scenario(document)
    assert sum(source.lease_share for source in scenario.sources) == 10
    assert scenario.initial_global_allocatable == 2


def test_load_broken_json_reports_scene(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text('{"cluster_capacity": 8,', encoding="utf-8")
    with pytest.raises(BusinessError) as caught:
        load_scenario(path)
    assert caught.value.code == ErrorCode.E_SCENE
    assert caught.value.field_path == "$"
