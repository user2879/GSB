from __future__ import annotations

import copy

from log_sampler.models import AcceptedRecord
from log_sampler.reservoir import build_bucket_history
from log_sampler.truncation import apply_global_budget


def _records(count: int, bucket: str = "edge-a|info") -> list[AcceptedRecord]:
    device, severity = bucket.split("|", 1)
    result = []
    for index in range(count):
        result.append(
            AcceptedRecord(
                log_id=f"log-{index:04d}",
                timestamp=100 + index,
                device_id=device,
                severity=severity,
                template_id=f"tmpl-{index % 3}",
                metric=(index + 1) / (count + 1),
                arrival_seq=1 + index // 3,
                record_index=index % 3,
            )
        )
    return result


def _history(config, records, capacity: int, budget: int = 8, priority: int = 10):
    return build_bucket_history(
        bucket_id="edge-a|info",
        records=records,
        capacity=capacity,
        priority=priority,
        config=config,
        profile="balanced",
        global_budget=budget,
        quantization={"decimals": 3, "tier": "fine"},
    )


def test_reservoir_is_deterministic_and_bounded(sample_bundle):
    records = _records(9)
    first = _history(sample_bundle["config"], records, 3)
    second = _history(sample_bundle["config"], records, 3)
    assert first == second
    assert first["final_slot_count"] == 3
    assert len(first["slots"]) == 3
    assert first["history"][0]["slot_no"] == 1
    assert first["history"][1]["slot_no"] == 2
    assert first["history"][2]["slot_no"] == 3
    assert all(event["slot_no"] <= 3 for event in first["history"] if event["admitted"])
    replacements = [
        event for event in first["history"] if event["replaced_log_id"] is not None
    ]
    assert replacements
    assert all(event["admitted"] for event in replacements)
    assert all(event["target_slot_no"] in {1, 2, 3} for event in replacements)


def test_capacity_change_changes_history_and_slots(sample_bundle):
    records = _records(9)
    cap_two = _history(sample_bundle["config"], records, 2)
    cap_three = _history(sample_bundle["config"], records, 3)
    assert cap_two["bucket_fingerprint"] != cap_three["bucket_fingerprint"]
    assert cap_two["history"] != cap_three["history"]
    assert [slot["log_id"] for slot in cap_two["slots"]] != [
        slot["log_id"] for slot in cap_three["slots"]
    ]


def test_budget_change_changes_random_source_and_members(sample_bundle):
    records = _records(9)
    low = _history(sample_bundle["config"], records, 3, budget=5)
    high = _history(sample_bundle["config"], records, 3, budget=8)
    assert low["random_source_digest"] != high["random_source_digest"]
    assert low["history"][3]["random_digest"] != high["history"][3]["random_digest"]
    assert low["history"] != high["history"]
    assert [slot["log_id"] for slot in low["slots"]] != [
        slot["log_id"] for slot in high["slots"]
    ]


def test_global_budget_truncates_and_audits_layers(sample_bundle):
    bucket_a = _history(sample_bundle["config"], _records(6), 3)
    bucket_b = _history(
        sample_bundle["config"],
        _records(6, bucket="edge-b|info"),
        3,
        priority=20,
    )
    truncation = apply_global_budget(
        {
            "edge-a|info": bucket_a,
            "edge-b|info": bucket_b,
        },
        global_budget=2,
    )
    assert truncation["consumed_budget"] == 2
    assert len(truncation["final_samples"]) == 2
    assert all(item["bucket_id"] == "edge-b|info" for item in truncation["final_samples"])
    assert truncation["global_truncated"]
    assert any(
        item["truncation_layer"] == "global"
        for item in truncation["audit"]
    )
    assert any(
        item["truncation_layer"] == "bucket"
        for item in truncation["audit"]
        if item["decision"] in {"bucket_rejected", "bucket_replaced"}
    )


def test_global_tie_is_broken_by_log_id(sample_bundle):
    def bucket(bucket_id: str, log_id: str):
        history = _history(sample_bundle["config"], _records(3), 2, priority=10)
        slot = {
            **history["slots"][0],
            "log_id": log_id,
            "bucket_id": bucket_id,
            "priority": 10,
            "arrival_ordinal": 1,
        }
        return {
            **history,
            "bucket_id": bucket_id,
            "priority": 10,
            "slots": [slot],
            "history": [
                {
                    "log_id": log_id,
                    "arrival_seq": 1,
                    "record_index": 0,
                    "arrival_ordinal": 1,
                    "slot_no": 1,
                    "admitted": True,
                    "decision": "admitted",
                    "basis": "test",
                    "replaced_log_id": None,
                    "target_slot_no": 1,
                }
            ],
        }

    truncation = apply_global_budget(
        {
            "z-bucket": bucket("z-bucket", "z-log"),
            "a-bucket": bucket("a-bucket", "a-log"),
        },
        global_budget=1,
    )
    assert [item["log_id"] for item in truncation["final_samples"]] == ["a-log"]
    assert truncation["final_samples"][0]["basis"].startswith("priority=10;admission_ordinal=1;log_id=a-log")

