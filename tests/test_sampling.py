from __future__ import annotations

from trace_diag.generator import generate_records
from trace_diag.sampling import sample_records, stable_bucket


def test_partition_bucket_is_stable():
    first = stable_bucket("client-7", 3, 2)
    assert first == stable_bucket("client-7", 3, 2)
    assert 0 <= first < 10_000


def test_head_and_tail_are_disjoint_and_deterministic(service_doc, policy_doc):
    service = service_doc.services[0]
    records = generate_records(service).records
    policy = policy_doc.policies[0]
    first = sample_records(records, policy, 1)
    second = sample_records(records, policy, 1)
    assert first.summary == second.summary
    assert {decision.record_index for decision in first.decisions if decision.reason == "head"} & {
        decision.record_index for decision in first.decisions if decision.reason == "tail"
    } == set()
    reasons = [decision.reason for decision in first.decisions]
    assert reasons.count("head") == first.summary["head_hits"]
    assert reasons.count("tail") == first.summary["tail_hits"]
    assert reasons.count("skip") == first.summary["skipped"]
    assert first.summary["sample_hits"] == first.summary["head_hits"] + first.summary["tail_hits"]


def test_client_key_conclusion_depends_only_on_declared_inputs(service_doc, policy_doc):
    records = generate_records(service_doc.services[0]).records
    policy = policy_doc.policies[0]
    changed_keys = [dict(record) for record in records]
    changed_keys[0]["client_key"] = "another-client"
    baseline = sample_records(records, policy, policy_doc.version).decisions[0]
    changed = sample_records(changed_keys, policy, policy_doc.version).decisions[0]
    assert baseline.bucket == stable_bucket(
        records[0]["client_key"], policy_doc.version, policy.scale_power
    )
    assert changed.bucket == stable_bucket(
        "another-client", policy_doc.version, policy.scale_power
    )
