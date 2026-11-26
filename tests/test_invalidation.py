from __future__ import annotations

from hopledger.invalidation import plan_invalidation, verify_digests_unchanged


NODE_IDS = ["client", "fwd1", "rev2", "origin"]


def test_invalidation_is_request_scoped_and_chain_ordered():
    stored = {
        ("req-old", "fwd1"): {"config_version": 1, "scale_factor": 1},
        ("req-old", "rev2"): {"config_version": 1, "scale_factor": 1000},
        ("req-old", "origin"): {"config_version": 1, "scale_factor": 1000},
        ("req-new", "fwd1"): {"config_version": 2, "scale_factor": 1},
        ("req-new", "rev2"): {"config_version": 1, "scale_factor": 1000},
        ("req-new", "origin"): {"config_version": 1, "scale_factor": 1000},
    }
    plan = plan_invalidation(
        request_ids=["req-new", "req-old"],
        chain_node_ids=NODE_IDS,
        stored_receipt_versions=stored,
        current_config_versions={"fwd1": 2, "rev2": 1, "origin": 1},
        current_scale_factors={"fwd1": 1, "rev2": 1000, "origin": 1000},
    )

    assert plan["affected_request_ids"] == ["req-old"]
    assert plan["redispatch_nodes"] == ["receipt:req-old:fwd1"]
    assert "e2e:req-old" in plan["invalid_nodes"]
    assert "attribution:req-old" in plan["invalid_nodes"]
    assert not any("req-new" in node for node in plan["invalid_nodes"])

    before = {
        "receipt:req-new:fwd1": "same",
        "conclusion:req-new": "same-conclusion",
        "attribution:req-new": "same-dominant-decision",
        "conclusion:req-old": "old",
        "attribution:req-old": "old",
    }
    after = dict(before)
    after["conclusion:req-old"] = "new"
    after["attribution:req-old"] = "new"
    violations = verify_digests_unchanged(
        before,
        after,
        allowed_changed=["conclusion:req-old", "attribution:req-old"],
    )
    assert violations == []
