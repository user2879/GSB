from __future__ import annotations

import copy

from gsb_index.codes import E_BUDGET_PENDING, E_SHARD_ENTRY_LIMIT
from gsb_index.edits import project_to_config
from gsb_index.engine import run_build


def _with_budget(sample_project, global_budget: int, per_shard: int) -> dict:
    raw = copy.deepcopy(project_to_config(sample_project))
    raw["budget"] = {
        "global_term_entry_budget": global_budget,
        "per_shard_term_entry_limit": per_shard,
    }
    return raw


def test_global_budget_leaves_stable_pending_queue(sample_project, write_config, tmp_path):
    config = write_config(_with_budget(sample_project, 30, 80), "budget-low.json")
    result = run_build(config, tmp_path / "work", mode="full", workers=3, global_version=1)
    assert result.ledger["entries_count"] == 27
    assert result.report["pending_shards"] == ["shard-b", "shard-c"]
    pending_codes = [
        item["code"] for item in result.failure_list if item["object_type"] == "shard"
    ]
    assert pending_codes == [E_BUDGET_PENDING, E_BUDGET_PENDING]
    assert result.report["scheduler"]["next_batch_priority"] == ["shard-b", "shard-c"]
    assert result.report["scheduler"]["pending"][0]["required_entries"] == 28


def test_next_batch_prioritizes_pending_before_stable_scan(sample_project, write_config, tmp_path):
    config = write_config(_with_budget(sample_project, 120, 80), "budget-priority.json")
    result = run_build(
        config,
        tmp_path / "work",
        mode="full",
        workers=3,
        global_version=1,
        prior_pending=["shard-c", "shard-b"],
    )
    assert result.report["scheduler"]["priority_order"][:2] == ["shard-b", "shard-c"]
    assert result.report["scheduler"]["dispatch_order"] == ["shard-a", "shard-b", "shard-c"]
    assert result.ledger["entries_count"] == 84


def test_per_shard_limit_aborts_batch_with_business_code(sample_project, write_config, tmp_path):
    config = write_config(_with_budget(sample_project, 240, 20), "budget-hard-limit.json")
    result = run_build(config, tmp_path / "work", mode="full", workers=3, global_version=1)
    assert result.ledger["entries_count"] == 0
    assert result.report["pending_shards"] == []
    codes = sorted(item["code"] for item in result.failure_list)
    assert codes == [E_SHARD_ENTRY_LIMIT, E_SHARD_ENTRY_LIMIT, E_SHARD_ENTRY_LIMIT]
    for item in result.failure_list:
        assert item["details"]["required_entries"] > item["details"]["per_shard_limit"]


def test_zero_global_budget_pending_without_scanning(sample_project, write_config, tmp_path):
    config = write_config(_with_budget(sample_project, 0, 80), "budget-zero.json")
    result = run_build(config, tmp_path / "work", mode="full", workers=3, global_version=1)
    assert result.report["pending_shards"] == ["shard-a", "shard-b", "shard-c"]
    assert result.report["scheduler"]["dispatch_order"] == []
    assert all(item["code"] == E_BUDGET_PENDING for item in result.failure_list)

