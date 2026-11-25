from __future__ import annotations

import copy

from gsb_index.codes import E_RECEIPT_STALE_INCOMPATIBLE
from gsb_index.edits import project_to_config
from gsb_index.engine import run_build


def test_stale_shard_receipt_is_recorded_with_coverage_not_dropped(sample_project, write_config, tmp_path):
    raw = copy.deepcopy(project_to_config(sample_project))
    raw["layout"]["shards"][0]["revision"] = 1
    config = write_config(raw, "stale-revision.json")
    result = run_build(config, tmp_path / "work", mode="full", workers=3, global_version=1)
    assert result.ledger["shard_entry_counts"] == {"shard-b": 28, "shard-c": 29}
    stale = [
        item
        for item in result.ledger["receipt_arbitrations"]
        if item["shard_id"] == "shard-a"
    ]
    assert len(stale) == 1
    record = stale[0]
    assert record["served_revision"] == 1
    assert record["current_revision"] == 2
    assert record["decision"] == "rejected"
    assert record["code"] == E_RECEIPT_STALE_INCOMPATIBLE
    assert record["covered_by"] == [2]
    assert record["counted"] is False
    assert record["raw_receipt"]["summary"]["shard_id"] == "shard-a"
    assert any(
        issue["code"] == E_RECEIPT_STALE_INCOMPATIBLE
        and issue["details"]["covered_by"] == [2]
        for issue in result.failure_list
    )
    assert result.ledger["arbitration_stats"]["rejected"] == 1

