from __future__ import annotations

import socket

from gsb_index.codes import E_SHARD_START_FAILED
from gsb_index.engine import run_build


def test_one_busy_port_does_not_abort_remaining_shards(sample_config, sample_project, tmp_path):
    blocked_port = sample_project.shard_map()["shard-a"].port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", blocked_port))
        blocker.listen(1)
        result = run_build(sample_config, tmp_path / "work", mode="full", workers=3, global_version=1)
    assert result.ledger["shard_entry_counts"] == {"shard-b": 28, "shard-c": 29}
    start_failures = [
        item
        for item in result.failure_list
        if item["code"] == E_SHARD_START_FAILED and item["object_id"] == "shard-a"
    ]
    assert len(start_failures) == 1
    assert result.ledger["entries_count"] == 57

