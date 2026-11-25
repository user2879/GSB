from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace

from gsb_index.cluster import ShardCluster
from gsb_index.codes import E_BUCKET_CAPACITY_EXCEEDED, E_ENTRY_LIMIT_EXCEEDED, E_SHARD_CONCURRENCY_EXCEEDED
from gsb_index.protocol import ShardClient, request_batch


def test_real_subprocess_shard_jsonl_and_hard_concurrency_rejection(sample_project, tmp_path):
    shards = tuple(replace(shard, concurrency_limit=1) if shard.shard_id == "shard-a" else shard for shard in sample_project.shards)
    project = replace(sample_project, shards=shards)
    shard = project.shard_map()["shard-a"]
    documents = [doc.to_dict() for doc in project.documents_by_shard()["shard-a"]]
    profile = project.profile_for_revision(project.current_revision).to_dict()
    cluster = ShardCluster(project, tmp_path, ["shard-a"])
    with cluster:
        client = ShardClient("127.0.0.1", shard.port, timeout=5.0)
        response = client.request(
            request_batch(
                request_id="normal-1",
                shard_id="shard-a",
                revision=project.current_revision,
                documents=documents,
                compression=profile,
                entry_limit=project.budget.per_shard_term_entry_limit,
            )
        )
        assert response["ok"] is True
        assert response["served_revision"] == shard.revision
        assert isinstance(response["elapsed_ms_monotonic"], int)
        assert response["summary"]["entries_count"] == 27
        assert response["summary"]["shard_id"] == "shard-a"
        assert len(response["summary"]["buckets"]) == project.compression.bucket_count
        assert any(bucket["encoded_terms"] for bucket in response["summary"]["buckets"])

        limited = client.request(
            request_batch(
                request_id="limit-1",
                shard_id="shard-a",
                revision=project.current_revision,
                documents=documents,
                compression=profile,
                entry_limit=10,
            )
        )
        assert limited["ok"] is False
        assert limited["code"] == E_ENTRY_LIMIT_EXCEEDED
        assert limited["required_entries"] == 27

        first_result: dict = {}

        def first_request() -> None:
            first_result["response"] = client.request(
                request_batch(
                    request_id="slow-1",
                    shard_id="shard-a",
                    revision=project.current_revision,
                    documents=documents,
                    compression=profile,
                    entry_limit=project.budget.per_shard_term_entry_limit,
                    debug_delay_ms=350,
                )
            )

        thread = threading.Thread(target=first_request)
        thread.start()
        time.sleep(0.12)
        rejected = client.request(
            request_batch(
                request_id="fast-2",
                shard_id="shard-a",
                revision=project.current_revision,
                documents=documents,
                compression=profile,
                entry_limit=project.budget.per_shard_term_entry_limit,
            ),
            timeout=3.0,
        )
        thread.join(timeout=3.0)
        assert first_result["response"]["ok"] is True
        assert rejected["ok"] is False
        assert rejected["code"] == E_SHARD_CONCURRENCY_EXCEEDED
        assert rejected["concurrency_limit"] == shard.concurrency_limit
    # The child process owned the port; after cluster shutdown the port is free again.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", shard.port))


def test_shard_enforces_term_bucket_capacity(sample_project, tmp_path):
    shards = tuple(
        replace(shard, term_bucket_capacity=shard.term_bucket_capacity // 16)
        if shard.shard_id == "shard-a"
        else shard
        for shard in sample_project.shards
    )
    project = replace(sample_project, shards=shards)
    shard = project.shard_map()["shard-a"]
    documents = [doc.to_dict() for doc in project.documents_by_shard()["shard-a"]]
    profile = project.profile_for_revision(project.current_revision).to_dict()
    with ShardCluster(project, tmp_path, ["shard-a"]):
        response = ShardClient("127.0.0.1", shard.port, timeout=5.0).request(
            request_batch(
                request_id="bucket-capacity-1",
                shard_id="shard-a",
                revision=project.current_revision,
                documents=documents,
                compression=profile,
                entry_limit=project.budget.per_shard_term_entry_limit,
            )
        )
    assert response["ok"] is False
    assert response["code"] == E_BUCKET_CAPACITY_EXCEEDED
