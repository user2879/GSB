from __future__ import annotations

import copy

import pytest

from topic_ledger.cluster import SubprocessCluster
from topic_ledger.errors import BusinessError, E_DUPLICATE_DELIVERY
from topic_ledger.samplegen import build_sample_bundle


@pytest.fixture
def cluster(tmp_path):
    bundle = build_sample_bundle()
    with SubprocessCluster.start(
        bundle["topology"], tmp_path / "cluster", override_free_ports=True
    ) as handles:
        yield handles


def _sample_request(bundle, *, batch_id="batch-worker-1", prior=None):
    return {
        "op": "sample",
        "batch_id": batch_id,
        "shard_id": "shard-worker",
        "term_counts": [[0, 3], [2, 2]],
        "document_refs": [{"doc_id": "doc-worker", "revision": 1}],
        "config": bundle["inference"],
        "vocab_size": len(bundle["vocab"]),
        "prior_receipt_digest": prior,
    }


def test_real_subprocess_jsonl_worker_duplicate_guard_and_timing(cluster) -> None:
    bundle = build_sample_bundle()
    first_response = cluster.request("node-a", _sample_request(bundle))
    assert first_response["ok"] is True
    receipt = first_response["receipt"]
    assert receipt["node_id"] == "node-a"
    assert isinstance(receipt["elapsed_monotonic_ms"], int)
    assert receipt["elapsed_monotonic_ms"] >= 0
    assert len(receipt["term_topic_counts"]) == 2 * bundle["inference"]["topic_count"]

    duplicate = cluster.request(
        "node-a", _sample_request(bundle, batch_id="batch-rejected-no-digest")
    )
    assert duplicate["ok"] is False
    assert duplicate["code"] == E_DUPLICATE_DELIVERY
    assert duplicate["batch_id"] == "batch-rejected-no-digest"
    assert duplicate["details"]["current_receipt_digest"] == receipt["receipt_digest"]

    rejections = cluster.request("node-a", {"op": "rejections"})["rejections"]
    assert rejections[-1]["batch_id"] == "batch-rejected-no-digest"
    assert rejections[-1]["code"] == E_DUPLICATE_DELIVERY

    accepted = cluster.request(
        "node-a",
        _sample_request(
            bundle,
            batch_id="batch-overwrite-with-digest",
            prior=receipt["receipt_digest"],
        ),
    )
    assert accepted["ok"] is True
    assert accepted["receipt"]["receipt_digest"] == receipt["receipt_digest"]


def test_cluster_sample_helper_raises_business_error(cluster) -> None:
    bundle = build_sample_bundle()
    cluster.sample("node-b", _sample_request(bundle, batch_id="helper-1"))
    with pytest.raises(BusinessError) as exc_info:
        cluster.sample("node-b", _sample_request(copy.deepcopy(bundle), batch_id="helper-2"))
    assert exc_info.value.code == E_DUPLICATE_DELIVERY
    assert exc_info.value.batch_id == "helper-2"
