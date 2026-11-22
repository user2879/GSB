from __future__ import annotations

from iidx.config import AppConfig
from iidx.loader import NormalizedDoc
from iidx.scheduler import run_scheduler


def _doc(doc_id: str, shard: str, tokens: int = 10) -> NormalizedDoc:
    terms = tuple("alpha" for _ in range(tokens))
    return NormalizedDoc(
        doc_id=doc_id,
        shard=shard,
        title_terms=terms,
        body_terms=(),
        token_terms=terms,
        token_count=tokens,
        raw_fingerprint=doc_id,
        normalized_fingerprint=doc_id,
        norm_cache_key=doc_id,
    )


def test_global_budget_exhaustion_marks_pending_queue() -> None:
    by_shard = {
        "shard-a": (_doc("doc-1", "shard-a", 10), _doc("doc-2", "shard-a", 10)),
        "shard-b": (_doc("doc-3", "shard-b", 10),),
    }
    config = AppConfig(
        batch_max_docs=1,
        thread_pool_size=2,
        concurrency_quota=2,
        global_token_budget=15,
        per_shard_token_quota=100,
    )
    result = run_scheduler(by_shard, config)
    assert result.failures[0]["code"] == "E_BUDGET_PENDING"
    assert result.accepted_doc_ids == ["doc-1"]
    assert result.pending_shards == ["shard-a", "shard-b"]
    assert result.ledger == [
        {
            "deduction_index": 1,
            "stable_order": 1,
            "shard": "shard-a",
            "batch_index": 0,
            "tokens": 10,
            "remaining": 5,
            "remaining_before": 15,
            "stable_key": "shard-a:00000000:schedule",
        }
    ]


def test_shard_quota_exhaustion_stops_only_that_shard() -> None:
    by_shard = {
        "shard-a": (_doc("doc-1", "shard-a", 10), _doc("doc-2", "shard-a", 10)),
        "shard-b": (_doc("doc-3", "shard-b", 5),),
    }
    config = AppConfig(
        batch_max_docs=1,
        thread_pool_size=2,
        concurrency_quota=2,
        global_token_budget=100,
        per_shard_token_quota=12,
    )
    result = run_scheduler(by_shard, config)
    codes = [item["code"] for item in result.failures]
    assert codes == ["E_SHARD_QUOTA_EXHAUSTED"]
    assert result.accepted_doc_ids == ["doc-1", "doc-3"]
    assert result.partial_shards == ["shard-a"]
    assert result.completed_shards == ["shard-b"]
    assert [item["deduction_index"] for item in result.ledger] == [1, 2]
    assert result.ledger[0]["remaining"] > result.ledger[1]["remaining"]


def test_ledger_and_failures_are_order_stable() -> None:
    by_shard = {
        "shard-b": (_doc("b-1", "shard-b", 8),),
        "shard-a": (_doc("a-1", "shard-a", 8), _doc("a-2", "shard-a", 8)),
    }
    config = AppConfig(
        batch_max_docs=1,
        global_token_budget=100,
        per_shard_token_quota=12,
        thread_pool_size=2,
        concurrency_quota=2,
    )
    first = run_scheduler(by_shard, config)
    second = run_scheduler(by_shard, config)
    assert first.ledger == second.ledger
    assert first.failures == second.failures
    assert [item["shard"] for item in first.ledger] == ["shard-a", "shard-b"]
