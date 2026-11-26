from __future__ import annotations

import copy
import time

import pytest

from topic_ledger.engine import ManualClock
from topic_ledger.errors import BusinessError, E_LEASE_EXPIRED, E_LEASE_GENERATION, E_LEDGER_FROZEN, E_LEDGER_MISMATCH
from topic_ledger.leases import LeaseManager, SlotPool, monotonic_ms
from topic_ledger.ledger import ReceiptLedger
from topic_ledger.samplegen import build_sample_bundle
from topic_ledger.sampling import compute_shard_receipt
from topic_ledger.scheduler import PendingShard, build_schedule


def test_lease_generation_mismatch_does_not_advance() -> None:
    clock = ManualClock(10_000)
    manager = LeaseManager(ttl_ms=100, clock=clock)
    lease = manager.issue(shard_id="s1", node_id="a", lease_limit=1, batch_id="b1")
    assert lease.generation == 1

    with pytest.raises(BusinessError) as exc_info:
        manager.renew(shard_id="s1", generation=999)
    assert exc_info.value.code == E_LEASE_GENERATION
    assert manager.generation == 1

    renewed = manager.renew(shard_id="s1", generation=1, ttl_ms=50)
    assert renewed.expires_at_ms == 10_050
    assert manager.generation == 1

    with pytest.raises(BusinessError):
        manager.release(shard_id="s1", generation=2)
    assert manager.active_count("a") == 1


def test_expired_lease_really_releases_inflight_slot() -> None:
    clock = ManualClock(5_000)
    pool = SlotPool(global_concurrency=1, node_quotas={"a": 1})
    ticket = pool.acquire("a", timeout_seconds=0.1)
    assert pool.metrics()["global_held"] == 1

    manager = LeaseManager(ttl_ms=5, clock=clock)
    manager.issue(shard_id="s1", node_id="a", lease_limit=1, batch_id="b1", ticket=ticket)
    clock.advance(6)
    expired = manager.recover_expired()

    assert expired[0].status == "expired"
    assert manager.ended[-1]["code"] == E_LEASE_EXPIRED
    assert ticket.released is True
    assert pool.metrics()["global_held"] == 0
    assert pool.global_slots.acquire(timeout=0.1) is True
    pool.global_slots.release()


def test_real_monotonic_ttl_recovery_releases_slot() -> None:
    pool = SlotPool(global_concurrency=1, node_quotas={"a": 1})
    ticket = pool.acquire("a", timeout_seconds=0.1)
    manager = LeaseManager(ttl_ms=1, clock=monotonic_ms)
    manager.issue(shard_id="real", node_id="a", lease_limit=1, batch_id="real-batch", ticket=ticket)
    time.sleep(0.02)
    expired = manager.recover_expired()
    assert len(expired) == 1
    assert ticket.released is True
    assert pool.metrics()["global_held"] == 0


def _receipt(bundle, shard_id, node_id, *, monotonic_ms, version=None):
    config = copy.deepcopy(bundle["inference"])
    if version is not None:
        config["topic_parameter_version"] = version
    return compute_shard_receipt(
        {
            "node_id": node_id,
            "shard_id": shard_id,
            "term_counts": [[0, 2], [1, 1]],
            "document_refs": [{"doc_id": f"doc-{shard_id}", "revision": 1}],
            "config": config,
            "vocab_size": len(bundle["vocab"]),
            "monotonic_clock_ms": monotonic_ms,
        }
    )


def test_ledger_authority_order_and_closure() -> None:
    bundle = build_sample_bundle()
    ledger = ReceiptLedger(term_ids=[0, 1], topic_count=bundle["inference"]["topic_count"])
    r1 = _receipt(bundle, "s1", "node-a", monotonic_ms=100)
    r2 = _receipt(bundle, "s2", "node-b", monotonic_ms=200)
    ledger.accept_receipt(r1, batch_id="b1")
    ledger.accept_receipt(r2, batch_id="b2")
    result = ledger.rebuild()
    assert result["all_closed"] is True
    term_zero = next(item for item in ledger.authority if item["term_id"] == 0)
    assert term_zero["authoritative_source"]["shard_id"] == "s2"
    assert term_zero["covered_sources"][0]["source"]["shard_id"] == "s1"
    assert ledger.overwrites[0]["rule"].startswith("topic_parameter_version_desc")


def test_ledger_node_lexicographic_tie_break_and_freeze() -> None:
    bundle = build_sample_bundle()
    ledger = ReceiptLedger(term_ids=[0, 1], topic_count=bundle["inference"]["topic_count"])
    ledger.accept_receipt(_receipt(bundle, "s1", "node-b", monotonic_ms=200), batch_id="b1")
    ledger.accept_receipt(_receipt(bundle, "s2", "node-a", monotonic_ms=200), batch_id="b2")
    ledger.rebuild()
    term_zero = next(item for item in ledger.authority if item["term_id"] == 0)
    assert term_zero["authoritative_source"]["node_id"] == "node-a"

    ledger.distribution[(0, 0)] += 1
    closure = ledger.verify_closure()
    assert closure["all_closed"] is False
    assert closure["mismatched_topics"] == [0]
    assert 0 in ledger.frozen_topics

    with pytest.raises(BusinessError) as exc_info:
        ledger.accept_receipt(_receipt(bundle, "s3", "node-c", monotonic_ms=300), batch_id="b3")
    assert exc_info.value.code == E_LEDGER_FROZEN
    assert "s3" not in ledger.receipts


def test_duplicate_receipt_requires_prior_digest() -> None:
    bundle = build_sample_bundle()
    ledger = ReceiptLedger(term_ids=[0, 1], topic_count=bundle["inference"]["topic_count"])
    receipt = _receipt(bundle, "s1", "node-a", monotonic_ms=100)
    ledger.accept_receipt(receipt, batch_id="batch-1")
    with pytest.raises(BusinessError) as exc_info:
        ledger.accept_receipt(copy.deepcopy(receipt), batch_id="batch-2")
    assert exc_info.value.code == "E_DUPLICATE_DELIVERY"
    assert ledger.rejections[-1]["batch_id"] == "batch-2"
    ledger.accept_receipt(
        copy.deepcopy(receipt), batch_id="batch-3", prior_receipt_digest=receipt["receipt_digest"]
    )


def test_budget_pending_order_and_next_round_priority() -> None:
    shards = ["s0", "s1", "s2", "s3"]
    result = build_schedule(
        ordered_shards=shards,
        routing={s: "a" for s in shards},
        costs={s: 10 for s in shards},
        batch_ids={s: f"b-{s}" for s in shards},
        budget_limit=25,
        lease_limits={"a": 4},
    )
    assert [item.shard_id for item in result.items] == ["s0", "s1"]
    assert [item.shard_id for item in result.pending] == ["s2", "s3"]
    assert all(item.reason == "E_BUDGET_PENDING" for item in result.pending)

    next_round = build_schedule(
        ordered_shards=shards,
        routing={s: "a" for s in shards},
        costs={s: 10 for s in shards},
        batch_ids={s: f"b2-{s}" for s in shards},
        budget_limit=20,
        lease_limits={"a": 4},
        existing_pending=result.pending,
    )
    assert [item.shard_id for item in next_round.items] == ["s2", "s3"]


def test_node_lease_limit_aborts_only_that_node_round() -> None:
    result = build_schedule(
        ordered_shards=["s0", "s1", "s2"],
        routing={"s0": "a", "s1": "a", "s2": "b"},
        costs={"s0": 1, "s1": 1, "s2": 1},
        batch_ids={s: f"b-{s}" for s in ["s0", "s1", "s2"]},
        budget_limit=100,
        lease_limits={"a": 1, "b": 1},
    )
    assert [item.shard_id for item in result.items] == ["s0", "s2"]
    assert [item.shard_id for item in result.pending] == ["s1"]
    assert result.pending[0].reason == "E_NODE_LEASE_LIMIT"
    assert result.node_blocked["a"] == "E_NODE_LEASE_LIMIT"


def test_ledger_mismatch_code_constant_matches_contract() -> None:
    assert E_LEDGER_MISMATCH == "E_LEDGER_MISMATCH"
