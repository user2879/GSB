from __future__ import annotations

import copy
import time

from topic_ledger.app import load_bundle
from topic_ledger.engine import CommandGateway, TopicLedgerEngine
from topic_ledger.replay import (
    compare_reports,
    final_full_rebuild_equivalence,
    replay_equivalence,
    run_command_stream,
)
from topic_ledger.samplegen import build_sample_bundle


def _bundle():
    return load_bundle(".")


def test_generated_command_stream_replays_field_for_field(tmp_path) -> None:
    bundle = _bundle()
    result = replay_equivalence(
        bundle,
        bundle["commands"],
        work_root=tmp_path / "replay",
        concurrency_left=1,
        concurrency_right=4,
    )
    assert result["equal"] is True
    assert result["comparison"]["left_sha256"] == result["comparison"]["right_sha256"]

    source = run_command_stream(
        bundle=bundle, state_dir=tmp_path / "source", commands=bundle["commands"], concurrency=2
    )
    final = final_full_rebuild_equivalence(source_run=source, work_dir=tmp_path / "final-full")
    assert final["equal"] is True


def test_real_subprocess_cluster_replay_matches(tmp_path) -> None:
    bundle = _bundle()
    result = replay_equivalence(
        bundle,
        bundle["commands"],
        work_root=tmp_path / "cluster-replay",
        use_cluster=True,
        concurrency_left=2,
        concurrency_right=3,
    )
    assert result["equal"] is True


def test_document_change_invalidates_only_referenced_shard(tmp_path) -> None:
    bundle = _bundle()
    commands = [
        {"command_id": "run-1", "op": "run_sampling", "batch_id": "b1"},
        {
            "command_id": "replace-doc",
            "op": "upsert_document",
            "document": {
                "doc_id": "doc-000",
                "shard_id": "shard-00",
                "revision": 2,
                "term_ids": [0, 0, 1, 2, 5, 8, 11],
            },
        },
        {"command_id": "run-2", "op": "run_sampling", "batch_id": "b2"},
    ]
    result = run_command_stream(
        bundle=bundle, state_dir=tmp_path / "precise", commands=commands, concurrency=2
    )
    report = result["report"]
    second_cache = report["invalidation_and_reuse"][1]
    assert [item["shard_id"] for item in second_cache["invalidated"]] == ["shard-00"]
    assert [item["shard_id"] for item in second_cache["reused"]] == [
        "shard-01",
        "shard-02",
        "shard-03",
    ]
    assert second_cache["reuse_hits"] == 3
    assert report["closure"]["all_closed"] is True


def test_persistent_cache_is_reused_by_a_new_engine(tmp_path) -> None:
    bundle = build_sample_bundle()
    root = tmp_path / "persistent"
    commands = [{"command_id": "run", "op": "run_sampling", "batch_id": "persist"}]
    first = run_command_stream(bundle=bundle, state_dir=root, commands=commands, concurrency=1)
    assert first["report"]["cache_totals"]["reuse_hit_count"] == 0

    second_engine = TopicLedgerEngine(
        topology=copy.deepcopy(bundle["topology"]),
        inference=copy.deepcopy(bundle["inference"]),
        vocab=copy.deepcopy(bundle["vocab"]),
        documents=copy.deepcopy(bundle["documents"]),
        state_dir=root / "engine",
        global_concurrency_override=1,
    )
    response = second_engine.handle_command(commands[0])
    assert response["ok"] is True
    report = second_engine.canonical_report()
    assert len(report["invalidation_and_reuse"][-1]["reused"]) == 4
    assert report["cache_totals"]["reuse_hit_count"] == 4


def test_budget_pending_then_refill_replay_is_identical(tmp_path) -> None:
    bundle = build_sample_bundle()
    commands = [
        {"command_id": "low-budget", "op": "set_config", "patch": {"sample_point_budget": 1}},
        {"command_id": "run-low", "op": "run_sampling", "batch_id": "low"},
        {"command_id": "refill", "op": "set_config", "patch": {"sample_point_budget": 100_000}},
        {"command_id": "run-full", "op": "run_sampling", "batch_id": "refill"},
    ]
    result = replay_equivalence(
        bundle, commands, work_root=tmp_path / "budget", concurrency_left=1, concurrency_right=3
    )
    assert result["equal"] is True
    assert result["left_report"]["pending"] == []
    assert result["left_report"]["code_counts"]["E_BUDGET_PENDING"] >= 4


def test_lease_failure_sequence_replays_identically(tmp_path) -> None:
    bundle = build_sample_bundle()
    commands = [
        {"command_id": "run", "op": "run_sampling", "batch_id": "lease-seq"},
        {"command_id": "issue", "op": "issue_lease", "shard_id": "shard-00", "batch_id": "explicit"},
        {"command_id": "bad-renew", "op": "renew_lease", "shard_id": "shard-00", "generation": 999},
        {"command_id": "advance", "op": "advance_clock_ms", "milliseconds": 251},
        {"command_id": "recover", "op": "recover_leases"},
        {"command_id": "rerun", "op": "run_sampling", "batch_id": "after-recovery"},
    ]
    result = replay_equivalence(
        bundle, commands, work_root=tmp_path / "lease-seq", concurrency_left=1, concurrency_right=2
    )
    assert result["equal"] is True
    codes = result["left_report"]["code_counts"]
    assert codes["E_LEASE_GENERATION"] == 1
    assert codes["E_LEASE_EXPIRED"] == 1
    assert result["left_report"]["pending"] == []


def test_snapshot_resume_preserves_pending_and_budget(tmp_path) -> None:
    bundle = build_sample_bundle()
    bundle["inference"]["sample_point_budget"] = 1
    root = tmp_path / "snapshot"
    run = run_command_stream(
        bundle=bundle,
        state_dir=root,
        commands=[{"command_id": "run", "op": "run_sampling", "batch_id": "snap"}],
        concurrency=1,
    )
    assert len(run["report"]["pending"]) == 4
    snapshot_root = root / "engine" / "snapshots"
    assert not list(snapshot_root.glob(".tmp-*"))

    restored = TopicLedgerEngine(
        topology=copy.deepcopy(bundle["topology"]),
        inference=copy.deepcopy(bundle["inference"]),
        vocab=copy.deepcopy(bundle["vocab"]),
        documents=copy.deepcopy(bundle["documents"]),
        state_dir=root / "engine",
    )
    decision = restored.restore_snapshot()
    assert decision["decision"] == "resume"
    assert len(restored.pending) == 4
    assert restored.consumed_budget == run["report"]["budget"]["consumed"]
    assert restored.global_version == run["report"]["global_version"]


def test_gateway_returns_stale_summary_with_config_hash(tmp_path) -> None:
    bundle = build_sample_bundle()
    bundle["inference"]["interaction_budget_ms"] = 1
    engine = TopicLedgerEngine(
        topology=bundle["topology"],
        inference=bundle["inference"],
        vocab=bundle["vocab"],
        documents=bundle["documents"],
        state_dir=tmp_path / "gateway",
    )
    gateway = CommandGateway(engine)
    response = gateway.handle({"command_id": "slow", "op": "sleep_ms", "milliseconds": 50})
    assert response["ok"] is False
    assert response["code"] == "E_STALE_SUMMARY"
    assert response["stale"] is True
    assert response["summary"]["config_hash"] == engine.config_hash
    time.sleep(0.08)
    gateway.close()


def test_compare_reports_reports_inequality() -> None:
    bundle = build_sample_bundle()
    engine = TopicLedgerEngine(
        topology=bundle["topology"],
        inference=bundle["inference"],
        vocab=bundle["vocab"],
        documents=bundle["documents"],
        state_dir="/tmp/topic-ledger-compare-negative",
    )
    engine.handle_command({"op": "run_sampling", "batch_id": "negative"})
    left = engine.canonical_report()
    right = copy.deepcopy(left)
    right["global_distribution"][0]["count"] += 1
    comparison = compare_reports(left, right)
    assert comparison["equal"] is False
    assert comparison["differences"]
    assert "$.global_distribution[0].count" in comparison["differences"][0]
