from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .commands import CommandState, diff_objects, load_all_receipts, load_commands, load_queries
from .engine import IndexEngine
from .snapshot import SnapshotStore


def run_stream_equivalence(sample_root: str | Path, output_root: str | Path | None = None) -> Dict[str, Any]:
    sample_root = Path(sample_root)
    output_root = Path(output_root or sample_root / "outputs")
    incremental_root = output_root / "incremental"
    if incremental_root.exists():
        shutil.rmtree(incremental_root)
    store = SnapshotStore(incremental_root / "snapshots")

    state = CommandState.from_sample_dir(sample_root)
    commands = load_commands(sample_root)
    receipts = load_all_receipts(sample_root)
    queries = load_queries(sample_root)

    steps: List[Dict[str, Any]] = []
    all_match = True
    budget_pending_seen = False
    invalidation_by_command: Dict[str, Any] = {}

    for command_index, command in enumerate(commands):
        state.apply(command)
        engine = IndexEngine.recover(
            state.config,
            store,
            receipts=receipts,
            queries=queries,
            receipt_preference=state.receipt_version,
        )
        blocks = state.blocks
        budget_events: List[Dict[str, Any]] = []
        report = engine.run_batch(blocks)
        if report["status"] == "E_POINT_BUDGET_PENDING":
            budget_pending_seen = True
            budget_events.append(
                {
                    "code": report["status"],
                    "scheduled_ids": report["scheduled_ids"],
                    "pending_ids": report["pending_ids"],
                    "point_budget": report["point_budget"],
                }
            )
            while report["status"] == "E_POINT_BUDGET_PENDING":
                engine = IndexEngine.recover(
                    state.config,
                    store,
                    receipts=receipts,
                    queries=queries,
                    receipt_preference=state.receipt_version,
                )
                report = engine.run_batch(state.blocks)
                if report["status"] == "E_POINT_BUDGET_PENDING":
                    budget_events.append(
                        {
                            "code": report["status"],
                            "scheduled_ids": report["scheduled_ids"],
                            "pending_ids": report["pending_ids"],
                            "point_budget": report["point_budget"],
                        }
                    )

        fresh_state = CommandState.from_sample_dir(sample_root)
        for prefix_command in commands[: command_index + 1]:
            fresh_state.apply(prefix_command)
        fresh_engine = IndexEngine(
            fresh_state.config,
            receipts=receipts,
            queries=queries,
            receipt_preference=fresh_state.receipt_version,
        )
        fresh_report = fresh_engine.run_batch(fresh_state.blocks, budget=10**9)
        invalidation_by_command[str(command.get("id"))] = report["invalidation"]

        differences = diff_objects(fresh_report["view"], report["view"])
        match = not differences
        all_match = all_match and match

        snapshot_roundtrip_match = None
        recovered_snapshot = store.load_current()
        if recovered_snapshot and "view" in recovered_snapshot.get("artifacts", {}):
            snapshot_differences = diff_objects(report["view"], recovered_snapshot["artifacts"]["view"])
            snapshot_roundtrip_match = not snapshot_differences
            all_match = all_match and snapshot_roundtrip_match
        else:
            snapshot_roundtrip_match = False
            all_match = False

        incremental_invalidation = report["invalidation"]
        fresh_order = fresh_report["invalidation"]["reprojected_blocks"]
        incremental_reprojected = incremental_invalidation["reprojected_blocks"]
        expected_common_order = [block_id for block_id in fresh_order if block_id in set(incremental_reprojected)]
        common_order_match = expected_common_order == incremental_reprojected
        reuse_order_match = incremental_invalidation["reused_order_is_canonical_subsequence"]
        recompute_order_match = incremental_invalidation["recomputed_order_is_canonical_subsequence"] and common_order_match
        all_match = all_match and common_order_match and reuse_order_match and recompute_order_match

        steps.append(
            {
                "command_index": command_index,
                "command_id": command.get("id"),
                "op": command["op"],
                "match": match,
                "field_difference_count": len(differences),
                "first_differences": differences[:8],
                "snapshot_roundtrip_match": snapshot_roundtrip_match,
                "common_recompute_order_match": common_order_match,
                "reuse_order_match": reuse_order_match,
                "recompute_order_match": recompute_order_match,
                "block_count": len(report["view"]["blocks"]),
                "bucket_count": len(report["buckets"]),
                "budget_events": budget_events,
                "recovery": report.get("recovery"),
                "invalidation_summary": {
                    "new_blocks": len(incremental_invalidation["new_blocks"]),
                    "content_changed_blocks": len(incremental_invalidation["content_changed_blocks"]),
                    "removed_blocks": len(incremental_invalidation["removed_blocks"]),
                    "bucket_key_changed_blocks": len(incremental_invalidation["bucket_key_changed_blocks"]),
                    "reprojected_blocks": len(incremental_invalidation["reprojected_blocks"]),
                    "requantized_blocks": len(incremental_invalidation["requantized_blocks"]),
                    "reused_blocks": len(incremental_invalidation["reused_blocks"]),
                    "invalidated_buckets": incremental_invalidation["invalidated_buckets"],
                    "placement_invalidated_buckets": incremental_invalidation["placement_invalidated_buckets"],
                },
                "observed_max_concurrency": report.get("observed_max_concurrency"),
                "snapshot_global_version": (report.get("manifest") or {}).get("global_version"),
            }
        )

    summary = {
        "schema": "incremental-equivalence-report/v1",
        "all_fields_match": all_match,
        "command_count": len(commands),
        "budget_pending_seen": budget_pending_seen,
        "step_count": len(steps),
        "steps": steps,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    import json

    (output_root / "equivalence_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "invalidation_report.json").write_text(
        json.dumps(invalidation_by_command, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def compare_command_sequence(sample_root: str | Path, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_root = Path(sample_root)
    receipts = load_all_receipts(sample_root)
    queries = load_queries(sample_root)
    state = CommandState.from_sample_dir(sample_root)
    differences_by_step: List[List[Dict[str, Any]]] = []

    with tempfile.TemporaryDirectory(prefix="sequence-snapshot-") as temporary:
        store = SnapshotStore(Path(temporary) / "snapshots")
        applied: List[Dict[str, Any]] = []
        for command in commands:
            command = dict(command)
            command.setdefault("id", f"generic-{len(applied) + 1:03d}")
            state.apply(command)
            applied.append(command)
            incremental = IndexEngine.recover(
                state.config,
                store,
                receipts=receipts,
                queries=queries,
                receipt_preference=state.receipt_version,
            ).run_batch(state.blocks, budget=10**9)

            fresh = CommandState.from_sample_dir(sample_root)
            for prefix_command in applied:
                fresh.apply(prefix_command)
            full = IndexEngine(
                fresh.config,
                receipts=receipts,
                queries=queries,
                receipt_preference=fresh.receipt_version,
            ).run_batch(fresh.blocks, budget=10**9)
            differences = diff_objects(full["view"], incremental["view"])
            differences_by_step.append(differences)
            if differences:
                break

    return {
        "all_fields_match": not any(differences_by_step),
        "steps": len(applied),
        "differences": differences_by_step,
    }
