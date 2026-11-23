from __future__ import annotations

from typing import Any

from .hot import HotBlockTable


def replay_audit(manifest: dict[str, Any], hot_capacity: int) -> dict[str, Any]:
    """Reconstruct hot counters and validate budget arithmetic from an audit log."""
    hot = HotBlockTable(hot_capacity)
    for event in manifest.get("hot_event_log", []):
        event_type = event["type"]
        if event_type == "invalidate":
            hot.invalidate(int(event["block_id"]), str(event.get("reason", "invalidation")))
        elif event_type == "recompute":
            hot.add_recompute(int(event["block_id"]), int(event["nnz"]))
        elif event_type == "prune":
            hot.prune(int(block_id) for block_id in event.get("valid_blocks", []))
        elif event_type == "clear":
            hot.clear()
        elif event_type == "resize":
            from .hot import HotEntry

            hot = HotBlockTable(
                int(event["capacity"]),
                [HotEntry.from_dict(item) for item in event.get("entries", [])],
            )
        else:
            raise ValueError(f"unknown hot event type: {event_type}")

    ledger = [dict(item) for item in manifest.get("budget_ledger", [])]
    sequence_valid = all(
        int(item["sequence"]) == index
        for index, item in enumerate(ledger, start=1)
    )
    arithmetic_valid = all(
        int(item["budget_before"]) - int(item["cost"]) == int(item["budget_after"])
        for item in ledger
    )
    stable_run_order = True
    previous_run: int | None = None
    previous_block: int | None = None
    for item in ledger:
        run_id = int(item["run_id"])
        block_id = int(item["block_id"])
        if run_id != previous_run:
            previous_run = run_id
            previous_block = None
        if previous_block is not None and block_id <= previous_block:
            stable_run_order = False
        previous_block = block_id
    return {
        "hot_table": hot.to_dict(),
        "hot_fingerprint": hot.fingerprint(),
        "budget_event_count": len(ledger),
        "budget_sequence_valid": sequence_valid,
        "budget_arithmetic_valid": arithmetic_valid,
        "budget_block_order_stable": stable_run_order,
    }
