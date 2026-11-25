from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .codes import MATCH_BUDGET_PENDING
from .ledger import atomic_write_json


def stable_work_key(item: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(item["image"]),
        int(item.get("rtl_index", item.get("column_seq", 0))),
        int(item.get("char_index", item.get("char_seq", 0))),
        int(item.get("global_detected_index", 0)),
    )


class BudgetScheduler:
    def __init__(self, budget: int, queue_path: str | Path | None = None, max_workers: int = 1):
        self.batch_budget = int(budget)
        self.remaining = int(budget)
        self.max_workers = max(1, int(max_workers))
        self.queue_path = Path(queue_path) if queue_path else None
        self.saved_pending: list[dict[str, Any]] = []
        self.total_consumed = 0
        self.batch_consumed = 0
        if self.queue_path and self.queue_path.exists():
            data = json.loads(self.queue_path.read_text(encoding="utf-8"))
            self.saved_pending = data.get("pending", [])
            self.total_consumed = int(data.get("total_consumed", 0))

    def _save(self, pending: list[dict[str, Any]]) -> None:
        if not self.queue_path:
            return
        atomic_write_json(self.queue_path, {
            "batch_budget": self.batch_budget,
            "remaining_budget": self.remaining,
            "batch_consumed": self.batch_consumed,
            "total_consumed": self.total_consumed,
            "pending": pending,
        })

    def admit(self, new_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        new_sorted = sorted(new_items, key=stable_work_key)
        by_id = {item["work_id"]: item for item in new_sorted}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for saved in self.saved_pending:
            work_id = saved["work_id"]
            item = by_id.get(work_id, saved)
            if work_id not in seen:
                ordered.append(item)
                seen.add(work_id)
        for item in new_sorted:
            if item["work_id"] not in seen:
                ordered.append(item)
                seen.add(item["work_id"])

        admitted: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        stopped = False
        for item in ordered:
            cost = int(item["point_cost"])
            if stopped:
                pending.append(self._pending_marker(item, cost, "earlier_item_insufficient_budget"))
                continue
            if cost <= self.remaining:
                self.remaining -= cost
                self.batch_consumed += cost
                self.total_consumed += cost
                admitted.append(item)
            else:
                stopped = True
                pending.append(self._pending_marker(item, cost, "insufficient_remaining_budget"))
        self.saved_pending = pending
        self._save(pending)
        return admitted, pending

    def _pending_marker(self, item: dict[str, Any], cost: int, reason: str) -> dict[str, Any]:
        return {
            "code": MATCH_BUDGET_PENDING,
            "status": MATCH_BUDGET_PENDING,
            "reason": reason,
            "priority": "next_batch_first",
            "point_cost": cost,
            "image": item.get("image"),
            "work_id": item.get("work_id"),
            "character_id": item.get("character_id"),
            "rtl_index": item.get("rtl_index"),
            "char_index": item.get("char_index"),
            "bbox": item.get("bbox"),
        }

    def run_pool(self, items: list[dict[str, Any]], worker: Callable[[dict[str, Any]], Any]) -> list[Any]:
        if not items:
            return []
        workers = min(self.max_workers, len(items))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(worker, items))

    def reset_batch(self, budget: int | None = None) -> None:
        if budget is not None:
            self.batch_budget = int(budget)
        self.remaining = self.batch_budget
        self.batch_consumed = 0
