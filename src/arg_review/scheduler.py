from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable


class BudgetScheduler:
    def __init__(self, config: dict[str, Any]) -> None:
        budget = config.get("budget", config)
        self.max_workers = max(1, int(budget.get("max_workers", 4)))
        self.global_limit = int(budget.get("global_arg_score_points", 10**9))
        self.event_quota = int(budget.get("per_event_arg_quota", 10**9))
        self.trigger_batch_limit = int(budget.get("max_trigger_matches_per_batch", 10**9))
        self.consumed_points = 0
        self.event_consumed: dict[str, int] = {}
        self.trigger_queue: list[str] = []
        self.arg_queue: list[dict[str, Any]] = []
        self.budget_decisions: list[dict[str, Any]] = []
        self.batch_reports: list[dict[str, Any]] = []
        self.peak_concurrency = 0
        self._active = 0
        self._lock = threading.Lock()

    def _enter(self) -> int:
        with self._lock:
            self._active += 1
            self.peak_concurrency = max(self.peak_concurrency, self._active)
            return self._active

    def _exit(self) -> None:
        with self._lock:
            self._active -= 1

    @staticmethod
    def unit_key(unit: dict[str, Any]) -> str:
        return f"{unit['event_id']}::{unit['role']}"

    def enqueue_trigger_events(self, event_ids: list[str]) -> None:
        existing = list(self.trigger_queue)
        seen = set(existing)
        for event_id in sorted(event_ids):
            if event_id not in seen:
                existing.append(event_id)
                seen.add(event_id)
        self.trigger_queue = existing

    def run_trigger_batch(
        self, trigger_fn: Callable[[str], dict[str, Any]]
    ) -> dict[str, Any]:
        batch_ids = self.trigger_queue[: self.trigger_batch_limit]
        self.trigger_queue = self.trigger_queue[len(batch_ids) :]
        results: dict[str, Any] = {}
        failures: dict[str, str] = {}

        def wrapped(event_id: str) -> dict[str, Any]:
            self._enter()
            try:
                return trigger_fn(event_id)
            finally:
                self._exit()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(wrapped, event_id): event_id for event_id in batch_ids}
            for future in as_completed(futures):
                event_id = futures[future]
                try:
                    results[event_id] = future.result()
                except Exception as exc:  # isolated event failure
                    failures[event_id] = f"{type(exc).__name__}:{exc}"

        report = {
            "stage": "trigger",
            "trigger_jobs_in_batch": len(batch_ids),
            "trigger_match_count_limit": self.trigger_batch_limit,
            "processed_event_ids": sorted(results),
            "failed_event_ids": sorted(failures),
            "pending_event_ids": list(self.trigger_queue),
            "pending_code": "E_ARG_PENDING" if self.trigger_queue else None,
            "peak_concurrency": self.peak_concurrency,
            "failures": failures,
        }
        self.batch_reports.append(report)
        return report

    def enqueue_argument_units(self, units: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for unit in units:
            item = dict(unit)
            item["unit_key"] = self.unit_key(item)
            item.setdefault("point_cost", max(1, int(item.get("candidate_count", 1))))
            normalized.append(item)
        normalized.sort(key=lambda item: (item["event_id"], item["role"], item["unit_key"]))

        pending_keys = {item["unit_key"] for item in self.arg_queue}
        merged = list(self.arg_queue)
        for item in normalized:
            if item["unit_key"] not in pending_keys:
                merged.append(item)
                pending_keys.add(item["unit_key"])
        self.arg_queue = merged

    def run_argument_batch(
        self, score_fn: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any]:
        admitted: list[dict[str, Any]] = []
        blocking_decision: dict[str, Any] | None = None
        consumed_before = self.consumed_points

        for unit in self.arg_queue:
            cost = int(unit["point_cost"])
            event_used = self.event_consumed.get(unit["event_id"], 0)
            reason = None
            if self.consumed_points + cost > self.global_limit:
                reason = "global_arg_score_budget"
            elif event_used + cost > self.event_quota:
                reason = "per_event_arg_quota"
            if reason:
                blocking_decision = {
                    "unit_key": unit["unit_key"],
                    "event_id": unit["event_id"],
                    "role": unit["role"],
                    "code": "E_ARG_BUDGET_EXCEEDED",
                    "business_code": "E_ARG_BUDGET_EXCEEDED",
                    "reason": reason,
                    "point_cost": cost,
                    "global_limit": self.global_limit,
                    "global_consumed_before": self.consumed_points,
                    "event_quota": self.event_quota,
                    "event_consumed_before": event_used,
                }
                self.budget_decisions.append(blocking_decision)
                break

            admitted.append(unit)
            self.consumed_points += cost
            self.event_consumed[unit["event_id"]] = event_used + cost
            self.budget_decisions.append(
                {
                    "unit_key": unit["unit_key"],
                    "event_id": unit["event_id"],
                    "role": unit["role"],
                    "status": "admitted",
                    "point_cost": cost,
                    "global_consumed_after": self.consumed_points,
                }
            )

        admitted_keys = {unit["unit_key"] for unit in admitted}
        results: dict[str, Any] = {}
        failures: dict[str, str] = {}

        def wrapped(unit: dict[str, Any]) -> dict[str, Any]:
            self._enter()
            try:
                return score_fn(unit)
            finally:
                self._exit()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(wrapped, unit): unit for unit in admitted}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    results[unit["unit_key"]] = future.result()
                except Exception as exc:
                    failures[unit["unit_key"]] = f"{type(exc).__name__}:{exc}"

        self.arg_queue = [
            unit for unit in self.arg_queue if unit["unit_key"] not in admitted_keys
        ]
        pending = [
            {
                "unit_key": unit["unit_key"],
                "event_id": unit["event_id"],
                "role": unit["role"],
                "code": "E_ARG_PENDING",
                "last_code": (
                    "E_ARG_BUDGET_EXCEEDED"
                    if blocking_decision
                    and unit["unit_key"] == blocking_decision["unit_key"]
                    else None
                ),
            }
            for unit in self.arg_queue
        ]
        report = {
            "stage": "argument_scoring",
            "admitted_unit_keys": [unit["unit_key"] for unit in admitted],
            "admitted_count": len(admitted),
            "point_cost_consumed_this_batch": self.consumed_points - consumed_before,
            "global_consumed_points": self.consumed_points,
            "global_arg_score_points": self.global_limit,
            "per_event_consumed": dict(sorted(self.event_consumed.items())),
            "failed_unit_keys": sorted(failures),
            "failures": failures,
            "budget_exceeded": blocking_decision,
            "pending_units": pending,
            "pending_count": len(pending),
            "peak_concurrency": self.peak_concurrency,
            "max_workers_configured": self.max_workers,
            "concurrency_limit_respected": self.peak_concurrency <= self.max_workers,
        }
        self.batch_reports.append(report)
        return report

    def budget_summary(self) -> dict[str, Any]:
        latest_arg_report = next(
            (report for report in reversed(self.batch_reports) if report["stage"] == "argument_scoring"),
            None,
        )
        return {
            "max_workers": self.max_workers,
            "peak_concurrency": self.peak_concurrency,
            "concurrency_limit_respected": self.peak_concurrency <= self.max_workers,
            "global_limit": self.global_limit,
            "consumed_points": self.consumed_points,
            "remaining_points": max(0, self.global_limit - self.consumed_points),
            "per_event_quota": self.event_quota,
            "per_event_consumed": dict(sorted(self.event_consumed.items())),
            "trigger_batch_limit": self.trigger_batch_limit,
            "trigger_queue_remaining": list(self.trigger_queue),
            "pending_unit_count": len(self.arg_queue),
            "pending_units": [self.unit_key(unit) for unit in self.arg_queue],
            "latest_argument_batch": latest_arg_report,
            "decisions": list(self.budget_decisions),
        }
