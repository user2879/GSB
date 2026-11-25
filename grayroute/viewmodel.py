"""Headless view-model: fast conclusion vs authoritative ledger recompute.

The fast path derives conclusions directly from in-memory planning data; the
authoritative path re-reads the committed SQLite ledger.  Both emit canonical
bytes and the engine asserts they are byte-identical, satisfying the
interactive-latency consistency requirement without requiring a display.
"""
from __future__ import annotations

import time
from .canonical import canonical_bytes
from .ledger import Ledger


class DashboardModel:
    def __init__(self, spec, hits, batches, run_result, invalid_versions):
        self.spec = spec
        self.hits = hits
        self.batches = batches
        self.run_result = run_result
        self.invalid_versions = invalid_versions

    def fast_conclusion(self) -> dict:
        """Deterministic quick conclusion built from in-memory structures."""
        invalid_requests = set()
        batch_views = []
        ordered_batches = sorted(
            self.batches,
            key=lambda b: (b.worker_id, b.param_version_seq, b.batch_index))
        for batch in ordered_batches:
            invalid = (batch.worker_id, batch.param_version_seq) \
                in self.invalid_versions
            if invalid:
                invalid_requests.update(batch.request_ids)
            adopted = self.run_result.adopted_by_batch.get(batch.batch_id, [])
            batch_views.append({
                "batch_id": batch.batch_id,
                "worker_id": batch.worker_id,
                "param_version_seq": batch.param_version_seq,
                "status": "invalid" if invalid else "valid",
                "request_ids": list(batch.request_ids),
                "receipt_summaries": sorted(
                    norm["content_summary"] for norm in adopted),
            })
        return {
            "total_batches": len(self.batches),
            "invalid_batches": sum(1 for b in batch_views
                                   if b["status"] == "invalid"),
            "invalid_requests": sorted(invalid_requests),
            "batches": batch_views,
        }

    @staticmethod
    def authoritative_conclusion(ledger: Ledger, planned_ids: set[str],
                                 invalid_versions: set[tuple[str, int]]) -> dict:
        invalid_rows = ledger.conn.execute(
            "SELECT DISTINCT worker_id, param_version_seq, batch_id, request_id "
            "FROM invalidation ORDER BY worker_id, param_version_seq, "
            "batch_id, request_id").fetchall()
        invalid_requests = sorted({row["request_id"] for row in invalid_rows})
        invalid_batches = {(row["worker_id"], row["param_version_seq"])
                           for row in invalid_rows}
        batch_views = []
        rows = ledger.conn.execute(
            "SELECT * FROM batch_attribution ORDER BY worker_id, "
            "param_version_seq, batch_index").fetchall()
        for row in rows:
            adopted = ledger.conn.execute(
                "SELECT content_summary FROM adopted_receipt WHERE batch_id=? "
                "ORDER BY request_id", (row["batch_id"],)).fetchall()
            invalid = (row["worker_id"], row["param_version_seq"]) \
                in invalid_batches
            batch_views.append({
                "batch_id": row["batch_id"],
                "worker_id": row["worker_id"],
                "param_version_seq": row["param_version_seq"],
                "status": "invalid" if invalid else "valid",
                "request_ids": sorted(__import__("json").loads(row["request_ids"])),
                "receipt_summaries": sorted(r["content_summary"] for r in adopted),
            })
        return {
            "total_batches": len(rows),
            "invalid_batches": sum(1 for b in batch_views
                                   if b["status"] == "invalid"),
            "invalid_requests": invalid_requests,
            "batches": batch_views,
        }

    def consistency_check(self, ledger: Ledger, budget_ms: int = 5000) -> dict:
        start = time.monotonic_ns() // 1_000_000
        fast = canonical_bytes(self.fast_conclusion())
        authoritative = canonical_bytes(self.authoritative_conclusion(
            ledger, {b.batch_id for b in self.batches},
            self.invalid_versions))
        elapsed = time.monotonic_ns() // 1_000_000 - start
        if fast != authoritative:
            raise AssertionError(
                "fast conclusion and authoritative ledger differ byte-wise")
        return {
            "byte_identical": True,
            "elapsed_ms": elapsed,
            "within_budget": elapsed <= budget_ms,
            "budget_ms": budget_ms,
        }
