"""Gateway orchestration: execute batches, merge receipts, commit the ledger."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .batching import Batch, build_batch_context
from .inference import output_summary_from_values
from .ledger import Ledger
from .merger import (check_batch_versions, merge_receipts,
                     propagate_invalidation)
from .receipt import encode_receipt, normalize_raw
from .scheduler import Scheduler, now_ms
from .sharding import Hit
from .spec import Spec


@dataclass
class RunResult:
    committed_batches: list[str]
    adopted_by_batch: dict[str, list[dict]]
    invalidated_requests: set[str]
    ledger_summary: str


class GatewayEngine:
    def __init__(self, spec: Spec, ledger: Ledger, pool,
                 max_global: int = 4, per_worker_quota: int = 2,
                 emit_format: int = 3):
        self.spec = spec
        self.ledger = ledger
        self.pool = pool
        self.scheduler = Scheduler(max_global, per_worker_quota)
        self.emit_format = emit_format
        self.requests_by_id = {q.request_id: q for q in spec.requests}
        self.known_request_ids = {q.request_id for q in spec.requests}

    def _execute_batch(self, batch: Batch, hits: dict[str, Hit],
                       invalid_worker_versions: set[tuple[str, int]],
                       timing: dict[str, tuple[int, int]]) -> None:
        if self.ledger.is_committed(batch.batch_id):
            return
        context = build_batch_context(self.spec, batch, self.requests_by_id)
        backend = self.spec.backends[batch.backend_id]
        raws: list[dict] = []
        for request_id in batch.request_ids:
            request = self.requests_by_id[request_id]
            response = self.pool.infer(
                batch.backend_id, request_id, request.vector,
                self.spec.model_dim, backend.hidden_dim,
                batch.param_summary)
            expected = context.output_summary_by_request[request_id]
            if response["summary"] != expected:
                raise RuntimeError(
                    f"backend output mismatch for {request_id}")
            elapsed = max(0, int(response.get("elapsed_ms", 1)))
            raw = encode_receipt(
                self.emit_format, context, request_id,
                response["summary"], elapsed, superseded_ref=None)
            raws.append(raw)

        normalized = [normalize_raw(raw, context, self.known_request_ids)
                      for raw in raws]
        check_batch_versions(batch, normalized)
        decision = merge_receipts(
            list(enumerate(normalized)), self.spec.merge_rule)

        adopted_record = {
            "batch_id": batch.batch_id,
            "request_id": decision.adopted.request_id,
            "source_format": decision.adopted.format_version,
            "content_summary": decision.adopted.content_summary(),
            "normalized": decision.adopted.to_dict(),
        }
        adopted_records = [{
            "batch_id": batch.batch_id,
            "request_id": norm.request_id,
            "source_format": norm.format_version,
            "content_summary": norm.content_summary(),
            "normalized": norm.to_dict(),
        } for norm in normalized]
        rejected_records = [{
            "batch_id": batch.batch_id,
            "request_id": norm.request_id,
            "source_format": norm.format_version,
            "param_version_seq": norm.param_version_seq,
            "reason": reason,
            "normalized": norm.to_dict(),
        } for norm, reason in decision.rejected]

        propagation = propagate_invalidation(
            (batch,),
            {batch.batch_id: normalized},
            invalid_worker_versions,
        )
        wait_start, wait_end = timing.get(batch.batch_id, (now_ms(), now_ms()))
        self.ledger.commit_batch(
            batch=batch, hits=hits, adopted=adopted_records,
            rejected=rejected_records, invalidations=propagation,
            wait_start_ms=wait_start, wait_end_ms=wait_end)

    def run(self, planned: tuple[Batch, ...], hits: dict[str, Hit],
            invalid_worker_versions: set[tuple[str, int]] | None = None
            ) -> RunResult:
        invalid_worker_versions = invalid_worker_versions or set()
        timing: dict[str, tuple[int, int]] = {}

        def worker(batch: Batch) -> None:
            start = now_ms()
            timing[batch.batch_id] = (start, now_ms())
            self._execute_batch(batch, hits, invalid_worker_versions, timing)

        runs = self.scheduler.run(planned, worker)
        for run in runs:
            timing[run.batch.batch_id] = (run.wait_start_ms, run.wait_end_ms)
        committed = self.ledger.committed_batch_ids()
        adopted = {
            bid: self.ledger.adopted_payloads(bid)
            for bid in sorted(committed)
        }
        invalidated = {
            row["request_id"] for row in
            self.ledger.conn.execute("SELECT request_id FROM invalidation")
        }
        return RunResult(
            committed_batches=sorted(committed),
            adopted_by_batch=adopted,
            invalidated_requests=invalidated,
            ledger_summary=self.ledger.ledger_content_summary(),
        )
