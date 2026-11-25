from __future__ import annotations

import json
from typing import Any

from .compute import forward_batch
from .errors import ReplayError
from .ledger import Ledger
from .models import Batch, RequestSpec, WorkerSpec


def _request_from_row(row: Any) -> RequestSpec:
    data = json.loads(row["request_json"])
    return RequestSpec(
        request_id=data["request_id"],
        worker_id=data["worker_id"],
        arrival_seq=int(data["arrival_seq"]),
        vector=tuple(int(value) for value in data["vector"]),
    )


def _mismatch(batch_id: str, field_name: str, expected: Any, actual: Any, index: int | None = None) -> ReplayError:
    return ReplayError(
        "恢复后重放发现首个不一致批次",
        batch_id=batch_id,
        field=field_name,
        index=index,
        expected=expected,
        actual=actual,
        code_name="E_REPLAY_MISMATCH",
    )


def replay_ledger(ledger: Ledger) -> list[dict[str, Any]]:
    """Recompute every committed receipt from frozen request/worker snapshots."""

    ledger.verify_digest()
    request_rows = ledger.conn.execute("SELECT * FROM requests").fetchall()
    request_lookup = {row["request_id"]: _request_from_row(row) for row in request_rows}
    batch_rows = ledger.conn.execute(
        "SELECT * FROM batches WHERE status='committed' ORDER BY worker_id,batch_index,batch_id"
    ).fetchall()
    reports: list[dict[str, Any]] = []

    for batch_row in batch_rows:
        batch_id = batch_row["batch_id"]
        arrival_seqs = tuple(json.loads(batch_row["arrival_seqs_json"]))
        request_ids = tuple(json.loads(batch_row["request_ids_json"]))
        ranks = {request_id: rank for rank, request_id in enumerate(request_ids)}
        batch = Batch(
            batch_id=batch_id,
            worker_id=batch_row["worker_id"],
            batch_index=batch_row["batch_index"],
            arrival_seqs=arrival_seqs,
            request_ids=request_ids,
            stable_ranks=ranks,
        )

        expected_membership = batch_row["membership_digest"]
        from .canonical import canonical_digest

        actual_membership = canonical_digest(batch.material())
        if actual_membership != expected_membership:
            raise _mismatch(batch_id, "membership_digest", expected_membership, actual_membership)

        assignment_rows = ledger.conn.execute(
            "SELECT * FROM assignments WHERE batch_id=? ORDER BY rank",
            (batch_id,),
        ).fetchall()
        assigned_ids = tuple(row["request_id"] for row in assignment_rows)
        if assigned_ids != request_ids:
            raise _mismatch(batch_id, "assignments.request_ids", list(request_ids), list(assigned_ids))
        for rank, row in enumerate(assignment_rows):
            if int(row["rank"]) != rank:
                raise _mismatch(batch_id, "assignments.rank", rank, row["rank"], index=rank)
            if int(row["arrival_seq"]) != arrival_seqs[rank]:
                raise _mismatch(batch_id, "assignments.arrival_seq", arrival_seqs[rank], row["arrival_seq"], index=rank)
            if row["worker_id"] != batch.worker_id:
                raise _mismatch(batch_id, "assignments.worker_id", batch.worker_id, row["worker_id"], index=rank)

        receipt_rows = ledger.conn.execute(
            "SELECT * FROM receipts WHERE batch_id=? ORDER BY revision",
            (batch_id,),
        ).fetchall()
        if not receipt_rows:
            raise _mismatch(batch_id, "receipts", "at_least_one", 0)

        latest_output: str | None = None
        for receipt_row in receipt_rows:
            normalized = json.loads(receipt_row["normalized_json"])
            snapshot_data = normalized.get("worker_snapshot")
            if snapshot_data is None and batch_row["worker_snapshot_json"]:
                snapshot_data = json.loads(batch_row["worker_snapshot_json"])
            if snapshot_data is None:
                raise _mismatch(batch_id, "worker_snapshot", "frozen_snapshot", None)
            worker = WorkerSpec(**snapshot_data)
            missing = [request_id for request_id in request_ids if request_id not in request_lookup]
            if missing:
                raise _mismatch(batch_id, "requests", sorted(request_lookup), missing[0])
            result = forward_batch(worker, batch, request_lookup)
            if result["output_digest"] != receipt_row["output_digest"]:
                raise _mismatch(
                    batch_id,
                    "receipts.output_digest",
                    receipt_row["output_digest"],
                    result["output_digest"],
                    index=int(receipt_row["revision"]),
                )
            if normalized.get("output_digest") and normalized["output_digest"] != receipt_row["output_digest"]:
                raise _mismatch(
                    batch_id,
                    "normalized.output_digest",
                    receipt_row["output_digest"],
                    normalized["output_digest"],
                )
            latest_output = result["output_digest"]

        if latest_output != batch_row["output_digest"]:
            raise _mismatch(batch_id, "batches.output_digest", batch_row["output_digest"], latest_output)
        first_receipt = receipt_rows[0]
        if batch_row["frozen_output_digest"] != first_receipt["output_digest"]:
            raise _mismatch(
                batch_id,
                "batches.frozen_output_digest",
                first_receipt["output_digest"],
                batch_row["frozen_output_digest"],
            )
        reports.append(
            {
                "batch_id": batch_id,
                "worker_id": batch.worker_id,
                "receipt_revisions": len(receipt_rows),
                "output_digest": latest_output,
                "frozen_output_digest": batch_row["frozen_output_digest"],
                "assignment_count": len(assignment_rows),
            }
        )
    return reports


def compare_expected_batch(batch_id: str, field_name: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        raise _mismatch(batch_id, field_name, expected, actual)
