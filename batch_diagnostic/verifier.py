"""Independent verification using only the ledger and raw declarations.

The verifier never imports engine/network/scheduler state: it reloads the three
JSON declarations, rebuilds the deterministic plan and parameter matrices,
recomputes forward outputs and invalidation, and reports the first offending
batch with ``E_VERIFY``.
"""

from __future__ import annotations

from .canonical import canonical_json
from .errors import VerifyError
from .inference import build_parameters, forward_batch
from .planning import build_plan
from .receipt import validate_chain
from .spec import load_spec


def verify_ledger(requests_path: str, workers_path: str, budget_path: str, ledger) -> dict:
    spec = load_spec(requests_path, workers_path, budget_path)
    plan = build_plan(spec)
    committed = set(ledger.committed_batch_ids())

    planned_ids = {batch.batch_id for batch in plan.batches}
    for batch_id in sorted(committed):
        if batch_id not in planned_ids:
            raise VerifyError(
                "ledger contains batch absent from deterministic plan",
                batch_id=batch_id,
                request_id=None,
                field="batch_id",
            )

    for batch in sorted(plan.batches, key=lambda item: (item.worker_id, item.batch_index)):
        if batch.batch_id not in committed:
            continue
        worker = spec.worker_by_id()[batch.worker_id]
        result = forward_batch(spec, worker, batch, build_parameters(worker))
        stored = ledger.load_batch(batch.batch_id)
        if stored["output_digest"] != result.output_digest:
            raise VerifyError(
                "independent output digest mismatch",
                batch_id=batch.batch_id,
                request_id=None,
                field="output_digest",
            )
        receipt_rows = ledger.load_receipts(batch.batch_id)
        from .replay import _receipt_from_dict

        receipts = [_receipt_from_dict(row) for row in receipt_rows]
        validate_chain(receipts)
        receipts_by_id = {item.request_id: item for item in receipts}
        for request_id in sorted(batch.request_ids):
            receipt = receipts_by_id.get(request_id)
            if receipt is None:
                raise VerifyError(
                    "missing request attribution receipt",
                    batch_id=batch.batch_id,
                    request_id=request_id,
                    field="request_id",
                )
            if receipt.output_digest != result.per_request[request_id]:
                raise VerifyError(
                    "independent per-request digest mismatch",
                    batch_id=batch.batch_id,
                    request_id=request_id,
                    field="output_digest",
                )
        if canonical_json(stored["param_digest"]) != canonical_json(result.param_digest):
            raise VerifyError(
                "parameter digest mismatch",
                batch_id=batch.batch_id,
                request_id=None,
                field="param_digest",
            )
    return {
        "verified_batches": sorted(committed),
        "worker_diff_available": bool(spec.workers),
        "status": "match",
    }
