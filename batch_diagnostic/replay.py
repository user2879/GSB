"""Post-recovery consistency replay against committed ledger content."""

from __future__ import annotations

from .errors import ReplayError
from .inference import build_parameters, forward_batch
from .planning import build_plan
from .receipt import NormalizedReceipt, validate_chain
from .spec import SystemSpec

COMPARED_BATCH_FIELDS = (
    "worker_id",
    "batch_index",
    "output_digest",
    "param_version",
    "param_digest",
    "intermediate_dim",
)


def _receipt_from_dict(raw: dict) -> NormalizedReceipt:
    return NormalizedReceipt(
        source_version=raw["source_version"],
        batch_id=raw["batch_id"],
        request_id=raw["request_id"],
        output_digest=raw["output_digest"],
        param_version=raw["param_version"],
        elapsed_ms=raw["elapsed_ms"],
        worker_id=raw["worker_id"],
        rank=raw["rank"],
        duplicate_ref=raw["duplicate_ref"],
        param_digest=raw["param_digest"],
        filled_fields=tuple(),
    )


def replay_ledger(spec: SystemSpec, ledger) -> dict:
    """Regenerate every committed batch and compare field by field.

    Raises ``E_REPLAY`` locating the first diverging batch, request and field.
    """

    plan = build_plan(spec)
    receipts_by_batch = {
        batch.batch_id: [_receipt_from_dict(item) for item in ledger.load_receipts(batch.batch_id)]
        for batch in plan.batches
    }
    for receipt_list in receipts_by_batch.values():
        validate_chain(receipt_list)

    for batch in sorted(plan.batches, key=lambda item: (item.worker_id, item.batch_index)):
        stored = ledger.load_batch(batch.batch_id)
        if stored is None:
            continue
        worker = spec.worker_by_id()[batch.worker_id]
        result = forward_batch(spec, worker, batch, build_parameters(worker))
        actual = {
            "worker_id": result.worker_id,
            "batch_index": batch.batch_index,
            "output_digest": result.output_digest,
            "param_version": result.param_version,
            "param_digest": result.param_digest,
            "intermediate_dim": result.intermediate_dim,
        }
        stored_view = {
            "worker_id": stored["worker_id"],
            "batch_index": stored["batch_index"],
            "output_digest": stored["output_digest"],
            "param_version": stored["param_version"],
            "param_digest": stored["param_digest"],
            "intermediate_dim": stored["intermediate_dim"],
        }
        for field_name in COMPARED_BATCH_FIELDS:
            if stored_view[field_name] != actual[field_name]:
                raise ReplayError(
                    "replay diverges from ledger",
                    batch_id=batch.batch_id,
                    request_id=None,
                    field=field_name,
                    expected=stored_view[field_name],
                    actual=actual[field_name],
                )
        stored_receipts = {
            item.request_id: item for item in receipts_by_batch[batch.batch_id]
        }
        for request_id in sorted(batch.request_ids):
            if request_id not in stored_receipts:
                raise ReplayError(
                    "replay diverges from ledger",
                    batch_id=batch.batch_id,
                    request_id=request_id,
                    field="<missing receipt>",
                )
            stored_receipt = stored_receipts[request_id]
            expected_digest = result.per_request[request_id]
            if stored_receipt.output_digest != expected_digest:
                raise ReplayError(
                    "replay diverges from ledger",
                    batch_id=batch.batch_id,
                    request_id=request_id,
                    field="output_digest",
                    expected=stored_receipt.output_digest,
                    actual=expected_digest,
                )
            if stored_receipt.param_digest != result.param_digest:
                raise ReplayError(
                    "replay diverges from ledger",
                    batch_id=batch.batch_id,
                    request_id=request_id,
                    field="param_digest",
                    expected=stored_receipt.param_digest,
                    actual=result.param_digest,
                )
    return {
        "replayed_batches": list(ledger.committed_batch_ids()),
        "status": "match",
    }
