"""Post-crash recovery and replay verification.

After a forced termination the only trustworthy state is what was actually
COMMIT-ed in SQLite.  Planned batches absent from the ledger are marked
``unknown`` and must be rerun; no exit report is fabricated.  Replay then
recomputes each unfinished batch and compares it field-by-field against the
ledger, raising :class:`ReplayError` (``E_REPLAY``) at the first mismatch.
"""
from __future__ import annotations

from .batching import Batch, build_batch_context
from .canonical import canonical_bytes
from .errors import ReplayError
from .ledger import Ledger
from .receipt import normalize_raw


def classify_batches(ledger: Ledger,
                     planned: tuple[Batch, ...]) -> tuple[list[Batch], list[Batch]]:
    committed, unknown = [], []
    for batch in planned:
        if ledger.is_committed(batch.batch_id):
            committed.append(batch)
        else:
            unknown.append(batch)
            ledger.mark_unknown(
                batch.batch_id,
                "not committed before forced termination; rerun required")
    return committed, unknown


def replay_batch(spec, ledger: Ledger, batch: Batch,
                 requests_by_id: dict, known_request_ids: set[str],
                 receipt_rows: list[dict]) -> None:
    """Recompute one batch and compare against ledger adopted receipts."""
    context = build_batch_context(spec, batch, requests_by_id)
    recomputed = {}
    for row in receipt_rows:
        normalized = normalize_raw(row, context, known_request_ids)
        recomputed[normalized.request_id] = normalized.to_dict()

    stored = {payload["request_id"]: payload
              for payload in ledger.adopted_payloads(batch.batch_id)}
    ledger_only = {"content_summary"}
    all_ids = sorted(set(recomputed) | set(stored))
    for request_id in all_ids:
        if request_id not in recomputed:
            raise ReplayError(
                "replay missing request recomputation",
                object_id=batch.batch_id, field="request_id",
                request_id=request_id)
        if request_id not in stored:
            raise ReplayError(
                "ledger missing request previously committed",
                object_id=batch.batch_id, field="request_id",
                request_id=request_id)
        expected = recomputed[request_id]
        actual = {k: v for k, v in stored[request_id].items()
                  if k not in ledger_only}
        fields = sorted(set(expected) | set(actual))
        for field_name in fields:
            if expected.get(field_name) != actual.get(field_name):
                raise ReplayError(
                    "replay mismatch against ledger",
                    object_id=batch.batch_id, field=field_name,
                    request_id=request_id,
                    expected=expected.get(field_name),
                    actual=actual.get(field_name))
        if canonical_bytes(expected) != canonical_bytes(actual):
            raise ReplayError(
                "canonical representation differs despite equal fields",
                object_id=batch.batch_id, field="canonical",
                request_id=request_id)


def replay_unfinished(spec, ledger: Ledger, planned: tuple[Batch, ...],
                      receipts_by_batch: dict[str, list[dict]]) -> dict:
    """Verify every already-committed batch replays byte-identically."""
    requests_by_id = {q.request_id: q for q in spec.requests}
    known = {q.request_id for q in spec.requests}
    report = {"verified": [], "unknown": []}
    for batch in planned:
        if not ledger.is_committed(batch.batch_id):
            report["unknown"].append(batch.batch_id)
            continue
        rows = receipts_by_batch.get(batch.batch_id)
        if rows is None:
            raise ReplayError(
                "no replay receipts available for committed batch",
                object_id=batch.batch_id, field="receipts")
        replay_batch(spec, ledger, batch, requests_by_id, known, rows)
        report["verified"].append(batch.batch_id)
    return report
