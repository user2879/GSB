"""Recovery and replay verification after forced termination.

Recovery: only fully committed batches are visible (each batch commits in a
single transaction). Planned batches missing from the ledger are UNKNOWN and
must be re-run. Replay re-executes every committed batch from the recorded
plan and compares the ledger row by row; the first divergence is reported as
E_REPLAY with the request sequence number and field.
"""

from __future__ import annotations

from typing import Dict, List

from .batching import build_plan, select_strategy
from .errors import ReplayError
from .fingerprint import fingerprints
from .forward import batch_digest, forward_scores, summarize
from .ledger import Ledger
from .models import Declaration
from .routing import route_requests

UNKNOWN = "UNKNOWN"


def recovery_scan(decl: Declaration, ledger: Ledger, version: int) -> dict:
    """Classify every planned batch as committed or UNKNOWN (needs re-run)."""
    routed = route_requests(decl)
    decision = select_strategy(decl, routed)
    batches = build_plan(decl, routed, decision.strategy)
    committed = set(ledger.committed_batch_ids(version))
    return {
        "committed": sorted(committed),
        "unknown": [b.batch_id for b in batches if b.batch_id not in committed],
        "cursor": ledger.cursor_info(version),
    }


def replay_verify(decl: Declaration, ledger: Ledger, version: int) -> dict:
    """Re-execute committed batches and diff against the ledger row by row.

    Returns a verification report; raises ReplayError (E_REPLAY) on the first
    inconsistent (request_seq, field).
    """
    fps = fingerprints(decl)
    params = {b.model_id: b.params for b in decl.backends}
    routed = route_requests(decl)
    routed_by_seq = {r.request.seq: r for r in routed}
    decision = select_strategy(decl, routed)
    plan = {b.batch_id: b for b in build_plan(decl, routed, decision.strategy)}

    rows = ledger.ledger_rows(version)
    compared = 0
    for row in rows:  # ordered by request_seq
        seq = row["request_seq"]
        batch = plan.get(row["batch_id"])
        if batch is None or seq not in batch.request_seqs:
            raise ReplayError(seq, "batch_id", "planned membership",
                              row["batch_id"])
        rr = routed_by_seq[seq]
        if rr.rule_id != row["rule_id"]:
            raise ReplayError(seq, "rule_id", rr.rule_id, row["rule_id"])
        if rr.backend_id != row["backend_id"]:
            raise ReplayError(seq, "backend_id", rr.backend_id,
                              row["backend_id"])
        if rr.variant != row["variant"]:
            raise ReplayError(seq, "variant", rr.variant, row["variant"])
        if fps[rr.backend_id] != row["fingerprint"]:
            raise ReplayError(seq, "fingerprint", fps[rr.backend_id],
                              row["fingerprint"])
        # Recompute this request's output summary deterministically.
        inputs = tuple(decl.requests[s].inputs for s in batch.request_seqs)
        scores = forward_scores(params[batch.backend_id], inputs)
        summaries = summarize(scores)
        offset = batch.request_seqs.index(seq)
        if summaries[offset] != row["output_summary"]:
            raise ReplayError(seq, "output_summary", summaries[offset],
                              row["output_summary"])
        compared += 1

    # Verify batch-level digests too.
    for brow in ledger.batch_rows(version):
        batch = plan[brow["batch_id"]]
        inputs = tuple(decl.requests[s].inputs for s in batch.request_seqs)
        scores = forward_scores(params[batch.backend_id], inputs)
        digest = batch_digest(summarize(scores))
        if digest != brow["output_digest"]:
            first_seq = batch.request_seqs[0]
            raise ReplayError(first_seq, "output_digest", digest,
                              brow["output_digest"])
    return {"compared_requests": compared,
            "compared_batches": len(ledger.batch_rows(version)),
            "status": "OK"}
