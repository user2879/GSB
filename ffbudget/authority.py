"""Stage 2: idempotent backfill adjudication and authoritative revision
selection.

For every slot group the receipts are ranked deterministically by
(rev_seq desc, effective_ms desc, batch_seq desc). A full tie on those three
fields is broken by the receipt content digest (lexicographic) and flagged as
a tie resolution. Redelivered batches whose receipts lose the deterministic
ranking are refused and never re-enter the scoring point ledger.
"""
from __future__ import annotations

from collections import Counter

from . import errors
from .fingerprint import content_fingerprint
from .models import AuthorityRecord, Receipt, RejectionRecord


def _adjudication_key(receipt: Receipt):
    return (-receipt.rev_seq, -receipt.effective_ms, -receipt.batch_seq,
            receipt.digest)


def adjudicate(receipts: list[Receipt], slots: dict):
    # Preserve repeated identical receipts (same batch_seq + content digest)
    # so exact redeliveries stay observable, while adjudication is driven by
    # the unique receipt set.
    delivery_counts = Counter((r.batch_seq, r.digest) for r in receipts)
    unique_receipts = list({r.digest: r for r in receipts}.values())

    groups: dict[str, list[Receipt]] = {}
    for receipt in unique_receipts:
        groups.setdefault(receipt.group_key, []).append(receipt)

    authorities: dict[str, AuthorityRecord] = {}
    rejections: list[RejectionRecord] = []
    backfill_ledger: list[dict] = []
    duplicate_delivery: list[dict] = []

    for group_key in sorted(groups):
        members = groups[group_key]
        slot = slots[members[0].slot_key]
        ordered = sorted(members, key=_adjudication_key)
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else None
        tie_resolved = (
            runner_up is not None
            and (winner.rev_seq, winner.effective_ms, winner.batch_seq)
            == (runner_up.rev_seq, runner_up.effective_ms,
                runner_up.batch_seq)
        )
        history = ordered[:slot.retention]
        pruned_receipts = ordered[slot.retention:]
        authority_fp = content_fingerprint([
            winner.batch_seq, winner.rev_seq, winner.slot_key,
            winner.effective_ms, list(winner.vector), winner.item_id,
            winner.digest])
        record = AuthorityRecord(
            group_key=group_key, slot_key=slot.key, item_id=winner.item_id,
            receipt=winner, authority_fp=authority_fp,
            tie_resolved=tie_resolved,
            history=[r.digest for r in history],
            pruned=[{"digest": r.digest, "batch_seq": r.batch_seq,
                     "rev_seq": r.rev_seq,
                     "code": errors.E_REVISION_PRUNED}
                    for r in pruned_receipts])
        authorities[group_key] = record

        for receipt in ordered[1:]:
            reason = _refusal_reason(receipt, winner)
            rejections.append(RejectionRecord(
                digest=receipt.digest, batch_seq=receipt.batch_seq,
                rev_seq=receipt.rev_seq, slot_key=receipt.slot_key,
                group_key=group_key, reason=reason,
                conflicting_digest=winner.digest,
                conflicting_rev_seq=winner.rev_seq,
                conflicting_batch_seq=winner.batch_seq))

        for receipt in history:
            backfill_ledger.append({
                "batch_seq": receipt.batch_seq, "slot_key": receipt.slot_key,
                "group_key": group_key, "rev_seq": receipt.rev_seq,
                "digest": receipt.digest, "points": receipt.points,
                "counted": True})
        for receipt in pruned_receipts:
            backfill_ledger.append({
                "batch_seq": receipt.batch_seq, "slot_key": receipt.slot_key,
                "group_key": group_key, "rev_seq": receipt.rev_seq,
                "digest": receipt.digest, "points": receipt.points,
                "counted": False, "code": errors.E_REVISION_PRUNED})

    for (batch_seq, digest), copies in sorted(delivery_counts.items()):
        if copies > 1:
            winner = next(r for r in unique_receipts if r.digest == digest)
            for _ in range(copies - 1):
                duplicate_delivery.append({
                    "code": errors.E_DUPLICATE_BATCH_DELIVERY,
                    "batch_seq": batch_seq, "slot_key": winner.slot_key,
                    "group_key": winner.group_key,
                    "rev_seq": winner.rev_seq, "digest": digest,
                    "points": winner.points, "counted": False,
                    "conflicting_digest": digest})

    rejections.sort(key=lambda item: (item.group_key, item.batch_seq,
                                      item.rev_seq, item.digest))
    backfill_ledger.sort(key=lambda item: (item["batch_seq"], item["group_key"],
                                           item["rev_seq"], item["digest"]))
    return authorities, rejections, backfill_ledger, duplicate_delivery


def _refusal_reason(receipt: Receipt, winner: Receipt) -> str:
    if receipt.rev_seq < winner.rev_seq:
        return errors.E_REV_SEQ_LOWER
    if receipt.effective_ms < winner.effective_ms:
        return errors.E_EFFECTIVE_EARLIER
    if receipt.batch_seq < winner.batch_seq:
        return errors.E_BATCH_SEQ_LOWER
    return errors.E_TIE_DIGEST
