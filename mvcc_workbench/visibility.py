from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .canonical import digest_json
from .errors import TaskTimedOut
from .model import Dataset, Query, ReadPoint, RecordVersion
from .value_ops import evaluate_predicate


@dataclass(frozen=True)
class VisibilityDecision:
    query_id: str
    read_point_id: str
    row_key: str
    snapshot_seq: int
    reader_txn_id: str
    visible_version_no: int | None
    visible_txn_id: str | None
    visible_value_digest: str | None
    visible_value_summary: dict[str, Any] | None
    outcome: str
    predicate_matched: bool | None
    predicate_detail: dict[str, Any] | None
    reason_chain: tuple[dict[str, Any], ...]
    reason_digest: str
    cost_units: int
    steps_examined: int
    predicate_columns: int
    visibility_policy: dict[str, Any]

    def as_dict(self, *, include_reason_chain: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "query_id": self.query_id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "snapshot_seq": self.snapshot_seq,
            "reader_txn_id": self.reader_txn_id,
            "visible_version_no": self.visible_version_no,
            "visible_txn_id": self.visible_txn_id,
            "visible_value_digest": self.visible_value_digest,
            "visible_value_summary": self.visible_value_summary,
            "outcome": self.outcome,
            "predicate_matched": self.predicate_matched,
            "predicate_detail": self.predicate_detail,
            "reason_digest": self.reason_digest,
            "cost_units": self.cost_units,
            "steps_examined": self.steps_examined,
            "predicate_columns": self.predicate_columns,
            "visibility_policy": self.visibility_policy,
        }
        if include_reason_chain:
            result["reason_chain"] = list(self.reason_chain)
        return result


def _check_deadline(deadline_ns: int | None) -> None:
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TaskTimedOut("visibility task exceeded its real monotonic deadline")


def _visible_step(
    record: RecordVersion,
    writer: Any,
    code: str,
    message: str,
    snapshot_seq: int,
) -> dict[str, Any]:
    return {
        "step": "VISIBLE",
        "version_no": record.version_no,
        "txn_id": record.txn_id,
        "code": code,
        "message": message,
        "writer_outcome": writer.outcome,
        "commit_seq": writer.commit_seq,
        "snapshot_seq": snapshot_seq,
    }


def _skip_step(
    record: RecordVersion,
    writer: Any,
    code: str,
    message: str,
    snapshot_seq: int,
) -> dict[str, Any]:
    return {
        "step": "SKIP",
        "version_no": record.version_no,
        "txn_id": record.txn_id,
        "code": code,
        "message": message,
        "writer_outcome": writer.outcome,
        "commit_seq": writer.commit_seq,
        "snapshot_seq": snapshot_seq,
    }


def select_visible_version(
    dataset: Dataset,
    query: Query,
    deadline_ns: int | None = None,
) -> tuple[RecordVersion | None, list[dict[str, Any]], ReadPoint]:
    read_point = dataset.read_points_by_id[query.read_point_id]
    reader_txn_id = read_point.reader_txn_id
    snapshot_seq = read_point.snapshot_seq
    reason_chain: list[dict[str, Any]] = []
    chain = dataset.chain(query.row_key)

    for record in reversed(chain):
        _check_deadline(deadline_ns)
        writer = dataset.txns_by_id[record.txn_id]
        if writer.outcome == "rolledback":
            reason_chain.append(
                _skip_step(
                    record,
                    writer,
                    "SKIP_ROLLEDBACK",
                    "writer transaction rolled back",
                    snapshot_seq,
                )
            )
            continue

        if record.txn_id == reader_txn_id:
            if writer.commit_seq is not None and writer.commit_seq <= snapshot_seq:
                reason_chain.append(
                    _visible_step(
                        record,
                        writer,
                        "VISIBLE_OWN_COMMITTED",
                        "own committed version is at or before the snapshot",
                        snapshot_seq,
                    )
                )
                return record, reason_chain, read_point
            if dataset.visibility_policy.own_uncommitted_visible and dataset.visibility_policy.own_write_ignores_snapshot:
                reason_chain.append(
                    _visible_step(
                        record,
                        writer,
                        "VISIBLE_OWN_WRITE_POLICY",
                        "explicit read-your-own-writes policy makes the self version visible",
                        snapshot_seq,
                    )
                )
                return record, reason_chain, read_point
            reason_chain.append(
                _skip_step(
                    record,
                    writer,
                    "SKIP_OWN_WRITE_BY_POLICY",
                    "self version is not visible under the declared policy and snapshot",
                    snapshot_seq,
                )
            )
            continue

        commit_seq = writer.commit_seq
        if commit_seq is not None and commit_seq <= snapshot_seq:
            reason_chain.append(
                _visible_step(
                    record,
                    writer,
                    "VISIBLE_COMMITTED_AT_SNAPSHOT",
                    "writer committed and commit_seq is not later than snapshot_seq",
                    snapshot_seq,
                )
            )
            return record, reason_chain, read_point
        reason_chain.append(
            _skip_step(
                record,
                writer,
                "SKIP_COMMIT_AFTER_SNAPSHOT",
                "writer commit_seq is later than snapshot_seq",
                snapshot_seq,
            )
        )

    return None, reason_chain, read_point


def compute_visibility(
    dataset: Dataset,
    query: Query,
    deadline_ns: int | None = None,
) -> VisibilityDecision:
    _check_deadline(deadline_ns)
    visible, reason_chain, read_point = select_visible_version(dataset, query, deadline_ns)
    steps_examined = len(reason_chain)
    predicate_columns = query.predicate.column_count
    matched: bool | None = None
    predicate_detail: dict[str, Any] | None = None
    if visible is None:
        reason_chain = list(reason_chain)
        reason_chain.append(
            {
                "step": "END",
                "code": "NO_VISIBLE_VERSION",
                "message": "the complete chain was examined without a visible version",
                "snapshot_seq": read_point.snapshot_seq,
                "steps_examined": steps_examined,
            }
        )
        outcome = "NO_VISIBLE_VERSION"
    else:
        _check_deadline(deadline_ns)
        matched, predicate_detail = evaluate_predicate(query.predicate, visible.value_summary)
        reason_chain = list(reason_chain)
        reason_chain.append(
            {
                "step": "PREDICATE",
                "version_no": visible.version_no,
                "txn_id": visible.txn_id,
                "code": "PREDICATE_MATCH" if matched else "PREDICATE_NO_MATCH",
                "message": "NumPy predicate evaluation completed",
                "detail": predicate_detail,
            }
        )
        outcome = "VISIBLE"
    reason_tuple = tuple(reason_chain)
    reason_digest = digest_json(list(reason_tuple))
    return VisibilityDecision(
        query_id=query.query_id,
        read_point_id=query.read_point_id,
        row_key=query.row_key,
        snapshot_seq=read_point.snapshot_seq,
        reader_txn_id=read_point.reader_txn_id,
        visible_version_no=visible.version_no if visible else None,
        visible_txn_id=visible.txn_id if visible else None,
        visible_value_digest=visible.value_digest if visible else None,
        visible_value_summary=dict(visible.value_summary) if visible else None,
        outcome=outcome,
        predicate_matched=matched,
        predicate_detail=predicate_detail,
        reason_chain=reason_tuple,
        reason_digest=reason_digest,
        cost_units=steps_examined + predicate_columns,
        steps_examined=steps_examined,
        predicate_columns=predicate_columns,
        visibility_policy=dataset.visibility_policy.as_dict(),
    )


def estimate_query_cost(dataset: Dataset, query: Query) -> int:
    visible, reason_chain, _ = select_visible_version(dataset, query)
    del visible
    return len(reason_chain) + query.predicate.column_count


def decisions_digest(decisions: list[VisibilityDecision] | tuple[VisibilityDecision, ...]) -> str:
    ordered = sorted(decisions, key=lambda item: (item.read_point_id, item.row_key, item.query_id))
    return digest_json([item.as_dict() for item in ordered])
