"""Stable invalidation propagation for transaction/chain evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .canonical import canonical_bytes, sha256_digest
from .errors import DivergenceError, SpecError
from .loader import validate_spec_object
from .model import (
    OUTCOME_COMMITTED,
    OUTCOME_ROLLEDBACK,
    Spec,
    SpecIndex,
    Transaction,
    VersionRecord,
)
from .visibility import QueryResult


@dataclass(frozen=True)
class OutcomeChange:
    tx_id: str
    new_outcome: str
    new_commit_seq: int | None = None


@dataclass(frozen=True)
class InvalidationRequest:
    outcome_changes: tuple[OutcomeChange, ...] = ()
    appended_records: tuple[VersionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_changes": [
                {
                    "tx_id": item.tx_id,
                    "new_outcome": item.new_outcome,
                    "new_commit_seq": item.new_commit_seq,
                }
                for item in sorted(self.outcome_changes, key=lambda x: x.tx_id)
            ],
            "appended_records": [
                item.to_dict()
                for item in sorted(self.appended_records, key=lambda x: (x.row_key, x.version))
            ],
        }


@dataclass(frozen=True)
class InvalidationPlan:
    affected_query_ids: tuple[str, ...]
    unaffected_query_ids: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    change_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "affected_query_ids": list(self.affected_query_ids),
            "unaffected_query_ids": list(self.unaffected_query_ids),
            "evidence": list(self.evidence),
            "change_digest": self.change_digest,
            "stable_order": "query_id,then cause_type,cause_id,row_key,version",
        }


def _validate_request(spec: Spec, index: SpecIndex, request: InvalidationRequest) -> None:
    seen_tx: set[str] = set()
    for change in sorted(request.outcome_changes, key=lambda x: x.tx_id):
        if change.tx_id in seen_tx:
            raise SpecError(
                f"事务结局变更重复: {change.tx_id}",
                entity_type="transaction",
                entity_id=change.tx_id,
            )
        seen_tx.add(change.tx_id)
        if change.tx_id not in index.tx_by_id:
            raise SpecError(
                f"结局变更引用不存在的事务: {change.tx_id}",
                entity_type="transaction",
                entity_id=change.tx_id,
            )
        if change.new_outcome not in {OUTCOME_COMMITTED, OUTCOME_ROLLEDBACK}:
            raise SpecError(
                "新结局必须是 committed 或 rolledback",
                entity_type="transaction",
                entity_id=change.tx_id,
            )
        current = index.tx_by_id[change.tx_id]
        if change.new_outcome == OUTCOME_COMMITTED:
            if type(change.new_commit_seq) is not int or change.new_commit_seq < current.start_seq:
                raise SpecError(
                    "提交结局的新 commit_seq 必须是不小于 start_seq 的整数",
                    entity_type="transaction",
                    entity_id=change.tx_id,
                )
        elif change.new_commit_seq is not None:
            raise SpecError(
                "回滚结局不得携带 commit_seq",
                entity_type="transaction",
                entity_id=change.tx_id,
            )

    seen_versions = {(record.row_key, record.version) for record in spec.records}
    for record in sorted(request.appended_records, key=lambda x: (x.row_key, x.version)):
        if record.row_key not in index.records_by_row_desc:
            raise SpecError(
                f"追加版本引用不存在的行键: {record.row_key}",
                entity_type="row_key",
                entity_id=record.row_key,
            )
        if record.tx_id not in index.tx_by_id:
            raise SpecError(
                f"追加版本引用不存在的事务: {record.tx_id}",
                entity_type="row_key",
                entity_id=record.row_key,
            )
        if (record.row_key, record.version) in seen_versions:
            raise SpecError(
                f"追加版本号已存在: {record.row_key}/{record.version}",
                entity_type="row_key",
                entity_id=record.row_key,
            )
        seen_versions.add((record.row_key, record.version))


def apply_changes(spec: Spec, request: InvalidationRequest) -> Spec:
    """Apply a validated evolution request and return a fully replaced Spec."""

    index = SpecIndex(spec)
    _validate_request(spec, index, request)
    changes = {item.tx_id: item for item in request.outcome_changes}
    transactions: list[Transaction] = []
    for tx in spec.transactions:
        change = changes.get(tx.id)
        if change is None:
            transactions.append(tx)
        else:
            transactions.append(
                Transaction(
                    id=tx.id,
                    start_seq=tx.start_seq,
                    outcome=change.new_outcome,
                    commit_seq=change.new_commit_seq,
                )
            )
    records = tuple(
        sorted(
            (*spec.records, *request.appended_records),
            key=lambda item: (item.row_key, item.version),
        )
    )
    candidate = Spec(
        name=spec.name,
        format_version=spec.format_version,
        transactions=tuple(sorted(transactions, key=lambda x: x.id)),
        records=records,
        read_points=spec.read_points,
        queries=spec.queries,
        budget=spec.budget,
        notes=spec.notes,
    )
    return validate_spec_object(candidate.to_dict(), allow_pending=True)


def _query_edge(index: SpecIndex, query_id: str) -> dict[str, Any]:
    query = index.query_by_id[query_id]
    read_point = index.read_point_by_id[query.read_point_id]
    return {
        "stage": "query",
        "query_id": query.id,
        "read_point_id": read_point.id,
        "snapshot_seq": read_point.snapshot_seq,
    }


def propagate_invalidation(
    spec: Spec,
    request: InvalidationRequest,
    *,
    index: SpecIndex | None = None,
) -> InvalidationPlan:
    current_index = index or SpecIndex(spec)
    _validate_request(spec, current_index, request)
    evidence: list[dict[str, Any]] = []
    affected: set[str] = set()

    for change in sorted(request.outcome_changes, key=lambda x: x.tx_id):
        current = current_index.tx_by_id[change.tx_id]
        tx_records = sorted(
            (record for record in spec.records if record.tx_id == change.tx_id),
            key=lambda item: (item.row_key, item.version),
        )
        for record in tx_records:
            for query in current_index.queries_by_row.get(record.row_key, ()):
                affected.add(query.id)
                evidence.append(
                    {
                        "query_id": query.id,
                        "row_key": record.row_key,
                        "read_point_id": query.read_point_id,
                        "cause_type": "TRANSACTION_OUTCOME",
                        "cause_id": change.tx_id,
                        "chain": [
                            {
                                "stage": "transaction",
                                "tx_id": change.tx_id,
                                "before_outcome": current.outcome,
                                "after_outcome": change.new_outcome,
                                "before_commit_seq": current.commit_seq,
                                "after_commit_seq": change.new_commit_seq,
                            },
                            {
                                "stage": "version",
                                "row_key": record.row_key,
                                "version": record.version,
                            },
                            {
                                "stage": "read_point",
                                "read_point_id": query.read_point_id,
                                "snapshot_seq": current_index.read_point_by_id[
                                    query.read_point_id
                                ].snapshot_seq,
                            },
                            _query_edge(current_index, query.id),
                        ],
                    }
                )

    for record in sorted(
        request.appended_records, key=lambda item: (item.row_key, item.version)
    ):
        for query in current_index.queries_by_row.get(record.row_key, ()):
            affected.add(query.id)
            evidence.append(
                {
                    "query_id": query.id,
                    "row_key": record.row_key,
                    "read_point_id": query.read_point_id,
                    "cause_type": "APPENDED_VERSION",
                    "cause_id": f"{record.row_key}/{record.version}",
                    "chain": [
                        {
                            "stage": "version",
                            "row_key": record.row_key,
                            "version": record.version,
                            "tx_id": record.tx_id,
                        },
                        {
                            "stage": "read_point",
                            "read_point_id": query.read_point_id,
                            "snapshot_seq": current_index.read_point_by_id[
                                query.read_point_id
                            ].snapshot_seq,
                        },
                        _query_edge(current_index, query.id),
                    ],
                }
            )

    evidence.sort(
        key=lambda item: (
            item["query_id"],
            item["cause_type"],
            item["cause_id"],
            item["row_key"],
        )
    )
    all_ids = {query.id for query in spec.queries}
    affected_order = tuple(sorted(affected))
    return InvalidationPlan(
        affected_query_ids=affected_order,
        unaffected_query_ids=tuple(sorted(all_ids - affected)),
        evidence=tuple(evidence),
        change_digest=sha256_digest(request.to_dict()),
    )


def mark_stale_results(
    results: Iterable[QueryResult],
    plan: InvalidationPlan,
) -> tuple[dict[str, Any], ...]:
    affected = set(plan.affected_query_ids)
    payload: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.query_id):
        item = result.to_dict()
        item["staleness"] = "STALE" if result.query_id in affected else "FRESH"
        payload.append(item)
    return tuple(payload)


def replace_affected_results(
    previous: Sequence[QueryResult],
    replacement: Sequence[QueryResult],
    plan: InvalidationPlan,
) -> tuple[QueryResult, ...]:
    previous_by_id = {item.query_id: item for item in previous}
    replacement_by_id = {item.query_id: item for item in replacement}
    expected_ids = set(previous_by_id) | set(replacement_by_id)
    if set(previous_by_id) != set(replacement_by_id):
        raise DivergenceError(
            "替换前后查询集合不一致",
            query_id=plan.affected_query_ids[0] if plan.affected_query_ids else "",
            row_key="",
            field="query_id",
            left=sorted(previous_by_id),
            right=sorted(replacement_by_id),
        )

    for query_id in plan.unaffected_query_ids:
        left = previous_by_id[query_id].to_dict()
        right = replacement_by_id[query_id].to_dict()
        if canonical_bytes(left) != canonical_bytes(right):
            raise DivergenceError(
                "未受影响判定在整体替换时发生变化",
                query_id=query_id,
                row_key=left["row_key"],
                field="reason_chain_digest",
                left=left["reason_chain_digest"],
                right=right["reason_chain_digest"],
            )
    for query_id in plan.affected_query_ids:
        if query_id not in replacement_by_id:
            raise DivergenceError(
                "受影响判定缺少替换结果",
                query_id=query_id,
                row_key=previous_by_id[query_id].row_key,
                field="query_id",
                left=previous_by_id[query_id].to_dict(),
                right=None,
            )
    if expected_ids != set(replacement_by_id):
        raise DivergenceError(
            "替换结果包含未知查询",
            query_id=sorted(expected_ids ^ set(replacement_by_id))[0],
            row_key="",
            field="query_id",
            left=sorted(expected_ids),
            right=sorted(replacement_by_id),
        )
    return tuple(replacement_by_id[query_id] for query_id in sorted(replacement_by_id))
