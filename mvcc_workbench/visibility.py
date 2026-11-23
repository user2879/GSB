"""Deterministic MVCC snapshot visibility and reason-chain construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, sha256_digest
from .errors import DivergenceError
from .model import (
    OUTCOME_COMMITTED,
    OUTCOME_PENDING,
    OUTCOME_ROLLEDBACK,
    Query,
    SpecIndex,
    VersionRecord,
)

RULE_VERSION = 1


@dataclass(frozen=True)
class ExclusionReason:
    step: int
    version: int
    tx_id: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "version": self.version,
            "tx_id": self.tx_id,
            "code": self.code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ResolvedVisibility:
    read_point_id: str
    row_key: str
    snapshot_seq: int
    visible_version: int | None
    visible_tx_id: str | None
    visible_value_digest: str | None
    visibility_basis: str | None
    backtrack_steps: int
    inspected_versions: int
    exclusion_reasons: tuple[ExclusionReason, ...]

    @property
    def has_visible_version(self) -> bool:
        return self.visible_version is not None

    def resolution_payload(self) -> dict[str, Any]:
        return {
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "snapshot_seq": self.snapshot_seq,
            "visible_version": self.visible_version,
            "visible_tx_id": self.visible_tx_id,
            "visible_value_digest": self.visible_value_digest,
            "visibility_basis": self.visibility_basis,
            "backtrack_steps": self.backtrack_steps,
            "inspected_versions": self.inspected_versions,
            "exclusion_reasons": [item.to_dict() for item in self.exclusion_reasons],
        }


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    read_point_id: str
    row_key: str
    snapshot_seq: int
    predicate: dict[str, Any]
    predicate_result: bool
    visible_version: int | None
    visible_tx_id: str | None
    visible_value_digest: str | None
    visibility_basis: str | None
    backtrack_steps: int
    inspected_versions: int
    exclusion_reasons: tuple[ExclusionReason, ...]
    terminal_code: str
    rule_version: int
    reason_chain_digest: str

    def reason_payload(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "query_id": self.query_id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "snapshot_seq": self.snapshot_seq,
            "predicate": self.predicate,
            "predicate_result": self.predicate_result,
            "visible_version": self.visible_version,
            "visible_tx_id": self.visible_tx_id,
            "visible_value_digest": self.visible_value_digest,
            "visibility_basis": self.visibility_basis,
            "backtrack_steps": self.backtrack_steps,
            "inspected_versions": self.inspected_versions,
            "exclusion_reasons": [item.to_dict() for item in self.exclusion_reasons],
            "terminal_code": self.terminal_code,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.reason_payload()
        payload["reason_chain_digest"] = self.reason_chain_digest
        return payload


@dataclass(frozen=True)
class MaterializedBatch:
    resolutions: tuple[ResolvedVisibility, ...]
    results: tuple[QueryResult, ...]
    unique_pair_count: int


@dataclass(frozen=True)
class ConsistencyReport:
    ok: bool
    compared_queries: int
    left_path: str
    right_path: str
    conclusion_digest: str
    code: str = "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "left_path": self.left_path,
            "right_path": self.right_path,
            "compared_queries": self.compared_queries,
            "conclusion_digest": self.conclusion_digest,
        }


def _exclusion(
    step: int,
    record: VersionRecord,
    code: str,
    detail: str,
) -> ExclusionReason:
    return ExclusionReason(
        step=step,
        version=record.version,
        tx_id=record.tx_id,
        code=code,
        detail=detail,
    )


def _ordered_chain(index: SpecIndex, row_key: str) -> tuple[VersionRecord, ...]:
    """Return the logical chain from its max-version head through prev links."""

    chain = index.chain(row_key)
    if not chain:
        return ()
    by_version = {record.version: record for record in chain}
    ordered: list[VersionRecord] = []
    current: int | None = chain[0].version
    seen: set[int] = set()
    while current is not None and current in by_version and current not in seen:
        seen.add(current)
        record = by_version[current]
        ordered.append(record)
        current = record.prev_version
    if len(ordered) != len(chain):
        # Branching or an unreachable fragment still has a deterministic order.
        tail = sorted(
            (record for record in chain if record.version not in seen),
            key=lambda item: item.version,
            reverse=True,
        )
        ordered.extend(tail)
    return tuple(ordered)


def resolve_visibility(
    index: SpecIndex,
    read_point_id: str,
    row_key: str,
) -> ResolvedVisibility:
    read_point = index.read_point_by_id[read_point_id]
    reasons: list[ExclusionReason] = []
    inspected = 0
    for step, record in enumerate(_ordered_chain(index, row_key), start=1):
        inspected = step
        tx = index.tx_by_id[record.tx_id]
        if tx.id == read_point.tx_id and tx.outcome == OUTCOME_PENDING:
            return ResolvedVisibility(
                read_point_id=read_point_id,
                row_key=row_key,
                snapshot_seq=read_point.snapshot_seq,
                visible_version=record.version,
                visible_tx_id=record.tx_id,
                visible_value_digest=record.value_digest,
                visibility_basis="SELF_UNCOMMITTED",
                backtrack_steps=len(reasons),
                inspected_versions=inspected,
                exclusion_reasons=tuple(reasons),
            )
        if tx.id != read_point.tx_id and tx.outcome == OUTCOME_PENDING:
            reasons.append(
                _exclusion(
                    step,
                    record,
                    "WRITER_PENDING_NOT_SELF",
                    f"写入事务 {tx.id} 未决，且读事务为 {read_point.tx_id}",
                )
            )
            continue
        if tx.outcome == OUTCOME_ROLLEDBACK:
            reasons.append(
                _exclusion(
                    step,
                    record,
                    "WRITER_ROLLED_BACK",
                    f"写入事务 {tx.id} 已回滚",
                )
            )
            continue
        if tx.outcome != OUTCOME_COMMITTED:
            reasons.append(
                _exclusion(
                    step,
                    record,
                    "WRITER_OUTCOME_UNKNOWN",
                    f"写入事务 {tx.id} 结局未知",
                )
            )
            continue
        if tx.commit_seq is None:
            reasons.append(
                _exclusion(
                    step,
                    record,
                    "COMMIT_SEQ_MISSING",
                    f"写入事务 {tx.id} 缺少提交序号",
                )
            )
            continue
        if tx.commit_seq > read_point.snapshot_seq:
            reasons.append(
                _exclusion(
                    step,
                    record,
                    "COMMIT_AFTER_SNAPSHOT",
                    (
                        f"提交序号 {tx.commit_seq} 晚于快照序号 "
                        f"{read_point.snapshot_seq}"
                    ),
                )
            )
            continue
        return ResolvedVisibility(
            read_point_id=read_point_id,
            row_key=row_key,
            snapshot_seq=read_point.snapshot_seq,
            visible_version=record.version,
            visible_tx_id=record.tx_id,
            visible_value_digest=record.value_digest,
            visibility_basis="COMMITTED_AT_OR_BEFORE_SNAPSHOT",
            backtrack_steps=len(reasons),
            inspected_versions=inspected,
            exclusion_reasons=tuple(reasons),
        )

    return ResolvedVisibility(
        read_point_id=read_point_id,
        row_key=row_key,
        snapshot_seq=read_point.snapshot_seq,
        visible_version=None,
        visible_tx_id=None,
        visible_value_digest=None,
        visibility_basis=None,
        backtrack_steps=len(reasons),
        inspected_versions=inspected,
        exclusion_reasons=tuple(reasons),
    )


def apply_predicate(resolved: ResolvedVisibility, predicate: Mapping[str, Any]) -> bool:
    if not resolved.has_visible_version:
        return False
    op = predicate.get("op")
    if op == "exists":
        return True
    if op == "version_eq":
        return resolved.visible_version == predicate.get("value")
    if op == "digest_eq":
        return resolved.visible_value_digest == predicate.get("value")
    if op == "tx_eq":
        return resolved.visible_tx_id == predicate.get("value")
    raise ValueError(f"unknown predicate op: {op!r}")


def build_query_result(
    query: Query,
    resolved: ResolvedVisibility,
) -> QueryResult:
    predicate_result = apply_predicate(resolved, query.predicate)
    if not resolved.has_visible_version:
        terminal_code = "NO_VISIBLE_VERSION"
    elif predicate_result:
        terminal_code = "VISIBLE_PREDICATE_TRUE"
    else:
        terminal_code = "VISIBLE_PREDICATE_FALSE"
    provisional = QueryResult(
        query_id=query.id,
        read_point_id=query.read_point_id,
        row_key=query.row_key,
        snapshot_seq=resolved.snapshot_seq,
        predicate=dict(query.predicate),
        predicate_result=predicate_result,
        visible_version=resolved.visible_version,
        visible_tx_id=resolved.visible_tx_id,
        visible_value_digest=resolved.visible_value_digest,
        visibility_basis=resolved.visibility_basis,
        backtrack_steps=resolved.backtrack_steps,
        inspected_versions=resolved.inspected_versions,
        exclusion_reasons=resolved.exclusion_reasons,
        terminal_code=terminal_code,
        rule_version=RULE_VERSION,
        reason_chain_digest="",
    )
    return dataclass_replace_digest(provisional, sha256_digest(provisional.reason_payload()))


def dataclass_replace_digest(result: QueryResult, digest: str) -> QueryResult:
    return QueryResult(
        query_id=result.query_id,
        read_point_id=result.read_point_id,
        row_key=result.row_key,
        snapshot_seq=result.snapshot_seq,
        predicate=result.predicate,
        predicate_result=result.predicate_result,
        visible_version=result.visible_version,
        visible_tx_id=result.visible_tx_id,
        visible_value_digest=result.visible_value_digest,
        visibility_basis=result.visibility_basis,
        backtrack_steps=result.backtrack_steps,
        inspected_versions=result.inspected_versions,
        exclusion_reasons=result.exclusion_reasons,
        terminal_code=result.terminal_code,
        rule_version=result.rule_version,
        reason_chain_digest=digest,
    )


def run_per_query(
    index: SpecIndex,
    queries: Iterable[Query] | None = None,
) -> tuple[QueryResult, ...]:
    selected = sorted(index.spec.queries if queries is None else queries, key=lambda q: q.id)
    results: list[QueryResult] = []
    for query in selected:
        resolved = resolve_visibility(index, query.read_point_id, query.row_key)
        results.append(build_query_result(query, resolved))
    return tuple(results)


def materialize_pairs(
    index: SpecIndex,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[ResolvedVisibility, ...]:
    selected = sorted(set(index.query_pairs if pairs is None else pairs))
    return tuple(
        resolve_visibility(index, read_point_id, row_key)
        for read_point_id, row_key in selected
    )


def run_materialized(
    index: SpecIndex,
    resolutions: Sequence[ResolvedVisibility] | None = None,
) -> MaterializedBatch:
    selected_resolutions = (
        materialize_pairs(index) if resolutions is None else tuple(resolutions)
    )
    by_pair = {
        (item.read_point_id, item.row_key): item for item in selected_resolutions
    }
    results: list[QueryResult] = []
    for query in sorted(index.spec.queries, key=lambda q: q.id):
        resolved = by_pair[(query.read_point_id, query.row_key)]
        results.append(build_query_result(query, resolved))
    return MaterializedBatch(
        resolutions=tuple(selected_resolutions),
        results=tuple(results),
        unique_pair_count=len(by_pair),
    )


def _reason_sequence(result: QueryResult) -> list[dict[str, Any]]:
    return [item.to_dict() for item in result.exclusion_reasons]


def _json_safe(value: Any) -> Any:
    if isinstance(value, QueryResult):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def compare_results(
    left: Sequence[QueryResult],
    right: Sequence[QueryResult],
    *,
    left_path: str = "per_query",
    right_path: str = "materialized",
) -> ConsistencyReport:
    left_by_id = {item.query_id: item for item in left}
    right_by_id = {item.query_id: item for item in right}
    all_ids = sorted(set(left_by_id) | set(right_by_id))
    for query_id in all_ids:
        left_item = left_by_id.get(query_id)
        right_item = right_by_id.get(query_id)
        if left_item is None or right_item is None:
            available = left_item or right_item
            raise DivergenceError(
                "两条判定路径的查询集合不一致",
                query_id=query_id,
                row_key=available.row_key if available else "",
                field="query_id",
                left=left_item.to_dict() if left_item else None,
                right=right_item.to_dict() if right_item else None,
            )
        checks = (
            ("visible_version", left_item.visible_version, right_item.visible_version),
            ("exclusion_reasons", _reason_sequence(left_item), _reason_sequence(right_item)),
            ("reason_chain_digest", left_item.reason_chain_digest, right_item.reason_chain_digest),
        )
        for field, left_value, right_value in checks:
            if left_value != right_value:
                raise DivergenceError(
                    "两条判定路径结果不一致",
                    query_id=query_id,
                    row_key=left_item.row_key,
                    field=field,
                    left=_json_safe(left_value),
                    right=_json_safe(right_value),
                )

    digest = sha256_digest(
        {
            "kind": "visibility-conclusions",
            "format_version": 1,
            "items": [item.to_dict() for item in sorted(left, key=lambda x: x.query_id)],
        }
    )
    return ConsistencyReport(
        ok=True,
        compared_queries=len(all_ids),
        left_path=left_path,
        right_path=right_path,
        conclusion_digest=digest,
    )


def result_map(results: Iterable[QueryResult]) -> dict[str, QueryResult]:
    return {result.query_id: result for result in results}


def result_digest_bytes(results: Sequence[QueryResult]) -> bytes:
    return canonical_json([item.to_dict() for item in sorted(results, key=lambda x: x.query_id)]).encode(
        "utf-8"
    )
