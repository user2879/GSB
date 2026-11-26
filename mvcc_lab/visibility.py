"""按读点快照的 MVCC 可见性推导、理由链与稳定裁决（业务要求 2、6）。

规则版本见 :data:`mvcc_lab.RULE_VERSION`。所有排序均为确定性排序，
理由链输出不依赖字典/集合遍历顺序。
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from . import (
    CAUSE_CHAIN_BROKEN,
    CAUSE_RULE_CONFLICT,
    CAUSE_TIE,
    REASON_CHAIN_BROKEN,
    REASON_COMMIT_TOO_LATE,
    REASON_ORDER,
    REASON_ROLLED_BACK,
    REASON_SUPERSEDED,
    REASON_UNCOMMITTED,
    RULE_VERSION,
)
from .digest import canonical, digest
from .model import COMMITTED, PENDING, ROLLED_BACK, RecordVersion, Spec


@dataclass(frozen=True)
class Excluded:
    version: int
    writer_txn_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    query_id: str
    read_point_id: str
    row_key: str
    visible_version: int | None
    visible_txn_id: str | None
    candidates: tuple[int, ...]
    excluded: tuple[Excluded, ...]
    reason_chain: tuple[str, ...]
    tie_applied: bool
    error_influenced: bool
    causes: tuple[str, ...]
    rule_version: str = RULE_VERSION

    def summary(self) -> dict:
        """账本与重放比较使用的稳定摘要。"""
        return {
            "query_id": self.query_id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "visible_version": self.visible_version,
            "visible_txn_id": self.visible_txn_id,
            "excluded": [
                {"version": e.version, "writer_txn_id": e.writer_txn_id,
                 "reasons": list(e.reasons)}
                for e in self.excluded
            ],
            "reason_chain": list(self.reason_chain),
            "tie_applied": self.tie_applied,
            "error_influenced": self.error_influenced,
            "causes": list(self.causes),
            "rule_version": self.rule_version,
        }

    def summary_digest(self) -> str:
        return digest(self.summary())


def _stable_reasons(reasons: set[str]) -> tuple[str, ...]:
    return tuple(r for r in REASON_ORDER if r in reasons)


def _version_status(rec: RecordVersion, spec: Spec, snapshot_seq: int,
                    reader_txn_id: str) -> tuple[bool, set[str]]:
    """返回 (基础可见性, 排除理由集合)。自身未提交版本对自身可见。"""
    txn = spec.txn(rec.writer_txn_id)
    reasons: set[str] = set()
    is_own = txn.txn_id == reader_txn_id
    if txn.outcome == ROLLED_BACK:
        reasons.add(REASON_ROLLED_BACK)
    if txn.outcome in (PENDING, ROLLED_BACK):
        if is_own and txn.outcome == PENDING:
            # 业务要求 2：本事务未提交版本对自身可见（回滚除外）。
            return True, reasons
        reasons.add(REASON_UNCOMMITTED)
        return False, reasons
    # outcome == committed
    assert txn.commit_seq is not None
    if txn.commit_seq > snapshot_seq:
        reasons.add(REASON_COMMIT_TOO_LATE)
        return False, reasons
    return True, reasons


def _tiebreak_key(rec: RecordVersion, spec: Spec):
    txn = spec.txn(rec.writer_txn_id)
    commit_seq = txn.commit_seq if txn.commit_seq is not None else -1
    # 显式决胜：提交序号降序 -> 版本号降序 -> 事务标识序（升序兜底）。
    return (-commit_seq, -rec.version, txn.txn_id)


def decide_query(spec: Spec, query, rule_version: str = RULE_VERSION,
                 forced_rule_conflict: bool = False) -> Decision:
    read_point = spec.read_point(query.read_point_id)
    chain = spec.row_chains.get(query.row_key, ())
    # 确定性候选顺序：版本号降序（链头在前），与遍历顺序无关。
    candidates = sorted(chain, key=lambda r: r.version, reverse=True)

    base_visible: list[tuple[RecordVersion, set[str]]] = []
    excluded: list[Excluded] = []
    chain_broken = False

    # 断链检测：任一非头版本的 prev_version 在链内缺失即断链。
    by_ver = {r.version: r for r in chain}
    for rec in candidates:
        if rec.prev_version is not None and rec.prev_version not in by_ver:
            chain_broken = True

    for rec in candidates:
        visible, reasons = _version_status(
            rec, spec, read_point.snapshot_seq, read_point.reader_txn_id
        )
        if visible:
            base_visible.append((rec, reasons))
        else:
            excluded.append(
                Excluded(
                    version=rec.version,
                    writer_txn_id=rec.writer_txn_id,
                    reasons=_stable_reasons(reasons),
                )
            )

    tie_applied = False
    error_influenced = False
    causes: set[str] = set()
    chosen: RecordVersion | None = None

    if base_visible:
        # 可见性得分：以提交序号为得分（未决自身版本记 -1）。
        def score(pair):
            rec = pair[0]
            txn = spec.txn(rec.writer_txn_id)
            return float(txn.commit_seq if txn.commit_seq is not None else -1)

        ordered = sorted(base_visible, key=lambda pair: _tiebreak_key(
            pair[0], spec))
        top = ordered[0]
        if len(ordered) > 1:
            second = ordered[1]
            gap = abs(score(top) - score(second))
            if spec.epsilon > 0 and (
                np.isclose(gap, 0.0, atol=spec.epsilon)
                or gap <= spec.epsilon
            ):
                tie_applied = True
                error_influenced = True
                causes.add(CAUSE_TIE)
        # 两个读点的定点快照边界差落在误差界内（同序号空间相邻读点）。
        if spec.epsilon > 0:
            for other in spec.read_points:
                if other.read_point_id == read_point.read_point_id:
                    continue
                boundary_gap = abs(
                    read_point.snapshot_seq - other.snapshot_seq
                )
                if boundary_gap <= spec.epsilon:
                    error_influenced = True
                    causes.add(CAUSE_TIE)
                    break
        chosen = top[0]
        # 其余基础可见候选被取代。
        chosen_versions = {chosen.version}
        for rec, _ in ordered[1:]:
            excluded.append(
                Excluded(
                    version=rec.version,
                    writer_txn_id=rec.writer_txn_id,
                    reasons=_stable_reasons({REASON_SUPERSEDED}),
                )
            )
        # 若被排除集合里也存在链上更老候选，统一以确定性顺序重排。
        _ = chosen_versions

    if chain_broken:
        error_influenced = True
        causes.add(CAUSE_CHAIN_BROKEN)
        for idx, exc in enumerate(excluded):
            if REASON_CHAIN_BROKEN not in exc.reasons:
                excluded[idx] = Excluded(
                    version=exc.version,
                    writer_txn_id=exc.writer_txn_id,
                    reasons=_stable_reasons(set(exc.reasons) |
                                            {REASON_CHAIN_BROKEN}),
                )
    if forced_rule_conflict or rule_version != RULE_VERSION:
        error_influenced = True
        causes.add(CAUSE_RULE_CONFLICT)

    excluded_sorted = tuple(
        sorted(excluded, key=lambda e: (-e.version, e.writer_txn_id))
    )
    reason_chain = _stable_reasons(
        {r for exc in excluded_sorted for r in exc.reasons}
    )
    return Decision(
        query_id=query.query_id,
        read_point_id=query.read_point_id,
        row_key=query.row_key,
        visible_version=None if chosen is None else chosen.version,
        visible_txn_id=None if chosen is None else chosen.writer_txn_id,
        candidates=tuple(r.version for r in candidates),
        excluded=excluded_sorted,
        reason_chain=reason_chain,
        tie_applied=tie_applied,
        error_influenced=error_influenced,
        causes=tuple(sorted(causes)),
        rule_version=rule_version if not forced_rule_conflict
        else f"{rule_version}!conflict",
    )


def decide_all(spec: Spec, rule_version: str = RULE_VERSION,
               forced_rule_conflict: bool = False) -> list[Decision]:
    # 确定性整体顺序：读点标识序 -> 行键序 -> 查询标识序。
    ordered = sorted(
        spec.queries,
        key=lambda q: (q.read_point_id, q.row_key, q.query_id),
    )
    return [
        decide_query(spec, q, rule_version, forced_rule_conflict)
        for q in ordered
    ]


def cause_proportions(decisions) -> dict[str, float]:
    """三类成因占比（按受影响判定计，分母为受影响判定总数）。"""
    names = (CAUSE_TIE, CAUSE_RULE_CONFLICT, CAUSE_CHAIN_BROKEN)
    counts = {name: 0 for name in names}
    influenced = 0
    for decision in decisions:
        if decision.error_influenced:
            influenced += 1
            for cause in decision.causes:
                counts[cause] = counts.get(cause, 0) + 1
    if influenced == 0:
        return {name: 0.0 for name in names}
    # numpy 保证浮点口径一致。
    arr = np.array([counts[name] for name in names], dtype=np.float64)
    props = np.round(arr / influenced, decimals=6)
    return {name: float(props[i]) for i, name in enumerate(names)}
