"""事务结局变化与版本链追加后的失效传播（业务要求 5）。

沿 事务 -> 版本 -> 行键 -> 查询 -> 读点 的引用关系推导全部直接与间接
失效判定，并按读点标识序、行键序稳定输出失效集合与每步传播依据。
"""

from dataclasses import dataclass

from .digest import digest
from .model import Spec
from .visibility import Decision, decide_query


@dataclass(frozen=True)
class InvalidationStep:
    step: int
    source_kind: str
    source_id: str
    relation: str
    target_kind: str
    target_id: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "relation": self.relation,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "reason": self.reason,
        }


def diff_transactions(old: Spec, new: Spec) -> dict[str, str]:
    """返回发生结局/提交序号变化的事务标识 -> 变化描述。"""
    changes: dict[str, str] = {}
    old_txn = {t.txn_id: t for t in old.transactions}
    for txn in sorted(new.transactions, key=lambda t: t.txn_id):
        prev = old_txn.get(txn.txn_id)
        if prev is None:
            changes[txn.txn_id] = "transaction_added"
            continue
        if (prev.outcome != txn.outcome or
                prev.commit_seq != txn.commit_seq):
            changes[txn.txn_id] = (
                f"outcome {prev.outcome}->{txn.outcome}; "
                f"commit_seq {prev.commit_seq}->{txn.commit_seq}"
            )
    for txn_id in sorted(set(old_txn) - {t.txn_id for t in new.transactions}):
        changes[txn_id] = "transaction_removed"
    return changes


def diff_chains(old: Spec, new: Spec) -> dict[str, str]:
    """返回发生追加（或结构变化）的行键 -> 变化描述。"""
    changes: dict[str, str] = {}
    rows = sorted(set(old.row_chains) | set(new.row_chains))
    for row_key in rows:
        old_chain = old.row_chains.get(row_key, ())
        new_chain = new.row_chains.get(row_key, ())
        old_versions = {r.version for r in old_chain}
        new_versions = {r.version for r in new_chain}
        if old_versions == new_versions:
            # 版本集合相同也要比较写入事务与值摘要（结局变化已在事务侧捕获）。
            if digest([r.__dict__ for r in old_chain]) != digest(
                [r.__dict__ for r in new_chain]
            ):
                changes[row_key] = "chain_metadata_changed"
            continue
        appended = sorted(new_versions - old_versions)
        removed = sorted(old_versions - new_versions)
        if appended and not removed and old_versions.issubset(new_versions):
            changes[row_key] = f"chain_appended versions={appended}"
        else:
            changes[row_key] = (
                f"chain_restructured added={appended} removed={removed}"
            )
    return changes


def propagate(old: Spec, new: Spec) -> dict:
    txn_changes = diff_transactions(old, new)
    chain_changes = diff_chains(old, new)

    steps: list[InvalidationStep] = []
    invalid_queries: set[str] = set()
    affected_rows: set[str] = set()

    def add_step(source_kind, source_id, relation, target_kind, target_id,
                 reason):
        steps.append(InvalidationStep(
            step=len(steps) + 1,
            source_kind=source_kind,
            source_id=source_id,
            relation=relation,
            target_kind=target_kind,
            target_id=target_id,
            reason=reason,
        ))

    # 第一层：事务变化 -> 该事务写入的版本 -> 所属行键。
    for txn_id, why in sorted(txn_changes.items()):
        for rec in sorted(new.records, key=lambda r: (r.row_key, r.version)):
            if rec.writer_txn_id != txn_id:
                continue
            add_step("transaction", txn_id, "wrote_version",
                     "version", f"{rec.row_key}#v{rec.version}", why)
            if rec.row_key not in affected_rows:
                add_step("version", f"{rec.row_key}#v{rec.version}",
                         "belongs_to_row", "row", rec.row_key, why)
                affected_rows.add(rec.row_key)

    # 第一层：版本链变化直接影响行键。
    for row_key, why in sorted(chain_changes.items()):
        if row_key not in affected_rows:
            add_step("chain", row_key, "chain_changed", "row", row_key, why)
            affected_rows.add(row_key)

    # 第二层：行键 -> 引用该行键的查询 -> 查询所属读点（间接失效）。
    rp_of_queries: dict[str, set[str]] = {}
    for query in sorted(new.queries,
                        key=lambda q: (q.read_point_id, q.row_key,
                                       q.query_id)):
        if query.row_key in affected_rows:
            add_step("row", query.row_key, "referenced_by_query",
                     "query", query.query_id, "row_input_changed")
            invalid_queries.add(query.query_id)
            rp_of_queries.setdefault(query.read_point_id, set()).add(
                query.query_id)
    for rp_id in sorted(rp_of_queries):
        add_step("query_group", rp_id, "queries_belong_to_readpoint",
                 "read_point", rp_id,
                 f"invalidated_queries={sorted(rp_of_queries[rp_id])}")

    # 稳定失效集合：读点标识序 -> 行键序 -> 查询标识序。
    invalid_set = sorted(
        (q for q in new.queries if q.query_id in invalid_queries),
        key=lambda q: (q.read_point_id, q.row_key, q.query_id),
    )
    return {
        "transaction_changes": txn_changes,
        "chain_changes": chain_changes,
        "affected_rows": sorted(affected_rows),
        "invalidated_query_ids": [q.query_id for q in invalid_set],
        "steps": [s.to_dict() for s in steps],
    }


def recompute_invalid_subset(new: Spec, invalidation_report: dict,
                             old_decisions: dict[str, Decision]
                             ) -> tuple[dict[str, Decision], list[dict]]:
    """仅重算失效查询；未失效查询保留原结论（逐字节可校验）。"""
    invalid_ids = set(invalidation_report["invalidated_query_ids"])
    merged: dict[str, Decision] = {}
    preserved: list[dict] = []
    for query in new.queries:
        if query.query_id in invalid_ids:
            merged[query.query_id] = decide_query(new, query)
        else:
            decision = old_decisions[query.query_id]
            merged[query.query_id] = decision
            preserved.append({
                "query_id": query.query_id,
                "read_point_id": query.read_point_id,
                "row_key": query.row_key,
                "summary_digest": decision.summary_digest(),
            })
    return merged, preserved


def verify_preserved_unchanged(old_decisions: dict[str, Decision],
                               preserved: list[dict]) -> None:
    for item in preserved:
        current = old_decisions[item["query_id"]].summary_digest()
        if current != item["summary_digest"]:
            raise AssertionError(
                f"未失效判定摘要发生变化: {item['query_id']}"
            )
