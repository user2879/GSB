"""权威重算：从段文件逐块重放扫描，与账本逐字节比较。

该路径不接触调度器和账本写入，只依赖段文件与扫描计划，因此可以独立复现
诊断结论。首个不一致位置以 ``E_DIVERGE`` 报告，携带块序号、列与字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .codecs import block_summary, canonical_values
from .errors import DivergenceError
from .planner import ScanPlan
from .segment import Segment
from .versioning import ParameterVersion


@dataclass
class RecomputedBlock:
    segment_id: str
    row_group: int
    column: str
    block_index: int
    rows_completed: int
    memory_cost: int
    result_digest: str
    canonical_bytes: bytes
    pushed_match: bool
    filter_values: list = field(default_factory=list)


def _row_predicate_holds(value, predicates) -> bool:
    for predicate in predicates:
        kind = predicate.kind
        target = predicate.values
        if kind == "eq" and not value == target[0]:
            return False
        if kind == "ne" and not value != target[0]:
            return False
        if kind == "gt" and not value > target[0]:
            return False
        if kind == "ge" and not value >= target[0]:
            return False
        if kind == "lt" and not value < target[0]:
            return False
        if kind == "le" and not value <= target[0]:
            return False
        if kind == "in" and value not in target:
            return False
    return True


def recompute(plan: ScanPlan, segments: dict[str, Segment]) -> list[RecomputedBlock]:
    """从段文件逐块重放候选块扫描。"""
    recomputed: list[RecomputedBlock] = []
    pushed_by_column: dict[str, list] = {}
    for predicate in plan.pushed:
        pushed_by_column.setdefault(predicate.column, []).append(predicate)

    for item in plan.candidate_blocks:
        segment = segments[item["segment_id"]]
        values = segment.read_block(item["row_group"], item["column"],
                                    item["block_index"])
        header = segment.header(item["row_group"], item["column"],
                                item["block_index"])
        predicates = pushed_by_column.get(item["column"], [])
        kept = [value for value in values
                if _row_predicate_holds(value, predicates)]
        recomputed.append(RecomputedBlock(
            segment_id=item["segment_id"],
            row_group=item["row_group"], column=item["column"],
            block_index=item["block_index"], rows_completed=len(values),
            memory_cost=header.pre_bytes,
            result_digest=block_summary(values),
            canonical_bytes=canonical_values(values),
            pushed_match=len(kept) == len(values),
            filter_values=kept))
    recomputed.sort(key=lambda block: (block.segment_id, block.row_group,
                                       block.column, block.block_index))
    return recomputed


def verify_ledger(plan: ScanPlan, segments: dict[str, Segment],
                  ledger, version: ParameterVersion) -> dict:
    """把账本当前版本轨迹与权威重算逐字段比较。"""
    recomputed = recompute(plan, segments)
    rows = ledger.active_rows(version)
    if len(rows) != len(recomputed):
        raise DivergenceError(
            "账本任务数与权威重算块数不一致",
            {"ledger_count": len(rows), "recomputed_count": len(recomputed),
             "segment_id": recomputed[0].segment_id if recomputed else None,
             "block_index": recomputed[0].block_index
             if recomputed else None,
             "column": recomputed[0].column if recomputed else None,
             "field": "task_count"})

    field_map = (
        ("rows_completed", "rows_completed"),
        ("memory_cost", "memory_cost"),
        ("result_digest", "result_digest"),
        ("segment_id", "segment_id"),
        ("row_group", "row_group"),
        ("column", "column"),
        ("block_index", "block_index"),
    )
    for row, block in zip(rows, recomputed):
        for row_field, block_field in field_map:
            if row[row_field] != getattr(block, block_field):
                raise DivergenceError(
                    "账本与权威重算不一致",
                    {"segment_id": block.segment_id,
                     "block_index": block.block_index,
                     "column": block.column, "field": row_field,
                     "ledger": row[row_field],
                     "recomputed": getattr(block, block_field)})
        if row["status"] != "COMPLETED":
            raise DivergenceError("权威重算发现失败任务",
                                  {"segment_id": block.segment_id,
                                   "block_index": block.block_index,
                                   "column": block.column,
                                   "field": "status"})
    return {"status": "MATCH",
            "blocks_compared": len(recomputed),
            "ledger_digest": ledger.digest(version),
            "recomputed_digests": [block.result_digest for block in
                                   recomputed]}


def result_rows(plan: ScanPlan,
                recomputed: list[RecomputedBlock]) -> list[dict]:
    """把重算结果按行组拼成结果行集合（仅投影列），用于演示展示。"""
    projected = plan.projected_columns
    grouped: dict[int, dict[str, list]] = {}
    for block in recomputed:
        grouped.setdefault(block.row_group, {})[block.column] = \
            block.filter_values or _reload(plan, block)
    rows: list[dict] = []
    for row_group in sorted(grouped):
        columns = grouped[row_group]
        if not columns:
            continue
        length = len(next(iter(columns.values())))
        for offset in range(length):
            rows.append({column: columns[column][offset]
                         for column in projected if column in columns})
    return rows


def _reload(plan: ScanPlan, block: RecomputedBlock) -> list:
    # 该辅助仅在没有下推谓词时使用；重算主路径已持有值
    return block.filter_values
