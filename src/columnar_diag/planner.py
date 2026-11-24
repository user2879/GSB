from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .errors import AppError
from .models import EncodedBlock, QueryPlan, ScanTask, SegmentSpec
from .predicate import (
    evaluate_node,
    leaf_columns,
    parse_where,
)
from .util import sha256_json


@dataclass(frozen=True)
class PlanBundle:
    plans: tuple[QueryPlan, ...]
    query_asts: dict[str, dict[str, Any]]
    tasks: tuple[ScanTask, ...]
    digest: str

    @property
    def plan_map(self) -> dict[str, QueryPlan]:
        return {plan.query_id: plan for plan in self.plans}


def _row_values(spec: SegmentSpec, row_group_id: str) -> dict[str, tuple[int, ...]]:
    values: dict[str, tuple[int, ...]] = {}
    for block in spec.blocks:
        if block.row_group == row_group_id:
            values[block.column] = block.values
    return values


def build_plans(
    spec: SegmentSpec,
    *,
    query_ids: set[str] | None = None,
) -> PlanBundle:
    if not spec.queries:
        return PlanBundle(plans=(), query_asts={}, tasks=(), digest=sha256_json([]))
    column_names = [column.name for column in spec.columns]
    column_set = set(column_names)
    plans: list[QueryPlan] = []
    query_asts: dict[str, dict[str, Any]] = {}
    tasks: list[ScanTask] = []
    ordinal = 1
    block_maps = {(block.column, block.row_group): block for block in spec.blocks}
    for query in spec.queries:
        if query_ids is not None and query.id not in query_ids:
            continue
        ast = parse_where(query.where, query_id=query.id)
        predicate_columns = tuple(leaf_columns(ast))
        for column in predicate_columns:
            if column not in column_set:
                raise AppError(
                    "E_SPEC",
                    f"谓词引用了不存在的列 {column!r}",
                    block_id=f"query:{query.id}",
                    field="where",
                )
        referenced = set(query.select) | set(predicate_columns)
        pruned = tuple(column for column in column_names if column not in referenced)
        candidate_blocks: list[str] = []
        row_group_details: list[dict[str, Any]] = []
        basis: list[str] = [
            f"column_clip:{','.join(pruned) if pruned else '<none>'}",
            f"predicate:and:{len(ast.get('children', []))}:{','.join(predicate_columns) if predicate_columns else '<none>'}",
        ]
        for group in spec.row_groups:
            group_values = _row_values(spec, group.id)
            matching_rows: list[int] = []
            for local_index in range(group.row_count):
                row = {
                    column: group_values[column][local_index]
                    for column in predicate_columns
                }
                if evaluate_node(row, ast):
                    matching_rows.append(local_index)
            group_candidates: list[str] = []
            if matching_rows:
                for column in column_names:
                    if column not in referenced:
                        continue
                    block = block_maps[(column, group.id)]
                    group_candidates.append(block.id)
                    candidate_blocks.append(block.id)
            row_group_details.append(
                {
                    "row_group_id": group.id,
                    "start_row": group.start_row,
                    "row_count": group.row_count,
                    "matched_rows": len(matching_rows),
                    "candidate_blocks": group_candidates,
                }
            )
            if matching_rows:
                basis.append(
                    f"row_group:{group.id}:predicate_match={len(matching_rows)}"
                )
            else:
                basis.append(f"row_group:{group.id}:pruned=no_predicate_match")
        for column in pruned:
            basis.append(f"column:{column}:pruned=not_referenced")
        digest_payload = {
            "query_id": query.id,
            "select": list(query.select),
            "predicate": ast,
            "predicate_columns": list(predicate_columns),
            "candidate_blocks": candidate_blocks,
            "pruned_columns": list(pruned),
            "row_groups": [
                {
                    "row_group_id": detail["row_group_id"],
                    "candidate_blocks": detail["candidate_blocks"],
                }
                for detail in row_group_details
            ],
        }
        digest = sha256_json(digest_payload)
        plan = QueryPlan(
            query_id=query.id,
            select=query.select,
            predicate_columns=predicate_columns,
            candidate_blocks=tuple(candidate_blocks),
            pruned_columns=pruned,
            row_groups=tuple(row_group_details),
            basis=tuple(basis),
            digest=digest,
        )
        plans.append(plan)
        query_asts[query.id] = ast
        for group in spec.row_groups:
            detail = next(
                item for item in row_group_details if item["row_group_id"] == group.id
            )
            for column in column_names:
                if column not in referenced:
                    continue
                block = block_maps[(column, group.id)]
                slot_ordinal = ordinal
                ordinal += 1
                if block.id not in detail["candidate_blocks"]:
                    continue
                task = ScanTask(
                    task_id=f"q:{query.id}|rg:{group.id}|block:{block.id}",
                    query_id=query.id,
                    segment_id=spec.segment_id,
                    row_group_id=group.id,
                    block_id=block.id,
                    column=block.column,
                    block_ordinal=slot_ordinal,
                    decode_memory_bytes=len(block.values) * 8,
                )
                tasks.append(task)
    plan_digest = sha256_json(
        [
            {
                "query_id": plan.query_id,
                "plan_digest": plan.digest,
                "candidate_blocks": list(plan.candidate_blocks),
            }
            for plan in plans
        ]
    )
    return PlanBundle(
        plans=tuple(plans),
        query_asts=query_asts,
        tasks=tuple(tasks),
        digest=plan_digest,
    )


def attach_task_memory(
    bundle: PlanBundle, encoded_blocks: dict[str, EncodedBlock]
) -> PlanBundle:
    tasks = tuple(
        ScanTask(
            task_id=task.task_id,
            query_id=task.query_id,
            segment_id=task.segment_id,
            row_group_id=task.row_group_id,
            block_id=task.block_id,
            column=task.column,
            block_ordinal=task.block_ordinal,
            decode_memory_bytes=encoded_blocks[task.block_id].pre_bytes,
        )
        for task in bundle.tasks
    )
    return PlanBundle(
        plans=bundle.plans,
        query_asts=bundle.query_asts,
        tasks=tasks,
        digest=bundle.digest,
    )


def task_result_payload(
    *,
    task: ScanTask,
    ast: dict[str, Any],
    values: tuple[int, ...],
    block_digest: str,
) -> dict[str, Any]:
    leaves = [
        child
        for child in ast.get("children", [])
        if child.get("column") == task.column
    ]
    if leaves:
        mask = [
            all(_evaluate_leaf(value, leaf) for leaf in leaves) for value in values
        ]
    else:
        mask = [True] * len(values)
    matched_rows = [index for index, matched in enumerate(mask) if matched]
    value_digest = sha256_json(list(values))
    result_json = {
        "task_id": task.task_id,
        "query_id": task.query_id,
        "row_group_id": task.row_group_id,
        "block_id": task.block_id,
        "column": task.column,
        "matched_row_indices": matched_rows,
        "value_digest": value_digest,
        "block_digest": block_digest,
    }
    result_digest = sha256_json(result_json)
    result_json = {**result_json, "result_digest": result_digest}
    return result_json


def _evaluate_leaf(value: int, leaf: dict[str, Any]) -> bool:
    from .predicate import evaluate_leaf

    return evaluate_leaf(value, leaf)


def merge_query_row_group(
    *,
    query_id: str,
    row_group_id: str,
    select: Iterable[str],
    predicate_columns: Iterable[str],
    task_payloads: Iterable[dict[str, Any]],
    row_values: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    payloads = list(task_payloads)
    if not payloads:
        rows: list[list[int]] = []
    else:
        row_count = len(next(iter(row_values.values())))
        predicate_columns_tuple = tuple(predicate_columns)
        selected_columns = tuple(select)
        if predicate_columns_tuple:
            masks: list[list[bool]] = []
            for column in predicate_columns_tuple:
                payload = next((item for item in payloads if item["column"] == column), None)
                if payload is None:
                    masks.append([False] * row_count)
                else:
                    matched = set(payload["matched_row_indices"])
                    masks.append([index in matched for index in range(row_count)])
            matching = [
                index
                for index in range(row_count)
                if all(mask[index] for mask in masks)
            ]
        else:
            matching = list(range(row_count))
        rows = [
            [row_values[column][index] for column in selected_columns]
            for index in matching
        ]
    digest_payload = {
        "query_id": query_id,
        "row_group_id": row_group_id,
        "select": list(select),
        "rows": rows,
    }
    return {
        **digest_payload,
        "row_count": len(rows),
        "result_digest": sha256_json(digest_payload),
    }


def plan_to_dict(
    plan: QueryPlan, ast: dict[str, Any]
) -> dict[str, Any]:
    return {**plan.to_dict(), "predicate": ast}


def bundle_to_dict(bundle: PlanBundle) -> dict[str, Any]:
    return {
        "plan_digest": bundle.digest,
        "plans": [
            plan_to_dict(plan, bundle.query_asts[plan.query_id])
            for plan in bundle.plans
        ],
        "tasks": [task.to_dict() for task in bundle.tasks],
    }
