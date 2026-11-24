from __future__ import annotations

from typing import Any

from .errors import AppError
from .scan import compute_query_results, run_block_task
from .util import canonical_json, sha256_json


def _first_ordinal_for_query_group(workspace: Any, query_id: str, row_group_id: str) -> int:
    for task in workspace.bundle.tasks:
        if task.query_id == query_id and task.row_group_id == row_group_id:
            return task.block_ordinal
    plan = workspace.bundle.plan_map.get(query_id)
    if plan:
        for detail in plan.row_groups:
            if detail["row_group_id"] == row_group_id:
                candidates = detail["candidate_blocks"]
                if candidates:
                    for task in workspace.bundle.tasks:
                        if task.block_id == candidates[0]:
                            return task.block_ordinal
    return 1


def authoritative_replay(
    workspace: Any,
    *,
    compare_query_results: bool = True,
    error_code: str = "E_DIVERGE",
) -> dict[str, Any]:
    """Recompute every expected block from the segment without reading ledger digests."""

    if error_code not in {"E_DIVERGE", "E_REPLAY"}:
        raise ValueError("error_code must be E_DIVERGE or E_REPLAY")

    ledger_rows = {
        row["task_id"]: row for row in workspace.ledger.current_task_rows()
    }
    decoded_row_values = {
        group.id: {
            block.column: workspace.image.values(block.id)
            for block in workspace.spec.blocks
            if block.row_group == group.id
        }
        for group in workspace.spec.row_groups
    }
    fresh_payloads: list[dict[str, Any]] = []
    checked_stats: list[dict[str, Any]] = []
    for task in sorted(workspace.bundle.tasks, key=lambda item: item.block_ordinal):
        row = ledger_rows.get(task.task_id)
        if row is None:
            raise AppError(
                error_code,
                "账本缺少扫描序列中的块结局",
                block_id=task.block_id,
                block_ordinal=task.block_ordinal,
                field="ledger_presence",
                basis=[f"task_id={task.task_id}"],
            )
        values = workspace.image.values(task.block_id)
        fresh = run_block_task(
            task,
            ast=workspace.bundle.query_asts[task.query_id],
            image=workspace.image,
        )
        fresh_payloads.append(fresh)
        expected_stats = {
            "scanned_rows": len(values),
            "matched_rows": len(fresh["matched_row_indices"]),
            "min_value": min(values) if values else None,
            "max_value": max(values) if values else None,
            "sum_value": sum(values),
        }
        comparisons = [
            ("completed_rows", int(row["completed_rows"]), len(values)),
            (
                "decode_memory_bytes",
                int(row["decode_memory_bytes"]),
                workspace.image.blocks[task.block_id].pre_bytes,
            ),
            (
                "block_digest",
                row["block_digest"],
                workspace.image.blocks[task.block_id].block_digest,
            ),
            ("result_digest", row["result_digest"], fresh["result_digest"]),
            (
                "result_json",
                canonical_json(row["result_json"]),
                canonical_json(fresh),
            ),
            (
                "stats",
                canonical_json(row["stats_json"]),
                canonical_json(expected_stats),
            ),
        ]
        for field, ledger_value, replay_value in comparisons:
            if ledger_value != replay_value:
                raise AppError(
                    error_code,
                    "账本逐块结局与段文件权威重算不一致",
                    block_id=task.block_id,
                    block_ordinal=task.block_ordinal,
                    field=field,
                    basis=[
                        f"task_id={task.task_id}",
                        f"ledger={ledger_value}",
                        f"replay={replay_value}",
                    ],
                    details={
                        "ledger_value": ledger_value,
                        "replay_value": replay_value,
                    },
                )
        checked_stats.append({"block_ordinal": task.block_ordinal, **expected_stats})
    expected_ids = {task.task_id for task in workspace.bundle.tasks}
    extra_ids = sorted(set(ledger_rows) - expected_ids)
    if extra_ids:
        row = ledger_rows[extra_ids[0]]
        raise AppError(
            error_code,
            "账本包含扫描序列之外的块结局",
            block_id=row["block_id"],
            block_ordinal=int(row["block_ordinal"]),
            field="ledger_task_id",
            basis=[f"task_id={extra_ids[0]}"],
        )
    checked_queries = 0
    if compare_query_results:
        recomputed = compute_query_results(
            workspace.spec,
            workspace.bundle,
            fresh_payloads,
            decoded_row_values=decoded_row_values,
        )
        ledger_queries = {
            (row["query_id"], row["row_group_id"]): row
            for row in workspace.ledger.current_query_results()
        }
        for key, fresh_result in sorted(recomputed.items()):
            row = ledger_queries.get(key)
            ordinal = _first_ordinal_for_query_group(workspace, key[0], key[1])
            if row is None:
                raise AppError(
                    error_code,
                    "账本缺少查询行组结论",
                    block_ordinal=ordinal,
                    field="query_result_presence",
                    basis=[f"query_id={key[0]}", f"row_group_id={key[1]}"],
                )
            if row["result_digest"] != fresh_result["result_digest"]:
                raise AppError(
                    error_code,
                    "查询行组结果摘要与权威重算不一致",
                    block_ordinal=ordinal,
                    field="query_result_digest",
                    basis=[
                        f"query_id={key[0]}",
                        f"row_group_id={key[1]}",
                        f"ledger={row['result_digest']}",
                        f"replay={fresh_result['result_digest']}",
                    ],
                    details={
                        "ledger_value": row["result_digest"],
                        "replay_value": fresh_result["result_digest"],
                    },
                )
            if row["rows"] != fresh_result["rows"]:
                raise AppError(
                    error_code,
                    "查询行集合与权威重算不一致",
                    block_ordinal=ordinal,
                    field="query_result_rows",
                    basis=[f"query_id={key[0]}", f"row_group_id={key[1]}"],
                )
            checked_queries += 1
        extra_queries = sorted(set(ledger_queries) - set(recomputed))
        if extra_queries:
            key = extra_queries[0]
            raise AppError(
                error_code,
                "账本包含计划之外的查询结论",
                block_ordinal=_first_ordinal_for_query_group(workspace, key[0], key[1]),
                field="query_result_extra",
                basis=[f"query_id={key[0]}", f"row_group_id={key[1]}"],
            )
    return {
        "ok": True,
        "checked_blocks": len(workspace.bundle.tasks),
        "checked_query_results": checked_queries,
        "checked_rows": sum(item["scanned_rows"] for item in checked_stats),
        "matched_rows": sum(item["matched_rows"] for item in checked_stats),
        "replay_digest": sha256_json(
            [
                {
                    "block_ordinal": item["block_ordinal"],
                    "result_digest": fresh["result_digest"],
                }
                for item, fresh in zip(checked_stats, fresh_payloads)
            ]
        ),
        "first_block_ordinal": (
            min((task.block_ordinal for task in workspace.bundle.tasks), default=None)
        ),
        "last_block_ordinal": (
            max((task.block_ordinal for task in workspace.bundle.tasks), default=None)
        ),
        "basis": "逐块从 segment.seg 解码并独立执行谓词，未读取账本摘要作为输入",
    }
