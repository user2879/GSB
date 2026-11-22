"""场景编排：加载输入、应用编辑序列、增量与全量运行、报告组装。"""
import copy
import json

from .config import config_hash
from .engine import GatherEngine
from .errors import E_CONFIG_INVALID, EGWError, sort_key
from .storage import apply_edit


def load_edits(path):
    edits = []
    if not path:
        return edits
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                edits.append(json.loads(line))
    return edits


def apply_state_edits(table, edits, config):
    """仅应用改变表结构/内容的编辑，返回 (表, 配置, 请求列表变更, 每步元数据)。"""
    cfg = copy.deepcopy(config)
    table = table.copy()
    agg_overrides = {}
    step_meta = []
    for index, edit in enumerate(edits):
        op = edit["op"]
        if op in ("replace_row", "replace_block", "append_block"):
            before_rows = set(table.rows)
            before_blocks = {bid: dict(m) for bid, m in table.block_offsets.items()}
            table = apply_edit(table, edit)
            changed_rows = set(table.rows) ^ before_rows
            if op == "replace_row":
                changed_rows.add(edit["row_id"])
            changed_blocks = set()
            for bid in set(table.block_offsets) | set(before_blocks):
                if before_blocks.get(bid) != table.block_offsets.get(bid):
                    changed_blocks.add(bid)
            step_meta.append(
                {
                    "index": index,
                    "op": op,
                    "changed_row_ids": sorted(changed_rows),
                    "changed_block_ids": sorted(changed_blocks),
                    "gear_changed": False,
                }
            )
        elif op == "switch_gear":
            cfg["quant_level"] = edit["quant_level"]
            cfg["fractional_bits"] = edit["fractional_bits"]
            step_meta.append(
                {
                    "index": index,
                    "op": op,
                    "changed_row_ids": [],
                    "changed_block_ids": [],
                    "gear_changed": True,
                }
            )
        elif op == "switch_bits":
            cfg["fractional_bits"] = edit["fractional_bits"]
            step_meta.append(
                {
                    "index": index,
                    "op": op,
                    "changed_row_ids": [],
                    "changed_block_ids": [],
                    "gear_changed": True,
                }
            )
        elif op == "switch_agg":
            agg_overrides[edit["request_id"]] = edit["agg_mode"]
            step_meta.append(
                {
                    "index": index,
                    "op": op,
                    "changed_row_ids": [],
                    "changed_block_ids": [],
                    "gear_changed": False,
                    "agg_switch": edit,
                }
            )
        else:
            raise EGWError(E_CONFIG_INVALID, f"未知编辑命令类型 {op}")
    return table, cfg, agg_overrides, step_meta


def apply_agg_overrides(requests, overrides):
    result = []
    for request in requests:
        if request.request_id in overrides:
            mode = overrides[request.request_id]
            if mode not in [request.agg_mode]:
                request.agg_mode = mode
        result.append(request)
    return requests


def build_report(config, run_outcome, load_failures, layout_probe_result, gear_comparisons):
    cfg_hash = config_hash(config)
    results = []
    for result in run_outcome["results"]:
        results.append(
            {
                "request_id": result["request_id"],
                "agg_mode": result["agg_mode"],
                "gear": result["gear"],
                "found_count": result["found_count"],
                "missing_count": result["missing_count"],
                "result_int": result["result_int"],
                "result_float": result["result_float"],
                "reference_float": result["reference_float"],
                "remainder_int": result["remainder_int"],
                "stage_errors": result["stage_errors"],
                "first_breach_stage": result["first_breach_stage"],
                "root_cause": result["root_cause"],
                "quant_residual_l1_total": result["quant_residual_l1_total"],
                "rounding_remainder_l1": result["rounding_remainder_l1"],
                "top_positions": result["top_positions"],
                "block_traversal": result["block_traversal"],
                "block_partials": result["block_partials"],
                "references": result["references"],
                "task_key": result["task_key"],
            }
        )

    all_failures = list(load_failures) + list(run_outcome["failures"])
    all_failures.sort(key=sort_key)

    return {
        "manifest": {
            "global_version": run_outcome["global_version"],
            "config_hash": cfg_hash,
            "quant_gear": {
                "quant_level": config["quant_level"],
                "fractional_bits": config["fractional_bits"],
            },
            "block_layout_fingerprint": run_outcome.get("block_layout_fingerprint"),
            "stage_versions": run_outcome["stage_versions"],
            "concurrency_used": run_outcome["concurrency_used"],
            "batch_budget": run_outcome["batch_budget"],
            "budget_consumed": run_outcome["budget_consumed"],
        },
        "reuse_ledger": run_outcome["reuse_ledger"],
        "pending_request_ids": run_outcome["pending_request_ids"],
        "requests": results,
        "failures": all_failures,
        "gear_comparison": gear_comparisons,
        "layout_probe": layout_probe_result,
        "error_codes": {
            "E_ROW_INVALID": "行记录非法（解析失败/字段缺失/重复ID/槽位冲突/负坐标）",
            "E_DIM_MISMATCH": "行向量维度与配置不一致",
            "E_NONFINITE_VECTOR": "向量含 NaN/Inf 非有限元素",
            "E_AGG_MODE_UNKNOWN": "聚合模式不在配置枚举内",
            "E_CONFIG_INVALID": "配置或编辑命令非法",
            "E_ID_MISSING": "Gather 序列中的词 ID 在表中不存在",
            "E_POINT_BUDGET_PENDING": "单批行访问点数预算不足，请求保留待办",
        },
    }


def run_full(config, table, requests, snapshot_dir, concurrency_override=None,
             force_rebuild=True, expected_version=None):
    engine = GatherEngine(config, snapshot_dir=snapshot_dir)
    outcome = engine.run(
        table,
        requests,
        force_rebuild=force_rebuild,
        concurrency_override=concurrency_override,
    )
    if expected_version is not None:
        outcome["global_version"] = expected_version
        engine.global_version = expected_version
    outcome["block_layout_fingerprint"] = engine.layout_fp
    return engine, outcome


def run_incremental(config, base_table, base_requests, edits, snapshot_dir, concurrency_override=None):
    """每条编辑后增量运行，返回 (最终引擎, 最终结果, 逐步账本)。"""
    cfg = copy.deepcopy(config)
    table = base_table.copy()
    requests = [copy.copy(req) for req in base_requests]
    engine = GatherEngine(cfg, snapshot_dir=snapshot_dir)
    ledgers = []
    outcome = engine.run(table, requests, force_rebuild=True, concurrency_override=concurrency_override)
    ledgers.append(
        {
            "step": -1,
            "op": "baseline",
            "reuse_ledger": outcome["reuse_ledger"],
            "failures": outcome["failures"],
            "global_version": outcome["global_version"],
        }
    )
    for index, edit in enumerate(edits):
        op = edit["op"]
        if op in ("replace_row", "replace_block", "append_block"):
            before_rows = set(table.rows)
            table = apply_edit(table, edit)
            if op == "replace_row":
                changed_rows = {edit["row_id"]}
            else:
                changed_rows = set(table.rows) ^ before_rows
            changed_blocks = {edit["block_id"]}
            block_scope = set() if op == "replace_row" else changed_blocks
            engine.invalidate_for_edit(table, edit, changed_rows, block_scope, False)
        elif op in ("switch_gear", "switch_bits"):
            if op == "switch_gear":
                cfg["quant_level"] = edit["quant_level"]
            cfg["fractional_bits"] = edit["fractional_bits"]
            engine.config = cfg
            engine.invalidate_for_edit(table, edit, [], [], True)
        elif op == "switch_agg":
            for request in requests:
                if request.request_id == edit["request_id"]:
                    request.agg_mode = edit["agg_mode"]
            engine.request_cache.pop(edit["request_id"], None)
        else:
            raise EGWError(E_CONFIG_INVALID, f"未知编辑命令类型 {op}")
        outcome = engine.run(table, requests, concurrency_override=concurrency_override)
        ledgers.append(
            {
                "step": index,
                "op": op,
                "reuse_ledger": outcome["reuse_ledger"],
                "failures": outcome["failures"],
                "global_version": outcome["global_version"],
            }
        )
    outcome["block_layout_fingerprint"] = engine.layout_fp
    return engine, outcome, ledgers, table, cfg, requests


def build_gear_comparisons(engine, table, requests):
    alt = engine.config.get("alternate_gear")
    if not alt:
        return []
    comparisons = []
    for request in sorted(requests, key=lambda item: str(item.request_id)):
        comparisons.append(engine.compare_gears(table, request, alt))
    return comparisons
