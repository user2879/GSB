"""定点 Gather 引擎：双路径计算、误差追踪、失效账本、预算调度与快照。"""
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import config_hash, gear_key
from .errors import (
    E_ID_MISSING,
    E_POINT_BUDGET_PENDING,
    failure,
    sort_key,
)
from .quantize import (
    accumulator_bounds,
    dequantize,
    divide_mean,
    quantize_row,
    saturating_add,
)
from .storage import all_block_fingerprints, block_layout_fingerprint, row_fingerprint


STAGES = ("quantize", "gather", "accumulate", "divide", "round")


def stable_task_key(request_id, stage_name):
    return f"{request_id}::{stage_name}"


def _max_abs_rel(diff, reference):
    abs_err = float(np.max(np.abs(diff)))
    denom = np.maximum(np.abs(reference), 1e-12)
    rel_err = float(np.max(np.abs(diff) / denom))
    return abs_err, rel_err


class QuantCache:
    def __init__(self, gear, headroom):
        self.gear = gear
        self.headroom = headroom
        self.level = gear["quant_level"]
        self.fbits = gear["fractional_bits"]
        self.rows = {}

    def get(self, table, row_id):
        entry = self.rows.get(row_id)
        if entry is not None:
            return entry
        row = table.rows[row_id]
        int_vector, step, residual = quantize_row(row.vector, self.level, self.fbits)
        entry = {
            "row_id": row_id,
            "block_id": row.block_id,
            "int": int_vector,
            "step": float(step),
            "residual": residual,
            "content_fp": None,
            "row_fingerprint": row_fingerprint(row.vector, int_vector, self.level, table.dim),
        }
        self.rows[row_id] = entry
        return entry

    def retain_verified(self, table):
        kept = {}
        for rid, item in self.rows.items():
            if rid not in table.rows:
                continue
            fresh, _, _ = quantize_row(
                table.rows[rid].vector, self.level, self.fbits
            )
            if np.array_equal(fresh, item["int"]):
                kept[rid] = item
        self.rows = kept


class GatherEngine:
    def __init__(self, config, snapshot_dir=None):
        self.config = config
        self.snapshot_dir = snapshot_dir
        self.manifest = None
        self.quant_cache = None
        self.block_fps = {}
        self.layout_fp = None
        self.request_cache = {}
        self.reuse_ledger = {"reused": [], "recomputed": []}
        self.global_version = 0
        self._load_snapshot()

    # ------------------------------------------------------------------ 快照
    def _load_snapshot(self):
        if not self.snapshot_dir:
            return
        manifest_path = os.path.join(self.snapshot_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("config_hash") != config_hash(self.config):
            return
        quant_path = os.path.join(self.snapshot_dir, "quant_cache.json")
        with open(quant_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        gear = payload["gear"]
        cache = QuantCache(gear, self.config["accumulator_headroom_bits"])
        for rid, item in payload["rows"].items():
            cache.rows[int(rid)] = {
                "row_id": int(rid),
                "block_id": item["block_id"],
                "int": np.asarray(item["int"], dtype=np.int64),
                "step": item["step"],
                "residual": np.asarray(item["residual"], dtype=np.float64),
                "content_fp": None,
                "row_fingerprint": item["row_fingerprint"],
            }
        with open(os.path.join(self.snapshot_dir, "request_cache.json"), "r", encoding="utf-8") as handle:
            self.request_cache = json.load(handle)
        self.quant_cache = cache
        self.manifest = manifest
        self.global_version = manifest["global_version"]

    def _atomic_commit(self, table, stage_versions):
        if not self.snapshot_dir:
            return
        parent = os.path.dirname(os.path.abspath(self.snapshot_dir))
        os.makedirs(parent, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix=".egw-snap-", dir=parent)
        quant_payload = {
            "gear": self.quant_cache.gear,
            "rows": {
                str(rid): {
                    "block_id": item["block_id"],
                    "int": item["int"].tolist(),
                    "step": item["step"],
                    "residual": item["residual"].tolist(),
                    "row_fingerprint": item["row_fingerprint"],
                }
                for rid, item in sorted(self.quant_cache.rows.items())
            },
        }
        with open(os.path.join(tmp_dir, "quant_cache.json"), "w", encoding="utf-8") as handle:
            json.dump(quant_payload, handle, sort_keys=True, ensure_ascii=False)
        with open(os.path.join(tmp_dir, "request_cache.json"), "w", encoding="utf-8") as handle:
            json.dump(self.request_cache, handle, sort_keys=True, ensure_ascii=False)
        manifest = {
            "global_version": self.global_version,
            "config_hash": config_hash(self.config),
            "quant_gear": self.quant_cache.gear,
            "block_layout_fingerprint": self.layout_fp,
            "stage_versions": stage_versions,
        }
        with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, ensure_ascii=False, indent=2)
        if os.path.exists(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir)
        os.replace(tmp_dir, self.snapshot_dir)
        self.manifest = manifest

    # ------------------------------------------------------------------ 量化
    def prepare_quantization(self, table, gear):
        target_key = gear_key(gear["quant_level"], gear["fractional_bits"])
        if self.quant_cache is None or gear_key(self.quant_cache.level, self.quant_cache.fbits) != target_key:
            self.quant_cache = QuantCache(gear, self.config["accumulator_headroom_bits"])
        else:
            self.quant_cache.retain_verified(table)
        for row_id in sorted(table.rows):
            self.quant_cache.get(table, row_id)
        self.block_fps = all_block_fingerprints(table, self.quant_cache)
        self.layout_fp = block_layout_fingerprint(self.block_fps)

    # ------------------------------------------------------------------ 失效
    def invalidate_for_edit(self, table, edit, changed_row_ids, changed_block_ids, gear_changed):
        if gear_changed:
            self.request_cache.clear()
            return
        invalid = set()
        for request_id, item in self.request_cache.items():
            refs_rows = set(item["references"]["row_ids"])
            refs_blocks = set(item["references"]["block_ids"])
            if refs_rows.intersection(changed_row_ids) or refs_blocks.intersection(changed_block_ids):
                invalid.add(request_id)
        for request_id in invalid:
            del self.request_cache[request_id]

    # ----------------------------------------------------- 定点主路径 Gather
    def run_gather(self, table, request, cache=None):
        """对单个请求执行完整定点路径；cache 给定时复用量化行。"""
        cache = cache or self.quant_cache
        cfg = self.config
        dim = table.dim
        acc_min, acc_max = accumulator_bounds(cache.level, cfg["accumulator_headroom_bits"])
        gathered_int = []
        gathered_float = []
        block_partials = {}
        block_traversal = []
        missing = []
        found_positions = []
        quant_residuals = []
        acc = np.zeros(dim, dtype=np.int64)
        float_acc = np.zeros(dim, dtype=np.float64)

        for position, row_id in enumerate(request.id_sequence):
            if row_id not in table.rows:
                missing.append(
                    failure(
                        E_ID_MISSING,
                        f"词 ID {row_id} 在表中不存在",
                        request_id=request.request_id,
                        position=position,
                        row_id=row_id,
                    )
                )
                continue
            entry = cache.get(table, row_id)
            int_row = entry["int"]
            gathered_int.append(int_row)
            float_row = dequantize(int_row, cache.fbits)
            gathered_float.append(float_row)
            block_id = entry["block_id"]
            if block_id not in block_partials:
                block_partials[block_id] = np.zeros(dim, dtype=np.int64)
                block_traversal.append(block_id)
            acc = saturating_add(acc, int_row, acc_min, acc_max)
            block_partials[block_id] = saturating_add(
                block_partials[block_id], int_row, acc_min, acc_max
            )
            float_acc = float_acc + float_row
            found_positions.append(position)
            quant_residuals.append(entry["residual"])

        found_n = len(found_positions)
        if found_n == 0:
            mean_q = np.zeros(dim, dtype=np.int64)
            remainder = np.zeros(dim, dtype=np.int64)
        elif request.agg_mode == "SUM":
            mean_q = acc
            remainder = np.zeros(dim, dtype=np.int64)
        else:
            mean_q, remainder = divide_mean(acc, found_n, cfg["rounding"])

        final_float = dequantize(mean_q, cache.fbits)

        # float64 参考路径：原始浮点输入，按相同顺序累加，均值用浮点除法
        ref_acc = np.zeros(dim, dtype=np.float64)
        for position, row_id in enumerate(request.id_sequence):
            if row_id in table.rows:
                ref_acc = ref_acc + table.rows[row_id].vector
        if request.agg_mode == "MEAN" and found_n > 0:
            ref_final = ref_acc / found_n
        else:
            ref_final = ref_acc

        # 每块部分和与最终和差值（反量化后 float64 域观测）
        block_report = []
        for block_id in block_traversal:
            partial_float = dequantize(block_partials[block_id], cache.fbits)
            block_report.append(
                {
                    "block_id": block_id,
                    "partial_sum_int": block_partials[block_id].tolist(),
                    "partial_sum_float": partial_float.tolist(),
                    "diff_vs_final_abs": float(np.max(np.abs(partial_float - final_float))),
                }
            )
        block_report.sort(key=lambda item: (-item["diff_vs_final_abs"], item["block_id"]))

        # ---------------- 五环节误差（float64 参考逐环节比对）
        ref_gathered = []
        for row_id in request.id_sequence:
            if row_id in table.rows:
                ref_gathered.append(table.rows[row_id].vector.astype(np.float64))

        def stage_metrics(diff, reference):
            abs_err, rel_err = _max_abs_rel(diff, reference)
            return {"max_abs_error": abs_err, "max_rel_error": rel_err}

        stage_errors = {}
        # 1 行量化：逐取行行最大量化误差
        if gathered_int:
            q_diffs = np.concatenate(
                [q - r for q, r in zip(gathered_float, ref_gathered)]
            )
            ref_flat = np.concatenate(ref_gathered)
            stage_errors["quantize"] = stage_metrics(q_diffs, ref_flat)
        else:
            stage_errors["quantize"] = {"max_abs_error": 0.0, "max_rel_error": 0.0}

        # 2 取行：取行不改变数值，误差等于量化误差
        if gathered_float:
            g_diffs = np.concatenate(
                [q - r for q, r in zip(gathered_float, ref_gathered)]
            )
            stage_errors["gather"] = stage_metrics(g_diffs, np.concatenate(ref_gathered))
        else:
            stage_errors["gather"] = {"max_abs_error": 0.0, "max_rel_error": 0.0}

        # 3 定点累加（含饱和）vs float64 参考累加
        stage_errors["accumulate"] = stage_metrics(
            dequantize(acc, cache.fbits) - ref_acc, ref_acc
        )

        # 4 均值除法 / 5 最终舍入
        if request.agg_mode == "MEAN" and found_n > 0:
            ref_mean_exact = ref_acc / found_n
            trunc_quotient = np.trunc(acc / found_n).astype(np.int64)
            stage_errors["divide"] = stage_metrics(
                dequantize(trunc_quotient, cache.fbits) - ref_mean_exact, ref_mean_exact
            )
            stage_errors["round"] = stage_metrics(final_float - ref_final, ref_final)
        else:
            stage_errors["divide"] = None
            stage_errors["round"] = None

        tolerance = cfg["tolerance"]
        first_breach = None
        for stage in STAGES:
            metrics = stage_errors[stage]
            if metrics is not None and metrics["max_abs_error"] > tolerance:
                first_breach = stage
                break

        # ---------------- 误差归因与高贡献位置
        position_contrib = []
        for idx, position in enumerate(found_positions):
            row_id = request.id_sequence[position]
            quant_l1 = float(np.sum(np.abs(quant_residuals[idx])))
            position_contrib.append(
                {
                    "position": position,
                    "row_id": row_id,
                    "quant_residual_l1": quant_l1,
                    "abs_int_row_sum": int(np.sum(np.abs(gathered_int[idx]))),
                }
            )
        rounding_remainder_l1 = float(np.sum(np.abs(remainder)))
        quant_residual_l1_total = float(
            sum(item["quant_residual_l1"] for item in position_contrib)
        )
        if request.agg_mode == "MEAN" and found_n > 0:
            if rounding_remainder_l1 > 0 and rounding_remainder_l1 >= quant_residual_l1_total:
                root_cause = "accumulate_rounding"
            else:
                root_cause = "quantization_truncation"
        else:
            root_cause = (
                "quantization_truncation" if quant_residual_l1_total > 0 else "none"
            )
        position_contrib.sort(
            key=lambda item: (-item["quant_residual_l1"], item["position"], item["row_id"])
        )
        top_positions = position_contrib[: cfg["top_k"]]

        referenced_rows = sorted(
            {request.id_sequence[p] for p in found_positions}
        )
        referenced_blocks = sorted({table.rows[rid].block_id for rid in referenced_rows})
        row_fp_set = sorted(cache.get(table, rid)["row_fingerprint"] for rid in referenced_rows)
        block_fp_set = sorted(self.block_fps[bid] for bid in referenced_blocks)

        return {
            "request_id": request.request_id,
            "agg_mode": request.agg_mode,
            "gear": {"quant_level": cache.level, "fractional_bits": cache.fbits},
            "found_count": found_n,
            "missing_count": len(missing),
            "gathered_int": [row.tolist() for row in gathered_int],
            "accum_int": acc.tolist(),
            "result_int": mean_q.tolist(),
            "result_float": final_float.tolist(),
            "reference_float": ref_final.tolist(),
            "remainder_int": remainder.tolist(),
            "stage_errors": stage_errors,
            "first_breach_stage": first_breach,
            "root_cause": root_cause,
            "quant_residual_l1_total": quant_residual_l1_total,
            "rounding_remainder_l1": rounding_remainder_l1,
            "top_positions": top_positions,
            "block_traversal": block_traversal,
            "block_partials": block_report,
            "references": {
                "row_ids": referenced_rows,
                "block_ids": referenced_blocks,
                "row_fingerprints": row_fp_set,
                "block_fingerprints": block_fp_set,
            },
            "task_key": stable_task_key(request.request_id, "gather"),
            "missing_failures": missing,
        }

    # --------------------------------------------------------------- 执行编排
    def run(self, table, requests, force_rebuild=False, concurrency_override=None):
        cfg = self.config
        gear = {
            "quant_level": cfg["quant_level"],
            "fractional_bits": cfg["fractional_bits"],
        }
        gear_changed = (
            self.quant_cache is None
            or self.quant_cache.gear != gear
        )
        if force_rebuild:
            self.quant_cache = None
            self.request_cache = {}
            self.global_version = 0
            gear_changed = True

        self.prepare_quantization(table, gear)

        ordered = sorted(requests, key=lambda item: str(item.request_id))
        reusable = {}
        todo = []
        for request in ordered:
            key = str(request.request_id)
            cached = self.request_cache.get(key)
            if (
                cached is not None
                and cached["agg_mode"] == request.agg_mode
                and cached["gear"] == gear
                and self._references_alive(table, cached)
            ):
                reusable[key] = cached
            else:
                todo.append(request)

        budget = cfg["point_budget"]
        pending = []
        runnable = []
        remaining = budget
        for request in todo:
            cost = len(request.id_sequence)
            if cost <= remaining:
                runnable.append(request)
                remaining -= cost
            else:
                pending.append(request)
        if remaining < 0:
            raise RuntimeError("预算调度内部错误")

        concurrency = concurrency_override or cfg["concurrency"]
        results_by_id = dict(reusable)
        failures = []
        recomputed_ids = []

        def _execute(request):
            result = self.run_gather(table, request)
            return request, result

        if concurrency == 1:
            for request in runnable:
                _, result = _execute(request)
                results_by_id[str(request.request_id)] = result
                recomputed_ids.append(str(request.request_id))
                failures.extend(result["missing_failures"])
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_execute, request) for request in runnable]
                gathered = [future.result() for future in futures]
            gathered.sort(key=lambda pair: str(pair[0].request_id))
            for request, result in gathered:
                results_by_id[str(request.request_id)] = result
                recomputed_ids.append(str(request.request_id))
                failures.extend(result["missing_failures"])

        pending_ids = []
        for request in pending:
            failures.append(
                failure(
                    E_POINT_BUDGET_PENDING,
                    f"请求需要 {len(request.id_sequence)} 点，超出单批预算 {budget} 后剩余额度，保留待办",
                    request_id=request.request_id,
                    required_points=len(request.id_sequence),
                    batch_budget=budget,
                )
            )
            pending_ids.append(str(request.request_id))
            results_by_id[str(request.request_id)] = {
                "request_id": request.request_id,
                "agg_mode": request.agg_mode,
                "gear": {"quant_level": self.quant_cache.level,
                         "fractional_bits": self.quant_cache.fbits},
                "found_count": 0,
                "missing_count": 0,
                "gathered_int": [],
                "accum_int": [0] * table.dim,
                "result_int": [0] * table.dim,
                "result_float": [0.0] * table.dim,
                "reference_float": [0.0] * table.dim,
                "remainder_int": [0] * table.dim,
                "stage_errors": {stage: None for stage in STAGES},
                "first_breach_stage": None,
                "root_cause": "pending",
                "quant_residual_l1_total": 0.0,
                "rounding_remainder_l1": 0.0,
                "top_positions": [],
                "block_traversal": [],
                "block_partials": [],
                "references": {"row_ids": [], "block_ids": [],
                               "row_fingerprints": [], "block_fingerprints": []},
                "task_key": stable_task_key(request.request_id, "gather"),
                "missing_failures": [],
                "pending": True,
            }

        recomputed_ids.sort(key=str)
        reused_ids = sorted(
            (rid for rid in reusable.keys() if rid not in pending_ids), key=str
        )
        pending_ids.sort(key=str)
        self.reuse_ledger = {"reused": reused_ids, "recomputed": recomputed_ids}

        for key in recomputed_ids:
            self.request_cache[key] = results_by_id[key]

        results = [results_by_id[str(req.request_id)] for req in ordered]
        failures.sort(key=sort_key)

        self.global_version += 1
        stage_versions = {
            stage: self._stage_version(stage, results) for stage in STAGES
        }
        self._atomic_commit(table, stage_versions)

        return {
            "global_version": self.global_version,
            "results": results,
            "failures": failures,
            "reuse_ledger": self.reuse_ledger,
            "stage_versions": stage_versions,
            "pending_request_ids": pending_ids,
            "concurrency_used": concurrency,
            "batch_budget": budget,
            "budget_consumed": budget - remaining,
        }

    def _references_alive(self, table, cached):
        for row_id in cached["references"]["row_ids"]:
            if row_id not in table.rows:
                return False
        for block_id in cached["references"]["block_ids"]:
            if block_id not in table.block_offsets:
                return False
        return True

    def _stage_version(self, stage, results):
        digest = hashlib.sha256()
        digest.update(stage.encode("utf-8"))
        for result in sorted(results, key=lambda item: str(item["request_id"])):
            digest.update(stable_task_key(result["request_id"], stage).encode("utf-8"))
            digest.update(json.dumps(result["stage_errors"].get(stage), sort_keys=True).encode("utf-8"))
            digest.update(str(result["result_int"]).encode("utf-8"))
        return digest.hexdigest()

    # ------------------------------------------------- 双档位对比与布局探针
    def compare_gears(self, table, request, alternate_gear):
        primary = self.run_gather(table, request)
        alt_cache = QuantCache(alternate_gear, self.config["accumulator_headroom_bits"])
        for row_id in sorted(table.rows):
            alt_cache.get(table, row_id)
        alt_engine_view = _AltEngineView(self, alt_cache, table)
        alternate = alt_engine_view.run_gather(table, request)
        primary_arr = np.asarray(primary["result_float"], dtype=np.float64)
        alternate_arr = np.asarray(alternate["result_float"], dtype=np.float64)
        discrepancy = float(np.max(np.abs(primary_arr - alternate_arr)))
        judgement_changed = (
            primary["first_breach_stage"] != alternate["first_breach_stage"]
        )
        return {
            "request_id": request.request_id,
            "primary_gear": primary["gear"],
            "alternate_gear": alternate["gear"],
            "primary_result_float": primary["result_float"],
            "alternate_result_float": alternate["result_float"],
            "max_abs_diff": discrepancy,
            "judgement_changed": judgement_changed,
            "primary_first_breach": primary["first_breach_stage"],
            "alternate_first_breach": alternate["first_breach_stage"],
            "primary_root_cause": primary["root_cause"],
            "alternate_root_cause": alternate["root_cause"],
            "top_positions": primary["top_positions"],
        }


class _AltEngineView(GatherEngine):
    """复用主引擎块指纹，仅替换量化档位的轻量视图。"""

    def __init__(self, parent, cache, table):
        self.config = parent.config
        self.snapshot_dir = None
        self.manifest = None
        self.quant_cache = cache
        self.block_fps = all_block_fingerprints(table, cache)
        self.layout_fp = parent.layout_fp
        self.request_cache = {}
        self.reuse_ledger = {"reused": [], "recomputed": []}
        self.global_version = parent.global_version
