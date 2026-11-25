"""End-to-end deterministic quantization + distillation calibration pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import cache as cache_mod
from . import calibration as cal_mod
from . import distill as dist_mod
from . import quantize as qnt_mod
from . import receipts as rcp_mod
from . import scheduler as sched_mod
from .errors import FailureList
from .hashing import fingerprint
from .models import CalibrationSet, QuantConfig, WeightLayer
from .snapshot import SnapshotStore


@dataclass
class PipelineInputs:
    layers: list[WeightLayer]
    calibration: CalibrationSet
    config: QuantConfig
    receipt_entries: list[rcp_mod.ReceiptEntry]
    input_failures: list[dict[str, Any]] | None = None


def config_fingerprint(config: QuantConfig) -> str:
    material = dict(config.defaults)
    material.update({
        "layers": config.layers,
        "distill": config.distill.__dict__,
        "workers": config.workers,
        "tiers": config.tiers,
        "default_tier": config.default_tier,
        "layer_tiers": config.layer_tiers,
        "receipts_dir": config.receipts_dir,
        "budgets": config.budgets.__dict__,
    })
    return fingerprint(material, prefix="cfg-")


def _layer_cal_outputs(layer: WeightLayer, cfg_material, groups,
                       calibrations, sample_inputs: np.ndarray,
                       failures: FailureList) -> np.ndarray:
    """Outputs after quantile calibration only (continuous scale/zero)."""
    from .models import LayerQuantConfig
    cfg = LayerQuantConfig(**cfg_material)
    flat = np.ascontiguousarray(layer.weights, dtype=np.float64).ravel()
    calibrated = np.empty_like(flat)
    for index, calibration in enumerate(calibrations):
        start = index * cfg.group_size
        end = start + cfg.group_size
        segment = flat[start:end]
        grid = 2.0 ** cfg.frac_bits
        # Continuous quantile mapping: no integer rounding, no clipping yet.
        mapped = segment / calibration.scale + calibration.zero
        calibrated[start:end] = (mapped - calibration.zero) * calibration.scale
    weights = calibrated.reshape(layer.weights.shape)
    return dist_mod.forward(sample_inputs, weights)


def _reference_path(layer: WeightLayer, base_cfg_material: dict[str, Any],
                    sample_inputs: np.ndarray,
                    failures: FailureList) -> np.ndarray:
    from .models import LayerQuantConfig
    base = LayerQuantConfig(**base_cfg_material)
    ref_cfg = dist_mod.reference_config(base)
    whole_group = layer.weights.ravel()
    ref_cfg = LayerQuantConfig(
        name=ref_cfg.name, bits=ref_cfg.bits, level=ref_cfg.level,
        group_size=layer.size, q_lo=ref_cfg.q_lo, q_hi=ref_cfg.q_hi,
        frac_bits=ref_cfg.frac_bits, scheme=ref_cfg.scheme,
        rounding=ref_cfg.rounding, saturation=ref_cfg.saturation)
    calibration = cal_mod.calibrate_group(0, whole_group, ref_cfg, failures)
    result = qnt_mod.quantize_group(whole_group, calibration, ref_cfg)
    weights = np.asarray(result.dequant, dtype=np.float64).reshape(layer.weights.shape)
    return dist_mod.forward(sample_inputs, weights)


class Pipeline:
    def __init__(self, store: SnapshotStore) -> None:
        self.store = store

    def run(self, inputs: PipelineInputs,
            previous_cache: dict[str, Any] | None = None) -> dict[str, Any]:
        failures = FailureList()
        previous_cache = previous_cache or {}
        self.workers = inputs.config.workers
        config_hash = config_fingerprint(inputs.config)
        content_versions = {layer.name: layer.content_version
                            for layer in inputs.layers}

        receipt_ledger = rcp_mod.build_receipt_ledger(inputs.receipt_entries)
        layer_artifacts: dict[str, dict[str, Any]] = {}
        reuse = cache_mod.ReuseLedger()
        new_cache: dict[str, Any] = {"layers": {}}

        tasks: list[sched_mod.Task] = []
        for layer in sorted(inputs.layers, key=lambda lyr: lyr.name):
            cfg = inputs.config.resolve_layer(layer.name)
            task_count = layer.size // cfg.group_size
            tasks.append(sched_mod.Task(layer.name, -1, task_count,
                                        {"groups": task_count}))
        schedule = sched_mod.schedule(
            tasks,
            inputs.config.budgets.global_cal_points,
            inputs.config.budgets.per_layer_quotas,
            failures,
        )
        allowed_layers = {task.layer_name for task in schedule.completed}

        for layer in sorted(inputs.layers, key=lambda lyr: lyr.name):
            if layer.name not in allowed_layers:
                continue
            artifact = self._process_layer(
                layer, inputs, previous_cache, new_cache, reuse, failures)
            layer_artifacts[layer.name] = artifact

        result = {
            "config_hash": config_hash,
            "content_versions": content_versions,
            "budget": {
                "global_budget": inputs.config.budgets.global_cal_points,
                "batch_quant_points": inputs.config.budgets.batch_quant_points,
                "consumed_cal_points": schedule.budget_consumed,
                "quota_aborted_layers": schedule.quota_aborted,
                "pending_tasks": [
                    {"layer": t.layer_name, "group_index": t.group_index,
                     "cost": t.cost} for t in schedule.pending],
                "quant_batches": {
                    layer.name: sched_mod.batch_quant_budget(
                        layer.size,
                        inputs.config.budgets.batch_quant_points)
                    for layer in inputs.layers},
            },
            "layers": layer_artifacts,
            "receipt_ledger": receipt_ledger,
            "reuse_ledger": reuse.to_list(),
            "canonical_invalidation_ledger": reuse.canonical_invalidation(),
            "failures": sorted(
                failures.to_list() + (inputs.input_failures or []),
                key=lambda item: (item["code"], item["scope"], item["target"],
                                  item["message"], item["detail"])),
            "cache": new_cache,
            "calibration_metadata": cal_mod.calibration_metadata(),
        }
        return result

    def _process_layer(self, layer: WeightLayer, inputs: PipelineInputs,
                       previous_cache: dict[str, Any],
                       new_cache: dict[str, Any],
                       reuse: cache_mod.ReuseLedger,
                       failures: FailureList) -> dict[str, Any]:
        cfg = inputs.config.resolve_layer(layer.name)
        cfg_material = cfg.to_material()
        groups = cal_mod.split_groups(layer.weights, cfg.group_size)
        prev_layer = previous_cache.get("layers", {}).get(layer.name, {})
        layer_cache: dict[str, Any] = {"keys": {}, "sample_refs": []}
        new_cache["layers"][layer.name] = layer_cache

        group_keys = [cache_mod.group_cache_keys(layer, cfg, group, index)
                      for index, group in enumerate(groups)]
        tier = inputs.config.tier_for(layer.name)
        sample_refs = inputs.calibration.sample_refs(tier)
        layer_cache["sample_refs"] = list(sample_refs)
        cal_fp = cache_mod.calibration_fingerprint(inputs.calibration, sample_refs)
        distill_material = inputs.config.distill.__dict__
        dst_key = cache_mod.distill_cache_key(
            layer, cfg, inputs.calibration, sample_refs, distill_material)
        rcp_material = inputs.config.receipts_dir
        rcp_key = cache_mod.receipt_cache_key(
            {"dir": rcp_material,
             "layer_receipt": inputs.receipt_entries and [
                 e.to_dict() for e in inputs.receipt_entries
                 if e.layer == layer.name]})

        cal_records, qnt_records, group_cache_entries = self._run_groups(
            layer, cfg, groups, group_keys, prev_layer, reuse, failures)
        layer_cache["groups"] = group_cache_entries

        dequant_weights = qnt_mod.assemble_dequant_matrix(
            layer.weights.shape, qnt_records)
        sample_inputs = inputs.calibration.inputs[
            [inputs.calibration.sample_ids.index(sid) for sid in sample_refs]]
        sample_ref = inputs.calibration.references[
            [inputs.calibration.sample_ids.index(sid) for sid in sample_refs]]
        original_out = dist_mod.forward(sample_inputs, layer.weights)
        quant_out = dist_mod.forward(sample_inputs, dequant_weights)
        requant_weights = qnt_mod.requant_only_matrix(
            layer.weights.shape, qnt_records, cal_records, cfg)
        requant_out = dist_mod.forward(sample_inputs, requant_weights)
        cal_out = _layer_cal_outputs(layer, cfg_material, groups,
                                     cal_records, sample_inputs, failures)
        ref_out = _reference_path(layer, cfg_material, sample_inputs, failures)

        prev_dst = prev_layer.get("distill")
        if prev_dst is not None and prev_dst.get("key") == dst_key:
            reuse.hit(layer.name, cache_mod.STAGE_DST, dst_key)
            distill_payload = prev_dst["payload"]
        else:
            reason = self._miss_reason(prev_dst, dst_key, layer, prev_layer)
            reuse.miss(layer.name, cache_mod.STAGE_DST, dst_key, reason,
                       old_key=prev_dst.get("key") if prev_dst else None)
            distill = dist_mod.distill_layer(
                layer, dequant_weights, original_out, sample_inputs,
                inputs.config.distill)
            distill_payload = distill.to_dict()
        layer_cache["distill"] = {"key": dst_key, "payload": distill_payload}

        prev_rcp = prev_layer.get("receipt")
        if prev_rcp is not None and prev_rcp.get("key") == rcp_key:
            reuse.hit(layer.name, cache_mod.STAGE_RCP, rcp_key)
        else:
            reuse.miss(layer.name, cache_mod.STAGE_RCP, rcp_key,
                       "receipt_version_or_dir_changed",
                       old_key=prev_rcp.get("key") if prev_rcp else None)
        layer_cache["receipt"] = {"key": rcp_key}

        corrected_weights = np.asarray(
            distill_payload["corrected_weights"],
            dtype=np.float64).reshape(layer.weights.shape)
        corrected_out = dist_mod.forward(sample_inputs, corrected_weights)
        attribution = dist_mod.stage_attribution(
            original_out, cal_out, requant_out, quant_out, corrected_out,
            ref_out, inputs.config.distill.tolerance)

        error_summary = self._error_summary(qnt_records)
        layer_cache["keys"] = {
            str(index): keys for index, keys in enumerate(group_keys)}
        return {
            "layer": layer.name,
            "content_version": layer.content_version,
            "shape": list(layer.weights.shape),
            "tier": tier,
            "sample_refs": list(sample_refs),
            "calibration_fingerprint": cal_fp,
            "config": cfg_material,
            "group_calibration": [c.to_dict() for c in cal_records],
            "group_quantization": [q.to_dict() for q in qnt_records],
            "dequant_weights": [float(x) for x in dequant_weights.ravel()],
            "error_summary": error_summary,
            "distillation": distill_payload,
            "stage_attribution": attribution,
            "cache_keys": {"distill": dst_key, "receipt": rcp_key,
                           "groups": group_keys},
        }

    def _miss_reason(self, prev, new_key, layer, prev_layer) -> str:
        if prev is None:
            return "first_build"
        return "fingerprint_changed"

    def _run_groups(self, layer, cfg, groups, group_keys, prev_layer,
                    reuse, failures):
        prev_groups = {item["group_index"]: item
                       for item in prev_layer.get("groups", [])}
        cal_records: list = []
        qnt_records: list = []
        group_cache_entries: list[dict[str, Any]] = []

        def work(task: sched_mod.Task):
            index = task.group_index
            group = groups[index]
            keys = group_keys[index]
            prev = prev_groups.get(index)
            cal_key = keys[cache_mod.STAGE_CAL]
            qnt_key = keys[cache_mod.STAGE_QNT]
            if prev is not None and prev.get("cal_key") == cal_key:
                reuse.hit(layer.name, cache_mod.STAGE_CAL, cal_key, index)
                calibration = self._hydrate_calibration(prev["cal_payload"])
            else:
                reuse.miss(layer.name, cache_mod.STAGE_CAL, cal_key,
                           "layer_or_group_content_or_params_changed",
                           old_key=prev.get("cal_key") if prev else None,
                           group_index=index)
                calibration = cal_mod.calibrate_group(index, group, cfg, failures)
            if prev is not None and prev.get("qnt_key") == qnt_key:
                reuse.hit(layer.name, cache_mod.STAGE_QNT, qnt_key, index)
                quant = self._hydrate_quant(prev["qnt_payload"])
                reuse.hit(layer.name, cache_mod.STAGE_ERR, qnt_key, index)
            else:
                reuse.miss(layer.name, cache_mod.STAGE_QNT, qnt_key,
                           "quant_level_frac_bits_or_content_changed",
                           old_key=prev.get("qnt_key") if prev else None,
                           group_index=index)
                quant = qnt_mod.quantize_group(group, calibration, cfg)
                reuse.miss(layer.name, cache_mod.STAGE_ERR,
                           keys[cache_mod.STAGE_ERR],
                           "derived_from_requantized_group",
                           old_key=prev.get("qnt_key") if prev else None,
                           group_index=index)
            return index, calibration, quant, cal_key, qnt_key

        tasks = [sched_mod.Task(layer.name, index, len(group), {"group": index})
                 for index, group in enumerate(groups)]
        results = sched_mod.run_map(work, tasks, self.workers)
        for index, calibration, quant, cal_key, qnt_key in results:
            cal_records.append(calibration)
            qnt_records.append(quant)
            group_cache_entries.append({
                "group_index": index,
                "cal_key": cal_key,
                "qnt_key": qnt_key,
                "cal_payload": calibration.to_dict(),
                "qnt_payload": quant.to_dict(),
            })
        return cal_records, qnt_records, group_cache_entries

    @staticmethod
    def _hydrate_calibration(payload: dict[str, Any]):
        return cal_mod.GroupCalibration(
            group_index=payload["group_index"],
            q_lo_value=payload["q_lo_value"],
            q_hi_value=payload["q_hi_value"],
            scale=payload["scale"],
            zero=payload["zero"],
            symmetric=payload["symmetric"],
            zero_scale_fallback=payload["zero_scale_fallback"],
            fingerprint=payload["fingerprint"])

    @staticmethod
    def _hydrate_quant(payload: dict[str, Any]):
        return qnt_mod.GroupQuantResult(
            group_index=payload["group_index"],
            raw_integers=payload["raw_integers"],
            integers=payload["integers"],
            saturated=payload["saturated"],
            dequant=payload["dequant"],
            abs_error=payload["abs_error"],
            rel_error=payload["rel_error"],
            error_kind=payload["error_kind"],
            zero_truth_branch=payload["zero_truth_branch"],
            fingerprint=payload["fingerprint"])

    @staticmethod
    def _error_summary(qnt_records) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        total_abs = 0.0
        total_rel = 0.0
        count = 0
        saturated = 0
        zero_branch = 0
        for record in qnt_records:
            saturated += len(record.saturated)
            zero_branch += sum(record.zero_truth_branch)
            for kind in record.error_kind:
                kinds[kind] = kinds.get(kind, 0) + 1
            total_abs += float(np.sum(record.abs_error))
            total_rel += float(np.sum(record.rel_error))
            count += len(record.abs_error)
        return {
            "element_count": count,
            "saturated_count": saturated,
            "zero_truth_branch_count": zero_branch,
            "mean_abs_error": total_abs / count if count else 0.0,
            "mean_rel_error": total_rel / count if count else 0.0,
            "error_kind_counts": dict(sorted(kinds.items())),
        }
