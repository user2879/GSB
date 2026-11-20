"""定点卷积主管线：调度、缓存、注入隔离、快照与误差归属的编排。"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import snapshot as snapshot_mod
from .attribution import AttributionReport, attribute_position
from .cache import (
    CacheEntry,
    PositionCache,
    build_accum_key,
    build_deq_key,
    build_trunc_key_with_desc,
    scale_fingerprint,
)
from .errors import (
    E_INJECT_NONFINITE_SCALE,
    E_INJECT_QBITS_OUT_OF_RANGE,
    FailureLedger,
)
from .injections import InjectionEngine
from .kernel import PositionResult, compute_position
from .models import OpDesc
from .quantize import dequantize, truncate
from .reference import reference_position
from .scheduler import BatchPlan, execute_tasks, plan_batches
from .tensorpack import TensorPack


@dataclass
class RunConfig:
    snapshot_dir: str
    threaded: bool = True
    run_id: str = "run"


@dataclass
class PipelineResult:
    desc: OpDesc
    positions: dict[int, dict] = field(default_factory=dict)
    attribution: AttributionReport = field(default_factory=AttributionReport)
    cache: PositionCache = field(default_factory=PositionCache)
    plan: BatchPlan | None = None
    snapshot_versions: list[int] = field(default_factory=list)
    ledger: FailureLedger | None = None
    injection_hits: list[dict] = field(default_factory=list)
    global_injection_artifacts: dict = field(default_factory=dict)
    concurrency_peak: int = 0
    completed_batches: int = 0
    consumed_budget: int = 0
    resumed: bool = False


def _keys_for(desc: OpDesc, position: PositionResult, scaled: float, step: float,
              rounding: str, frac_bits: int, scale_fp: str,
              saturation: str, qbits: int) -> tuple[str, str, str]:
    accum_key = build_accum_key(
        desc,
        position.window_fingerprint,
        position.act_row_ids,
        position.weight_row_ids,
        position.raw_sum,
        saturation=saturation,
    )
    trunc_key = build_trunc_key_with_desc(
        desc, accum_key, position.buffered, qbits=qbits, saturation=saturation
    )
    deq_key = build_deq_key(trunc_key, scaled, step, rounding, frac_bits, scale_fp)
    return accum_key, trunc_key, deq_key


def run_pipeline(
    desc: OpDesc,
    pack: TensorPack,
    ledger: FailureLedger,
    injection_engine: InjectionEngine,
    config: RunConfig,
) -> PipelineResult:
    result = PipelineResult(desc=desc, ledger=ledger)

    # 全局注入审计（数据越界、非法策略组合），不影响其余张量进入计算。
    result.global_injection_artifacts = injection_engine.audit_global()

    state = snapshot_mod.load_snapshot(
        config.snapshot_dir,
        config_hash=desc.config_hash,
        package_fingerprint=pack.package_fingerprint,
        ledger=ledger,
    )
    if state is not None:
        result.cache = PositionCache.restore(state.payload.get("cache", {}))
        result.resumed = True
        next_batch = state.completed_batches
        version = state.version
        result.consumed_budget = state.consumed_budget
        result.snapshot_versions.append(version)
        for key, item in state.payload.get("positions", {}).items():
            result.positions[int(key)] = item
    else:
        result.cache = PositionCache()
        next_batch = 0
        version = 0

    plan = plan_batches(
        desc.total_positions, desc.macs_per_position, desc.mac_budget, ledger
    )
    result.plan = plan
    scale_fp = scale_fingerprint(desc)

    in_flight_peak = 0

    def worker(task) -> dict:
        nonlocal in_flight_peak
        position = task.position
        batch = task.batch
        if task.stage != "kernel":
            # 三阶段在同一 worker 内完成；truncate/dequant 任务仅占位以保持
            # 稳定排序键语义，真正的阶段产物在 kernel 任务里一并生成。
            return {"position": position, "stage": task.stage, "placeholder": True}

        try:
            return _process_position(
                desc, pack, position, batch, result.cache, injection_engine,
                ledger, scale_fp,
            )
        except Exception as exc:  # 单位置异常隔离，不中断其余位置
            from .errors import E_DESC_INVALID
            ledger.add(
                E_DESC_INVALID,
                f"位置 {position} 计算异常被隔离: {type(exc).__name__}: {exc}",
                "kernel",
                (batch, position, 9),
                position=position,
                batch=batch,
            )
            return {"position": position, "failed": True, "error": str(exc)}

    for batch_no, tasks in enumerate(plan.batches):
        if batch_no < next_batch:
            continue
        kernel_tasks = [task for task in tasks if task.stage == "kernel"]
        if kernel_tasks:
            in_flight_peak = max(
                in_flight_peak, min(desc.concurrency, len(kernel_tasks))
            )
        ordered = execute_tasks(kernel_tasks, worker, desc.concurrency, config.threaded)
        for item in ordered:
            position = item["position"]
            result.positions[position] = item
            if not item.get("failed"):
                result.attribution.add(item["attribution"])

        result.completed_batches = batch_no + 1
        result.consumed_budget += plan.budget_ledger[batch_no]["allocated"]
        should_commit = (
            (batch_no + 1) % desc.snapshot_every == 0
            or batch_no == len(plan.batches) - 1
        )
        if should_commit:
            version += 1
            snapshot_mod.commit_snapshot(
                config.snapshot_dir,
                version=version,
                config_hash=desc.config_hash,
                package_fingerprint=pack.package_fingerprint,
                completed_batches=result.completed_batches,
                consumed_budget=result.consumed_budget,
                payload={
                    "cache": result.cache.snapshot(),
                    "positions": {str(key): value for key, value in result.positions.items()},
                    "run_id": config.run_id,
                },
            )
            result.snapshot_versions.append(version)

    result.concurrency_peak = in_flight_peak
    result.injection_hits = list(injection_engine.hits)
    ledger.sort()
    return result


def _strip_attribution(item: dict) -> dict:
    cleaned = dict(item)
    cleaned.pop("attribution", None)
    return cleaned


def _process_position(
    desc: OpDesc,
    pack: TensorPack,
    position: int,
    batch: int,
    cache: PositionCache,
    engine: InjectionEngine,
    ledger: FailureLedger,
    scale_fp: str,
) -> dict:
    from .kernel import split_position

    _batch_index, out_channel, _oh, _ow = split_position(desc, position)

    # 注入：量化位宽越界在计算前拦截，隔离该位置的截断/反量化阶段。
    qbits_override = engine.qbits_override(position, batch)
    if qbits_override is not None:
        qbits, injection = qbits_override
        return {
            "position": position,
            "batch": batch,
            "failed": True,
            "failure_code": E_INJECT_QBITS_OUT_OF_RANGE,
            "injection_id": injection.injection_id,
        }

    position_result = compute_position(desc, pack, position)
    trunc = truncate(desc, position_result)

    # 注入：非有限缩放因子在反量化环节命中，隔离该位置的反量化。
    bad_scale = engine.nonfinite_scale(position, batch)
    if bad_scale is not None:
        injection = next(
            item for item in engine.for_position(position) if item.kind == "nonfinite_scale"
        )
        return {
            "position": position,
            "batch": batch,
            "failed": True,
            "failure_code": E_INJECT_NONFINITE_SCALE,
            "injection_id": injection.injection_id,
            "kernel": position_result.to_dict(),
            "truncation": trunc.to_dict(),
        }

    deq = dequantize(desc, trunc, out_channel)
    exact_sum, _raw_truth, truth_deq = reference_position(desc, pack, position)
    attribution = attribute_position(
        desc, position_result, trunc, deq, exact_sum, truth_deq
    )

    accum_key, trunc_key, deq_key = _keys_for(
        desc,
        position_result,
        deq.scaled,
        deq.step,
        deq.rounding,
        deq.frac_bits,
        scale_fp,
        desc.saturation,
        desc.qbits,
    )
    cached = cache.lookup(position, accum_key, trunc_key, deq_key)
    if cached is not None:
        cache.record_reuse(position)
        payload = cached.payload
    else:
        cache.record_recompute(position, "missing_or_invalidated_cache")
        payload = {
            "kernel": position_result.to_dict(),
            "truncation": trunc.to_dict(),
            "dequant": deq.to_dict(),
            "reference": {
                "exact_sum": exact_sum,
                "truth_deq": truth_deq,
            },
        }
        cache.put(
            CacheEntry(
                position=position,
                accum_key=accum_key,
                trunc_key=trunc_key,
                deq_key=deq_key,
                payload=payload,
            )
        )

    return {
        "position": position,
        "batch": batch,
        "reused": cached is not None,
        "attribution": attribution,
        **payload,
    }
