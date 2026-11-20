"""端到端演示：覆盖全部验收点并输出诊断报告。

阶段：
  A. 加载校验（描述/张量包/校准集/注入清单）
  B. 串行 vs 线程池一致性
  C. 预算调度与快照续跑
  D. 饱和策略与舍入模式切换的真实取值变化
  E. 内容指纹缓存的位置级精确失效
  F. 注入隔离与业务码、诊断报告落盘
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from fxpconv import (
    FailureLedger,
    InjectionEngine,
    PositionCache,
    RunConfig,
    load_calibration,
    load_descriptor,
    load_injections,
    load_tensor_pack,
    run_pipeline,
)
from fxpconv.cache import scale_fingerprint
from fxpconv.errors import (
    E_BATCH_BUDGET_EXHAUSTED,
    E_SNAPSHOT_STALE,
)
from fxpconv.kernel import (
    compute_position,
    gather_window_fingerprint,
    positions_referencing_activation_cell,
    split_position,
)
from fxpconv.pipeline import _process_position
from fxpconv.quantize import dequantize, truncate
from fxpconv.report import build_report, ensure_writable_dirs, write_report
from fxpconv.tensorpack import Tensor, expected_main_shapes

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
SNAPSHOTS = ROOT / "snapshots"
REPORTS = ROOT / "reports"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_all(snapshot_name: str):
    ledger = FailureLedger()
    desc, desc_ledger = load_descriptor(SAMPLES / "operator_desc.json")
    ledger._items.extend(desc_ledger._items)
    assert desc is not None, desc_ledger.to_list()

    shapes = expected_main_shapes(desc)
    pack, pack_ledger = load_tensor_pack(SAMPLES / "tensor_pack.jsonl", desc, shapes)
    ledger._items.extend(pack_ledger._items)
    assert pack is not None, pack_ledger.to_list()

    calibration, calib_ledger = load_calibration(SAMPLES / "calibration.json", desc)
    ledger._items.extend(calib_ledger._items)
    assert calibration, calib_ledger.to_list()

    injections = load_injections(SAMPLES / "injections.json", desc, ledger)
    engine = InjectionEngine(injections, desc, ledger)

    snapshot_dir = SNAPSHOTS / snapshot_name
    return desc, pack, calibration, injections, engine, ledger, snapshot_dir


def position_payloads(result) -> dict:
    return {
        pos: {
            "buffered": item["kernel"]["buffered"],
            "truncated": item["truncation"]["truncated"],
            "deq": item["dequant"]["value"],
            "attribution": item["attribution"].to_dict(),
        }
        for pos, item in result.positions.items()
        if not item.get("failed")
    }


def demo_serial_vs_threaded(desc, pack, engine, ledger):
    section("B. 串行 vs 线程池一致性（稳定排序键）")
    all_injections = list(engine.global_injections()) + [
        injection for injections in engine.by_position.values() for injection in injections
    ]
    serial_ledger = FailureLedger()
    serial_engine = InjectionEngine(list(all_injections), desc, serial_ledger)
    serial = run_pipeline(
        desc, pack, serial_ledger, serial_engine,
        RunConfig(snapshot_dir=str(SNAPSHOTS / "serial"), threaded=False, run_id="serial"),
    )
    thread_ledger = FailureLedger()
    thread_engine = InjectionEngine(list(all_injections), desc, thread_ledger)
    threaded = run_pipeline(
        desc, pack, thread_ledger, thread_engine,
        RunConfig(snapshot_dir=str(SNAPSHOTS / "threaded"), threaded=True, run_id="threaded"),
    )
    serial_payload = position_payloads(serial)
    thread_payload = position_payloads(threaded)
    assert serial_payload == thread_payload, "串行与线程池位置产物不一致"
    assert serial.plan.budget_ledger == threaded.plan.budget_ledger, "预算扣减序列不一致"
    assert serial_ledger.to_list() == thread_ledger.to_list(), "失败清单顺序不一致"
    print(f"位置产物逐元素一致: {len(serial_payload)} 个位置")
    print(f"预算扣减序列一致: {len(serial.plan.budget_ledger)} 个批次")
    print(f"失败清单顺序一致: {len(serial_ledger)} 条")
    print(f"线程池配置并发={desc.concurrency}，实测峰值 in-flight={threaded.concurrency_peak}")
    return serial, threaded


def demo_budget_and_snapshot(desc, pack, engine, ledger):
    section("C. 单批乘加点预算调度与快照整体替换/续跑")
    run_dir = SNAPSHOTS / "main"
    shutil.rmtree(run_dir, ignore_errors=True)
    result = run_pipeline(
        desc, pack, ledger, engine,
        RunConfig(snapshot_dir=str(run_dir), threaded=True, run_id="main"),
    )
    batches = result.plan.batches
    exhaust = [item for item in ledger if item.code == E_BATCH_BUDGET_EXHAUSTED]
    print(f"总位置={desc.total_positions}，单位置 MAC={desc.macs_per_position}，"
          f"单批预算={desc.mac_budget}")
    print(f"切出批次数={len(batches)}，预算耗尽事件={len(exhaust)}")
    for item in exhaust[:4]:
        print(f"  - {item.code}: {item.message}")
    print(f"快照版本序列={result.snapshot_versions}，最终已完成批次={result.completed_batches}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"manifest: version={manifest['version']} completed_batches="
          f"{manifest['completed_batches']} consumed_budget={manifest['consumed_budget']}")

    # 续跑：相同配置与张量指纹，应判定为可复用快照。
    resume_ledger = FailureLedger()
    resume_engine = InjectionEngine(
        list(engine.global_injections())
        + [inj for injections in engine.by_position.values() for inj in injections],
        desc, resume_ledger,
    )
    resumed = run_pipeline(
        desc, pack, resume_ledger, resume_engine,
        RunConfig(snapshot_dir=str(run_dir), threaded=True, run_id="resume"),
    )
    print(f"二次启动 resumed={resumed.resumed}，快照版本继续递增={resumed.snapshot_versions}")
    return result


def demo_snapshot_stale(desc, pack):
    section("C2. 快照陈旧判定 E_SNAPSHOT_STALE")
    run_dir = SNAPSHOTS / "stale_check"
    shutil.rmtree(run_dir, ignore_errors=True)
    ledger = FailureLedger()
    engine = InjectionEngine([], desc, ledger)
    run_pipeline(desc, pack, ledger, engine,
                 RunConfig(snapshot_dir=str(run_dir), run_id="stale-1"))
    altered = {**desc.raw, "rounding": "trunc_zero"}
    from fxpconv.models import OpDesc
    stale_desc = OpDesc(
        name=desc.name, raw=altered, batch_n=desc.batch_n, in_channels=desc.in_channels,
        in_h=desc.in_h, in_w=desc.in_w, out_channels=desc.out_channels, k_h=desc.k_h,
        k_w=desc.k_w, stride_h=desc.stride_h, stride_w=desc.stride_w, pad_h=desc.pad_h,
        pad_w=desc.pad_w, acc_bits=desc.acc_bits, qbits=desc.qbits,
        saturation=desc.saturation, rounding="trunc_zero", frac_bits=desc.frac_bits,
        out_low=desc.out_low, out_high=desc.out_high, concurrency=desc.concurrency,
        mac_budget=desc.mac_budget, snapshot_every=desc.snapshot_every,
        tolerance=desc.tolerance, act_scale=desc.act_scale,
        weight_scales=list(desc.weight_scales), scale_sources=desc.scale_sources,
    )
    stale_ledger = FailureLedger()
    stale_engine = InjectionEngine([], stale_desc, stale_ledger)
    stale_result = run_pipeline(stale_desc, pack, stale_ledger, stale_engine,
                                RunConfig(snapshot_dir=str(run_dir), run_id="stale-2"))
    codes = stale_ledger.codes()
    assert E_SNAPSHOT_STALE in codes, codes
    print(f"配置哈希变化后命中 {E_SNAPSHOT_STALE}，已从头重算，失败码={codes}")


def demo_saturation_and_rounding(desc, pack):
    section("D. 累加溢出/饱和截断与三种舍入模式的真实取值变化")
    overflow_positions = []
    boundary_positions = []
    for position in range(desc.total_positions):
        result = compute_position(desc, pack, position)
        if result.overflow is not None:
            overflow_positions.append((position, result))
        trunc = truncate(desc, result)
        if trunc.boundary_sample:
            boundary_positions.append(position)
    print(f"发生累加缓冲溢出的位置数={len(overflow_positions)}")
    if overflow_positions:
        position, result = overflow_positions[0]
        event = result.overflow
        print(f"  示例位置 {position}: raw_sum={result.raw_sum} buffered={result.buffered}")
        print(f"  首处溢出坐标(ic,kh,kw)={event.coord} 方向={event.direction} "
              f"溢出前部分和={event.partial_before}")
    print(f"量化边界样本位置数={len(boundary_positions)}（前5个={boundary_positions[:5]}）")

    # 饱和策略真实改变截断值：saturate vs wrap（wrap 需 >=16 位组合，这里构造对照描述）
    trunc_candidates = []
    for position in range(desc.total_positions):
        candidate = compute_position(desc, pack, position)
        candidate_trunc = truncate(desc, candidate)
        if candidate_trunc.clamped:
            trunc_candidates.append((position, candidate, candidate_trunc))
    probe = (trunc_candidates[0][0] if trunc_candidates
             else (overflow_positions[0][0] if overflow_positions else 0))
    pos_result = compute_position(desc, pack, probe)
    sat_trunc = truncate(desc, pos_result)
    from fxpconv.kernel import _twos_complement_wrap
    wrap_value = _twos_complement_wrap(pos_result.buffered, desc.qbits)
    print(f"位置 {probe} 截断对照: saturate={sat_trunc.truncated} wrap(对照)={wrap_value} "
          f"策略真实改值={sat_trunc.truncated != wrap_value}")

    # 三种舍入模式 + frac_bits 切换
    channels = split_position(desc, probe)[1]
    # 选一个 scaled 未落在定点网格整数倍上的位置，保证三种舍入可观测到差异。
    rounding_probe = None
    for position in range(desc.total_positions):
        candidate = compute_position(desc, pack, position)
        candidate_trunc = truncate(desc, candidate)
        oc = split_position(desc, position)[1]
        scaled = candidate_trunc.truncated * desc.effective_scale(oc)
        if scaled / (2 ** -desc.frac_bits) != int(scaled / (2 ** -desc.frac_bits)):
            rounding_probe = (position, candidate, candidate_trunc, oc, scaled)
            break
    if rounding_probe is None:
        rounding_probe = (probe, pos_result, sat_trunc, channels,
                          sat_trunc.truncated * desc.effective_scale(channels))
    r_position, r_pos, r_trunc, r_channels, r_scaled = rounding_probe
    deq_values = {}
    for rounding in ("trunc_zero", "round_nearest_even", "floor"):
        deq = dequantize(desc, r_trunc, r_channels, rounding=rounding)
        deq_values[rounding] = deq.value
    deq_frac4 = dequantize(desc, r_trunc, r_channels, frac_bits=4)
    deq_frac8 = dequantize(desc, r_trunc, r_channels, frac_bits=8)
    frac_values = {4: deq_frac4.value, 6: deq_values["round_nearest_even"], 8: deq_frac8.value}
    # 单点可能因输出区间裁剪落到同一网格点，故在全位置上统计位宽切换的真实差异。
    frac_diff_positions = 0
    rounding_diff_positions = 0
    for scan_position in range(desc.total_positions):
        scan_pos = compute_position(desc, pack, scan_position)
        scan_trunc = truncate(desc, scan_pos)
        scan_oc = split_position(desc, scan_position)[1]
        scan_frac = {
            bits: dequantize(desc, scan_trunc, scan_oc, frac_bits=bits).value
            for bits in (4, 6, 8)
        }
        scan_round = {
            mode: dequantize(desc, scan_trunc, scan_oc, rounding=mode).value
            for mode in ("trunc_zero", "round_nearest_even", "floor")
        }
        if len(set(scan_frac.values())) >= 2:
            frac_diff_positions += 1
        if len(set(scan_round.values())) >= 2:
            rounding_diff_positions += 1
    frac_changed = frac_diff_positions > 0
    print(f"位置 {r_position} scaled={r_scaled:.7f} 反量化: "
          f"零舍入={deq_values['trunc_zero']:.6f} "
          f"就近偶={deq_values['round_nearest_even']:.6f} "
          f"向负无穷={deq_values['floor']:.6f}")
    print(f"单位量化阶跃: frac6={2 ** -6:.6f} frac4={2 ** -4:.6f} frac8={2 ** -8:.6f}")
    print(f"探针位置三档取值={frac_values}；全量位宽切换产生差异的位置数={frac_diff_positions}")
    print(f"全量舍入模式产生差异的位置数={rounding_diff_positions}")
    assert frac_changed, "定点小数位宽切换必须真实改变反量化值"
    assert rounding_diff_positions > 0, "舍入模式切换必须真实改变反量化值"
    assert sat_trunc.truncated != wrap_value, "饱和策略必须真实改变截断整数取值"
    return overflow_positions


def demo_cache_invalidation(desc, pack):
    section("E. 内容指纹缓存的位置级精确失效")
    ledger = FailureLedger()
    engine = InjectionEngine([], desc, ledger)
    cache = PositionCache()
    scale_fp = scale_fingerprint(desc)

    for position in range(desc.total_positions):
        _process_position(desc, pack, position, position // desc.positions_per_sample,
                          cache, engine, ledger, scale_fp)
    baseline_recomputed = len(cache.recomputed)
    print(f"首轮全部重算位置数={baseline_recomputed}，缓存条目={len(cache.entries)}")

    # 第二轮：输入与策略完全不变，必须全部复用、零重算。
    cache.reused.clear()
    cache.recomputed.clear()
    for position in range(desc.total_positions):
        _process_position(desc, pack, position, position // desc.positions_per_sample,
                          cache, engine, ledger, scale_fp)
    print(f"无变化第二轮: 复用={len(cache.reused)} 重算={len(cache.recomputed)}")
    assert len(cache.recomputed) == 0

    # 替换单个激活单元，仅引用该窗口的位置及其下游失效。
    activation = pack.get("activation")
    altered_data = list(activation.data)
    flat_index = (0 * desc.in_channels + 0) * desc.in_h * desc.in_w + 2 * desc.in_w + 2
    altered_data[flat_index] = altered_data[flat_index] + 7
    altered_act = Tensor("activation", activation.shape, activation.dtype, True,
                         tuple(altered_data), activation.row_ids)
    altered_pack = type(pack)(
        tensors={"activation": altered_act, "weight": pack.get("weight")},
        package_fingerprint=pack.package_fingerprint + ":altered",
    )
    expected_affected = positions_referencing_activation_cell(desc, 0, 0, 2, 2)
    actual_affected = set()
    for position in range(desc.total_positions):
        old_fp = gather_window_fingerprint(desc, pack, position)
        new_fp = gather_window_fingerprint(desc, altered_pack, position)
        if old_fp != new_fp:
            actual_affected.add(position)
    assert actual_affected == expected_affected
    print(f"替换激活单元 n0/ic0/h2/w2: 几何受影响位置={len(expected_affected)} "
          f"指纹实测受影响={len(actual_affected)}（一致）")

    cache.reused.clear()
    cache.recomputed.clear()
    for position in range(desc.total_positions):
        _process_position(desc, altered_pack, position,
                          position // desc.positions_per_sample,
                          cache, engine, ledger, scale_fp)
    recomputed_set = set(cache.recomputed)
    assert recomputed_set == expected_affected, (recomputed_set ^ expected_affected)
    print(f"替换后重算={len(recomputed_set)} 复用={len(cache.reused)}，"
          f"未失效位置零重算成立")

    # 舍入模式切换：仅舍入结果真正变化的位置失效。
    from fxpconv.models import OpDesc
    rounding_desc = OpDesc(
        name=desc.name, raw={**desc.raw, "rounding": "trunc_zero"},
        batch_n=desc.batch_n, in_channels=desc.in_channels, in_h=desc.in_h,
        in_w=desc.in_w, out_channels=desc.out_channels, k_h=desc.k_h, k_w=desc.k_w,
        stride_h=desc.stride_h, stride_w=desc.stride_w, pad_h=desc.pad_h,
        pad_w=desc.pad_w, acc_bits=desc.acc_bits, qbits=desc.qbits,
        saturation=desc.saturation, rounding="trunc_zero", frac_bits=desc.frac_bits,
        out_low=desc.out_low, out_high=desc.out_high, concurrency=desc.concurrency,
        mac_budget=desc.mac_budget, snapshot_every=desc.snapshot_every,
        tolerance=desc.tolerance, act_scale=desc.act_scale,
        weight_scales=list(desc.weight_scales), scale_sources=desc.scale_sources,
    )
    cache.reused.clear()
    cache.recomputed.clear()
    for position in range(desc.total_positions):
        _process_position(rounding_desc, pack, position,
                          position // desc.positions_per_sample,
                          cache, engine, ledger, scale_fp)
    print(f"切到零舍入后: 重算={len(cache.recomputed)} 复用={len(cache.reused)}，"
          f"稳定位置保持复用")
    return expected_affected


def check_calibration(desc, pack, calibration):
    section("A2. 校准样本 float64 参考路径比对")
    from fxpconv.reference import reference_position
    summary = []
    for sample in calibration:
        override = Tensor("activation", pack.get("activation").shape, "int8", True,
                          sample.activation, pack.get("activation").row_ids)
        max_err = 0.0
        for position in range(desc.total_positions):
            _, _, truth = reference_position(desc, pack, position, activation_override=override)
            err = abs(truth - sample.reference_output[position])
            max_err = max(max_err, err)
        summary.append({"sample_id": sample.sample_id, "max_abs_vs_reference": max_err})
        print(f"  {sample.sample_id}: 与 float64 参考最大绝对偏差={max_err:.3e}")
        assert max_err < 1e-9
    return summary


def main() -> int:
    ensure_writable_dirs(SNAPSHOTS, REPORTS)
    section("A. 加载校验层")
    desc, pack, calibration, injections, engine, ledger, snapshot_dir = load_all("main")
    print(f"描述校验通过: {desc.name} config_hash={desc.config_hash}")
    print(f"张量包指纹={pack.package_fingerprint} 张量={sorted(pack.tensors)}")
    print(f"校准样本={len(calibration)} 注入项={len(injections)}")

    calibration_summary = check_calibration(desc, pack, calibration)
    serial, threaded = demo_serial_vs_threaded(desc, pack, engine, FailureLedger())
    main_result = demo_budget_and_snapshot(desc, pack, engine, ledger)
    demo_snapshot_stale(desc, pack)
    demo_saturation_and_rounding(desc, pack)
    demo_cache_invalidation(desc, pack)

    section("F. 注入隔离与诊断报告")
    hit_ids = [hit["id"] for hit in main_result.injection_hits]
    print(f"注入命中编号={hit_ids}")
    global_codes = [item.code for item in ledger if item.injection_id in ("INJ-03", "INJ-04")]
    position_codes = [item.code for item in ledger if item.position is not None]
    print(f"全局注入业务码={global_codes}")
    print(f"单位置注入业务码={sorted(set(position_codes))}")

    report = build_report(main_result, calibration_summary=calibration_summary)
    report_path = write_report(report, REPORTS / "diagnostic_report.json")
    print(f"诊断报告已写入 {report_path}")
    print(f"失败码统计={report['failure_code_counts']}")
    print(f"三环节超差计数={report['attribution']['exceed_counts']}")
    print(f"误差直方图={report['attribution']['error_histogram']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
