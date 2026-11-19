"""固定序列可重复演示入口。

序列覆盖：素材生成（低/高漂移、温漂、丢帧丢样本）-> 同步估计 ->
空洞填补 -> 增量改漂移模型 -> 改延迟补偿 -> 改空洞策略 ->
参考通道切换 -> 快照替换 -> 强制终止 -> 恢复后继续增量。

输出 reports/demo_report.json 与 reports/node_hashes.json；
两次相同序列运行报告逐字节一致（报告不含时间戳/绝对路径随机量）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from .config import (
    GAP_DRIFT,
    PARAM_DELAY_COMP,
    PARAM_DRIFT_MODEL,
    PARAM_GAP_POLICY,
    PARAM_REFERENCE,
    DRIFT_THERMAL,
    default_config,
)
from .engine import Engine
from .graph import node_id
from .hashing import canonical_bytes

CRASH_ARG = "--crash-phase"


def _node_hash_table(engine: Engine) -> dict[str, Any]:
    out = {}
    for nid, st in sorted(engine.graph.nodes.items()):
        out[nid] = {
            "version": st.version,
            "status": st.status,
            "content_hash": st.content_hash,
            "reuse_source": st.reuse_source,
            "recompute_count": st.recompute_count,
        }
    return out


def _drift_report(engine: Engine) -> list[dict[str, Any]]:
    rows = []
    for ch in range(len(engine.cfg.truths)):
        est = engine._estimations[ch]
        truth = engine.cfg.truth(ch)
        rows.append({
            "channel": ch,
            "truth_ppm": truth.ppm,
            "truth_thermal_ppm": truth.thermal_ppm,
            "truth_delay_samples": truth.delay_samples,
            "estimated_delay_samples": round(est.delay_samples, 4),
            "delay_error_abs": round(
                abs(est.delay_samples - truth.delay_samples), 4),
            "fused_ppm": round(est.fused_ppm, 3),
            "audio_bias_samples": round(est.audio_bias_samples, 4),
            "audio_bias_ci": [round(v, 4) for v in est.audio_bias_ci],
            "video_bias_frames": round(est.video_bias_frames, 5),
            "segment_rms_residual": [round(s.rms_residual, 5)
                                     for s in est.segments],
            "segment_ppm": [round(s.ppm, 3) for s in est.segments],
            "segment_ppm_ci": [[round(a, 3), round(b, 3)]
                               for s in est.segments for a, b in [s.ppm_ci]],
            "error_codes": sorted({e.code for e in est.errors}),
        })
    return rows


def _gap_report(engine: Engine) -> list[dict[str, Any]]:
    rows = []
    for ch in range(len(engine.cfg.truths)):
        g = engine._gaps[ch]
        rows.append({
            "channel": ch,
            "policy": g.policy,
            "drift_version": g.drift_version,
            "audio_hash": g.audio_hash,
            "video_hash": g.video_hash,
            "records": [r.to_dict() for r in g.records],
            "error_codes": sorted({e.code for e in g.errors}),
        })
    return rows


def _timeline_report(engine: Engine) -> list[dict[str, Any]]:
    rows = []
    for ch in range(len(engine.cfg.truths)):
        d = engine._duals[ch]
        rows.append({
            "channel": ch,
            "reference_channel": d.timeline.reference_channel,
            "max_ref_diff_sec": round(d.max_ref_diff_sec, 9),
            "map_hash": engine.graph.nodes[node_id("map", ch)].content_hash,
            "error_codes": sorted({e.code for e in d.errors}),
        })
    return rows


def run_sequence(ws_dir: str, crash: bool = False) -> dict[str, Any]:
    cfg = default_config()
    engine = Engine(ws_dir, cfg)

    if os.path.exists(os.path.join(ws_dir, "store", "shutdown.marker")):
        os.remove(os.path.join(ws_dir, "store", "shutdown.marker"))

    initial = engine.initial_build()
    steps: list[dict[str, Any]] = [
        {"step": "initial_build",
         "closure_size": len(initial["closure"]),
         "format_error_codes": sorted({e.code for e in initial["format_errors"]})}]

    # 1) 增量修改通道 2 的漂移模型 linear -> thermal
    before = dict(engine.graph.reusable_hashes())
    r = engine.apply_change(PARAM_DRIFT_MODEL, 2, DRIFT_THERMAL)
    steps.append({"step": "change_drift_model_ch2",
                  "closure": r["closure"],
                  "reused_unchanged": _reused(before, engine, r["closure"])})

    # 2) 通道 3 延迟补偿开关 关
    before = dict(engine.graph.reusable_hashes())
    r = engine.apply_change(PARAM_DELAY_COMP, 3, False)
    steps.append({"step": "toggle_delay_comp_ch3",
                  "closure": r["closure"],
                  "reused_unchanged": _reused(before, engine, r["closure"])})

    # 3) 通道 0 空洞策略 -> drift_extrapolate
    before = dict(engine.graph.reusable_hashes())
    r = engine.apply_change(PARAM_GAP_POLICY, 0, GAP_DRIFT)
    steps.append({"step": "change_gap_policy_ch0",
                  "closure": r["closure"],
                  "reused_unchanged": _reused(before, engine, r["closure"])})

    # 4) 参考通道切换 0 -> 1（drift/gapfill 必须复用且哈希不变）
    before = dict(engine.graph.reusable_hashes())
    if crash:
        engine.apply_change(PARAM_REFERENCE, None, 1, simulate_crash=True)
        return {}  # 硬退出，不可达
    r = engine.apply_change(PARAM_REFERENCE, None, 1)
    ref_step = {"step": "switch_reference_to_ch1",
                "closure": r["closure"],
                "reused_unchanged": _reused(before, engine, r["closure"]),
                "drift_gapfill_hashes_preserved":
                    _drift_gapfill_preserved(before, engine)}
    steps.append(ref_step)

    # 5) 再切换到通道 2（验证可连续增量）
    before = dict(engine.graph.reusable_hashes())
    r = engine.apply_change(PARAM_REFERENCE, None, 2)
    steps.append({"step": "switch_reference_to_ch2",
                  "closure": r["closure"],
                  "reused_unchanged": _reused(before, engine, r["closure"])})

    report = {
        "schema": "syncbench-demo-report/1",
        "seed": cfg.seed,
        "final_reference_channel": engine.cfg.reference_channel,
        "param_version": engine.cfg.param_version(),
        "steps": steps,
        "drift_report": _drift_report(engine),
        "gap_report": _gap_report(engine),
        "timeline_report": _timeline_report(engine),
        "node_hashes": _node_hash_table(engine),
        "snapshot": engine.snapshots.current.to_dict(),
        "all_error_codes": _all_error_codes(engine),
    }
    shutdown = engine.controlled_shutdown()
    report["shutdown"] = shutdown
    return report


def _reused(before: dict[str, str], engine: Engine,
            closure: list[str]) -> list[str]:
    """闭包外哈希保持不变的节点清单（复用证据）。"""
    closed = set(closure)
    preserved = []
    for nid, old_hash in sorted(before.items()):
        if nid in closed:
            continue
        if engine.graph.nodes[nid].content_hash == old_hash:
            preserved.append(nid)
    return preserved


def _drift_gapfill_preserved(before: dict[str, str],
                             engine: Engine) -> dict[str, str]:
    out = {}
    for nid, old_hash in before.items():
        if nid.startswith(("drift:", "gapfill:")):
            if engine.graph.nodes[nid].content_hash == old_hash:
                out[nid] = old_hash
    return out


def _all_error_codes(engine: Engine) -> dict[str, list[str]]:
    codes: dict[str, list[str]] = {}
    for ch in range(len(engine.cfg.truths)):
        c: set[str] = set()
        c.update(e.code for e in engine._estimations[ch].errors)
        c.update(e.code for e in engine._gaps[ch].errors)
        c.update(e.code for e in engine._duals[ch].errors)
        codes[f"ch{ch}"] = sorted(c)
    if engine.snapshots.current and engine.snapshots.current.errors:
        codes.setdefault("snapshot", []).extend(
            sorted({e.code for e in engine.snapshots.current.errors}))
    return codes


def run_with_crash_recovery(ws_dir: str) -> dict[str, Any]:
    """完整演示：先在参考切换处硬终止，再新进程恢复并继续同一序列。"""
    # 阶段 A：跑到参考切换即被 SIGKILL（os._exit 137）
    proc = subprocess.run(
        [sys.executable, "-m", "syncbench.cli", "demo", "--workspace", ws_dir,
         CRASH_ARG],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    crash_rc = proc.returncode

    # 阶段 B：新引擎进程恢复
    cfg = default_config()
    engine = Engine(ws_dir, cfg)
    recovery = engine.restore_from_store()

    # 阶段 C：恢复后继续增量修改（与未中断序列相同的剩余步骤）
    before = dict(engine.graph.reusable_hashes())
    r = engine.apply_change(PARAM_REFERENCE, None, 2)
    post_step = {"step": "post_recovery_switch_reference_to_ch2",
                 "closure": r["closure"],
                 "reused_unchanged": _reused(before, engine, r["closure"])}

    report = {
        "schema": "syncbench-demo-report/1",
        "seed": cfg.seed,
        "final_reference_channel": engine.cfg.reference_channel,
        "param_version": engine.cfg.param_version(),
        "crash_exit_code": crash_rc,
        "recovery": recovery,
        "steps": [post_step],
        "drift_report": _drift_report(engine),
        "gap_report": _gap_report(engine),
        "timeline_report": _timeline_report(engine),
        "node_hashes": _node_hash_table(engine),
        "snapshot": engine.snapshots.current.to_dict(),
        "all_error_codes": _all_error_codes(engine),
    }
    report["shutdown"] = engine.controlled_shutdown()
    return report


def write_reports(ws_dir: str, report: dict[str, Any]) -> tuple[str, str]:
    reports_dir = os.path.join(ws_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "demo_report.json")
    hashes_path = os.path.join(reports_dir, "node_hashes.json")
    with open(report_path, "wb") as fh:
        fh.write(canonical_bytes(report))
    with open(hashes_path, "wb") as fh:
        fh.write(canonical_bytes(report["node_hashes"]))
    return report_path, hashes_path
