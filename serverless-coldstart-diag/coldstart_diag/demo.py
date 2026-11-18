"""可重复演示：全量推进（含强制终止恢复）-> 修改预热比例 -> 缩放因子扰动。

运行：python -m coldstart_diag.demo
运行前清空 workspace 下的旧账本与快照。
"""

import json
import shutil
from pathlib import Path

from .engine import Engine, aggregate
from .ledger import Ledger
from .replay import replay_run
from .samples import write_samples
from .spec import load_manifest
from .versioning import assert_monotonic, compare_version_stats, replace_view
from .workspace import list_snapshots, write_snapshot
from .analytics import cold_ratio_heat_hist, wait_percentiles


def _run_full(ledger, run_id, manifest, fid, fver):
    seq = manifest["call_sequence"]
    ledger.start_run(run_id, manifest, len(seq))
    engine = Engine(manifest, fid, fver)
    results = []
    for item in sorted(seq, key=lambda x: x["minute"]):
        r = engine.run_minute(item)
        ledger.append_minute(run_id, r)
        results.append(r)
    ledger.finalize_run(run_id)
    return [r.to_dict() for r in results], aggregate(results, manifest["scale"])


def _run_crash(ledger, run_id, manifest, fid, fver, commit_until):
    """模拟在 commit_until 之后被强制终止：之后的分钟不提交。"""
    seq = manifest["call_sequence"]
    ledger.start_run(run_id, manifest, len(seq))
    engine = Engine(manifest, fid, fver)
    for item in sorted(seq, key=lambda x: x["minute"]):
        r = engine.run_minute(item)
        if item["minute"] <= commit_until:
            ledger.append_minute(run_id, r)
        # 强制终止：进程被杀，未提交分钟不留任何痕迹
    # 不调用 finalize_run，保持 RUNNING 以模拟崩溃


def run_demo(workspace_dir):
    workspace = Path(workspace_dir)
    sample_dir = workspace.parent / "samples"
    if not sample_dir.exists():
        sample_dir = workspace / "samples"
    snap_dir = workspace / "snapshots"
    ledger_path = workspace / "ledger.sqlite"
    # 运行前清空旧账本与快照
    if ledger_path.exists():
        ledger_path.unlink()
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    write_samples(sample_dir)

    summary = {"scenarios": []}
    fid = "fn-order-processor"

    with Ledger(ledger_path) as ledger:
        # ---- 场景 1a：强制终止与恢复 ----
        m1 = load_manifest(sample_dir / "manifest_v1.json")
        crash_id = "run-v1-crash"
        _run_crash(ledger, crash_id, m1, fid, 1, commit_until=4)
        recovery = ledger.recover(crash_id)

        # 重跑：只重放未决分钟，已提交分钟保持不变（确定性重算结果一致）
        engine = Engine(m1, fid, 1)
        for item in sorted(m1["call_sequence"], key=lambda x: x["minute"]):
            r = engine.run_minute(item)
            ledger.append_minute(crash_id, r)
        ledger.finalize_run(crash_id)
        replay_ok = replay_run(
            sample_dir / "manifest_v1.json", ledger, crash_id,
            m1["call_sequence"], fid, 1,
        )

        # ---- 场景 1b：首次全量推进 ----
        payloads_v1, agg_v1 = _run_full(ledger, "run-v1-full", m1, fid, 1)
        write_snapshot(snap_dir, "run-v1-full", m1, payloads_v1, agg_v1, recovery)
        summary["scenarios"].append(
            {
                "scenario": "S1_full_push_with_crash_recovery",
                "declaration_version": 1,
                "recovery": recovery,
                "replay": replay_ok,
                "aggregate": agg_v1,
            }
        )

        # ---- 场景 2：修改预热比例后推进，旧结论标记陈旧并整体替换 ----
        m2 = load_manifest(sample_dir / "manifest_v2.json")
        assert_monotonic(m1, m2)
        stale_count = ledger.mark_stale("run-v1-full", 1)
        payloads_v2, agg_v2 = _run_full(ledger, "run-v2-full", m2, fid, 2)
        replacement = replace_view(payloads_v1, payloads_v2)
        write_snapshot(snap_dir, "run-v2-full", m2, payloads_v2, agg_v2)
        summary["scenarios"].append(
            {
                "scenario": "S2_warmup_ratio_changed_full_replace",
                "declaration_version": 2,
                "stale_minutes_marked": stale_count,
                "changed_minutes": replacement["changed_minutes"],
                "change_count": len(replacement["changes"]),
                "sample_changes": replacement["changes"][:5],
                "aggregate": agg_v2,
                "version_compare_v1_v2": compare_version_stats(agg_v1, agg_v2),
            }
        )

        # ---- 场景 3：缩放因子扰动后再推进 ----
        m3 = load_manifest(sample_dir / "manifest_v3.json")
        assert_monotonic(m2, m3)
        stale_count_3 = ledger.mark_stale("run-v2-full", 2)
        payloads_v3, agg_v3 = _run_full(ledger, "run-v3-full", m3, fid, 3)
        write_snapshot(snap_dir, "run-v3-full", m3, payloads_v3, agg_v3)
        # 用新缩放因子重放旧账本 -> 必须单独区分报告配置不一致
        config_mismatch = None
        try:
            replay_run(
                sample_dir / "manifest_v3.json", ledger, "run-v1-crash",
                m3["call_sequence"], fid, 1,
            )
        except Exception as exc:  # noqa: BLE001 - 演示需展示错误码
            config_mismatch = {"code": exc.code, "details": exc.details}
        replay_v3 = replay_run(
            sample_dir / "manifest_v3.json", ledger, "run-v3-full",
            m3["call_sequence"], fid, 3,
        )
        summary["scenarios"].append(
            {
                "scenario": "S3_scale_factor_perturbation",
                "declaration_version": 3,
                "stale_minutes_marked": stale_count_3,
                "config_mismatch_report": config_mismatch,
                "replay_current": replay_v3,
                "aggregate": agg_v3,
            }
        )
        summary["analytics"] = {
            "current_declaration": 3,
            "wait_percentiles": wait_percentiles(payloads_v3),
            "cold_ratio_heat_hist": cold_ratio_heat_hist(payloads_v3, m3["scale"]),
        }
        summary["snapshots"] = list_snapshots(snap_dir)
        summary["runs"] = ledger.list_runs()

    out_path = workspace / "diagnostic_summary.json"
    out_path.write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return summary, out_path


def _print_human(summary):
    lines = []
    for sc in summary["scenarios"]:
        lines.append(f"== {sc['scenario']} (declaration v{sc['declaration_version']}) ==")
        if "recovery" in sc and sc["recovery"]:
            rec = sc["recovery"]
            lines.append(
                f"  恢复: 已恢复 {rec['recovered_minutes']} 分钟, "
                f"未决 {rec['pending_minutes']} -> 重跑后状态 {rec['status']}"
            )
        if "replay" in sc:
            lines.append(f"  重放: {sc['replay']['status']} ({sc['replay']['minutes_verified']} 分钟)")
        if "stale_minutes_marked" in sc:
            lines.append(f"  陈旧分钟标记: {sc['stale_minutes_marked']}")
        if "changed_minutes" in sc:
            lines.append(f"  整体替换变化分钟: {sc['changed_minutes']} (字段变更 {sc['change_count']} 项)")
        if "config_mismatch_report" in sc and sc["config_mismatch_report"]:
            lines.append(f"  配置不一致: {sc['config_mismatch_report']['code']}")
        agg = sc["aggregate"]
        lines.append(
            f"  聚合: 总调用 {agg['total_calls']}, 冷调用 {agg['total_cold_calls']}, "
            f"整体冷启动比例 {agg['overall_cold_ratio_q']}/{agg['scale']}, "
            f"误差界 {agg['agg_error_bound_ulp']} ulp, 平均等待 {agg['avg_wait_ms']} ms"
        )
    return "\n".join(lines)


def main():
    workspace = Path(__file__).resolve().parent.parent / "workspace"
    summary, out_path = run_demo(workspace)
    print(_print_human(summary))
    print(f"\n诊断总结已写入: {out_path}")


if __name__ == "__main__":
    main()
