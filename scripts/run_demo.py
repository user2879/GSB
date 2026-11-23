"""端到端验收演示：在确定性本地样本上跑通全部业务要求并落盘诊断报告。

输出目录:
    reports/        诊断报告 JSON + 人类可读摘要
    work/demo_*     各运行的缓存与快照
不访问网络/外部目标。
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvrb.diagnostics import cross_version_report
from mvrb.exchange_v2 import read_v2
from mvrb.inject import apply_injections, load_injection_config
from mvrb.pipeline import ReconstructionPipeline
from mvrb.utils import canonical_json, read_json, write_json
from mvrb.workbench import Workbench

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "demo")
REPORTS = os.path.join(ROOT, "reports")


def _reset(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    _reset(REPORTS)
    report_bundle = {}

    # 1) 双版本交换往返 ----------------------------------------------------
    section("1. 双版本交换文件往返（v1 定宽 / v2 JSON + 旁路）")
    ir_v2 = read_v2(os.path.join(SAMPLE, "result_v2.json"))
    diag = cross_version_report(ir_v2)
    print(f"v2 无损往返: {diag['v2_lossless']}")
    print(f"带旁路精确还原: {diag['exact_restore_with_sidecar']}")
    print(f"匹配得分最大偏差: {diag['max_score_abs_diff']}")
    print(f"旁路命中 feature/match/voxel = "
          f"{diag['sidecar_hits']['features']}/"
          f"{diag['sidecar_hits']['matches']}/"
          f"{diag['sidecar_hits']['voxel_points']}")
    report_bundle["cross_version"] = diag

    # 2) 全量 vs 增量失效边界 ---------------------------------------------
    section("2. 增量失效边界（图像/相机/阈值/体素编辑）")
    inc_sample = os.path.join(ROOT, "work", "inc_sample")
    inc_work = os.path.join(ROOT, "work", "inc_work")
    shutil.copytree(SAMPLE, inc_sample, dirs_exist_ok=True)
    shutil.rmtree(inc_work, ignore_errors=True)
    ReconstructionPipeline(inc_sample, inc_work).run()
    from mvrb.camera import load_cameras
    from mvrb.synthetic import make_scene, render_view, save_png
    cams = load_cameras(os.path.join(inc_sample, "cameras.json"))
    moved = [(lid, (x + 0.3, y, z)) for lid, (x, y, z) in make_scene()]
    save_png(os.path.join(inc_sample, "images", "view_02.png"),
             render_view(cams[2], moved))
    after_image = ReconstructionPipeline(inc_sample, inc_work).run()
    print("替换视图2后失效阶段:", sorted({i["stage"] for i in after_image.invalidated}))

    cfg = read_json(os.path.join(inc_sample, "recon_config.json"))
    cfg["feature_threshold"] = 145
    write_json(os.path.join(inc_sample, "recon_config.json"), cfg)
    after_threshold = ReconstructionPipeline(inc_sample, inc_work).run()
    print("调阈值后失效阶段:", sorted({i["stage"] for i in after_threshold.invalidated}))

    cfg["voxel_size"] = 0.75
    write_json(os.path.join(inc_sample, "recon_config.json"), cfg)
    incremental = ReconstructionPipeline(inc_sample, inc_work).run()
    print("调体素后失效阶段:", sorted({i["stage"] for i in incremental.invalidated}))

    fresh = ReconstructionPipeline(inc_sample,
                                   os.path.join(ROOT, "work", "fresh_work")).run()
    equal = canonical_json(_sig(incremental)) == canonical_json(_sig(fresh))
    print("增量运行与空白全量逐字段一致:", equal)
    report_bundle["incremental_equivalence"] = {
        "equal_to_clean_rebuild": equal,
        "image_edit_invalidated": sorted({i["stage"] for i in after_image.invalidated}),
        "threshold_edit_invalidated": sorted({i["stage"]
                                              for i in after_threshold.invalidated}),
        "voxel_edit_invalidated": sorted({i["stage"] for i in incremental.invalidated}),
        "reused_cache_keys": len(incremental.cache_ledger["reused"]),
        "computed_cache_keys": len(incremental.cache_ledger["computed"]),
    }

    # 3) 预算调度 ----------------------------------------------------------
    section("3. 全局预算 / 单视图配额调度")
    budget_sample = os.path.join(ROOT, "work", "budget_sample")
    shutil.copytree(SAMPLE, budget_sample, dirs_exist_ok=True)
    cfg = read_json(os.path.join(budget_sample, "recon_config.json"))
    cfg.update(point_budget=30, max_workers=4, view_point_quota=10_000)
    write_json(os.path.join(budget_sample, "recon_config.json"), cfg)
    budget_result = ReconstructionPipeline(
        budget_sample, os.path.join(ROOT, "work", "budget_work")).run()
    print("待办队列:", budget_result.schedule_report["todo_queue"])
    print("峰值并发:", budget_result.schedule_report["peak_concurrency"])
    print("预算失败码:", sorted({f.code for f in budget_result.failures.sorted()}))
    report_bundle["budget"] = budget_result.schedule_report

    # 4) 快照恢复判定 ------------------------------------------------------
    section("4. 阶段化快照整体替换与 E_SNAPSHOT_STALE 恢复判定")
    snap_sample = os.path.join(ROOT, "work", "snap_sample")
    snap_work = os.path.join(ROOT, "work", "snap_work")
    shutil.copytree(SAMPLE, snap_sample, dirs_exist_ok=True)
    shutil.rmtree(snap_work, ignore_errors=True)
    wb = Workbench(snap_sample, snap_work)
    run1 = wb.run()
    run2 = Workbench(snap_sample, snap_work).run()
    print("首次提交:", run1["stage_commit_actions"])
    print("二次运行提交(应全 unchanged):", run2["stage_commit_actions"])
    cfg = read_json(os.path.join(snap_sample, "recon_config.json"))
    cfg["feature_threshold"] = 120
    write_json(os.path.join(snap_sample, "recon_config.json"), cfg)
    run3 = Workbench(snap_sample, snap_work).run()
    print("配置变更后 stale:", run3["resume"]["stale"],
          "起点阶段:", run3["resume"]["start_stage"])
    report_bundle["snapshot"] = {
        "first_commit": run1["stage_commit_actions"],
        "second_commit": run2["stage_commit_actions"],
        "stale_after_config": run3["resume"],
        "manifest_versions": run3["manifest"]["stage_versions"],
    }

    # 5) 并发 vs 串行逐字段一致 -------------------------------------------
    section("5. 并发执行与串行执行逐字段一致")
    serial = ReconstructionPipeline(SAMPLE,
                                    os.path.join(ROOT, "work", "serial"),
                                    force_serial=True).run()
    parallel = ReconstructionPipeline(SAMPLE,
                                      os.path.join(ROOT, "work", "parallel"),
                                      force_serial=False).run()
    conc_equal = canonical_json(_sig(serial)) == canonical_json(_sig(parallel))
    print("串行/并行逐字段一致:", conc_equal,
          "| 并行峰值并发:", parallel.schedule_report["peak_concurrency"])
    report_bundle["concurrency"] = {"equal": conc_equal,
                                    "peak_concurrency":
                                        parallel.schedule_report["peak_concurrency"]}

    # 6) 异常注入诊断 ------------------------------------------------------
    section("6. 异常注入隔离与诊断（真实输入故障 -> 业务码）")
    injection_cfg = [
        {"id": "INJ-001", "fault": "truncated_png", "view": 1,
         "stage": "features", "keep_bytes": 24},
        {"id": "INJ-002", "fault": "missing_camera", "view": 2, "stage": "load"},
        {"id": "INJ-003", "fault": "disparity_overflow", "view": 0,
         "stage": "triangulate", "forced_min_disparity": 1e9},
        {"id": "INJ-004", "fault": "zero_voxel_size", "stage": "voxel"},
        {"id": "INJ-005", "fault": "non_finite_feature", "view": 0,
         "stage": "validate"},
    ]
    cfg_path = os.path.join(REPORTS, "injection_config.json")
    write_json(cfg_path, {"injections": injection_cfg})
    # 每项注入独立隔离运行，避免多个真实故障在同一管线内相互屏蔽
    hit_rows = []
    all_codes = set()
    for item in injection_cfg:
        broken = os.path.join(ROOT, "work", f"broken_{item['id']}")
        broken_work = os.path.join(ROOT, "work", f"broken_work_{item['id']}")
        shutil.rmtree(broken, ignore_errors=True)
        shutil.rmtree(broken_work, ignore_errors=True)  # 独立缓存，杜绝跨注入复用
        single_cfg = os.path.join(REPORTS, f"injection_{item['id']}.json")
        write_json(single_cfg, {"injections": [item]})
        manifest = apply_injections(SAMPLE, broken, load_injection_config(single_cfg))
        injected = ReconstructionPipeline(broken, broken_work).run()
        codes = {f.code for f in injected.failures.sorted()}
        if item["fault"] == "non_finite_feature":
            v2_ir = read_v2(os.path.join(broken, "result_v2.json"))
            codes |= {f.code for f in v2_ir.validate().sorted()}
        all_codes |= codes
        hit = manifest["hits"][0]
        hit_rows.append({
            "injection_id": hit["id"],
            "fault": hit["fault"],
            "expected_code": hit["code"],
            "applied": hit["applied"],
            "observed": hit["code"] in codes,
        })
    all_codes = sorted(all_codes)
    for row in hit_rows:
        print(f"{row['injection_id']} {row['fault']:20s} -> "
              f"{row['expected_code']:28s} observed={row['observed']}")
    report_bundle["injection"] = {"hits": hit_rows, "all_failure_codes": all_codes}

    # 落盘报告
    write_json(os.path.join(REPORTS, "diagnostic_report.json"), report_bundle)
    _write_summary(os.path.join(REPORTS, "diagnostic_report.txt"), report_bundle)
    section("诊断报告已写入 reports/diagnostic_report.json / .txt")
    return 0


def _sig(result) -> dict:
    return {
        "features": {v: [(f.feature_id, round(f.x, 6), round(f.y, 6), f.landmark_id)
                         for f in vf.features]
                     for v, vf in sorted(result.view_features.items())},
        "matches": [(m.view_a, m.view_b, m.pairs) for m in result.pair_matches],
        "points": [p.to_dict() for p in result.points3d],
        "depth": {v: d.summary for v, d in sorted(result.depth_maps.items())},
        "voxels": [v.to_dict() for v in result.voxels],
        "failures": result.failures.dicts(),
    }


def _write_summary(path: str, bundle: dict) -> None:
    lines = ["MVRB 验收诊断摘要", "=" * 40]
    cv = bundle["cross_version"]
    lines += [
        f"[跨版本] v2无损={cv['v2_lossless']} 旁路精确还原={cv['exact_restore_with_sidecar']} "
        f"最大得分偏差={cv['max_score_abs_diff']}",
        f"[增量等价] {bundle['incremental_equivalence']['equal_to_clean_rebuild']} "
        f"复用={bundle['incremental_equivalence']['reused_cache_keys']} "
        f"重算={bundle['incremental_equivalence']['computed_cache_keys']}",
        f"[预算] 待办={bundle['budget']['todo_queue']} 峰值并发={bundle['budget']['peak_concurrency']}",
        f"[快照] stale={bundle['snapshot']['stale_after_config']['stale']}",
        f"[并发一致] {bundle['concurrency']['equal']}",
        "[异常注入]",
    ]
    for row in bundle["injection"]["hits"]:
        lines.append(f"  {row['injection_id']} {row['fault']} -> {row['expected_code']} "
                     f"observed={row['observed']}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
