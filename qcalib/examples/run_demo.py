"""端到端示例：解析两版本 -> 双向转换 -> 逐组标定 -> 饱和截断 ->
误差归因 -> 预算调度 -> 增量/全量一致性 -> 快照与诊断报告。

用法（在 qcalib/ 目录下）：
    python examples/generate_samples.py --out samples
    python examples/run_demo.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcalib import (  # noqa: E402
    CalibrationRunner,
    downgrade_v2_to_v1,
    load_calibration_config,
    load_feature_table,
    read_v1,
    read_v2,
    read_sidecar,
    upgrade_v1_to_v2,
    write_sidecar,
    write_v1,
)
from qcalib.pipeline import Pipeline  # noqa: E402
from qcalib.store import StageStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
WORK = ROOT / "work_demo"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    # 1) 加载校验
    section("1. 配置/特征表加载校验")
    table, table_failures = load_feature_table(SAMPLES / "feature_table.jsonl")
    cfg, cfg_failures = load_calibration_config(SAMPLES / "groups_config.json", table)
    print(f"特征表: {len(table.row_ids)} 行 x {len(table.columns)} 列，失败 {len(table_failures)} 条")
    print(f"列组: {[g.group_id for g in cfg.groups]}，配置失败 {len(cfg_failures)} 条")

    # 2) 两版本解析 + 双向转换
    section("2. v1/v2 解析与双向转换")
    v1_entries, f1 = read_v1(SAMPLES / "calib_v1.txt")
    v2_entries, f2 = read_v2(SAMPLES / "calib_v2.json")
    print(f"v1 解析 {len(v1_entries)} 条（失败 {f1.codes}）；v2 解析 {len(v2_entries)} 条（失败 {f2.codes}）")
    upgraded = upgrade_v1_to_v2(v1_entries)
    downgraded, sidecar = downgrade_v2_to_v1(v2_entries)
    out_dir = WORK / "exchange"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_v1(out_dir / "downgraded_v1.txt", downgraded)
    write_sidecar(out_dir / "v2_to_v1_sidecar.json", sidecar)
    reloaded_side = read_sidecar(out_dir / "v2_to_v1_sidecar.json")
    print(f"v2->v1 写出 {len(downgraded)} 行；旁路可再解析条目 {len(reloaded_side['entries'])}")
    print(f"版本标识示例: v1={v1_entries[0].version}")

    # 3) 全量标定管线
    section("3. 全量标定 + 定点重缩放 + 饱和截断 + 误差归因")
    shutil.rmtree(WORK / "full", ignore_errors=True)
    full_report = CalibrationRunner(table, cfg, WORK / "full", max_workers=4).run(force_full=True)
    for gid, columns in full_report["groups"].items():
        for item in columns:
            r = item["rescale"]
            e = item["error"]
            print(
                f"{gid}/{item['column']} fbits={r['fbits']} scale={item['calibration']['scale']:.4f} "
                f"zp={item['calibration']['zero_point']} 饱和={r['n_saturated']} "
                f"超差比例={e['outlier_ratio']:.2f} 结论={e['layer_conclusion']}"
            )
    print("失败清单:", [f["code"] for f in full_report["failures"]])

    # 4) 预算约束调度
    section("4. 单批元素预算约束（不足 -> E_CALIB_BUDGET_PENDING）")
    shutil.rmtree(WORK / "budget", ignore_errors=True)
    costs = [
        sum(1 for v in table.column_vector(c) if v is not None)
        for g in cfg.groups for c in g.columns
    ]
    tight_budget = sum(sorted(costs)[:2])
    pipe = Pipeline(table, cfg, StageStore(WORK / "budget/cache"),
                    element_budget=tight_budget, max_workers=4)
    budget_result = pipe.run(force_full=True)
    print(f"预算={tight_budget}，完成 {len(budget_result.calibration)} 列，待办 {len(budget_result.pending)} 列")
    print("待办排序键:", [p["sort_key"] for p in budget_result.pending])

    # 5) 增量运行 + 与全量逐字段一致
    section("5. 单行修正后的增量运行与全量一致性")
    edited, _ = load_feature_table(SAMPLES / "feature_table.jsonl")
    edited.values[edited.row_ids[2]][cfg.groups[0].columns[0]] = 2.5
    inc_report = CalibrationRunner(edited, cfg, WORK / "full").run()
    fresh_report = CalibrationRunner(edited, cfg, WORK / "fresh", max_workers=4).run(force_full=True)
    same = json.dumps(inc_report["groups"], sort_keys=True) == json.dumps(
        fresh_report["groups"], sort_keys=True
    )
    print(f"增量失效键 {len(inc_report['invalidated'])}，复用键 {len(inc_report['reused'])}")
    print(f"增量与空白全量逐字段一致: {same}")

    section("完成")
    print(f"诊断报告: {WORK / 'full/snapshot/report.json'}")
    print(f"阶段快照: {WORK / 'full/snapshot/'}")


if __name__ == "__main__":
    main()
