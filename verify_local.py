#!/usr/bin/env python3
"""本地端到端验证脚本（无需修改源码即可更换输入参数）。

覆盖验收点
----------
1. 确定性样本生成（可重复）；
2. 多介质透射渲染 + 双/单精度误差定位 + 截断统计；
3. 分块中断（写 INTERRUPT 标记，仅在分块边界生效）与约定退出码；
4. 重启恢复：版本一致只渲缺失块，并比对分块内容摘要；
5. 原子输出与版本变更归档；
6. 非法输入返回 E_xxx 且不产生输出文件。

用法
----
python verify_local.py \
    --sample-dir samples --state-dir state --output-dir output \
    [--width 32] [--height 24] [--supersample 2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from offscreen_render.errors import EXIT_INTERRUPTED  # noqa: E402
from offscreen_render.persistence import (  # noqa: E402
    compute_version_id,
    interrupt_marker_exists,
    load_tile,
    read_version,
)
from offscreen_render.samplegen import generate_sample  # noqa: E402
from offscreen_render.scene import load_scene  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"


class Reporter:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((PASS if ok else FAIL, name, detail))
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> int:
        failed = [r for r in self.results if r[0] == FAIL]
        print("\n" + "=" * 68)
        print(f"验证项 {len(self.results)}，通过 {len(self.results) - len(failed)}，"
              f"失败 {len(failed)}")
        return 1 if failed else 0


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "offscreen_render.cli", *args],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default="samples")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=24)
    ap.add_argument("--supersample", type=int, default=2)
    args = ap.parse_args()

    sample_dir = os.path.join(ROOT, args.sample_dir)
    state_dir = os.path.join(ROOT, args.state_dir)
    output_dir = os.path.join(ROOT, args.output_dir)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    rpt = Reporter()

    # --- 1. 确定性样本生成 -------------------------------------------------
    print("\n[1] 生成确定性本地样本")
    paths = generate_sample(sample_dir, state_dir, output_dir,
                            args.width, args.height, args.supersample)
    scene_path = paths["scene"]
    rpt.check("样本三文件齐全", all(os.path.exists(p) for p in paths.values()))
    paths2 = generate_sample(sample_dir + "_dup", state_dir + "_dup",
                             output_dir + "_dup", args.width, args.height,
                             args.supersample)
    g1, g2 = np.load(paths["grid"]), np.load(paths2["grid"])
    deterministic = all(np.array_equal(g1[k], g2[k])
                        for k in ("sigma_t", "albedo", "ior"))
    rpt.check("固定种子栅格可重复", deterministic)

    scene = load_scene(scene_path)
    version = compute_version_id(scene)

    # --- 2. 期望透射率与探针光线 ------------------------------------------
    print("\n[2] 期望透射率（解析指数积分）")
    expected = json.load(open(paths["expected"], encoding="utf-8"))
    for probe in expected["probes"]:
        t = probe["expected_transmittance"]
        rpt.check(f"探针 {probe['name']} 透射率∈[0,1]",
                  0.0 <= t <= 1.0, f"T={t:.6e}, 段数={len(probe['segments'])}")

    # --- 3. 中断与退出码 ---------------------------------------------------
    print("\n[3] 中断标记（分块边界生效）与退出码")
    # 预置中断标记后启动：应快速以 130 退出且不产出最终图像。
    from offscreen_render.persistence import write_interrupt_marker, remove_interrupt_marker
    write_interrupt_marker(state_dir)
    proc = _run_cli("render", "--scene", scene_path, "--resume")
    rpt.check("中断渲染退出码为 130", proc.returncode == EXIT_INTERRUPTED,
              f"got={proc.returncode}")
    payload = json.loads(proc.stdout)
    rpt.check("中断返回 E_4001 业务码", payload.get("error_code") == "E_4001")
    rpt.check("中断后未冒充完成写出图像",
              not os.path.exists(os.path.join(output_dir, "grayscale.pgm")))
    remove_interrupt_marker(state_dir)

    # --- 4. 完整渲染与原子输出 --------------------------------------------
    print("\n[4] 完整渲染、原子输出与误差/截断报告")
    t0 = time.time()
    proc = _run_cli("render", "--scene", scene_path)
    elapsed = time.time() - t0
    rpt.check("完整渲染退出码为 0", proc.returncode == 0,
              proc.stderr[:200] if proc.returncode else f"{elapsed:.1f}s")
    image_path = os.path.join(output_dir, "grayscale.pgm")
    report_path = os.path.join(output_dir, "report.json")
    rpt.check("PGM 与报告均已写出",
              os.path.exists(image_path) and os.path.exists(report_path))
    rpt.check("无残留临时文件",
              not any(".tmp." in n for n in os.listdir(output_dir)))
    with open(image_path, "rb") as fh:
        header = fh.readline().strip()
    rpt.check("PGM 魔数 P5", header == b"P5")

    report = json.load(open(report_path, encoding="utf-8"))
    rpt.check("报告版本标识一致", report["version_id"] == version)
    term = report["termination_stats"]
    rpt.check("截断统计自洽（各类之和=总光线数）",
              term["completed"] + term["cutoff"] + term["max_steps"]
              == report["total_rays"],
              f"cutoff={term['cutoff']}, max_steps={term['max_steps']}")
    rpt.check("存在被截断光线并计入统计", term["cutoff"] > 0,
              f"cutoff_ratio={term['cutoff_ratio']:.3f}")
    prec = report["precision"]
    rpt.check("双/单精度误差已记录", prec["max_abs_transmittance_diff"] > 0.0,
              f"max_diff={prec['max_abs_transmittance_diff']:.3e}")
    rpt.check("首次超差点含像素与介质定位",
              all(k in prec["first_exceeds"][0]
                  for k in ("pixel", "box_name", "step", "abs_diff"))
              if prec["first_exceeds"] else False,
              f"exceed_count={prec['first_exceed_count']}")

    # --- 5. 分块摘要比对与恢复（只渲缺失块） -------------------------------
    print("\n[5] 分块持久化、摘要校验与缺失块恢复")
    rpt.check("状态目录版本文件一致", read_version(state_dir) == version)
    tile_files = [n for n in os.listdir(state_dir)
                  if n.startswith("tile_") and n.endswith(".json")]
    expected_indices = build_tile_count(scene)
    rpt.check("全部分块已持久化",
              len(tile_files) == len(expected_indices),
              f"{len(tile_files)} / {len(expected_indices)} 块")
    # 摘要有效性：逐块重新加载并校验
    digest_ok = True
    for idx in build_tile_count(scene):
        if load_tile(state_dir, idx, version) is None:
            digest_ok = False
            break
    rpt.check("所有分块版本与摘要有效", digest_ok)

    # 删除一块模拟缺失，重跑应只恢复该块（restored=其余，rendered=1）
    missing_idx = sorted(expected_indices)[len(tile_files) // 2]
    os.remove(os.path.join(state_dir, f"tile_{missing_idx:05d}.json"))
    proc2 = _run_cli("render", "--scene", scene_path)
    out2 = json.loads(proc2.stdout)
    rpt.check("重启仅渲染缺失块",
              out2.get("rendered_tiles") == 1 and out2.get("restored_tiles")
              == len(tile_files) - 1,
              f"rendered={out2.get('rendered_tiles')}, "
              f"restored={out2.get('restored_tiles')}")
    rpt.check("恢复后状态为 resumed", out2.get("status") == "resumed")

    # 重跑后输出图像与首版一致（确定性 + 摘要一致）
    img1 = open(image_path, "rb").read()
    proc3 = _run_cli("render", "--scene", scene_path)
    img2 = open(image_path, "rb").read()
    rpt.check("全量恢复后图像逐字节一致", img1 == img2)

    # --- 6. 版本变更归档 ---------------------------------------------------
    print("\n[6] 版本标识变化触发归档而非覆盖")
    archive_dir = os.path.join(state_dir, "archive")
    before = set(os.listdir(archive_dir)) if os.path.isdir(archive_dir) else set()
    # 修改步长 -> 版本标识改变
    doc = json.load(open(scene_path, encoding="utf-8"))
    doc["render"]["step_length"] = doc["render"]["step_length"] * 0.5
    changed_path = os.path.join(sample_dir, "scene_changed.json")
    json.dump(doc, open(changed_path, "w", encoding="utf-8"))
    _run_cli("render", "--scene", changed_path)
    after = set(os.listdir(archive_dir)) if os.path.isdir(archive_dir) else set()
    rpt.check("旧版本分块已归档", len(after - before) > 0,
              f"归档新增 {len(after - before)} 个文件")

    # --- 7. 非法输入 E_xxx 且无输出文件 -----------------------------------
    print("\n[7] 非法输入业务码与零副作用")
    bad = json.load(open(scene_path, encoding="utf-8"))
    bad["volumes"][0]["sigma_t"] = -3.0
    bad_path = os.path.join(sample_dir, "scene_bad.json")
    json.dump(bad, open(bad_path, "w", encoding="utf-8"))
    out_before = set(os.listdir(output_dir))
    proc_bad = _run_cli("render", "--scene", bad_path)
    out_after = set(os.listdir(output_dir))
    bad_payload = json.loads(proc_bad.stdout)
    rpt.check("非法消光系数返回 E_1003",
              bad_payload.get("error_code") == "E_1003")
    rpt.check("非法输入不产生任何新输出文件", out_before == out_after)

    # 中断标记残留检查
    rpt.check("成功完成后中断标记被清除",
              not interrupt_marker_exists(state_dir))

    return rpt.summary()


def build_tile_count(scene):
    from offscreen_render.scheduler import build_tile_specs
    return [s.index for s in build_tile_specs(scene)]


if __name__ == "__main__":
    raise SystemExit(main())
