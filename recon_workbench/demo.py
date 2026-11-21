"""端到端验收演示：生成样本 -> 双精度重建 -> 增量 -> 注入 -> 快照恢复。

运行：python demo.py
所有产物写入系统临时目录下的 rwb_demo_*，秒级完成，无网络依赖。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

from recon_workbench.artifact_store import ArtifactStore
from recon_workbench.config import RebuildConfig
from recon_workbench.editing import apply_edit_sequence
from recon_workbench.engine import WorkbenchEngine
from recon_workbench.errors import WorkbenchError
from recon_workbench.generate_sample import generate
from recon_workbench.injection import apply_injections, load_injection_config
from recon_workbench.loader import input_fingerprint, load_sample
from recon_workbench.precision_analysis import analyze
from recon_workbench.runner import ReconstructionRunner
from recon_workbench.utils import atomic_write_json
from recon_workbench.visualization import render_all


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    root = tempfile.mkdtemp(prefix="rwb_demo_")
    sample = os.path.join(root, "sample")
    work = os.path.join(root, "work")
    generate(sample)
    config = RebuildConfig()
    atomic_write_json(
        os.path.join(sample, "rebuild_config.json"), config.to_dict()
    )
    print("工作目录:", root)

    section("1. 全量双精度重建与逐级误差归因")
    load = load_sample(sample)
    store = ArtifactStore(work)
    tracks: dict = {}
    for precision in ("float32", "float64"):
        raw = WorkbenchEngine(
            load, config, store, concurrency=config.thread_count
        ).run(precision, version=1)
        tracks[precision] = {
            "features_flat": raw.features_flat,
            "matches_flat": raw.matches_flat,
            "tripoints_flat": raw.tripoints_flat,
            "depth_maps": raw.depth_maps,
            "cloud": raw.cloud, "raw": raw,
        }
    analysis = analyze(tracks, config)
    for stage in analysis["stage_errors"]:
        print(
            f"  {stage['stage']:12s} abs={stage['max_abs_error']:.3e} "
            f"rel={stage['max_rel_error']:.3e} n={stage['item_count']}"
            f"{'  <== 首次超容差' if stage['first_breach'] else ''}"
        )
    print("  边界样本数:", len(analysis["boundary_samples"]))
    viz = os.path.join(root, "viz")
    render_all(load.views, tracks, viz)
    print("  可视化产物:", len(os.listdir(viz)), "张 PNG @", viz)

    section("2. 增量重算（替换视图 2 图像）")
    edited = os.path.join(root, "edited")
    apply_edit_sequence(
        sample, edited,
        [{"command": "replace_image", "view": 2, "x": 40, "y": 40,
          "size": 18, "color": [255, 0, 0]}],
    )
    engine2 = WorkbenchEngine(
        load_sample(edited), config, ArtifactStore(work), concurrency=1
    )
    engine2.run("float64", version=2)
    decision = engine2.last_context.decision
    print("  复用缓存键:", len(decision.reused))
    print("  重算缓存键:", len(decision.recomputed))

    section("3. 异常注入隔离")
    injection_path = os.path.join(root, "injections.json")
    atomic_write_json(injection_path, {
        "injections": [
            {"type": "truncate_png", "view": 1, "keep_ratio": 0.3},
            {"type": "nonfinite_descriptor", "view": 0},
            {"type": "zero_voxel_size", "view": 2},
        ]
    })
    items = load_injection_config(injection_path)
    injected = os.path.join(root, "injected")
    hits = apply_injections(sample, injected, items)
    injected_load = load_sample(injected)
    from recon_workbench.cli import _attach_injection_markers

    _attach_injection_markers(injected, injected_load)
    cfg_inj = RebuildConfig()
    if any(i["type"] == "zero_voxel_size" for i in items):
        object.__setattr__(cfg_inj, "voxel_size", 0.0)
    report = ReconstructionRunner(
        injected_load, cfg_inj, os.path.join(root, "inj_work"),
        injection_hits=hits, concurrency=1,
    ).execute(report_path=os.path.join(root, "report.json")).report
    for hit in report["injection_hits"]:
        print(f"  {hit['injection_id']} {hit['injection_type']:24s} "
              f"hit={hit['hit']} code={hit['expected_code']}")

    section("4. 快照不匹配恢复")
    snap_store = ArtifactStore(work)
    try:
        snap_store.verify_resume(
            1, "ffffffffffffffff",
            input_fingerprint(load_sample(sample).views),
        )
    except WorkbenchError as exc:
        print("  捕获业务码:", exc.code, "|", exc.reason)
    print("  恢复策略：废弃不匹配版本 -> 从最近有效快照整体替换重算")

    section("诊断报告位置")
    print(" ", os.path.join(root, "report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
