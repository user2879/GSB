"""CLI 入口：生成样本、运行演示流程（转换-编辑-增量-全量一致性-报告）。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _demo_edits() -> list[dict]:
    return [
        {"op": "modify_params", "index": 0,
         "params": {"features": ["f_age", "f_click", "f_price", "f_depth"]}},
        {"op": "add_step", "index": 3,
         "step": {"name": "bucket",
                  "params": {"feature": "f_depth", "boundaries": [0.5],
                             "output": "f_depth_bucket"}}},
        {"op": "reorder_steps", "order": [0, 2, 1, 3]},
        {"op": "replace_entity", "entity": "ent_001",
         "features": {"f_age": 0.5, "f_click": 0.25, "f_price": 0.75,
                      "f_ctr": 0.4, "f_depth": 0.6}},
        {"op": "merge_entities", "source": "ent_010", "target": "ent_011"},
        {"op": "remove_step", "index": 3},
    ]


def cmd_generate() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.gen_samples import generate
    generate(ROOT / "data")


def cmd_run(inject: str | None, resume: bool = False) -> None:
    from app.report import write_report
    from app.runner import PipelineRunner

    if not resume:
        import shutil
        shutil.rmtree(ROOT / "output", ignore_errors=True)

    runner = PipelineRunner(ROOT, inject_config=inject)
    runner.start()
    print(f"[启动] 快照状态: {runner.snapshot_status}")

    if resume:
        inc = runner.run(incremental=True)
        full = runner.run(incremental=False)
        same = (inc.final_values == full.final_values
                and inc.final_missing == full.final_missing
                and [s.summary for s in inc.step_results]
                == [s.summary for s in full.step_results])
        print(f"[续跑] 快照恢复后增量 vs 全量: {'一致' if same else '不一致!'}")
        write_report(runner, ROOT / "output" / "report.json")
        if not same:
            sys.exit(1)
        return

    for target in ("v1", "v3", "v2"):
        runner.convert(target)
        print(f"[转换] -> {target}，降级损失清单 {len(runner.ledger)} 条")

    for edit in _demo_edits():
        ok = runner.apply_edit(edit)
        result = runner.run(incremental=True)
        print(f"[编辑] {edit['op']}: {'接受' if ok else '拒绝(E_VERSION_UNSUPPORTED)'} "
              f"复用步骤 {result.reuse} 重算步骤 {result.recompute}")

    runner.apply_injected_edits()
    runner.run(incremental=True)

    digest_inc = runner.state_digest()

    import tempfile
    fresh = PipelineRunner(ROOT, inject_config=inject, use_cache=False,
                           out_dir=Path(tempfile.mkdtemp(prefix="rp_full_")))
    fresh.start()
    for target in ("v1", "v3", "v2"):
        fresh.convert(target)
    for edit in _demo_edits():
        fresh.apply_edit(edit)
    fresh.apply_injected_edits()
    fresh.run(incremental=False)
    digest_full = fresh.state_digest()

    consistent = digest_inc == digest_full
    print(f"[一致性] 增量 vs 全量重建: {'一致' if consistent else '不一致!'}")

    report = write_report(runner, ROOT / "output" / "report.json")
    print(f"[报告] output/report.json 已写出："
          f"注入命中 {report['injected_hits']}，失败 {len(report['failures'])} 条，"
          f"降级损失 {len(report['downgrade_ledger'])} 条")
    if not consistent:
        sys.exit(1)


def cmd_test() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT)
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="推荐特征管道多版本互操作应用")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="生成固定种子样本")
    run_p = sub.add_parser("run", help="运行演示流程")
    run_p.add_argument("--inject", default=str(ROOT / "config" / "inject_config.json"))
    run_p.add_argument("--resume", action="store_true", help="从快照续跑")
    sub.add_parser("test", help="运行测试套件")
    args = parser.parse_args()
    if args.cmd == "generate":
        cmd_generate()
    elif args.cmd == "run":
        cmd_run(args.inject, args.resume)
    elif args.cmd == "test":
        cmd_test()


if __name__ == "__main__":
    main()
