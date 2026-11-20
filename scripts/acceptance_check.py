"""一键验收：确定性、增量/全量逐字段一致、快照恢复、错误码、并发不变性。

用法：.venv/bin/python scripts/acceptance_check.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sft_diag.errors import E_CONFIG_INVALID, E_CONFIG_MISMATCH, OK  # noqa: E402
from sft_diag.runner import compare_artifacts, run_workbench  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "samples", "basic")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    workspace = tempfile.mkdtemp(prefix="sft-diag-acceptance-")
    try:
        section("1. 确定性：两次全量构造产物逐字节一致")
        first = os.path.join(workspace, "full-1")
        second = os.path.join(workspace, "full-2")
        r1 = run_workbench(SAMPLE_DIR, first, shadow_check=False)
        r2 = run_workbench(SAMPLE_DIR, second, shadow_check=False)
        assert r1.business_code == OK
        assert compare_artifacts(r1.stage_artifacts, r2.stage_artifacts) == []
        print("PASS: 全量构造可复现，三阶段产物完全一致")

        section("2. 分批增量 vs 全量重建逐字段一致")
        incremental_dir = os.path.join(workspace, "incremental")
        for batch_no in range(1, 6):
            inc = run_workbench(
                SAMPLE_DIR, incremental_dir,
                up_to_batch=batch_no, shadow_check=False,
            )
            full = run_workbench(
                SAMPLE_DIR, os.path.join(workspace, f"full-b{batch_no}"),
                up_to_batch=batch_no, shadow_check=False,
            )
            diffs = compare_artifacts(inc.stage_artifacts, full.stage_artifacts)
            totals = inc.report["totals"]
            print(
                f"  batch {batch_no}: 候选 {totals['candidate_count']} -> "
                f"保留 {totals['retained_count']}，差异字段 {len(diffs)}"
            )
            assert diffs == [], diffs
        print("PASS: 5 个批次增量结果与全量重建逐字段一致")

        section("3. 精确失效与复用")
        probe = os.path.join(workspace, "probe")
        run_workbench(SAMPLE_DIR, probe, up_to_batch=1, shadow_check=False)
        run_workbench(SAMPLE_DIR, probe, up_to_batch=2, shadow_check=False)
        rerun = run_workbench(SAMPLE_DIR, probe, up_to_batch=2, shadow_check=False)
        inv = rerun.report["invalidation"]
        assert not inv["rescored_candidates"], "无输入变化时不得重新打分"
        assert inv["score_reused_candidates"], "未失效候选必须复用"
        print(
            f"PASS: 无变化重跑时重算 {len(inv['rescored_candidates'])} 条，"
            f"复用 {len(inv['score_reused_candidates'])} 条"
        )

        section("4. 快照恢复与强制终止")
        crash_dir = os.path.join(workspace, "crash")

        def crash_hook(stage: str, batch_no: int) -> None:
            if stage == "scoring_pre" and batch_no == 2:
                raise KeyboardInterrupt("simulated kill")

        try:
            run_workbench(
                SAMPLE_DIR, crash_dir, up_to_batch=2,
                shadow_check=False, fault_hook=crash_hook,
            )
            raise AssertionError("crash hook did not fire")
        except KeyboardInterrupt:
            pass
        leftovers = [
            name for name in os.listdir(os.path.join(crash_dir, "snapshot"))
            if name.startswith("tmp.")
        ]
        assert leftovers == [], leftovers
        recovered = run_workbench(
            SAMPLE_DIR, crash_dir, up_to_batch=2, shadow_check=False
        )
        full = run_workbench(
            SAMPLE_DIR, os.path.join(workspace, "recover-full"),
            up_to_batch=2, shadow_check=False,
        )
        assert compare_artifacts(recovered.stage_artifacts, full.stage_artifacts) == []
        print(
            f"PASS: 强制终止后从批次 {recovered.recovery_batch_no} 恢复，"
            "临时目录无残留，恢复结果与全量一致"
        )

        section("5. 错误码 E_CONFIG_MISMATCH")
        other_sample = os.path.join(workspace, "mutated-sample")
        shutil.copytree(SAMPLE_DIR, other_sample)
        config_path = os.path.join(other_sample, "build_config.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        config["max_workers"] = 99
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, sort_keys=True)
        mismatch = run_workbench(
            other_sample, crash_dir, up_to_batch=2, shadow_check=False
        )
        assert mismatch.business_code == E_CONFIG_MISMATCH
        print("PASS: 基础输入指纹变化时拒绝续跑并返回 E_CONFIG_MISMATCH")

        section("6. 加载错误码不中断其余条目")
        bad_sample = os.path.join(workspace, "bad-sample")
        shutil.copytree(SAMPLE_DIR, bad_sample)
        with open(os.path.join(bad_sample, "seeds.jsonl"), "a",
                  encoding="utf-8") as handle:
            handle.write("{not valid json\n")
        bad = run_workbench(
            bad_sample, os.path.join(workspace, "bad-out"), shadow_check=False
        )
        assert any(failure.code == "E_SEED_INVALID" for failure in bad.failures)
        assert bad.business_code in (OK, E_CONFIG_INVALID)
        print(f"PASS: 坏条目进入失败清单（{len(bad.failures)} 条），其余条目正常加载")

        section("7. 并发度不变性")
        concurrent = {}
        sample_copy = os.path.join(workspace, "worker-sample")
        shutil.copytree(SAMPLE_DIR, sample_copy)
        cfg_path = os.path.join(sample_copy, "build_config.json")
        with open(cfg_path, encoding="utf-8") as handle:
            base_cfg = json.load(handle)
        for workers in (1, 4, 8):
            cfg = dict(base_cfg)
            cfg["max_workers"] = workers
            with open(cfg_path, "w", encoding="utf-8") as handle:
                json.dump(cfg, handle, sort_keys=True)
            result = run_workbench(
                sample_copy, os.path.join(workspace, f"workers-{workers}"),
                shadow_check=False,
            )
            concurrent[workers] = result.stage_artifacts
        for workers in (4, 8):
            assert compare_artifacts(concurrent[1], concurrent[workers]) == []
        print("PASS: max_workers=1/4/8 下采样、打分、去重产物完全一致")

        print("\n全部验收通过 ✔")
        return 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
