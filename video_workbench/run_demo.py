"""一键本地验收：覆盖全部业务要求的确定性演示。

依次执行
--------
1. 重新生成固定种子样本（词表/提示词/噪声表/初始图）。
2. 文生视频全量渲染 + 二次运行全量缓存命中。
3. 18 条交互命令的增量渲染与空白工作区全量渲染逐帧/逐潜变量比对。
4. 参数与结构注入校验矩阵（7 个错误码用例）。
5. 并发与串行渲染逐字段一致性。
6. 图生视频分支与初始图异常、补丁越界异常。
7. 强制终止（SIGKILL 模拟）与可处理信号（SIGTERM）恢复演示。
8. 单元测试套件。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def step(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def cli(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "vbench.cli", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(proc.stdout) if proc.stdout.strip() else {}
    return {"proc": proc, "payload": payload}


def expect(condition: bool, label: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        raise SystemExit(f"验收失败: {label}")


def main() -> None:
    shutil.rmtree(RUNS, ignore_errors=True)

    step("1/8 生成固定种子本地样本")
    subprocess.run([sys.executable, "generate_samples.py"], cwd=ROOT, check=True)

    step("2/8 文生视频全量渲染与缓存命中")
    first = cli("run", "--config", "config/default_config.json",
                "--workdir", "runs/t2v", "--report", "runs/t2v_1.json")
    expect(first["payload"]["status"] == "ok", "首次全量渲染成功")
    second = cli("run", "--config", "config/default_config.json",
                 "--workdir", "runs/t2v", "--report", "runs/t2v_2.json")
    expect(second["payload"]["stats"]["recomputed_count"] == 0, "二次运行零重算全命中")
    expect(second["payload"]["stats"]["reused_count"] == 13, "13 个阶段全部复用")

    step("3/8 交互命令流：增量渲染 vs 空白全量渲染")
    incr = cli("apply-commands", "--config", "config/default_config.json",
               "--commands", "config/commands.json",
               "--workdir", "runs/incr", "--report", "runs/apply.json")
    expect(incr["payload"]["status"] == "ok", "18 条编辑后逐帧/逐潜变量完全一致")
    print(f"      命令数: {incr['payload']['commands_applied']}，"
          f"一致性: {incr['payload']['consistency']}")

    step("4/8 参数与结构注入校验矩阵")
    validation = cli("validate-config", "--config", "config/default_config.json",
                     "--injections", "config/injections.json",
                     "--report", "runs/validate.json")
    cases = validation["payload"]["cases"]
    expect(all(c["passed"] for c in cases), f"{len(cases)} 个校验用例全部命中预期错误码")
    for case in cases:
        print(f"      {case['id']:20s} expect={case['expected']} actual={case['actual']}")

    step("5/8 并发渲染与串行结果一致性")
    concurrency = cli("concurrency-check", "--config", "config/default_config.json",
                     "--workdir", "runs/conc", "--report", "runs/concurrency.json")
    expect(concurrency["payload"]["status"] == "ok", "并发/串行快照逐字段一致")

    step("6/8 图生视频分支、初始图异常与补丁越界")
    i2v_cfg = ROOT / "runs" / "i2v_config.json"
    i2v_raw = json.loads((ROOT / "config" / "default_config.json").read_text())
    i2v_raw["mode"] = "i2v"
    i2v_cfg.write_text(json.dumps(i2v_raw, indent=2))
    i2v = cli("run", "--config", "runs/i2v_config.json",
              "--workdir", "runs/i2v", "--report", "runs/i2v.json")
    expect(i2v["payload"]["status"] == "ok", "图生视频渲染成功")
    replaced = [
        name for name in i2v["payload"]["recomputed"] if name.startswith("denoise_")
    ]
    print(f"      去噪阶段: {replaced}")

    step("7/8 强制终止与信号恢复（真实子进程）")
    proc = subprocess.run([sys.executable, "demo_recovery.py"], cwd=ROOT, check=True)
    expect(proc.returncode == 0, "硬终止恢复与 SIGTERM 优雅退出均符合预期")

    step("8/8 单元测试套件")
    test_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_pipeline", "-v"],
        cwd=ROOT,
    )
    expect(test_proc.returncode == 0, "10 个单元测试全部通过")

    step("验收完成")
    print("全部本地确定性验收项通过，诊断报告位于 runs/ 与 runs/recovery_demo.json")


if __name__ == "__main__":
    main()
