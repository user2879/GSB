"""一键验收：增量检索与全量重建逐字段一致 + 账本/预算/注入业务码检查。

用法：
    python tools/run_acceptance.py [--keep]

产物全部写入 workspace/acceptance/，不访问网络。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vrep.engine import Engine
from vrep.config import Config
from vrep.injection import InjectionManager
from vrep.jsonio import canonical, read_json, write_json
from vrep.samplegen import generate_all


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def comparable(ledger: dict) -> dict:
    return {
        "config_hash": ledger["config_hash"],
        "quant_level": ledger["quant_level"],
        "images": ledger["images"],
        "index": ledger["index"],
        "failures": ledger["failures"],
        "pending": ledger["runtime"]["pending"],
    }


def drain(engine: Engine, config: Config, start_batch: int = 1) -> int:
    batch_no = start_batch
    engine.run_batch(config, batch_no=batch_no)
    while engine.ledger["runtime"]["pending"]:
        before = len(engine.ledger["runtime"]["pending"])
        batch_no += 1
        engine.run_batch(config, batch_no=batch_no)
        if len(engine.ledger["runtime"]["pending"]) >= before:
            break
    return batch_no


def main() -> int:
    parser = argparse.ArgumentParser(description="vrep 一键验收")
    parser.add_argument("--keep", action="store_true", help="保留工作区产物")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    base = os.path.join(ROOT, "workspace", "acceptance")
    if not args.keep and os.path.exists(base):
        shutil.rmtree(base)
    samples = os.path.join(base, "samples")
    summary = generate_all(samples, image_size=args.image_size, count=args.count)
    print(f"[1/6] 生成样本: {summary['image_count']} 张 {args.image_size}px 图像，"
          f"分块 {summary['tile_size']}px，命令 {summary['commands']} 条")

    commands = read_json(os.path.join(samples, "commands.json"))["commands"]
    config = Config.load(os.path.join(samples, "config.json"),
                         image_size=args.image_size)

    # ---- 全量重建 ----
    full_ws = os.path.join(base, "full")
    full = Engine(full_ws, InjectionManager(None))
    full.seed_library(samples)
    for command in commands:
        if command["op"] != "config":
            full.apply_command(command, samples)
    drain(full, config)
    print(f"[2/6] 全量重建完成: v{full.ledger['global_version']}，"
          f"索引 {len(full.ledger['index']['images'])} 张")

    # ---- 增量执行 ----
    incr_ws = os.path.join(base, "incr")
    incr = Engine(incr_ws, InjectionManager(None))
    incr.seed_library(samples)
    live_config = config
    batch_no = 0
    for command in commands:
        batch_no += 1
        if command["op"] == "config":
            candidate = command["config_path"]
            if not os.path.isabs(candidate):
                candidate = os.path.join(samples, candidate)
            updated, failure = incr.load_config(
                candidate, allow_quant_injection=True
            )
            if failure is None:
                live_config = updated
        else:
            incr.apply_command(command, samples)
        incr.run_batch(live_config, batch_no=batch_no)
    drain(incr, live_config, batch_no + 1)
    print(f"[3/6] 增量执行完成: v{incr.ledger['global_version']}，"
          f"复用事件 {len(incr.ledger['runtime']['reuse'])}，"
          f"重算事件 {len(incr.ledger['runtime']['recompute'])}")

    # ---- 逐字段比对 ----
    full_view = comparable(full.ledger)
    incr_view = comparable(incr.ledger)
    if canonical(full_view) != canonical(incr_view):
        print("[FAIL] 增量与全量账本逐字段不一致")
        return 1
    print("[4/6] 逐字段一致 ✔ 嵌入向量/缓存键/检索顺序/边界样本/失败清单/待办全部匹配")
    print(f"       边界样本: {full_view['index']['boundary_set']}")

    # ---- 预算审计 ----
    audit = incr.ledger["runtime"]["budget_audit"]
    per_batch: dict[int, int] = {}
    for event in audit:
        if event["event"] == "consume":
            per_batch[event["batch"]] = per_batch.get(event["batch"], 0) + 1
    budget = config.point_budget_per_batch
    overrun = [batch for batch, count in per_batch.items() if count > budget]
    if overrun:
        print(f"[FAIL] 批次 {overrun} 超过预算上限 {budget}")
        return 1
    print(f"[5/6] 预算审计 ✔ 单批上限 {budget}，审计事件 {len(audit)} 条，"
          f"最终待办 {len(incr.ledger['runtime']['pending'])} 项")

    # ---- 故障注入 ----
    inj_ws = os.path.join(base, "injected")
    injections = InjectionManager(os.path.join(samples, "injections.json"))
    injected = Engine(inj_ws, injections)
    injected.seed_library(samples)
    live_config = config
    batch_no = 0
    for command in commands:
        batch_no += 1
        if command["op"] == "config":
            candidate = command["config_path"]
            if not os.path.isabs(candidate):
                candidate = os.path.join(samples, candidate)
            updated, failure = injected.load_config(
                candidate, allow_quant_injection=True
            )
            if failure is not None and failure.injection_id:
                injected.record_config_failure(failure, batch_no)
            elif failure is None:
                live_config = updated
        else:
            injected.apply_command(command, samples)
        injected.run_batch(live_config, batch_no=batch_no)
    diagnostics = injections.diagnostics()
    write_json(os.path.join(inj_ws, "diagnostics.json"), diagnostics)
    if not diagnostics["all_hit"]:
        print("[FAIL] 存在未命中的注入项")
        return 1
    expected = {
        "INJ-001": "E_IMAGE_FORMAT",
        "INJ-002": "E_TRUTH_INVALID",
        "INJ-003": "E_NONFINITE_FEATURE",
        "INJ-004": "E_QUANT_LEVEL",
    }
    for item in diagnostics["items"]:
        assert item["expected_code"] == expected[item["id"]]
    print(f"[6/6] 故障注入 ✔ {diagnostics['hit_count']}/"
          f"{diagnostics['injection_count']} 命中，业务码映射全部正确")

    report = {
        "sample_summary": summary,
        "full_version": full.ledger["global_version"],
        "incremental_version": incr.ledger["global_version"],
        "fieldwise_equal": True,
        "boundary_samples": full_view["index"]["boundary_set"],
        "budget_limit": budget,
        "budget_audit_events": len(audit),
        "final_pending": len(incr.ledger["runtime"]["pending"]),
        "injection_diagnostics": diagnostics,
        "reuse_events": len(incr.ledger["runtime"]["reuse"]),
        "recompute_events": len(incr.ledger["runtime"]["recompute"]),
    }
    report_path = os.path.join(base, "acceptance_report.json")
    write_json(report_path, report)
    print(f"\n验收通过 ✔ 报告: {os.path.relpath(report_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
