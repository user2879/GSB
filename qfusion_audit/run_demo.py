"""一键验收演示（不依赖 pytest，直接 python run_demo.py）。

覆盖验收四件套：
1. 跨位宽/跨档位量化审计账本与环节归因；
2. 排序翻转归因与边界对随档位收紧增加；
3. 单批预算待办续跑与足额预算逐字段一致；
4. 快照整体替换、版本识别（E_SNAPSHOT_STALE）与审计报告。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

from qfusion.config import load_config
from qfusion.errors import E_SNAPSHOT_STALE
from qfusion.loader import load_stream
from qfusion.pipeline import Pipeline
from qfusion.serialization import canonical_json, read_json
from qfusion.snapshot import SnapshotStore

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(ROOT, "samples", "input")
TMP = os.path.join(ROOT, "samples", "snapshot_tmp")
COMMIT = os.path.join(ROOT, "samples", "snapshot_commit")


def reset_dirs():
    for d in (TMP, COMMIT):
        os.makedirs(d, exist_ok=True)
        for name in os.listdir(d):
            p = os.path.join(d, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


def run(cfg, stream, budget):
    store = SnapshotStore(TMP, COMMIT)
    manifest, stale = store.validate(cfg.hash())
    start = 1 if manifest is None else manifest["global_version"] + 1
    return Pipeline(cfg, store).run(stream, budget, start_version=start),         [f.to_dict() for f in stale]


def payloads():
    cur = os.path.realpath(os.path.join(COMMIT, "current"))
    return {s: read_json(os.path.join(cur, s, "data.json"))
            for s in ("ledger", "flips", "boundary")}


def main():
    reset_dirs()
    cfg, cfg_failures = load_config(os.path.join(INPUT, "fusion_config.json"))
    assert not cfg_failures, cfg_failures
    stream = load_stream(os.path.join(INPUT, "candidates.jsonl"), cfg)
    assert not stream.failures, stream.failures

    print("== 1) 小预算多批续跑（制造 E_RECOMPUTE_BUDGET_PENDING 待办） ==")
    report_tiny, _ = run(cfg, stream, budget=13)
    print(f"批次数={len(report_tiny['batch_ledgers'])} "
          f"总消耗点={report_tiny['budget_consumed_points']} "
          f"累计保留项={report_tiny['budget_pending_count']}")
    tiny = payloads()

    print("== 2) 全新临时目录一次性足额预算，比对逐字段一致 ==")
    full_work = tempfile.mkdtemp(prefix="qfusion_full_")
    try:
        full_store = SnapshotStore(os.path.join(full_work, "tmp"),
                                   os.path.join(full_work, "commit"))
        Pipeline(cfg, full_store).run(stream, 1_000_000)
        cur_full = os.path.realpath(os.path.join(
            full_work, "commit", "current"))
        full = {s: read_json(os.path.join(cur_full, s, "data.json"))
                for s in ("ledger", "flips", "boundary")}
        for stage in ("ledger", "flips", "boundary"):
            assert canonical_json(tiny[stage]) == canonical_json(full[stage])
        print("续跑与足额预算 ledger/flips/boundary 逐字段一致: OK")
    finally:
        shutil.rmtree(full_work, ignore_errors=True)

    print("== 3) 跨档位账本与翻转归因 ==")
    print(f"逐环节最大定点差值(公共单位)="
          f"{report_tiny['stage_max_fixed_point_diff']}")
    print(f"翻转候选总数={report_tiny['flipped_candidate_count']} "
          f"(同位宽跨档位={report_tiny['flipped_same_frac_gear_count']}, "
          f"跨位宽={report_tiny['flipped_cross_frac_count']})")
    sample_flip = next(
        f for req in tiny["flips"]["requests"] for f in req["flips"])
    print("翻转样例:", json.dumps(sample_flip, ensure_ascii=False,
                                  sort_keys=True))

    print("== 4) 边界对随档位收紧单调增加 ==")
    counts = {}
    for req in tiny["boundary"]["requests"]:
        for pair in req["boundary_pairs"]:
            key = (pair["frac_bits"], pair["gear"])
            counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f"  位宽={key[0]} 档位={key[1]} 边界对数={counts[key]}")

    print("== 5) 快照版本识别：切换配置副本触发 E_SNAPSHOT_STALE ==")
    cfg_v2, _ = load_config(os.path.join(
        ROOT, "samples", "config_versions",
        "fusion_config_v2_weight.json"))
    store = SnapshotStore(TMP, COMMIT)
    manifest, failures = store.validate(cfg_v2.hash())
    assert manifest is None
    assert any(f.code == E_SNAPSHOT_STALE for f in failures)
    print("配置哈希不符 -> E_SNAPSHOT_STALE，旧产物已隔离:",
          failures[0].reason)

    print("\n全部验收点通过。审计报告: samples/ 下各 run 输出与快照产物。")


if __name__ == "__main__":
    main()
