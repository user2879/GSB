"""端到端实验入口：加载配置/张量包/注入清单，跑多批并输出可复现报告。

用法：
    python scripts/run_experiment.py \
        --config configs/operator_config.json \
        --pack data/sample_pack.fatp \
        --injections configs/injection_manifest.json \
        --batches 3
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fablock.config import OperatorConfig
from fablock.dataset import SampleDataset
from fablock.injection import load_manifest
from fablock.pipeline import Pipeline
from fablock.snapshot import SnapshotManager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="分块注意力算子验证实验")
    parser.add_argument("--config", default=os.path.join(
        ROOT, "configs", "operator_config.json"))
    parser.add_argument("--pack", default=os.path.join(
        ROOT, "data", "sample_pack.fatp"))
    parser.add_argument("--injections", default=None)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--snapshots", default=os.path.join(ROOT, "snapshots"))
    parser.add_argument("--reports", default=os.path.join(ROOT, "reports"))
    args = parser.parse_args()

    cfg = OperatorConfig.load(args.config)
    issues = cfg.validate()
    if issues:
        print("配置校验未通过:")
        for i in issues:
            print(" ", i.to_dict())
        sys.exit(2)

    import hashlib
    with open(args.pack, "rb") as fh:
        pack_fp = hashlib.sha256(fh.read()).hexdigest()[:16]
    dataset = SampleDataset.from_pack(args.pack, cfg.query_block_len,
                                      cfg.key_block_len,
                                      cfg.burst_query_blocks)
    injections = load_manifest(args.injections) if args.injections else []
    snapshot = SnapshotManager(args.snapshots, cfg.config_hash(), pack_fp,
                               cfg.quantization_level_bits)
    pipeline = Pipeline(cfg, dataset, injections, snapshot_mgr=snapshot,
                        reports_dir=args.reports, tensor_pack_fp=pack_fp)
    pipeline.install_signal_handlers()
    reports = pipeline.run(args.batches)

    for r in reports:
        if r.get("batch") is None:
            print("续跑判定:", r["resume_issues"])
            continue
        cmp_stats = r.get("compare", {})
        print(f"批次 {r['batch']}: "
              f"超差块={cmp_stats.get('out_of_tolerance_count', '-')} "
              f"复用={len(r.get('cache', {}).get('reuse', []))} "
              f"重算={len(r.get('cache', {}).get('recompute', []))} "
              f"失败={len(r['failures'])} 注入命中={len(r['injection_hits'])} "
              f"溢出={len(r['overflows'])} 并发峰值={r['peak_concurrency']}")
    manifest = snapshot.load_manifest()
    print(f"快照清单: 游标={manifest['completed_batch_cursor']} "
          f"产物版本={manifest['artifact_versions']}")
    print(f"报告目录: {args.reports}")


if __name__ == "__main__":
    main()
