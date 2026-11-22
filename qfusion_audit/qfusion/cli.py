"""命令行入口：生成样本、运行审计、输出 JSON 报告。"""
from __future__ import annotations

import argparse
import json
import os

from .config import load_config
from .loader import load_stream
from .pipeline import Pipeline
from .snapshot import SnapshotStore
from .serialization import write_json


def run(input_dir: str, tmp_dir: str, commit_dir: str,
        batch_budget: int | None = None,
        report_path: str | None = None) -> dict:
    cfg, cfg_failures = load_config(os.path.join(input_dir,
                                                 "fusion_config.json"))
    if cfg is None:
        return {"status": "config_invalid",
                "failures": [f.to_dict() for f in cfg_failures]}

    store = SnapshotStore(tmp_dir, commit_dir)
    manifest, stale_failures = store.validate(cfg.hash())
    if stale_failures:
        # E_SNAPSHOT_STALE：旧产物已隔离丢弃，下面全量重建。
        pass
    start_version = 1 if manifest is None else manifest["global_version"] + 1

    stream = load_stream(os.path.join(input_dir, "candidates.jsonl"), cfg)
    pipeline = Pipeline(cfg, store)
    budget = cfg.recompute_budget if batch_budget is None else batch_budget
    report = pipeline.run(stream, budget, start_version=start_version)
    report["config_failures"] = [f.to_dict() for f in cfg_failures]
    report["snapshot_failures"] = [f.to_dict() for f in stale_failures]

    if report_path:
        write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="多路召回融合量化溯源")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--input-dir",
                        default=os.path.join(base, "samples", "input"))
    parser.add_argument("--tmp-dir",
                        default=os.path.join(base, "samples", "snapshot_tmp"))
    parser.add_argument("--commit-dir",
                        default=os.path.join(base, "samples", "snapshot_commit"))
    parser.add_argument("--batch-budget", type=int, default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report = run(args.input_dir, args.tmp_dir, args.commit_dir,
                 args.batch_budget, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
