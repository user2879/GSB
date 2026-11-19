"""命令行入口：在给定样本目录上执行一次（可续跑）流水线。

用法:
  python examples/run_pipeline.py --sample-dir examples/sample_data \
      --snapshot-dir examples/output/cli/snapshot \
      --report examples/output/cli/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segpos import ComponentConfig, Dictionary, ProbabilityTable, IncrementalPipeline  # noqa: E402
from segpos.errors import ConfigMismatchError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中文分词与词性标注增量流水线")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--table-version", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    config = ComponentConfig.load(sample_dir / "config.json")
    table = ProbabilityTable.load(
        sample_dir / "prob_table.json", version=args.table_version
    )
    dictionary = Dictionary.load(
        sample_dir / "dictionary.txt", allowed_pos=set(table.tags)
    )
    pipe = IncrementalPipeline(
        config=config,
        dictionary=dictionary,
        prob_table=table,
        snapshot_dir=args.snapshot_dir,
        sentences_path=sample_dir / "sentences.txt",
    )
    pipe.install_signal_handlers()
    try:
        outcome = pipe.run(args.report)
    except ConfigMismatchError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    finally:
        pipe.restore_signal_handlers()

    print(json.dumps(outcome.report, ensure_ascii=False, indent=2, sort_keys=True))
    if outcome.stopped_early:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
