"""Command line entry point for the full pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_pipeline.pipeline import Pipeline
from feature_pipeline.snapshot import SnapshotStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-source feature pipeline")
    parser.add_argument("--sample-dir", default="samples")
    parser.add_argument("--config", default="samples/merge_config.json")
    parser.add_argument("--metadata", default="samples/feature_metadata.csv")
    parser.add_argument("--work-dir", default="artifacts/run")
    parser.add_argument(
        "--recovery-info",
        action="store_true",
        help="print recovery point from the last fully replaced snapshot and exit",
    )
    args = parser.parse_args()

    if args.recovery_info:
        store = SnapshotStore(Path(args.work_dir) / "snapshot_store")
        print(json.dumps(store.recovery_point(), ensure_ascii=False, indent=2,
                         sort_keys=True))
        return 0

    pipeline = Pipeline(
        sample_dir=args.sample_dir,
        config_path=args.config,
        metadata_path=args.metadata,
        work_dir=args.work_dir,
    )
    summary = pipeline.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
