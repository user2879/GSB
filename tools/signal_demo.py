"""Demonstrate graceful termination: the runner sends itself SIGTERM after
starting the pipeline and must still finish the batch and manifest swap with
exit code 0."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_pipeline.pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pipeline = Pipeline(
        sample_dir=root / "samples",
        config_path=root / "samples" / "merge_config.json",
        metadata_path=root / "samples" / "feature_metadata.csv",
        work_dir=args.work_dir,
    )

    def send_term() -> None:
        time.sleep(0.05)
        os.kill(os.getpid(), 15)  # SIGTERM

    threading.Thread(target=send_term, daemon=True).start()
    summary = pipeline.run()
    assert summary["graceful_exit_requested"]
    print(f"received {summary['exit_signal']}")
    print("graceful shutdown after batch and manifest replacement")
    print(f"committed global_version={summary['global_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
