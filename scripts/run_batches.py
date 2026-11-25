"""按预算分批运行链路（可被强制 kill，仅依赖快照恢复）。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from voicecmd.config import load_config
from voicecmd.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--config", default="configs/pipeline.json")
    parser.add_argument("--injection", default=None)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-batches", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(Path(args.base_dir) / args.config)
    injection = None
    if args.injection:
        from voicecmd.injection import load_injection_config
        injection = load_injection_config(Path(args.base_dir) / args.injection)
    pipeline = Pipeline(cfg, args.base_dir, injection_config=injection)
    pipeline.load_all()
    for _ in range(args.max_batches):
        pipeline.run_batch(budget=args.budget)
        if not pipeline.pending and not pipeline.invalid_windows:
            break


if __name__ == "__main__":
    main()
