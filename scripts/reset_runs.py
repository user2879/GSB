from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pagecache.fsutil import remove_tree


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear historical local run artifacts")
    parser.add_argument("--include-acceptance", action="store_true")
    parser.add_argument("--include-samples", action="store_true")
    args = parser.parse_args()
    targets = [ROOT / "artifacts" / "runs"]
    if args.include_acceptance:
        targets.append(ROOT / "artifacts" / "acceptance")
    if args.include_samples:
        targets.append(ROOT / "samples" / "data")
    for target in targets:
        remove_tree(target)
        print(f"cleared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
