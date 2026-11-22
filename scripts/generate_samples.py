from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowdiag.config import load_config
from slowdiag.generator import check_repeatable_generation, generate_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local source samples")
    parser.add_argument("--workspace", default="workbench")
    parser.add_argument("--config", default=None)
    parser.add_argument("--check-repeat", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve(strict=False)
    config_path = (
        Path(args.config).resolve(strict=False)
        if args.config
        else workspace / "slow-sources.json"
    )
    config = load_config(config_path, workspace)
    summaries = generate_all(config)
    output = {
        "business_code": "OK",
        "sources": summaries,
        "repeatability": check_repeatable_generation(config)
        if args.check_repeat
        else None,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
