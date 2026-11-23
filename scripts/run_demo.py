#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaseaudit.demo import run_demo  # noqa: E402
from leaseaudit.errors import BusinessError  # noqa: E402
from leaseaudit.logging_utils import StructuredLogger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic loopback lease demo")
    parser.add_argument("--workdir", default=str(ROOT / "artifacts" / "demo"))
    parser.add_argument("--template", default=str(ROOT / "scenarios" / "demo_scene.json"))
    parser.add_argument("--base-port", type=int, default=55441)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logger = StructuredLogger(Path(args.workdir) / "structured.jsonl")
    try:
        summary = run_demo(
            args.workdir,
            template_path=args.template,
            base_port=args.base_port,
            fresh=not args.keep_existing,
        )
    except BusinessError as exc:
        logger.error(exc)
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True))
        return 2
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    report = Path(args.workdir) / "report.txt"
    print(report.read_text(encoding="utf-8"), end="")
    return 0 if summary["overall_code"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
