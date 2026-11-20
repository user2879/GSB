"""命令行: full / interactive / inject 三种模式。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_full_rebuild, run_interactive


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="本地确定性精排与重排组件")
    parser.add_argument("mode", choices=["full", "interactive"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--commands")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    if args.mode == "full":
        report = run_full_rebuild(args.input, args.snapshot)
    else:
        if not args.commands:
            parser.error("interactive 模式需要 --commands")
        report = run_interactive(args.input, args.snapshot, args.commands)
    write_report(report, args.report)
    print(f"报告已写入 {args.report}")


if __name__ == "__main__":
    main()
