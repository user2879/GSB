"""命令行入口：全量运行、交互回放、恢复点查询与状态清理。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .app import ProofreadApp
from .errors import ProofreadError


def _load_commands(path: str | Path) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            commands.append(json.loads(line))
    return commands


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="中文文本校对与数值误差诊断应用")
    parser.add_argument("--base-dir", default=".", help="工作区根目录")
    sub = parser.add_subparsers(dest="action", required=True)

    run_p = sub.add_parser("run", help="全量/续跑校对流水线")
    run_p.add_argument("--resume", action="store_true", help="依据快照续跑")
    run_p.add_argument("--clean", action="store_true", help="运行前清空状态目录")

    replay_p = sub.add_parser("replay", help="按命令流回放交互调参")
    replay_p.add_argument("--commands", required=True, help="JSONL 交互命令文件")
    replay_p.add_argument("--clean", action="store_true")

    sub.add_parser("recovery", help="输出快照恢复点信息")
    sub.add_parser("clean", help="清空状态与输出目录")

    args = parser.parse_args(argv)
    app = ProofreadApp(args.base_dir)
    app.controller.install()

    try:
        if args.action == "clean":
            app.clean_state()
            print("已清理 state/ 缓存与快照")
            return 0

        if args.action == "recovery":
            print(json.dumps(app.snapshots.recovery_point(),
                             ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.action == "run":
            if args.clean:
                app.clean_state()
            report = app.run(resume=args.resume)
            print(json.dumps({
                "samples": report["sample_count"],
                "failures": len(report["failures"]),
                "cache": report["cache"]["reused_count"],
                "global_version": report["snapshot"]["global_version"],
                "graceful_exit": report.get("graceful_exit", False),
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if args.action == "replay":
            if args.clean:
                app.clean_state()
            commands = _load_commands(args.commands)
            traces = app.replay(commands, run_after_each=False)
            # 交互调参后配置哈希变化：旧快照不可续跑，重置快照但保留内容缓存，
            # 由缓存键命中情况体现增量复用与失效
            app.snapshots.reset()
            report = app.run(resume=False)
            out = Path(args.base_dir) / "output" / "replay_trace.json"
            out.write_text(json.dumps(traces, ensure_ascii=False, indent=2,
                                      sort_keys=True), encoding="utf-8")
            print(json.dumps({
                "commands": len(traces),
                "report_samples": report["sample_count"],
                "cache": report["cache"],
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except ProofreadError as exc:
        print(json.dumps(exc.to_item(), ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
