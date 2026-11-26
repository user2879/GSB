"""命令行入口：python -m mvcc_lab --config <JSON> --ledger <路径>。

异常注入全部经命令行参数传入，不改动源码：
--inject none|silent|port_busy|missing_field|rule_conflict
--kill-after <N>      轮次意图提交后强制终止（模拟 SIGKILL）
--replay              恢复后执行一致性重放校验
--report <路径>       输出判定报告（规范 JSON）
--batch-size <N>      执行器批次大小
"""

import argparse
import json
import sys

from .errors import MvccError
from .loader import load_spec
from .runner import Runner, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvcc_lab")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument(
        "--inject",
        choices=("none", "silent", "silent_hold", "port_busy",
                 "missing_field", "rule_conflict"),
        default="none",
    )
    parser.add_argument("--kill-after", type=int, default=None)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_spec(args.config)
        runner = Runner(
            spec=spec,
            config_path=args.config,
            ledger_path=args.ledger,
            injection=args.inject,
            batch_size=args.batch_size,
            kill_after=args.kill_after,
            replay=args.replay,
        )
        try:
            report = runner.run_full_round()
        finally:
            runner.close()
    except MvccError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 2
    if args.report:
        write_report(report, args.report)
    console = {
        key: report[key]
        for key in ("rule_version", "round_id", "identity", "recovery",
                    "peaks", "queries_committed", "content_digest",
                    "cause_proportions")
        if key in report
    }
    if "replay_check" in report:
        console["replay_check"] = report["replay_check"]
    print(json.dumps(console, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
