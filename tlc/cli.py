"""命令行入口：gen-data / run / rebuild。"""

import argparse
import json
import os
import sys

from .datagen import generate
from .errors import BizError
from .pipeline import rebuild, run_batch


def _write_report(path, report):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(report, sort_keys=False, ensure_ascii=False, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tlc", description="跨任务迁移学习组件")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen-data", help="生成固定种子合成样本")
    p_gen.add_argument("--out", required=True)

    p_run = sub.add_parser("run", help="增量运行一个批次")
    p_run.add_argument("--data", required=True)
    p_run.add_argument("--batch", type=int, required=True)
    p_run.add_argument("--state", required=True)
    p_run.add_argument("--report", required=True)

    p_re = sub.add_parser("rebuild", help="从空白状态全量重放批次序列")
    p_re.add_argument("--data", required=True)
    p_re.add_argument("--batches", type=int, required=True)
    p_re.add_argument("--state", required=True)
    p_re.add_argument("--report", required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "gen-data":
            generate(args.out)
            print(f"样本数据已生成: {args.out}")
            return 0
        if args.cmd == "run":
            report = run_batch(args.data, args.batch, args.state)
        else:
            report = rebuild(args.data, args.batches, args.state)
        _write_report(args.report, report)
        print(f"批次 {report['batch']} 完成, run_key={report['run_key'][:12]}..., "
              f"codes={report['codes']}")
        return 0
    except BizError as err:
        report = {"error_code": err.code, "message": err.message}
        if getattr(args, "report", None):
            _write_report(args.report, report)
        print(f"{err.code}: {err.message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
