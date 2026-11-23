"""mvrb 命令行入口：generate / run / convert / diagnose / inject。"""
from __future__ import annotations

import argparse
import os
import sys

from .convert import convert_file, downgrade_file
from .diagnostics import cross_version_report
from .exchange_v1 import read_v1
from .exchange_v2 import read_v2
from .inject import apply_injections, load_injection_config
from .pipeline import ReconstructionPipeline
from .utils import write_json
from .workbench import Workbench


def cmd_generate(args) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.argv = ["generate_sample", "--out", args.out,
                "--views", str(args.views), "--seed", str(args.seed)]
    import scripts.generate_sample as gen
    return gen.main()


def cmd_run(args) -> int:
    report = Workbench(args.sample, args.work, force_serial=args.serial).run()
    write_json(args.report, report)
    print(f"report -> {args.report}")
    print(f"counts={report['counts']} failures={len(report['failures'])}")
    return 0


def cmd_convert(args) -> int:
    if args.direction == "v1-to-v2":
        convert_file(args.input, args.output, args.sidecar)
    else:
        if not args.sidecar:
            raise SystemExit("v2-to-v1 requires --sidecar output path")
        downgrade_file(args.input, args.output, args.sidecar)
    print(f"converted {args.input} -> {args.output}")
    return 0


def cmd_diagnose(args) -> int:
    ir = read_v2(args.input) if args.input.endswith(".json") else None
    if ir is None:
        ir = read_v1(args.input)
        from .convert import upgrade_v1_to_v2
        ir = upgrade_v1_to_v2(ir)
    report = cross_version_report(ir)
    write_json(args.output, report)
    print(f"diagnostics -> {args.output}")
    return 0


def cmd_inject(args) -> int:
    items = load_injection_config(args.config)
    manifest = apply_injections(args.sample, args.output_sample, items)
    result = ReconstructionPipeline(args.output_sample,
                                    os.path.join(args.output_sample, "_pipeline")).run()
    report = {"manifest": manifest,
              "failure_codes": sorted({f.code for f in result.failures.sorted()}),
              "failures": result.failures.dicts()}
    write_json(args.report, report)
    print(f"injection report -> {args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvrb", description="多视图三维几何重建工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="生成确定性本地样本")
    g.add_argument("--out", required=True)
    g.add_argument("--views", type=int, default=6)
    g.add_argument("--seed", type=int, default=20261124)
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="运行重建管线并产出快照/诊断")
    r.add_argument("--sample", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--report", required=True)
    r.add_argument("--serial", action="store_true")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("convert", help="v1/v2 交换文件双向转换")
    c.add_argument("direction", choices=["v1-to-v2", "v2-to-v1"])
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--sidecar", default=None)
    c.set_defaults(func=cmd_convert)

    d = sub.add_parser("diagnose", help="跨版本一致性诊断")
    d.add_argument("--input", required=True)
    d.add_argument("--output", required=True)
    d.set_defaults(func=cmd_diagnose)

    i = sub.add_parser("inject", help="异常注入并输出命中报告")
    i.add_argument("--sample", required=True)
    i.add_argument("--output-sample", required=True)
    i.add_argument("--config", required=True)
    i.add_argument("--report", required=True)
    i.set_defaults(func=cmd_inject)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
