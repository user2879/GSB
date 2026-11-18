"""命令行入口：样本生成、管线运行、恢复点查询与独立格式转换。"""

import argparse
import json
import sys

from .errors import PipelineError
from .pipeline import Pipeline
from .samplegen import generate_dataset
from .snapshot import SnapshotStore


def _print(obj):
    print(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2))


def cmd_generate(args):
    names = generate_dataset(args.input, args.count,
                             (args.width, args.height))
    _print({"generated": names, "count": len(names)})
    return 0


def cmd_run(args):
    pipeline = Pipeline(
        input_dir=args.input,
        work_dir=args.workdir,
        config_path=args.config,
        spec_path=args.spec,
        injection_path=args.injections,
        format_version=args.format_version,
        max_workers_override=args.workers,
    )
    report = pipeline.run()
    _print({
        "report": args.workdir + "/report.json",
        "run_id": report["run_id"],
        "interrupted": report["interrupted"],
        "failures": report["failures"]["count"],
        "global_version": report["snapshot"]["global_version"],
        "content_hash": report["content_hash"],
    })
    if report["interrupted"]:
        return 0
    return 0


def cmd_recovery(args):
    store = SnapshotStore(args.workdir)
    _print(store.recovery_point())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="xmodal",
        description="本地确定性跨模态对齐与检索管线")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性图文样本")
    gen.add_argument("--input", required=True)
    gen.add_argument("--count", type=int, default=8)
    gen.add_argument("--width", type=int, default=64)
    gen.add_argument("--height", type=int, default=64)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="运行完整管线")
    run.add_argument("--input", required=True)
    run.add_argument("--workdir", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--spec", required=True)
    run.add_argument("--injections")
    run.add_argument("--format-version", type=int, choices=(1, 2))
    run.add_argument("--workers", type=int)
    run.set_defaults(func=cmd_run)

    rec = sub.add_parser("recovery", help="输出快照恢复点信息")
    rec.add_argument("--workdir", required=True)
    rec.set_defaults(func=cmd_recovery)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PipelineError as exc:
        _print({"error": exc.code, "message": str(exc),
                "exc_type": exc.exc_type})
        return 2


if __name__ == "__main__":
    sys.exit(main())
