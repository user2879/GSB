"""命令行入口：``python -m colbench.cli <command>``。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import BenchError
from .gui import launch
from .sample_data import generate
from .workbench import Workbench


def _samples(workspace: Path) -> list[Path]:
    return sorted((workspace / "samples").glob("segment_*.json"))


def cmd_init(args) -> int:
    workspace = Path(args.workspace)
    paths = generate(workspace / "samples")
    bench = Workbench(workspace)
    bench.build_from_specs(paths)
    print(json.dumps({"status": "OK", "segments": [p.name for p in paths],
                      "data_dir": str(bench.data_dir)},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_scan(args) -> int:
    bench = Workbench(args.workspace)
    bench.reopen_segments()
    query = Path(args.query).read_text(encoding="utf-8")
    sql = json.loads(query).get("sql", query)
    params = _BudgetParams_cls(args.parallelism, args.memory, args.quota)
    model = bench.run_scan(sql, params)
    mode = launch(model, force_text=args.text)
    print(f"[render={mode}]")
    return 0


def cmd_verify(args) -> int:
    from .ledger import Ledger
    from .replay import verify_ledger
    bench = Workbench(args.workspace)
    bench.reopen_segments()
    plan = bench.plan(Path(args.query).read_text(encoding="utf-8")
                      and json.loads(Path(args.query).read_text(
                          encoding="utf-8"))["sql"])
    with Ledger(bench.db_path) as ledger:
        report = verify_ledger(plan, bench.segments, ledger,
                               bench.registry.current())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_gui(args) -> int:
    bench = Workbench(args.workspace)
    bench.reopen_segments()
    query = json.loads((bench.workspace / "samples" / "q_range.json")
                       .read_text(encoding="utf-8"))["sql"]
    model = bench.run_scan(query, _BudgetParams_cls(2, 4096, 1))
    launch(model, force_text=args.text)
    return 0


def _BudgetParams_cls(parallelism: int, memory: int, quota: int):
    from .versioning import BudgetParams
    return BudgetParams(global_parallelism=parallelism,
                        per_column_memory=memory,
                        per_segment_concurrency=quota)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="列式存储压缩与扫描诊断工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="生成确定性样本并构建段文件")
    init.add_argument("--workspace", default="data/demo")
    init.set_defaults(func=cmd_init)

    scan = sub.add_parser("scan", help="执行扫描并渲染诊断视图")
    scan.add_argument("--workspace", default="data/demo")
    scan.add_argument("--query", required=True)
    scan.add_argument("--parallelism", type=int, default=2)
    scan.add_argument("--memory", type=int, default=4096)
    scan.add_argument("--quota", type=int, default=1)
    scan.add_argument("--text", action="store_true")
    scan.set_defaults(func=cmd_scan)

    verify = sub.add_parser("verify", help="权威重算并校验账本")
    verify.add_argument("--workspace", default="data/demo")
    verify.add_argument("--query", required=True)
    verify.set_defaults(func=cmd_verify)

    gui = sub.add_parser("gui", help="打开诊断工作台")
    gui.add_argument("--workspace", default="data/demo")
    gui.add_argument("--text", action="store_true")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BenchError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
