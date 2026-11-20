"""gradcomp 命令行入口。

所有命令以稳定键序的 JSON 打印结果；业务错误打印错误对象并返回
errors.EXIT_CODES 中对应的退出码。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import Config, dumps, load_config
from .data import data_summary, initial_parameters
from .digest import array_digest
from .engine import train
from .errors import EXIT_CODES, GradCompError
from .replay import replay_resume
from .snapshot import list_snapshots, load_latest


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                     default=_default))


def _default(obj: Any) -> Any:
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"不可序列化对象: {type(obj)}")


def _cmd_gen_config(args: argparse.Namespace) -> int:
    from .config import validate
    raw: dict[str, Any] = {}
    if args.file:
        raw = json.loads(Path(args.file).read_text(encoding="utf-8"))
    overrides = {k: v for k, v in vars(args).items()
                 if k in Config.__dataclass_fields__ and v is not None}
    raw.update(overrides)
    cfg = validate(raw)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(cfg) + "\n", encoding="utf-8")
    _print({"written": str(out), "config": cfg.summary()})
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    params = initial_parameters(cfg)
    _print({"config": cfg.summary(),
            "data_summary": data_summary(cfg),
            "initial_parameter_digest": array_digest(params)})
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = train(cfg, workdir=args.workdir)
    _print(_report_dict(report))
    return 0


def _cmd_snapshots(args: argparse.Namespace) -> int:
    _print(list_snapshots(args.workdir))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    snap, pending = load_latest(args.workdir)
    check = replay_resume(cfg, snap, args.extra_steps)
    payload = check.to_dict()
    payload["pending_or_unknown_steps"] = pending
    _print(payload)
    return 0 if check.matched else EXIT_CODES["E_REPLAY"]


def _cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui
    launch_gui(load_config(args.config) if args.config else Config(),
               args.workdir)
    return 0


def _report_dict(report) -> dict[str, Any]:
    return {
        "config": report.config,
        "data_summary": report.data_summary,
        "steps": report.steps,
        "final_parameters_digest": report.final_parameters_digest,
        "final_parameter_version": report.final_parameter_version,
        "snapshot_files": report.snapshot_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradcomp",
        description="分布式数据并行训练梯度压缩通信仿真台")
    sub = parser.add_subparsers(dest="command", required=True)

    common_fields = list(Config.__dataclass_fields__.keys())

    g = sub.add_parser("gen-config", help="生成样本配置 JSON")
    g.add_argument("--file", help="基于已有配置覆盖")
    g.add_argument("--output", required=True)
    for name in common_fields:
        g.add_argument(f"--{name.replace('_', '-')}",
                       dest=name, type=_coerce(Config, name))
    g.set_defaults(func=_cmd_gen_config)

    s = sub.add_parser("summary", help="输出确定性数据摘要")
    s.add_argument("config")
    s.set_defaults(func=_cmd_summary)

    t = sub.add_parser("train", help="执行训练并写快照")
    t.add_argument("config")
    t.add_argument("--workdir", required=True)
    t.set_defaults(func=_cmd_train)

    ls = sub.add_parser("snapshots", help="列出快照与未完成文件")
    ls.add_argument("workdir")
    ls.set_defaults(func=_cmd_snapshots)

    r = sub.add_parser("replay", help="从最新快照恢复并做重放校验")
    r.add_argument("config")
    r.add_argument("--workdir", required=True)
    r.add_argument("--extra-steps", type=int, default=2)
    r.set_defaults(func=_cmd_replay)

    ui = sub.add_parser("gui", help="启动 Tkinter 界面")
    ui.add_argument("--config")
    ui.add_argument("--workdir", default="workdir")
    ui.set_defaults(func=_cmd_gui)
    return parser


def _coerce(cls: type, name: str):
    annotation = cls.__dataclass_fields__[name].type
    if name in ("compressed",):
        return lambda v: v.lower() in ("1", "true", "yes", "on")
    return int if name in ("workers", "dimension", "seed", "samples",
                           "scale_exponent", "block_size", "base_port",
                           "steps") else float if name == "learning_rate" else str


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GradCompError as exc:
        _print(exc.to_dict())
        return EXIT_CODES[exc.code]


if __name__ == "__main__":
    sys.exit(main())
