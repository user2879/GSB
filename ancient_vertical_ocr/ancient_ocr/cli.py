"""命令行入口：初始化、运行、命令流、强制终止演示、恢复、报告导出。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from .config import Config, write_default_config
from .errors import E_RECOVERY_FALLBACK
from .loader import load_command_stream
from .pipeline import Pipeline, bootstrap_demo_workspace
from .snapshot import SnapshotStore


def _project_paths(root: Path) -> dict[str, Path]:
    return {
        "samples": root / "samples",
        "config": root / "config" / "config.json",
        "snapshots": root / "snapshots",
        "reports": root / "reports",
        "commands": root / "commands",
    }


def cmd_init(args) -> None:
    root = Path(args.root)
    paths = _project_paths(root)
    for name in ("samples", "snapshots", "reports", "commands"):
        paths[name].mkdir(parents=True, exist_ok=True)
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    import shutil
    if paths["samples"].exists():
        shutil.rmtree(paths["samples"])
    paths["samples"].mkdir(parents=True)
    write_default_config(paths["config"])
    cfg = Config.load(paths["config"])
    ids = bootstrap_demo_workspace(root, cfg)
    demo_commands = [
        {"op": "run"},
        {"op": "replace_keypoint_version", "sample_id": "page_a", "version": "annot-v2"},
        {"op": "run"},
        {"op": "replace_image", "sample_id": "page_b", "seed": 8088},
        {"op": "run"},
        {"op": "add_sample", "sample_id": "page_d", "seed": 2002},
        {"op": "run"},
        {"op": "remove_sample", "sample_id": "page_c"},
        {"op": "run"},
    ]
    (paths["commands"] / "demo_commands.json").write_text(
        json.dumps(demo_commands, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"initialized": True, "samples": ids,
                      "config": str(paths["config"])}, ensure_ascii=False, indent=2))


def _build_pipeline(root: Path, config_overrides: dict | None = None) -> Pipeline:
    paths = _project_paths(root)
    cfg = Config.load(paths["config"])
    if config_overrides:
        merged = dict(cfg.to_dict())
        merged.update(config_overrides)
        cfg = Config.from_dict(merged)
    return Pipeline(paths["samples"], cfg, paths["snapshots"])


def cmd_run(args) -> None:
    root = Path(args.root)
    paths = _project_paths(root)
    overrides = json.loads(args.overrides) if args.overrides else None
    pipeline = _build_pipeline(root, overrides)
    state = pipeline.execute()
    report_path = paths["reports"] / f"run-{int(time.time())}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"ok": True, "global_version": state["global_version"],
                      "report": str(report_path),
                      "issues": len(state["issues"]),
                      "todos": len(state["scheduler"]["todos"])},
                     ensure_ascii=False, indent=2))


def cmd_commands(args) -> None:
    root = Path(args.root)
    paths = _project_paths(root)
    commands = load_command_stream(args.file)
    states = []
    pipeline = _build_pipeline(root)
    for index, command in enumerate(commands):
        op = command["op"]
        if op in ("set_window", "set_quant", "set_budget"):
            cfg_path = paths["config"]
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            mapping = {
                "set_window": ("window_size", "window_stride"),
                "set_quant": ("frac_bits", "quant_level"),
                "set_budget": ("scan_row_budget", "compare_point_budget",
                               "adjudication_budget"),
            }[op]
            for key in mapping:
                if key in command:
                    payload[key] = command[key]
            cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            old_cache = pipeline.cache
            pipeline = _build_pipeline(root)
            pipeline.cache = old_cache
            continue
        if op not in ("run", "crash", "recover"):
            pipeline.apply_command(command)
        if op == "crash":
            _simulate_crash(pipeline, command)
            continue
        if op == "recover":
            info = SnapshotStore(paths["snapshots"]).recover(
                Config.load(paths["config"]).fingerprint())
            states.append({"index": index, "op": op,
                           "recoverable": info.recoverable,
                           "stage": info.stage,
                           "global_version": info.global_version,
                           "completed_units": len(info.completed_units),
                           "issue": info.issue.to_dict() if info.issue else None})
            continue
        state = pipeline.execute()
        states.append({"index": index, "op": op,
                       "global_version": state["global_version"],
                       "cache_digest": state["cache_state_digest"],
                       "invalidated": state["invalidation"]["invalidated"],
                       "reused": state["invalidation"]["reused"],
                       "todos": state["scheduler"]["todos"],
                       "issues": state["issues"]})
    out = paths["commands"] / "command_stream_report.json"
    out.write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "steps": len(states), "report": str(out)},
                     ensure_ascii=False, indent=2))


def _simulate_crash(pipeline: Pipeline, command: dict) -> None:
    """真实强制终止：先提交一次完整快照，再 SIGKILL 自身；由外部包装器重跑恢复。"""
    pipeline.execute()
    if command.get("after_commit", True):
        os.kill(os.getpid(), signal.SIGKILL)


def cmd_crash_demo(args) -> None:
    root = Path(args.root)
    paths = _project_paths(root)
    marker = root / ".crash_marker"
    if marker.exists():
        pipeline = _build_pipeline(root)
        pipeline.execute()
        sys.stderr.write("[crash-demo] 快照已提交，现在发送 SIGKILL 强制终止工作进程\n")
        sys.stderr.flush()
        marker.unlink()
        os.kill(os.getpid(), signal.SIGKILL)
        return
    cfg = Config.load(paths["config"])
    info = SnapshotStore(paths["snapshots"]).recover(cfg.fingerprint())
    pipeline = _build_pipeline(root)
    state = pipeline.execute()
    report = {
        "recovery": {
            "recoverable": info.recoverable,
            "stage": info.stage,
            "global_version": info.global_version,
            "completed_units": info.completed_units,
            "issue": info.issue.to_dict() if info.issue else None,
        },
        "post_recovery_global_version": state["global_version"],
        "cache_state_digest": state["cache_state_digest"],
    }
    out = paths["reports"] / "crash_recovery_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_gui(args) -> None:
    from .gui import launch_gui
    launch_gui(Path(args.root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="竖排古籍版面识别确定性演示")
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--root", default=".")
    p_init.set_defaults(func=cmd_init)
    p_run = sub.add_parser("run")
    p_run.add_argument("--root", default=".")
    p_run.add_argument("--overrides", default=None)
    p_run.set_defaults(func=cmd_run)
    p_cmd = sub.add_parser("commands")
    p_cmd.add_argument("--root", default=".")
    p_cmd.add_argument("file")
    p_cmd.set_defaults(func=cmd_commands)
    p_crash = sub.add_parser("crash-demo")
    p_crash.add_argument("--root", default=".")
    p_crash.set_defaults(func=cmd_crash_demo)
    p_gui = sub.add_parser("gui")
    p_gui.add_argument("--root", default=".")
    p_gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # 让 CLI 始终给出结构化错误
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
