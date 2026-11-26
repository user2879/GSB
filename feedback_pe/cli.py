from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .commands import apply_command, load_command_stream
from .config import load_config
from .demo import run_demo, write_json
from .engine import Workspace
from .generator import generate_project_files
from .reporting import build_report
from .samples import load_samples_jsonl
from .runner import run_batch


def ensure_local_tk_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    library = root / ".sysroot" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
    if not library.exists():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if str(library) in current.split(os.pathsep):
        return
    os.environ["LD_LIBRARY_PATH"] = str(library) + (os.pathsep + current if current else "")
    tcl = root / ".sysroot" / "root" / "usr" / "share" / "tcltk" / "tcl8.6"
    tk = root / ".sysroot" / "root" / "usr" / "share" / "tcltk" / "tk8.6"
    if tcl.exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl))
    if tk.exists():
        os.environ.setdefault("TK_LIBRARY", str(tk))
    os.execv(sys.executable, [sys.executable, "-m", "feedback_pe.cli", *sys.argv[1:]])


def main() -> None:
    ensure_local_tk_runtime()
    parser = argparse.ArgumentParser(description="Deterministic fixed-point PE diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate")
    sub.add_parser("demo")

    validate = sub.add_parser("validate")
    validate.add_argument("--config", default="configs/demo_config.json")
    validate.add_argument("--samples", default="samples/attention_samples.jsonl")
    validate.add_argument("--out", default="out/load_validation.json")

    stream = sub.add_parser("run-stream")
    stream.add_argument("--config", default="configs/demo_config.json")
    stream.add_argument("--samples", default="samples/attention_samples.jsonl")
    stream.add_argument("--stream", default="commands/demo_commands.jsonl")
    stream.add_argument("--state", default="state/cli")
    stream.add_argument("--out", default="out/cli_report.json")

    sub.add_parser("ui")
    args = parser.parse_args()

    if args.command == "generate":
        for name, path in generate_project_files(Path.cwd()).items():
            print(f"generated {name}: {path}")
        return
    if args.command == "demo":
        print(json.dumps(run_demo(Path.cwd()), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "validate":
        config, errors = load_config(args.config)
        if errors:
            write_json(args.out, {"config_errors": errors})
            raise SystemExit(2)
        result = load_samples_jsonl(args.samples, int(config["dimension"]))
        payload = {
            "accepted_count": len(result.samples),
            "failure_count": len(result.failures),
            "failures": result.failures,
            "code_counts": result.code_counts,
        }
        write_json(args.out, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "run-stream":
        config, config_errors = load_config(args.config)
        if config_errors:
            raise SystemExit(json.dumps(config_errors, ensure_ascii=False))
        loaded = load_samples_jsonl(args.samples, int(config["dimension"]))
        samples = {item["id"]: item for item in loaded.samples}
        commands, command_failures = load_command_stream(args.stream)
        workspace = Workspace(args.state)
        summary = None
        for command in commands:
            result = apply_command(config, samples, command)
            if result["run_batch"]:
                summary = run_batch(workspace, config, list(samples.values()), loaded.failures)
        report = build_report(config, list(samples.values()), loaded.failures, workspace, summary)
        report["command_failures"] = command_failures
        write_json(args.out, report)
        print(args.out)
        return
    if args.command == "ui":
        from .tkview import main as ui_main
        ui_main()


if __name__ == "__main__":
    main()
