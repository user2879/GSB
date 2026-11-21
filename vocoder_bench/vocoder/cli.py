"""Command line entry point for the vocoder diagnostic workbench."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .generate_samples import generate
from .runner import PipelineRunner
from .session import InteractiveSession


def cmd_generate(args) -> int:
    info = generate(args.output)
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    runner = PipelineRunner(args.input, cfg, args.workdir)
    runner.reload_inputs()
    report = runner.run_full()
    runner.save_report(report, Path(args.workdir) / "report_full.json")
    print(json.dumps({
        "version": report["version"],
        "reuse": report["reuse"],
        "failures": report["failures"],
        "utterance_count": len(report["utterances"]),
    }, indent=2, sort_keys=True))
    runner.shutdown()
    return 0


def cmd_stream(args) -> int:
    cfg = load_config(args.config)
    commands = json.loads(Path(args.commands).read_text(encoding="utf-8"))
    session = InteractiveSession(args.input, cfg, args.workdir)
    responses = session.run_command_stream(commands)
    out = Path(args.workdir) / "interactive_responses.json"
    out.write_text(json.dumps(responses, indent=2, sort_keys=True),
                   encoding="utf-8")
    final = next((r for r in reversed(responses) if r.get("code") == "SHUTDOWN"), None)
    pending = sum(1 for r in responses if r.get("code") == "E_INTERACT_BUDGET_PENDING")
    ok = sum(1 for r in responses if r.get("code") == "OK")
    summary = None
    if final and final.get("report"):
        rep = final["report"]
        summary = {
            "version": rep["version"],
            "reuse": rep["reuse"],
            "failures": [(f["utterance_id"], f["code"]) for f in rep["failures"]],
        }
    print(json.dumps({"responses": len(responses), "ok": ok, "pending": pending,
                      "out": str(out),
                      "exit_code": None if not final else final.get("exit_code"),
                      "final": summary}, indent=2, sort_keys=True))
    return 0 if final and final.get("exit_code") == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vocoder waveform diagnostic workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic samples")
    gen.add_argument("output")
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="full synthesis + diagnostics")
    run.add_argument("--input", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--workdir", required=True)
    run.set_defaults(func=cmd_run)

    stream = sub.add_parser("stream", help="replay edit command stream")
    stream.add_argument("--input", required=True)
    stream.add_argument("--config", required=True)
    stream.add_argument("--commands", required=True)
    stream.add_argument("--workdir", required=True)
    stream.set_defaults(func=cmd_stream)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
