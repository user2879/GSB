"""Command line entry points.

    python -m workbench.cli generate-samples --out samples
    python -m workbench.cli run --samples samples --run-dir run [--inject config/injection.yaml]
    python -m workbench.cli interactive --samples samples --run-dir run
    python -m workbench.cli verify --work-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from . import injection, loader
from .interactive import Workbench


def load_dataset(samples_dir):
    vocab, lexicon = loader.load_vocabulary(os.path.join(samples_dir, "vocab.json"))
    tools, tool_failures = loader.load_tools(os.path.join(samples_dir, "tools.json"))
    tool_schemas = {name: t["param_schema"] for name, t in tools.items()}
    turns, turn_failures = loader.load_turns(
        os.path.join(samples_dir, "turns.jsonl"), vocab, tool_schemas)
    acoustics, acoustic_failures = loader.load_acoustics(
        os.path.join(samples_dir, "acoustics.jsonl"))
    config, config_failures = loader.load_config(
        os.path.join(samples_dir, "fusion.yaml"))
    cross_failures = loader.cross_check(turns, acoustics)
    failures = turn_failures + acoustic_failures + tool_failures + config_failures + cross_failures
    return {
        "vocab": vocab, "lexicon": lexicon, "tools": tools, "turns": turns,
        "acoustics": acoustics, "config": config, "load_failures": failures,
    }


def cmd_generate_samples(args):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from generate_samples import generate
    print(json.dumps(generate(args.out), ensure_ascii=False))


def cmd_run(args):
    dataset = load_dataset(args.samples)
    turns = dataset["turns"]
    acoustics = dataset["acoustics"]
    tools = dataset["tools"]
    config = dataset["config"]
    failures = list(dataset["load_failures"])
    hits = []
    if args.inject:
        injections = injection.load_injections(args.inject)
        turns, acoustics, tools, config, hits, config_failures = injection.apply_injections(
            turns, acoustics, tools, config, injections)
        failures.extend(config_failures)

    from .cache import StageCache
    from .pipeline import Pipeline
    pipeline = Pipeline(turns, acoustics, tools, dataset["lexicon"], config,
                        cache=StageCache())
    report = pipeline.run()
    report["failures"] = failures + report["failures"]
    report["injection_hits"] = hits

    os.makedirs(args.run_dir, exist_ok=True)
    report_path = os.path.join(args.run_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=1)

    summary = {
        "report": report_path,
        "completed_turns": report["summary"]["completed_turns"],
        "total_turns": report["summary"]["total_turns"],
        "pending_turns": report["summary"]["pending_turns"],
        "flips": report["summary"]["flips"],
        "failures": len(report["failures"]),
        "failure_codes": sorted({f["code"] for f in report["failures"]}),
        "injection_hits": [h["id"] for h in hits],
        "confidence_distribution": report["summary"]["confidence_distribution"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def cmd_interactive(args):
    dataset = load_dataset(args.samples)
    workbench = Workbench(
        dataset["turns"], dataset["acoustics"], dataset["tools"], dataset["lexicon"],
        dataset["config"], dataset["load_failures"], args.run_dir)
    if workbench.startup_findings:
        print(json.dumps({"startup_findings": workbench.startup_findings}, ensure_ascii=False))
    print("workbench ready; commands: set-weight|set-temperature|set-precision|"
          "replace-turn|add-tool|remove-tool|run|summary|quit")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "quit":
            break
        response = workbench.apply(line)
        print(json.dumps(response, ensure_ascii=False, default=str))


def cmd_verify(args):
    from .verify import run_verification
    result = run_verification(args.work_dir)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    sys.exit(0 if result["all_passed"] else 1)


def main():
    parser = argparse.ArgumentParser(prog="workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-samples")
    p.add_argument("--out", default="samples")
    p.set_defaults(func=cmd_generate_samples)

    p = sub.add_parser("run")
    p.add_argument("--samples", default="samples")
    p.add_argument("--run-dir", default="run")
    p.add_argument("--inject", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("interactive")
    p.add_argument("--samples", default="samples")
    p.add_argument("--run-dir", default="run")
    p.set_defaults(func=cmd_interactive)

    p = sub.add_parser("verify")
    p.add_argument("--work-dir", default="verify_out")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
