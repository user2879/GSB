from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .build import run_full_build
from .commands import CommandState, load_all_receipts, load_commands, load_queries
from .diagnostics import build_failure_manifest, run_fault_injections, write_injection_configs
from .replay import run_stream_equivalence
from .sample import generate_sample


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")


def cmd_generate(args: argparse.Namespace) -> int:
    summary = generate_sample(args.root, seed=args.seed)
    write_injection_configs(args.root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    report = run_full_build(args.root, budget=args.budget)
    print(json.dumps({"output": str(Path(args.root) / "outputs"), "blocks": len(report["view"]["blocks"]), "buckets": len(report["buckets"])}, ensure_ascii=False))
    return 0


def cmd_failures(args: argparse.Namespace) -> int:
    manifest = build_failure_manifest(args.root)
    print(json.dumps({"all_hit": manifest["all_hit"], "cases": len(manifest["cases"])}, ensure_ascii=False, indent=2))
    return 0 if manifest["all_hit"] else 4


def cmd_replay(args: argparse.Namespace) -> int:
    report = run_stream_equivalence(args.root)
    print(json.dumps({"all_fields_match": report["all_fields_match"], "steps": report["step_count"], "budget_pending_seen": report["budget_pending_seen"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_fields_match"] else 2


def cmd_faults(args: argparse.Namespace) -> int:
    report = run_fault_injections(args.root)
    print(json.dumps({"all_hit": report["all_hit"], "hit_count": report["hit_count"], "total": report["total"]}, ensure_ascii=False, indent=2))
    return 0 if report["all_hit"] else 3


def cmd_validate_inputs(args: argparse.Namespace) -> int:
    root = Path(args.root)
    state = CommandState.from_sample_dir(root)
    commands = load_commands(root)
    receipts = load_all_receipts(root)
    result: Dict[str, Any] = {
        "code": "OK",
        "paragraph_count": len(state.records),
        "block_count_after_current_config": len(state.blocks),
        "command_count": len(commands),
        "receipt_count": len(receipts),
        "config_hash": state.config.config_hash,
        "param_fingerprint": state.config.param_fingerprint,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense long-text retrieval index toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-sample", help="generate deterministic sample workspace")
    generate.add_argument("--root", default="sample_workspace")
    generate.add_argument("--seed", type=int, default=239)
    generate.set_defaults(func=cmd_generate)

    validate = sub.add_parser("validate-inputs", help="load and validate generated inputs")
    validate.add_argument("--root", default="sample_workspace")
    validate.set_defaults(func=cmd_validate_inputs)

    build = sub.add_parser("build", help="run a blank full build")
    build.add_argument("--root", default="sample_workspace")
    build.add_argument("--budget", type=int, default=None)
    build.set_defaults(func=cmd_build)

    replay = sub.add_parser("replay", help="replay command stream and compare incremental with full build")
    replay.add_argument("--root", default="sample_workspace")
    replay.set_defaults(func=cmd_replay)

    faults = sub.add_parser("faults", help="run isolated fault injections")
    faults.add_argument("--root", default="sample_workspace")
    faults.set_defaults(func=cmd_faults)

    failures = sub.add_parser("failure-manifest", help="validate illegal inputs and write failure manifest")
    failures.add_argument("--root", default="sample_workspace")
    failures.set_defaults(func=cmd_failures)
    return parser


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
