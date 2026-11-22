from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from .engine import Workbench
from .errors import SimulatedCrash, ValidationError
from .io import write_json_atomic
from .samplegen import generate_samples


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _run_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": report.get("mode"),
        "semantic_digest": report.get("semantic_digest"),
        "snapshot_version": (report.get("snapshot") or {}).get("global_version"),
        "snapshot_stale": report.get("snapshot_stale"),
        "sentence_count": len(report.get("sentences", [])),
        "ok_count": sum(1 for item in report.get("sentences", []) if item.get("status") == "OK"),
        "failed_count": len(report.get("failures", [])),
        "pending_count": len(report.get("pending", [])),
        "mode_difference_count": report.get("mode_differences", {}).get("different_sentence_count", 0),
        "boundary_tie_count": len(report.get("boundary_ties", [])),
        "failure_codes": sorted({item.get("code") for item in report.get("failures", [])}),
        "pending_codes": sorted({item.get("code") for item in report.get("pending", [])}),
    }


def _build_workbench(args: argparse.Namespace) -> Workbench:
    return Workbench(
        args.samples,
        snapshot_dir=None if args.no_snapshot else args.snapshot_dir,
        sentence_file=args.sentences,
        model_file=args.model,
        config_file=args.config,
        vocabulary_file=args.vocabulary,
        injection_file=args.injections,
        report_path=None if args.no_report else args.report,
    )


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", default="samples", help="sample directory")
    parser.add_argument("--sentences", default=None, help="sentence JSONL path override")
    parser.add_argument("--model", default=None, help="model JSON path override")
    parser.add_argument("--config", default=None, help="config JSON path override")
    parser.add_argument("--vocabulary", default=None, help="vocabulary JSON path override")
    parser.add_argument("--injections", default=None, help="injection JSON path override")
    parser.add_argument("--snapshot-dir", default=".depworkbench-snapshots", help="snapshot root")
    parser.add_argument("--no-snapshot", action="store_true", help="disable snapshot persistence")
    parser.add_argument("--report", default="artifacts/report.json", help="JSON report path")
    parser.add_argument("--no-report", action="store_true", help="do not write the JSON report")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("greedy", "exhaustive"), default=None)
    parser.add_argument("--resume", action="store_true", help="resume only when snapshot hashes match")
    parser.add_argument("--fresh", action="store_true", help="ignore all cached products")
    parser.add_argument("--expected-version", type=int, default=None)
    parser.add_argument(
        "--crash-after-sentences",
        type=int,
        default=None,
        help="test hook: commit the current chunk, then raise SimulatedCrash",
    )
    parser.add_argument(
        "--crash-partial-steps",
        type=int,
        default=None,
        help="test hook: retain this many transition steps for the last sentence before crashing",
    )


def cmd_generate(args: argparse.Namespace) -> int:
    metadata = generate_samples(args.out, seed=args.seed, count=args.count)
    _print_json(metadata)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    workbench = Workbench(
        args.samples,
        sentence_file=args.sentences,
        model_file=args.model,
        config_file=args.config,
        vocabulary_file=args.vocabulary,
        injection_file=args.injections,
    )
    project = workbench.load_project()
    result = {
        "valid_sentence_count": len(project.sentences),
        "invalid_sentence_count": len(project.failures),
        "model_hash": project.model.fingerprint,
        "config_hash": project.config.fingerprint,
        "corpus_hash": project.corpus_fingerprint,
        "injection_count": len(project.injections),
        "failures": [failure.to_dict() for failure in project.failures],
    }
    _print_json(result)
    return 0 if not project.failures else 2


def cmd_run(args: argparse.Namespace) -> int:
    workbench = _build_workbench(args)
    report = workbench.run(
        mode_override=args.mode,
        resume=args.resume,
        fresh=args.fresh,
        expected_version=args.expected_version,
        crash_after_sentences=args.crash_after_sentences,
        crash_partial_steps=args.crash_partial_steps,
        write_report=not args.no_report,
    )
    _print_json(_run_summary(report))
    return 0


def _split_edit_payload(text: str) -> tuple[list[str], list[str]]:
    if "|" not in text:
        raise ValueError("edit payload must use: <words> | <pos_tags>")
    left, right = text.split("|", 1)
    words = left.strip().split()
    pos_tags = right.strip().split()
    if not words or not pos_tags:
        raise ValueError("both words and pos_tags must be non-empty")
    return words, pos_tags


def _print_repl_help() -> None:
    print(
        "\n".join(
            [
                "commands:",
                "  run [greedy|exhaustive] [fresh|resume]",
                "  replace <sid> <words> | <pos_tags>",
                "  add <sid> <words> | <pos_tags>",
                "  delete <sid>",
                "  weight <ACTION> <index> <finite-value>",
                "  constraint <name> <json-value>",
                "  config <name> <json-value>",
                "  mode <greedy|exhaustive>",
                "  show",
                "  help",
                "  quit",
            ]
        )
    )


def cmd_repl(args: argparse.Namespace) -> int:
    workbench = _build_workbench(args)
    last_report: dict[str, Any] | None = None
    _print_repl_help()
    while True:
        try:
            raw = input("depworkbench> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not raw:
            continue
        command, _, remainder = raw.partition(" ")
        command = command.lower()
        remainder = remainder.strip()
        try:
            if command in {"quit", "exit"}:
                return 0
            if command == "help":
                _print_repl_help()
                continue
            if command == "show":
                if last_report is None:
                    print("no report yet")
                else:
                    _print_json(_run_summary(last_report))
                continue
            if command == "run":
                tokens = remainder.split()
                mode = None
                fresh = False
                resume = False
                for token in tokens:
                    if token in {"greedy", "exhaustive"}:
                        mode = token
                    elif token == "fresh":
                        fresh = True
                    elif token == "resume":
                        resume = True
                    else:
                        raise ValueError(f"unknown run option {token!r}")
                last_report = workbench.run(
                    mode_override=mode,
                    fresh=fresh,
                    resume=resume,
                    write_report=True,
                )
                _print_json(_run_summary(last_report))
                continue
            if command in {"replace", "add"}:
                sid, _, payload = remainder.partition(" ")
                words, pos_tags = _split_edit_payload(payload)
                if not sid:
                    raise ValueError("sentence id is required")
                if command == "replace":
                    workbench.replace_sentence(sid, words, pos_tags)
                else:
                    workbench.add_sentence(sid, words, pos_tags)
                print(f"{command}d {sid}")
                continue
            if command == "delete":
                workbench.delete_sentence(remainder)
                print(f"deleted {remainder}")
                continue
            if command == "weight":
                action, index_text, value_text = remainder.split()
                workbench.set_action_weight(action, int(index_text), float(value_text))
                print(f"updated {action}[{index_text}]")
                continue
            if command == "constraint":
                name, value_text = remainder.split(maxsplit=1)
                workbench.set_constraint(name, json.loads(value_text))
                print(f"updated constraint {name}")
                continue
            if command == "config":
                name, value_text = remainder.split(maxsplit=1)
                workbench.set_config(name, json.loads(value_text))
                print(f"updated config {name}")
                continue
            if command == "mode":
                workbench.set_mode(remainder)
                print(f"mode set to {remainder}")
                continue
            raise ValueError(f"unknown command {command!r}")
        except (ValueError, KeyError, IndexError, json.JSONDecodeError, ValidationError) as exc:
            print(f"error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depworkbench",
        description="Deterministic Chinese transition-based dependency parsing workbench",
    )
    parser.add_argument("--version", action="version", version=f"depworkbench {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--out", default="samples")
    generate.add_argument("--seed", type=int, default=20261123)
    generate.add_argument("--count", type=int, default=10)
    generate.set_defaults(func=cmd_generate)

    validate = subparsers.add_parser("validate", help="validate sample, model and config inputs")
    _add_project_args(validate)
    validate.set_defaults(func=cmd_validate)

    run = subparsers.add_parser("run", help="run one deterministic workbench pass")
    _add_project_args(run)
    _add_run_args(run)
    run.set_defaults(func=cmd_run)

    repl = subparsers.add_parser("repl", help="interactive edit/run loop")
    _add_project_args(repl)
    _add_run_args(repl)
    repl.set_defaults(func=cmd_repl)

    verify = subparsers.add_parser("verify", help="run the reproducible acceptance scenarios")
    verify.add_argument("--samples", default="samples")
    verify.add_argument("--out", default="artifacts")
    verify.add_argument("--seed", type=int, default=20261123)
    verify.add_argument("--keep-work", action="store_true")
    verify.set_defaults(func=cmd_verify)
    return parser


def cmd_verify(args: argparse.Namespace) -> int:
    from .verify import run_verification

    result = run_verification(
        args.samples,
        args.out,
        seed=args.seed,
        keep_work=args.keep_work,
    )
    _print_json(result["summary"])
    return 0 if result["summary"]["all_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SimulatedCrash as exc:
        print(json.dumps({"event": "simulated_crash", "message": str(exc)}, ensure_ascii=False))
        return 75
    except ValidationError as exc:
        print(
            json.dumps(
                {"code": exc.code, "message": exc.message, "details": exc.details},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
