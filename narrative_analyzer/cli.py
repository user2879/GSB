"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .canonical import pretty_json_bytes, sha256_bytes
from .engine import replay
from .errors import EXIT_CODES, BusinessError, ERROR_CODES
from .inject import load_injection, run_injection
from .pathcheck import analyze_paths
from .samplegen import build_demo_bundle
from .sensitivity import parse_scan_config, scan_sensitivity
from .sequence import load_sequence
from .snapshot import verify_snapshot_file
from .spec import load_spec


def _write_or_print(value: Any, output: str | None) -> None:
    data = pretty_json_bytes(value)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(str(path))
    else:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def _load_scan_file(path: str | None):
    if path is None:
        return parse_scan_config(None)
    try:
        raw = Path(path).read_text(encoding="utf-8")
        value = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise BusinessError(
            "E_SPEC_INVALID",
            "cannot read sensitivity configuration",
            {"path": path, "error": str(exc)},
        ) from exc
    return parse_scan_config(value)


def _cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    _write_or_print(
        {
            "business_code": "OK",
            "spec_sha256": spec.source_sha256,
            "seed": spec.seed,
            "components": list(spec.state.components),
            "node_count": len(spec.nodes),
            "edge_count": len(spec.edges),
            "condition_count": len(spec.conditions),
            "coefficient_count": len(spec.coefficients),
        },
        args.output,
    )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    selections, sequence_sha256 = load_sequence(args.sequence)
    if args.inject:
        plan = load_injection(args.inject)
        report = run_injection(spec, selections, plan)
    else:
        report = replay(spec, selections)
    report["sequence_sha256"] = sequence_sha256
    _write_or_print(report, args.output)
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    selections, sequence_sha256 = load_sequence(args.sequence)
    config = _load_scan_file(args.config)
    report = scan_sensitivity(spec, selections, config)
    report["sequence_sha256"] = sequence_sha256
    _write_or_print(report, args.output)
    return 0


def _cmd_paths(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    report = analyze_paths(spec, max_paths=args.max_paths, raise_on_diverge=args.strict)
    _write_or_print(report, args.output)
    return 0


def _cmd_snapshot_check(args: argparse.Namespace) -> int:
    report = verify_snapshot_file(args.snapshot)
    _write_or_print(report, args.output)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    artifacts = build_demo_bundle()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(artifacts):
        (out_dir / name).write_bytes(artifacts[name])
    verification = verify_snapshot_file(out_dir / "snapshot.json")
    result = {
        "business_code": "OK",
        "out_dir": str(out_dir),
        "artifact_count": len(artifacts),
        "snapshot_checked_artifacts": verification["artifact_count"],
        "demo_summary_sha256": sha256_bytes(artifacts["demo_summary.json"]),
        "snapshot_sha256": sha256_bytes(artifacts["snapshot.json"]),
    }
    _write_or_print(result, args.output)
    return 0


def _cmd_error_codes(_: argparse.Namespace) -> int:
    _write_or_print(
        {
            "business_code": "OK",
            "error_codes": [
                {"code": code, "exit_code": EXIT_CODES[code]} for code in ERROR_CODES
            ],
        },
        None,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="narrative-analyzer",
        description="Offline deterministic narrative branch and causal analysis toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="parse and validate a YAML narrative spec")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--output")
    validate.set_defaults(func=_cmd_validate)

    replay_parser = subparsers.add_parser("replay", help="deterministically replay a selection sequence")
    replay_parser.add_argument("--spec", required=True)
    replay_parser.add_argument("--sequence", required=True)
    replay_parser.add_argument("--inject", help="optional injection YAML/JSON")
    replay_parser.add_argument("--output")
    replay_parser.set_defaults(func=_cmd_replay)

    sensitivity = subparsers.add_parser("sensitivity", help="scan coefficient perturbations")
    sensitivity.add_argument("--spec", required=True)
    sensitivity.add_argument("--sequence", required=True)
    sensitivity.add_argument("--config")
    sensitivity.add_argument("--output")
    sensitivity.set_defaults(func=_cmd_sensitivity)

    paths = subparsers.add_parser("paths", help="audit branch-merge path consistency")
    paths.add_argument("--spec", required=True)
    paths.add_argument("--output")
    paths.add_argument("--max-paths", type=int, default=1024)
    paths.add_argument("--strict", action="store_true", help="return E_PATH_DIVERGE when divergent")
    paths.set_defaults(func=_cmd_paths)

    snapshot_check = subparsers.add_parser("snapshot-check", help="detect stale snapshot artifacts")
    snapshot_check.add_argument("--snapshot", required=True)
    snapshot_check.add_argument("--output")
    snapshot_check.set_defaults(func=_cmd_snapshot_check)

    demo = subparsers.add_parser("demo", help="generate and verify the reproducible sample corpus")
    demo.add_argument("--out-dir", default="artifacts/demo")
    demo.add_argument("--output")
    demo.set_defaults(func=_cmd_demo)

    error_codes = subparsers.add_parser("error-codes", help="list stable business error codes")
    error_codes.set_defaults(func=_cmd_error_codes)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as exc:
        payload = exc.to_dict()
        _write_or_print(payload, None if not hasattr(args, "output") else args.output)
        return EXIT_CODES[exc.code]
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        payload = {
            "business_code": "E_INTERNAL",
            "message": str(exc),
            "details": {"exception_type": type(exc).__name__},
        }
        _write_or_print(payload, None if not hasattr(args, "output") else args.output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

