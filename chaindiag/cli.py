from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .acceptance import run_acceptance
from .errors import ChainFormatError
from .exchange import e1_to_e2
from .model import read_json, validate_chain, validate_config
from .pipeline import engine_from_data
from .samplegen import write_sample_bundle
from .runtime import verify_write_permission
from .snapshot import simulate_hard_kill


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def load_chain(path: str | Path, sidecar: str | Path | None = None) -> Any:
    path = Path(path)
    if path.suffix == ".txt":
        if sidecar:
            sidecar_path = Path(sidecar)
        else:
            sidecar_path = path.with_name(path.name[:-4] + ".sidecar.json") if path.name.endswith(".txt") else path.with_suffix(".sidecar.json")
        sidecar_data = read_json(sidecar_path) if sidecar_path.exists() else None
        return e1_to_e2(path.read_text(encoding="utf-8"), sidecar_data)
    return read_json(path)


def cmd_generate(args: argparse.Namespace) -> int:
    paths = write_sample_bundle(args.dir)
    _print({"status": "ok", "generated": {name: str(path) for name, path in paths.items()}})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    sample = Path(args.sample_dir)
    config = read_json(sample / "config.json")
    chain = load_chain(args.chain or sample / "chain.e2.json", args.sidecar)
    validate_config(config)
    operators, _ = validate_chain(chain)
    _print({"status": "ok", "operator_count": len(operators), "operator_seqs": [op["seq"] for op in operators]})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    sample = Path(args.sample_dir)
    write_check = verify_write_permission(Path(args.report).parent) if args.report else None
    config = read_json(sample / "config.json")
    chain = load_chain(args.chain or sample / "chain.e2.json", args.sidecar)
    tokens = read_json(sample / "tokens.json")
    segments = read_json(sample / "segments.json")
    injections = None
    injection_path = sample / "injections.json"
    if injection_path.exists():
        injections = read_json(injection_path)
    engine = engine_from_data(
        config=config,
        chain=chain,
        tokens=tokens,
        segments=segments,
        injections=injections if args.with_injections else None,
        serial=args.serial,
        overrides={"max_workers": args.workers, "point_budget": args.budget, "operator_quota": args.quota},
    )
    report = engine.run(with_width_reference=not args.no_width_reference)
    if write_check is not None:
        report["write_permission"] = write_check
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print({"status": report["status"], "tokens": len(report["token_results"]),
            "report": args.report, "diagnostics": report["diagnostics"][:3]})
    return 0 if report["status"] in ("ok", "fault_injected") else 1


def cmd_acceptance(args: argparse.Namespace) -> int:
    report = run_acceptance(args.output_dir)
    _print({"status": report["status"], "checks": report["checks"], "artifact_dir": report["artifact_dir"]})
    return 0 if report["status"] == "ok" else 1


def cmd_hard_kill(args: argparse.Namespace) -> int:
    simulate_hard_kill(args.dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chaindiag", description="Fixed-point embedding/aggregation chain diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate deterministic samples")
    generate.add_argument("dir", nargs="?", default="samples")
    generate.set_defaults(func=cmd_generate)

    validate = sub.add_parser("validate", help="validate JSON sample and chain")
    validate.add_argument("--sample-dir", default="samples")
    validate.add_argument("--chain", help="optional chain JSON path")
    validate.add_argument("--sidecar", help="e1 bypass sidecar JSON")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run one deterministic diagnosis")
    run.add_argument("--sample-dir", default="samples")
    run.add_argument("--chain")
    run.add_argument("--sidecar")
    run.add_argument("--report", default="artifacts/verification/run.json")
    run.add_argument("--serial", action="store_true")
    run.add_argument("--with-injections", action="store_true")
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--budget", type=int, default=None)
    run.add_argument("--quota", type=int, default=None)
    run.add_argument("--no-width-reference", action="store_true")
    run.set_defaults(func=cmd_run)

    acceptance = sub.add_parser("acceptance", help="run all local acceptance checks and write reports")
    acceptance.add_argument("--output-dir", default="artifacts/verification")
    acceptance.set_defaults(func=cmd_acceptance)

    hard_kill = sub.add_parser("hard-kill")
    hard_kill.add_argument("dir")
    hard_kill.set_defaults(func=cmd_hard_kill)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ChainFormatError as error:
        _print(error.to_report())
        return 2


if __name__ == "__main__":
    sys.exit(main())
