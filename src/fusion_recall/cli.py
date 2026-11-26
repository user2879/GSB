from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .doctor import environment_report
from .engine import Engine, prepare_config
from .generator import generate_sample_files
from .loading import load_documents, load_queries
from .commands import parse_commands
from .config import validate_config_safe
from .store import atomic_write_json
from .verification import write_verification_report
from .vocab import load_vocabulary


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run_sample(root: Path) -> dict:
    paths = generate_sample_files(root)
    workspace = root / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    vocab = load_vocabulary(paths["vocab"])
    raw_config = json.loads(Path(paths["config"]).read_text(encoding="utf-8"))
    config = prepare_config(raw_config, vocab)
    documents = load_documents(config, vocab, path=paths["documents"])
    queries = load_queries(config, vocab, path=paths["queries"])
    commands = parse_commands(config, vocab, path=paths["commands"])
    engine = Engine(
        config,
        vocab,
        queries["queries"],
        root=workspace,
        input_failures=documents["failures"] + queries["failures"] + commands["failures"],
    )
    engine.cold_build(documents["documents"])
    engine.apply_commands(commands["commands"])
    report = engine.run_batch(unlimited=True, workers=4)
    engine.write_snapshot(report)
    atomic_write_json(root / "runs" / "last_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fusion-recall")
    parser.add_argument("--root", default=".", help="project root")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check Python, NumPy, loopback port and staging directory")
    sub.add_parser("generate", help="generate deterministic config and local samples")
    validate_parser = sub.add_parser("validate-config", help="validate a pipeline config file")
    validate_parser.add_argument("--config", default="configs/pipeline.json")
    sub.add_parser("run-sample", help="run incremental command sequence and write snapshot/report")
    sub.add_parser("verify", help="run parity, cache, budget, concurrency and recovery checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    if args.command == "doctor":
        report = environment_report(root)
        _print_json(report)
        return 0 if report["ready"] else 1
    if args.command == "generate":
        paths = generate_sample_files(root)
        _print_json({"generated": paths})
        return 0
    if args.command == "validate-config":
        raw_config = json.loads((root / args.config).read_text(encoding="utf-8"))
        config, failure = validate_config_safe(raw_config)
        if failure is not None:
            _print_json({"ok": False, "failure": failure})
            return 1
        _print_json(
            {
                "ok": True,
                "config_fingerprint": config["config_fingerprint"],
                "full_config_fingerprint": config["full_config_fingerprint"],
            }
        )
        return 0
    if args.command == "run-sample":
        report = run_sample(root)
        _print_json(
            {
                "ok": True,
                "counts": report["counts"],
                "business_codes": report["business_codes"],
                "snapshot_shutdown": report["shutdown"],
                "ledger": str(root / "workspace" / "ledger.json"),
            }
        )
        return 0
    if args.command == "verify":
        output = write_verification_report(root)
        report = json.loads(output.read_text(encoding="utf-8"))
        _print_json(
            {
                "ok": report["ok"],
                "report": str(output),
                "parity": report["parity"],
                "budget_first_ready": report["budget_first_batch"]["ready"],
                "recovery_decision": report["recovery"]["recovery_result"]["decision"],
            }
        )
        return 0 if report["ok"] and report["recovery"]["recovered_to_v2"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
