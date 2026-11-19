from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .artifact import convert_index, decode_with_compat, load_meta, write_index
from .compat import CompatTable
from .config import Config
from .engine import RetrievalEngine
from .errors import PipelineError
from .postings import canonical_postings_digest
from .sample_data import generate_samples
from .util import atomic_write_text, deep_merge, read_json, sha256_json
from .verification import run_consistency_suite


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _build(args: argparse.Namespace) -> int:
    outcome = RetrievalEngine(args.state).build(
        args.docs,
        args.queries,
        args.config,
        args.compat,
        output_dir=args.out,
        fresh=args.fresh,
    )
    _print(outcome.result)
    return 0 if outcome.status == "OK" else 3


def _edit(args: argparse.Namespace) -> int:
    engine = RetrievalEngine(args.state)
    if args.command_file:
        outcomes = engine.apply_edit_file(args.command_file, output_dir=args.out, budget_ms=args.budget_ms)
        payload = [outcome.result for outcome in outcomes]
        if not outcomes:
            _print({"status": "OK", "message": "no edit commands"})
            return 0
        _print(payload)
        status = outcomes[-1].status
    else:
        if not args.command_json:
            raise PipelineError("E_EDIT_INVALID", "provide --command-file or --command-json")
        command = json.loads(args.command_json)
        outcome = engine.apply_edit(command, output_dir=args.out, budget_ms=args.budget_ms)
        _print(outcome.result)
        status = outcome.status
    return 0 if status in {"OK", "OK_IDEMPOTENT"} else 3


def _resume(args: argparse.Namespace) -> int:
    current = RetrievalEngine(args.state).resume(args.config, args.version)
    _print(
        {
            "status": "OK",
            "global_version": current.version,
            "snapshot": str(current.snapshot_root),
            "manifest": current.manifest,
        }
    )
    return 0


def _convert(args: argparse.Namespace) -> int:
    engine = RetrievalEngine(args.state)
    current = engine.manager.load_current()
    if current is None:
        raise PipelineError("E_INDEX_INVALID", "no committed snapshot to convert")
    compat_raw = read_json(current.input_path("compat.json"))
    config_raw = read_json(current.input_path("config.json"))
    compat = CompatTable.from_mapping(compat_raw, file_hash=sha256_json(compat_raw))
    source_root = current.stage_path("compressed")
    source_meta = load_meta(source_root)
    patch = {"index_format_version": args.target_format}
    if args.tier is not None:
        patch["active_compression_tier"] = args.tier
    if args.drop_positions:
        patch["keep_positions"] = False
    target_config = Config.from_mapping(deep_merge(config_raw, patch), compat)
    encoded = convert_index(
        source_root,
        target_config,
        args.target_format,
        source_meta,
        target_tier=target_config.active_tier,
        keep_positions=target_config.keep_positions,
    )
    target_root = Path(args.out)
    if target_root.exists() and args.replace:
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    write_index(encoded, target_root)
    decoded = decode_with_compat(target_root, compat)
    source_decoded = decode_with_compat(source_root, compat)
    source_digest = canonical_postings_digest(source_decoded.postings)
    target_digest = canonical_postings_digest(decoded.postings)
    source_tier_semantic = compat.tier_semantic(source_meta["format_version"], int(source_meta["compression_tier"]))
    target_tier_semantic = encoded.meta["tier_semantic"]
    result = {
        "status": "OK" if source_digest == target_digest else "E_INDEX_INVALID",
        "source_format": source_meta["format_version"],
        "target_format": encoded.meta["format_version"],
        "compression_tier": encoded.meta["compression_tier"],
        "tier_semantic": encoded.meta["tier_semantic"],
        "tier_mapping_applied": {
            "source": source_tier_semantic,
            "target": target_tier_semantic,
            "mapped": source_tier_semantic != target_tier_semantic,
        },
        "keep_positions": encoded.meta["keep_positions"],
        "index_fingerprint": encoded.fingerprint,
        "source_logical_digest": source_digest,
        "target_logical_digest": target_digest,
        "logical_content_preserved": source_digest == target_digest,
        "degradation_losses": encoded.meta["degradation_losses"],
    }
    atomic_write_text(target_root / "conversion_report.json", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _print(result)
    return 0 if result["status"] == "OK" else 2


def _verify(args: argparse.Namespace) -> int:
    samples = Path(args.samples)
    if not (samples / "edits.jsonl").is_file():
        generate_samples(samples, args.seed)
    report = run_consistency_suite(samples, args.out)
    _print(
        {
            "status": "OK" if report["ok"] else "E_INDEX_INVALID",
            "overall": "PASS" if report["ok"] else "FAIL",
            "checks": report["checks"],
            "summary": report["summary"],
            "report_json": str(Path(args.out) / "consistency_report.json"),
            "report_md": str(Path(args.out) / "consistency_report.md"),
        }
    )
    return 0 if report["ok"] else 2


def _generate(args: argparse.Namespace) -> int:
    _print(generate_samples(args.out, args.seed))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="cm_fusion", description="Deterministic cross-modal retrieval pipeline")
    subparsers = result.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--out", default="samples")
    generate.add_argument("--seed", type=int, default=20261120)
    generate.set_defaults(func=_generate)

    build = subparsers.add_parser("build", help="build a snapshot from source files")
    build.add_argument("--state", required=True)
    build.add_argument("--docs", required=True)
    build.add_argument("--queries", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--compat", required=True)
    build.add_argument("--out")
    build.add_argument("--fresh", action="store_true", help="allow adding a new full snapshot over existing state")
    build.set_defaults(func=_build)

    edit = subparsers.add_parser("edit", help="apply one edit command or a JSONL edit stream")
    edit.add_argument("--state", required=True)
    edit.add_argument("--command-file")
    edit.add_argument("--command-json")
    edit.add_argument("--out")
    edit.add_argument("--budget-ms", type=float)
    edit.set_defaults(func=_edit)

    resume = subparsers.add_parser("resume", help="validate and resume the committed snapshot")
    resume.add_argument("--state", required=True)
    resume.add_argument("--config")
    resume.add_argument("--version", type=int)
    resume.set_defaults(func=_resume)

    convert = subparsers.add_parser("convert", help="convert the committed index to v1 or v2")
    convert.add_argument("--state", required=True)
    convert.add_argument("--out", required=True)
    convert.add_argument("--target-format", choices=("v1", "v2"), required=True)
    convert.add_argument("--tier", type=int, choices=(1, 2))
    convert.add_argument("--drop-positions", action="store_true")
    convert.add_argument("--replace", action="store_true")
    convert.set_defaults(func=_convert)

    verify = subparsers.add_parser("verify", help="run the complete consistency suite")
    verify.add_argument("--samples", default="samples")
    verify.add_argument("--out", default="reports/demo")
    verify.add_argument("--seed", type=int, default=20261120)
    verify.set_defaults(func=_verify)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except PipelineError as exc:
        _print({"status": exc.code, "code": exc.code, "message": exc.message, "details": exc.details})
        return 2
    except json.JSONDecodeError as exc:
        _print({"status": "E_INDEX_INVALID", "code": "E_INDEX_INVALID", "message": f"invalid JSON: {exc}"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
