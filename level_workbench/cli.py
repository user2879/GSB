"""Command-line entry points for generation, verification, demo and compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

from .canonical import canonical_json_bytes, grid_text, sha256_value, write_canonical_json, write_text
from .counterexample import replay_minimal_counterexample
from .errors import WorkbenchError
from .injection import apply_spec_patches, load_injection_file, normalize_injection
from .runner import (
    compare_directories,
    pin_thread_environment,
    run_case,
    run_demo,
    write_demo_artifacts,
)
from .spec import load_spec_file, parse_spec
from .verifier import verify_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="level-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the fixed local sample sequence")
    demo.add_argument("--out", required=True, help="directory for reports, levels and hashes")

    generate = subparsers.add_parser("generate", help="generate one level from YAML")
    generate.add_argument("--spec", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--inject", default=None)

    verify = subparsers.add_parser("verify", help="independently verify a level artifact")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--level", required=True)

    validate = subparsers.add_parser("validate-spec", help="parse and validate a YAML specification")
    validate.add_argument("--spec", required=True)

    compare = subparsers.add_parser("compare", help="compare two demo output trees byte-for-byte")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)

    replay = subparsers.add_parser("replay-counterexample", help="replay a serialized minimal counterexample")
    replay.add_argument("--spec", required=True)
    replay.add_argument("--counterexample", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    pin_thread_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            report = run_demo()
            out = Path(args.out)
            manifest = write_demo_artifacts(out, report)
            print(
                _json_text(
                    {
                        "status": "pass" if report["determinism"]["case_reports_byte_identical"] else "fail",
                        "out": out.as_posix(),
                        "cases": report["summary"],
                        "determinism": report["determinism"],
                        "run_sha256": manifest["run_sha256"],
                        "report_sha256": manifest["files"].get("report.json"),
                    }
                )
            )
            return 0 if report["determinism"]["case_reports_byte_identical"] else 2

        if args.command == "generate":
            raw = load_spec_file(args.spec)
            injection_raw = load_injection_file(args.inject) if args.inject else None
            injection = normalize_injection(injection_raw)
            patched = apply_spec_patches(raw, injection)
            # Parse before touching the output directory, guaranteeing that an
            # invalid spec creates no partial grid file.
            parse_spec(patched)
            result = run_case(
                {
                    "id": "cli-generate",
                    "kind": "cli",
                    "spec": raw,
                    "injection": injection,
                }
            )
            _write_generate_artifacts(Path(args.out), result)
            print(_json_text(_result_summary(result)))
            return 0 if result["status"] == "pass" else 2

        if args.command == "verify":
            spec = parse_spec(load_spec_file(args.spec))
            level_doc = json.loads(Path(args.level).read_text(encoding="utf-8"))
            result = verify_artifact(spec, level_doc)
            print(_json_text(result))
            return 0 if result["status"] == "pass" else 2

        if args.command == "validate-spec":
            spec = parse_spec(load_spec_file(args.spec))
            print(
                _json_text(
                    {
                        "status": "pass",
                        "spec_sha256": sha256_value(spec.normalized),
                        "spec": spec.normalized,
                    }
                )
            )
            return 0

        if args.command == "compare":
            result = compare_directories(Path(args.a), Path(args.b))
            print(_json_text(result))
            return 0 if result["identical"] else 2

        if args.command == "replay-counterexample":
            spec = parse_spec(load_spec_file(args.spec))
            counterexample = json.loads(Path(args.counterexample).read_text(encoding="utf-8"))
            result = replay_minimal_counterexample(spec, counterexample)
            print(_json_text(result))
            return 0 if result.get("reproduced") else 2

        parser.error("unknown command")
        return 2
    except WorkbenchError as exc:
        print(_json_text({"status": "error", "error": exc.to_dict()}))
        return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(_json_text({"status": "error", "error": {"code": "E_IO", "message": str(exc)}}))
        return 2


def _write_generate_artifacts(out: Path, result: dict[str, Any]) -> None:
    level_doc = result.get("level_document")
    if level_doc is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    write_canonical_json(out / "level.json", level_doc)
    write_text(out / "grid.txt", grid_text(level_doc["grid"]["cells"]))
    _write_yaml(out / "effective_spec.yaml", result.get("effective_spec"))
    _write_yaml(out / "injection.yaml", result.get("injection"))
    write_canonical_json(out / "result.json", result)


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(value, sort_keys=True, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8", newline="\n")


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "code": result["code"],
        "message": result["message"],
        "first_violation": result.get("first_violation"),
        "minimal_counterexample": result.get("minimal_counterexample"),
        "hashes": result.get("hashes", {}),
    }


def _json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
