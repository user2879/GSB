"""Command line interface for the deterministic render workbench."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import load_workbench
from .errors import WorkbenchError
from .workbench import Workbench, load_edit_sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "workbench.yaml"
DEFAULT_EDITS = ROOT / "configs" / "edit_sequence.yaml"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> Workbench:
    result = load_workbench(path)
    if not result.can_render:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(2)
    return Workbench(path)


def command_validate(args: argparse.Namespace) -> int:
    result = load_workbench(args.config)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.can_render else 2


def command_run(args: argparse.Namespace) -> int:
    result = load_workbench(args.config, temp_dir_override=args.temp_dir)
    if not result.can_render:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    workbench = Workbench(args.config, temp_dir=args.temp_dir, batch_size=args.batch_size)
    if args.edits:
        workbench.apply_edits(load_edit_sequence(args.edits))
    report = workbench.run()
    _write_json(Path(args.report), report.to_dict())
    payload = report.to_dict()
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "business_code": payload.get("business_code"),
                "run_signature": payload.get("run_signature"),
                "run_fingerprint": payload.get("run_fingerprint"),
                "canonical_fingerprint": report.canonical_fingerprint(),
                "frames": len(payload.get("frames") or []),
                "strategy": (payload.get("strategy") or {}).get("selected", {}).get("policy_id"),
                "reuse": payload.get("reuse"),
                "cache": {
                    key: (payload.get("cache") or {}).get(key)
                    for key in ("reuses", "evictions", "recomputes", "recompute_matches", "recompute_mismatches")
                },
                "report": str(Path(args.report).resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.status == "ok" else 2


def command_demo(args: argparse.Namespace) -> int:
    config = Path(args.config).resolve()
    edits = load_edit_sequence(args.edits)
    incremental_dir = Path(args.incremental_dir).resolve()
    fresh_dir = Path(args.fresh_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    shutil.rmtree(incremental_dir, ignore_errors=True)
    shutil.rmtree(fresh_dir, ignore_errors=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    incremental = Workbench(config, temp_dir=incremental_dir, batch_size=args.batch_size)
    sequence_reports = incremental.run_edit_sequence(edits)
    final_incremental = sequence_reports[-1]

    fresh = Workbench(config, temp_dir=fresh_dir, batch_size=args.batch_size)
    fresh.apply_edits(edits)
    full_rebuild = fresh.run()

    canonical_equal = final_incremental.canonical_dict() == full_rebuild.canonical_dict()
    frame_byte_equal = True
    frame_comparisons: list[dict[str, Any]] = []
    for incremental_frame, fresh_frame in zip(
        final_incremental.to_dict()["frames"], full_rebuild.to_dict()["frames"]
    ):
        incremental_path = incremental_dir / incremental_frame["png_relative_path"]
        fresh_path = fresh_dir / fresh_frame["png_relative_path"]
        incremental_bytes = incremental_path.read_bytes()
        fresh_bytes = fresh_path.read_bytes()
        equal = incremental_bytes == fresh_bytes
        frame_byte_equal = frame_byte_equal and equal
        frame_comparisons.append(
            {
                "frame_id": incremental_frame["frame_id"],
                "png_sha256": incremental_frame["png_sha256"],
                "byte_equal": equal,
                "bytes": len(incremental_bytes),
            }
        )

    differences: list[str] = []
    left = final_incremental.canonical_dict()
    right = full_rebuild.canonical_dict()
    for key in left:
        if left[key] != right[key]:
            differences.append(key)

    equivalence = {
        "status": "ok" if canonical_equal and frame_byte_equal and not differences else "failed",
        "canonical_equal": canonical_equal,
        "all_png_bytes_equal": frame_byte_equal,
        "canonical_differences": differences,
        "incremental_canonical_fingerprint": final_incremental.canonical_fingerprint(),
        "full_rebuild_canonical_fingerprint": full_rebuild.canonical_fingerprint(),
        "incremental_run_fingerprint": final_incremental.to_dict().get("run_fingerprint"),
        "full_rebuild_run_fingerprint": full_rebuild.to_dict().get("run_fingerprint"),
        "edit_count": len(edits),
        "incremental_sequence_runs": len(sequence_reports),
        "final_incremental_reuse": final_incremental.to_dict().get("reuse"),
        "final_incremental_invalidation_summary": final_incremental.to_dict()
        .get("invalidation", {})
        .get("summary"),
        "final_incremental_cache": final_incremental.to_dict().get("cache"),
        "frames": frame_comparisons,
    }
    _write_json(report_dir / "equivalence.json", equivalence)
    _write_json(report_dir / "incremental_final.json", final_incremental.to_dict())
    _write_json(report_dir / "full_rebuild.json", full_rebuild.to_dict())
    print(json.dumps(equivalence, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if equivalence["status"] == "ok" else 2


def command_preview_probe(args: argparse.Namespace) -> int:
    workbench = Workbench(args.config, temp_dir=args.temp_dir, batch_size=args.batch_size)
    handle = workbench.run_async()
    initial = workbench.preview(args.budget_ms)
    report = handle.result(timeout=args.timeout_seconds)
    final = workbench.preview(0)
    payload = {
        "initial_preview": initial.to_dict(),
        "final_preview": final.to_dict(),
        "render_status": report.status,
        "render_business_code": report.business_code,
    }
    _write_json(Path(args.report), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="renderlab", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate YAML, tables and sample inputs")
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate.set_defaults(function=command_validate)

    run = subparsers.add_parser("run", help="execute one deterministic render plan")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--temp-dir", default=None)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--edits", default=None, help="optional edit-sequence YAML applied before the run")
    run.add_argument("--report", default=str(ROOT / "reports" / "run_report.json"))
    run.set_defaults(function=command_run)

    demo = subparsers.add_parser("demo", help="run the edit sequence incrementally and compare with a cold rebuild")
    demo.add_argument("--config", default=str(DEFAULT_CONFIG))
    demo.add_argument("--edits", default=str(DEFAULT_EDITS))
    demo.add_argument("--batch-size", type=int, default=2)
    demo.add_argument("--incremental-dir", default=str(ROOT / ".renderlab" / "demo-incremental"))
    demo.add_argument("--fresh-dir", default=str(ROOT / ".renderlab" / "demo-fresh"))
    demo.add_argument("--report-dir", default=str(ROOT / "reports"))
    demo.set_defaults(function=command_demo)

    preview = subparsers.add_parser("preview-probe", help="probe preview-budget behaviour during an async render")
    preview.add_argument("--config", default=str(DEFAULT_CONFIG))
    preview.add_argument("--temp-dir", default=str(ROOT / ".renderlab" / "preview-probe"))
    preview.add_argument("--batch-size", type=int, default=2)
    preview.add_argument("--budget-ms", type=int, default=0)
    preview.add_argument("--timeout-seconds", type=float, default=60.0)
    preview.add_argument("--report", default=str(ROOT / "reports" / "preview_probe.json"))
    preview.set_defaults(function=command_preview_probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except WorkbenchError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "business_code": exc.code,
                    "component": exc.component,
                    "item": exc.item,
                    "message": exc.message,
                    "details": exc.details,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
