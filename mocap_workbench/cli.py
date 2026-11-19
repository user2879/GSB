"""Command-line entry points for the desktop workbench and headless audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .artifacts import export_snapshot
from .constants import APP_NAME, APP_VERSION, DEFAULT_SEED, FRAME_COUNTS
from .detection import compute_evidence, detect, thresholds_from_mapping, write_detection
from .errors import BusinessError, ErrorCode
from .generator import generate_bundle, load_bundle
from .io import atomic_write_json
from .pipeline import build_full_snapshot
from .service import WorkbenchService


ERROR_EXIT = {
    ErrorCode.E_FRAME_MISSING: 20,
    ErrorCode.E_THRESHOLD_RANGE: 21,
    ErrorCode.E_KEYFRAME_INVALID: 22,
    ErrorCode.E_QUAT_INVALID: 23,
    ErrorCode.E_CLOSURE_MISMATCH: 24,
    ErrorCode.E_SNAPSHOT_STALE: 25,
    ErrorCode.E_BUDGET_EXCEEDED: 26,
}


def _parse_thresholds(values: list[str] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise BusinessError(
                ErrorCode.E_THRESHOLD_RANGE,
                "threshold arguments must use key=value",
                {"argument": value},
            )
        key, text = value.split("=", 1)
        try:
            result[key] = float(text)
        except ValueError as exc:
            raise BusinessError(
                ErrorCode.E_THRESHOLD_RANGE,
                "threshold value is not numeric",
                {"key": key, "value": text},
            ) from exc
    return result


def _parse_frames(values: list[int] | None) -> frozenset[int]:
    return frozenset(int(value) for value in (values or []))


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _cmd_generate(args: argparse.Namespace) -> int:
    frame_count = args.frames[0]
    bundle = generate_bundle(Path(args.out), frame_count, args.seed)
    _print(
        {
            "ok": True,
            "frame_count": frame_count,
            "seed": args.seed,
            "defect_count": len(bundle.defects),
            "paths": {key: str(Path(value).resolve()) for key, value in bundle.paths().items()},
        }
    )
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    bundle = load_bundle(Path(args.bundle))
    thresholds = thresholds_from_mapping(_parse_thresholds(args.threshold))
    evidence = compute_evidence(bundle.record)
    detection = detect(bundle.record, bundle.defects, thresholds, evidence)
    output = Path(args.out)
    write_detection(output, detection, evidence, bundle.record)
    _print(
        {
            "ok": True,
            "output": str(output.resolve()),
            "thresholds": thresholds.to_dict(),
            "detection": detection.comparison,
            "detection_sha256": detection.detection_hash,
        }
    )
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    bundle = load_bundle(Path(args.bundle))
    thresholds = thresholds_from_mapping(_parse_thresholds(args.threshold))
    keyframes = _parse_frames(args.keyframe)
    suppressed = _parse_frames(args.suppress_auto)
    snapshot = build_full_snapshot(
        bundle.record,
        bundle.truth,
        bundle.defects,
        thresholds,
        keyframes,
        0,
        committed=False,
        suppressed_auto=suppressed,
    )
    paths = export_snapshot(snapshot, Path(args.out))
    _print(
        {
            "ok": True,
            "version": snapshot.version.to_dict(),
            "trusted": snapshot.evaluation.trusted,
            "evaluation": snapshot.evaluation.summary,
            "paths": paths,
        }
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .verification import run_verification

    frames = tuple(FRAME_COUNTS) if args.frames == ["all"] else tuple(int(v) for v in args.frames)
    for value in frames:
        if value not in FRAME_COUNTS:
            raise BusinessError(
                ErrorCode.E_FRAME_MISSING,
                "verification frame tier is unsupported",
                {"frame_count": value},
            )
    report = run_verification(Path(args.workspace), frames, args.seed, keep_duplicate=args.keep_duplicate)
    _print(
        {
            "ok": report["ok"],
            "elapsed_seconds": report["elapsed_seconds"],
            "report": str((Path(args.workspace) / "artifacts" / "verification_report.json").resolve()),
            "tiers": [
                {
                    "frame_count": tier["frame_count"],
                    "ok": tier["ok"],
                    "checks": tier["checks"],
                }
                for tier in report["tiers"]
            ],
        }
    )
    return 0 if report["ok"] else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    service = WorkbenchService.open(
        Path(args.workspace),
        args.frames,
        args.seed,
        namespace=f"demo-{args.frames}-{args.seed}",
        regenerate=False,
    )
    try:
        paths = service.export()
        _print(
            {
                "ok": True,
                "frame_count": service.snapshot.frame_count,
                "version": service.snapshot.version.to_dict(),
                "trusted": service.snapshot.evaluation.trusted,
                "restored": service.restored,
                "detection": service.snapshot.detection.comparison,
                "evaluation": service.snapshot.evaluation.summary,
                "paths": paths,
            }
        )
    finally:
        service.close()
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    from .gui import run_gui

    return run_gui(Path(args.workspace), args.frames, args.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mocap-workbench",
        description=f"{APP_NAME} {APP_VERSION} — deterministic waste-frame audit and sparse key-pose repair",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic synthetic record bundle")
    generate.add_argument("--frames", nargs=1, type=int, choices=FRAME_COUNTS, required=True)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--out", default="data/generated")
    generate.set_defaults(handler=_cmd_generate)

    detect_parser = subparsers.add_parser("detect", help="run detector and write per-frame evidence")
    detect_parser.add_argument("--bundle", required=True)
    detect_parser.add_argument("--threshold", action="append", help="key=value; repeatable")
    detect_parser.add_argument("--out", default="outputs/detection.json")
    detect_parser.set_defaults(handler=_cmd_detect)

    repair = subparsers.add_parser("repair", help="run a full sparse-keyframe repair and export audit files")
    repair.add_argument("--bundle", required=True)
    repair.add_argument("--threshold", action="append", help="key=value; repeatable")
    repair.add_argument("--keyframe", nargs="*", type=int, default=[])
    repair.add_argument("--suppress-auto", nargs="*", type=int, default=[])
    repair.add_argument("--out", default="outputs/repair")
    repair.set_defaults(handler=_cmd_repair)

    verify = subparsers.add_parser("verify", help="run full three-tier local acceptance verification")
    verify.add_argument("--workspace", default=".")
    verify.add_argument("--frames", nargs="+", default=["all"])
    verify.add_argument("--seed", type=int, default=DEFAULT_SEED)
    verify.add_argument("--keep-duplicate", action="store_true")
    verify.set_defaults(handler=_cmd_verify)

    demo = subparsers.add_parser("demo", help="open/generate one tier and export a deterministic demo")
    demo.add_argument("--workspace", default=".")
    demo.add_argument("--frames", type=int, choices=FRAME_COUNTS, default=1200)
    demo.add_argument("--seed", type=int, default=DEFAULT_SEED)
    demo.set_defaults(handler=_cmd_demo)

    app = subparsers.add_parser("app", help="launch the PySide6 desktop workbench")
    app.add_argument("--workspace", default=".")
    app.add_argument("--frames", type=int, choices=FRAME_COUNTS, default=1200)
    app.add_argument("--seed", type=int, default=DEFAULT_SEED)
    app.set_defaults(handler=_cmd_app)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BusinessError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return ERROR_EXIT.get(exc.code, 1)
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "code": "INTERRUPTED", "message": "interrupted"}, ensure_ascii=False))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
