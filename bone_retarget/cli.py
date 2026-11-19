"""Command-line interface for conversion, migration, retargeting, and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .binary_format import animation_to_binary, parse_binary
from .consistency import Tolerances
from .errors import ErrorCode, PipelineError
from .hashing import animation_semantic_sha256, sha256_bytes
from .json_format import (
    LATEST_SCHEMA_VERSION,
    animation_from_document,
    animation_to_document,
    document_json_bytes,
    migrate_json_document,
    parse_json_bytes,
)
from .models import Animation, Skeleton
from .resample import resample_to_fps
from .retarget import retarget_motion, target_transforms_document
from .verify import run_verification


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _write_json(path: Path, value: Any) -> str:
    data = document_json_bytes(value, pretty=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _load_animation(path: Path) -> Animation:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_json_bytes(data)
    if suffix in {".bin", ".btrj", ".trajectory"}:
        return parse_binary(data).animation
    raise PipelineError(
        ErrorCode.E_FIELD_MISSING,
        "cannot infer animation format from file extension",
        field="input",
        details={"suffix": suffix, "supported": [".json", ".bin", ".btrj", ".trajectory"]},
    )


def _load_skeleton(path: Path) -> Skeleton:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "skeleton file is not valid JSON",
            field="source_skeleton",
            details={"error": str(exc)},
        ) from exc
    return Skeleton.from_dict(document)


def _command_verify(args: argparse.Namespace) -> int:
    report = run_verification(args.out, clean=not args.no_clean)
    summary = {
        "status": report["status"],
        "case_count": report["case_count"],
        "anomaly_count": report["anomaly_count"],
        "error_coverage": report["error_coverage"],
        "manifest_sha256": report["manifest_sha256"],
        "report": str(Path(args.out) / "report.json"),
    }
    _print_json(summary)
    return 0 if report["status"] == "PASS" else 1


def _command_migrate(args: argparse.Namespace) -> int:
    source_path = Path(args.input)
    try:
        document = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "input is not valid JSON",
            field="input",
            details={"error": str(exc)},
        ) from exc
    first = migrate_json_document(document)
    second = migrate_json_document(first.document)
    output_path = Path(args.output)
    data = document_json_bytes(first.document, pretty=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    report = first.to_dict()
    report.update(
        {
            "idempotent": first.after_sha256 == second.before_sha256 == second.after_sha256,
            "second_before_sha256": second.before_sha256,
            "second_after_sha256": second.after_sha256,
            "output_file_sha256": sha256_bytes(data),
        }
    )
    _print_json(report)
    return 0


def _command_convert(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    input_data = input_path.read_bytes()
    source_suffix = input_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_suffix == ".json":
        try:
            document = json.loads(input_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "input is not valid JSON",
                field="input",
                details={"error": str(exc)},
            ) from exc
        migration = migrate_json_document(document)
        source_animation = animation_from_document(document)
        binary_data, metadata = animation_to_binary(
            source_animation,
            endian=args.endian,
            index_width=args.index_width,
            quat_order=args.quat_order,
            format_version=args.binary_version,
        )
        output_path.write_bytes(binary_data)
        parsed = parse_binary(binary_data)
        report = {
            "direction": "json_to_binary",
            "source_schema_version": migration.input_version,
            "migrated_schema_version": migration.output_version,
            "source_artifact_sha256": sha256_bytes(input_data),
            "destination_artifact_sha256": sha256_bytes(binary_data),
            "source_semantic_sha256": animation_semantic_sha256(source_animation),
            "destination_semantic_sha256": animation_semantic_sha256(parsed.animation),
            "equal_within_tolerance": animation_semantic_sha256(source_animation)
            == animation_semantic_sha256(parsed.animation),
            "joint_count": parsed.animation.joint_count,
            "frame_count": parsed.animation.frame_count,
            "metadata": metadata.to_dict(),
        }
    elif source_suffix in {".bin", ".btrj", ".trajectory"}:
        parsed = parse_binary(input_data)
        document = animation_to_document(
            parsed.animation,
            schema_version=args.schema_version,
            quat_order=args.quat_order,
        )
        output_data = document_json_bytes(document, pretty=True)
        output_path.write_bytes(output_data)
        reparsed = animation_from_document(document)
        report = {
            "direction": "binary_to_json",
            "source_artifact_sha256": sha256_bytes(input_data),
            "destination_artifact_sha256": sha256_bytes(output_data),
            "source_semantic_sha256": animation_semantic_sha256(parsed.animation),
            "destination_semantic_sha256": animation_semantic_sha256(reparsed),
            "equal_within_tolerance": animation_semantic_sha256(parsed.animation)
            == animation_semantic_sha256(reparsed),
            "joint_count": reparsed.joint_count,
            "frame_count": reparsed.frame_count,
            "output_schema_version": args.schema_version,
            "metadata": parsed.metadata.to_dict(),
        }
    else:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "cannot infer conversion direction from file extension",
            field="input",
            details={"suffix": source_suffix},
        )
    _print_json(report)
    return 0 if report["equal_within_tolerance"] else 1


def _command_retarget(args: argparse.Namespace) -> int:
    animation = _load_animation(Path(args.source_animation))
    source = _load_skeleton(Path(args.source_skeleton))
    target = _load_skeleton(Path(args.target_skeleton))
    tolerances = Tolerances(
        rotation_rad=args.rotation_tolerance,
        translation=args.translation_tolerance,
        time=args.time_tolerance,
        retarget_residual_rad=args.residual_tolerance,
    )
    resample_report: dict[str, Any] | None = None
    if args.target_fps is not None:
        animation, resample_info = resample_to_fps(animation, args.target_fps)
        resample_report = resample_info.to_dict()
    result = retarget_motion(animation, source, target, tolerances=tolerances)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_json_path = output_dir / "target_motion.json"
    target_json_hash = _write_json(
        target_json_path,
        animation_to_document(result.animation, schema_version=3, quat_order="wxyz"),
    )
    target_binary_data, target_binary_meta = animation_to_binary(
        result.animation,
        endian=args.endian,
        index_width=args.index_width,
        quat_order=args.quat_order,
    )
    target_binary_path = output_dir / "target_trajectory.bin"
    target_binary_path.write_bytes(target_binary_data)
    report_path = output_dir / "retarget_report.json"
    report_hash = _write_json(report_path, result.report)
    transforms_path = output_dir / "target_transforms.json"
    transforms_hash = _write_json(transforms_path, target_transforms_document(result))
    summary = {
        "joint_count": result.animation.joint_count,
        "frame_count": result.animation.frame_count,
        "root_translation_scale": result.report["root_translation_scale"],
        "max_residual_rad": result.report["max_residual_rad"],
        "max_global_deviation_rad": result.report["max_global_deviation_rad"],
        "flagged_joints": result.report["flagged_joints"],
        "resample": resample_report,
        "target_motion": str(target_json_path),
        "target_motion_sha256": target_json_hash,
        "target_binary": str(target_binary_path),
        "target_binary_metadata": target_binary_meta.to_dict(),
        "retarget_report": str(report_path),
        "retarget_report_sha256": report_hash,
        "target_transforms": str(transforms_path),
        "target_transforms_sha256": transforms_hash,
    }
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bone-retarget",
        description="Deterministic skeleton animation format and retargeting pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="run the complete deterministic verification suite")
    verify_parser.add_argument("--out", required=True, help="output sample/report directory")
    verify_parser.add_argument("--no-clean", action="store_true", help="keep existing output directory contents")
    verify_parser.set_defaults(handler=_command_verify)

    migrate_parser = subparsers.add_parser("migrate", help="migrate a JSON animation document to schema v3")
    migrate_parser.add_argument("--input", required=True)
    migrate_parser.add_argument("--output", required=True)
    migrate_parser.set_defaults(handler=_command_migrate)

    convert_parser = subparsers.add_parser("convert", help="convert JSON animation to binary or binary to JSON")
    convert_parser.add_argument("--input", required=True)
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--schema-version", type=int, default=LATEST_SCHEMA_VERSION)
    convert_parser.add_argument("--endian", choices=("little", "big"), default="little")
    convert_parser.add_argument("--index-width", type=int, choices=(2, 4), default=4)
    convert_parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    convert_parser.add_argument("--binary-version", type=int, choices=(1, 2), default=2)
    convert_parser.set_defaults(handler=_command_convert)

    retarget_parser = subparsers.add_parser("retarget", help="retarget a source animation onto a target skeleton")
    retarget_parser.add_argument("--source-animation", required=True)
    retarget_parser.add_argument("--source-skeleton", required=True)
    retarget_parser.add_argument("--target-skeleton", required=True)
    retarget_parser.add_argument("--output-dir", required=True)
    retarget_parser.add_argument("--target-fps", type=float, default=None)
    retarget_parser.add_argument("--endian", choices=("little", "big"), default="little")
    retarget_parser.add_argument("--index-width", type=int, choices=(2, 4), default=4)
    retarget_parser.add_argument("--quat-order", choices=("wxyz", "xyzw"), default="wxyz")
    retarget_parser.add_argument("--rotation-tolerance", type=float, default=1e-6)
    retarget_parser.add_argument("--translation-tolerance", type=float, default=1e-6)
    retarget_parser.add_argument("--time-tolerance", type=float, default=1e-9)
    retarget_parser.add_argument("--residual-tolerance", type=float, default=1e-6)
    retarget_parser.set_defaults(handler=_command_retarget)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        payload = {
            "code": "E_FIELD_MISSING",
            "message": str(exc),
            "joint": None,
            "frame": None,
            "field": None,
            "details": {"exception_type": type(exc).__name__},
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
