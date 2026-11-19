"""Deterministic end-to-end sample generation and verification."""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .binary_format import (
    animation_to_binary,
    binary_to_json_document,
    parse_binary,
)
from .consistency import Tolerances, compare_animations
from .deterministic import generate_animation, generate_skeleton
from .errors import ErrorCode, PipelineError
from .hashing import animation_semantic_sha256, sha256_bytes, sha256_json
from .json_format import (
    animation_from_document,
    animation_to_document,
    document_json_bytes,
    migrate_json_document,
)
from .math3d import quat_from_axis_angle, quat_multiply
from .models import Animation, Skeleton
from .resample import resample_to_fps
from .retarget import retarget_motion, target_transforms_document

TARGET_FPS = 30.0
PAIR_SEEDS = ((11, 31), (17, 47), (23, 59))
CASE_VARIANTS: tuple[tuple[int, str, int, str], ...] = (
    (1, "little", 2, "wxyz"),
    (2, "big", 4, "xyzw"),
    (3, "little", 4, "xyzw"),
    (1, "big", 2, "wxyz"),
    (2, "big", 4, "wxyz"),
    (3, "little", 2, "xyzw"),
)
REPORT_EXCLUDED_NAMES = {"report.json", "report.txt", "manifest.sha256", "hashes.sha256"}


@dataclass
class CaseInputs:
    source_skeleton: Skeleton
    target_skeleton: Skeleton
    source_seed: int
    target_seed: int


def _write_json(path: Path, document: Any, *, allow_nan: bool = False) -> str:
    data = document_json_bytes(document, pretty=True, allow_nan=allow_nan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _relative(output: Path, path: Path) -> str:
    return path.relative_to(output).as_posix()


def _comparison_ok(*comparisons: dict[str, Any]) -> bool:
    return all(bool(item["equal"]) for item in comparisons)


def _case_name(case_index: int, source_seed: int, target_seed: int, schema: int, endian: str, width: int, order: str) -> str:
    return (
        f"case_{case_index:02d}_p{source_seed}_{target_seed}"
        f"_s{schema}_{endian}_iw{width}_{order}"
    )


def _run_case(
    output: Path,
    case_index: int,
    inputs: CaseInputs,
    schema_version: int,
    endian: str,
    index_width: int,
    quat_order: str,
    tolerances: Tolerances,
) -> dict[str, Any]:
    case_name = _case_name(
        case_index,
        inputs.source_seed,
        inputs.target_seed,
        schema_version,
        endian,
        index_width,
        quat_order,
    )
    case_dir = output / "cases" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    source = inputs.source_skeleton
    target = inputs.target_skeleton
    animation_seed = inputs.source_seed * 1009 + inputs.target_seed * 9176 + 41
    raw_animation = generate_animation(source, animation_seed, frame_count=7, fps=12.0)
    raw_document = animation_to_document(
        raw_animation,
        schema_version=schema_version,
        quat_order=quat_order,
    )
    raw_path = case_dir / f"source_schema_v{schema_version}.json"
    raw_file_hash = _write_json(raw_path, raw_document)

    migration_first = migrate_json_document(raw_document)
    migration_second = migrate_json_document(migration_first.document)
    migrated_path = case_dir / "source_migrated_v3.json"
    migrated_file_hash = _write_json(migrated_path, migration_first.document)
    migration_idempotent = (
        sha256_json(migration_first.document) == sha256_json(migration_second.document)
        and migration_second.before_sha256 == migration_second.after_sha256
    )

    baseline = animation_from_document(raw_document)

    channel_order = (
        tuple(range(1, source.joint_count)) + (0,) if case_index % 2 == 1 else None
    )
    source_binary, source_writer_meta = animation_to_binary(
        baseline,
        endian=endian,
        index_width=index_width,
        quat_order=quat_order,
        channel_joint_order=channel_order,
    )
    source_binary_path = case_dir / "source_trajectory.bin"
    source_binary_file_hash = _write_bytes(source_binary_path, source_binary)
    parsed_binary = parse_binary(source_binary)
    source_binary_comparison = compare_animations(
        baseline,
        parsed_binary.animation,
        tolerances=tolerances,
        left_name="source_json_semantics",
        right_name="source_binary_roundtrip",
        raise_on_divergence=True,
    ).to_dict()

    back_document, _, back_conversion = binary_to_json_document(
        source_binary, schema_version=3, quat_order="wxyz"
    )
    back_path = case_dir / "binary_back_to_json_v3.json"
    back_file_hash = _write_json(back_path, back_document)
    back_animation = animation_from_document(back_document)
    back_comparison = compare_animations(
        parsed_binary.animation,
        back_animation,
        tolerances=tolerances,
        left_name="source_binary",
        right_name="binary_to_json",
        raise_on_divergence=True,
    ).to_dict()

    resampled, resample_report = resample_to_fps(baseline, TARGET_FPS)
    resampled_json_document = animation_to_document(
        resampled,
        schema_version=schema_version,
        quat_order=quat_order,
    )
    resampled_json_path = case_dir / f"resampled_schema_v{schema_version}.json"
    resampled_json_file_hash = _write_json(resampled_json_path, resampled_json_document)
    resampled_json_animation = animation_from_document(resampled_json_document)
    resampled_json_comparison = compare_animations(
        resampled,
        resampled_json_animation,
        tolerances=tolerances,
        left_name="resampled_direct",
        right_name="resampled_json_roundtrip",
        raise_on_divergence=True,
    ).to_dict()

    resampled_binary, resampled_binary_meta = animation_to_binary(
        resampled,
        endian=endian,
        index_width=index_width,
        quat_order=quat_order,
        channel_joint_order=channel_order,
    )
    resampled_binary_path = case_dir / "resampled_trajectory.bin"
    resampled_binary_file_hash = _write_bytes(resampled_binary_path, resampled_binary)
    resampled_binary_animation = parse_binary(resampled_binary).animation
    resampled_binary_comparison = compare_animations(
        resampled,
        resampled_binary_animation,
        tolerances=tolerances,
        left_name="resampled_direct",
        right_name="resampled_binary_roundtrip",
        raise_on_divergence=True,
    ).to_dict()

    direct = retarget_motion(resampled, source, target, tolerances=tolerances)
    from_json = retarget_motion(
        resampled_json_animation, source, target, tolerances=tolerances
    )
    from_binary = retarget_motion(
        resampled_binary_animation, source, target, tolerances=tolerances
    )
    json_retarget_comparison = compare_animations(
        direct.animation,
        from_json.animation,
        tolerances=tolerances,
        left_name="direct_retarget",
        right_name="json_roundtrip_retarget",
        raise_on_divergence=True,
    ).to_dict()
    binary_retarget_comparison = compare_animations(
        direct.animation,
        from_binary.animation,
        tolerances=tolerances,
        left_name="direct_retarget",
        right_name="binary_roundtrip_retarget",
        raise_on_divergence=True,
    ).to_dict()

    target_document = animation_to_document(
        direct.animation, schema_version=3, quat_order="wxyz"
    )
    target_json_path = case_dir / "target_motion.json"
    target_json_file_hash = _write_json(target_json_path, target_document)
    target_json_animation = animation_from_document(target_document)
    target_json_comparison = compare_animations(
        direct.animation,
        target_json_animation,
        tolerances=tolerances,
        left_name="target_direct",
        right_name="target_json_roundtrip",
        raise_on_divergence=True,
    ).to_dict()

    target_binary, target_binary_meta = animation_to_binary(
        direct.animation,
        endian=endian,
        index_width=index_width,
        quat_order=quat_order,
        channel_joint_order=channel_order,
    )
    target_binary_path = case_dir / "target_trajectory.bin"
    target_binary_file_hash = _write_bytes(target_binary_path, target_binary)
    target_binary_animation = parse_binary(target_binary).animation
    target_binary_comparison = compare_animations(
        direct.animation,
        target_binary_animation,
        tolerances=tolerances,
        left_name="target_direct",
        right_name="target_binary_roundtrip",
        raise_on_divergence=True,
    ).to_dict()

    retarget_report_path = case_dir / "retarget_report.json"
    retarget_report_hash = _write_json(retarget_report_path, direct.report)
    target_transforms_path = case_dir / "target_transforms.json"
    target_transforms = target_transforms_document(direct)
    target_transforms_hash = _write_json(target_transforms_path, target_transforms)

    source_semantic_hash = animation_semantic_sha256(baseline)
    binary_semantic_hash = animation_semantic_sha256(parsed_binary.animation)
    conversion_reports = {
        "json_to_binary": {
            "source_artifact_sha256": raw_file_hash,
            "destination_artifact_sha256": source_binary_file_hash,
            "source_semantic_sha256": source_semantic_hash,
            "destination_semantic_sha256": binary_semantic_hash,
            "equal_within_tolerance": source_semantic_hash == binary_semantic_hash,
            "metadata": source_writer_meta.to_dict(),
        },
        "binary_to_json": back_conversion.to_dict(),
    }

    joint_residuals = direct.report["joints"]
    checks = [
        migration_idempotent,
        source_binary_comparison["equal"],
        back_comparison["equal"],
        resampled_json_comparison["equal"],
        resampled_binary_comparison["equal"],
        json_retarget_comparison["equal"],
        binary_retarget_comparison["equal"],
        target_json_comparison["equal"],
        target_binary_comparison["equal"],
        direct.report["max_residual_rad"] <= tolerances.retarget_residual_rad,
        direct.report["max_global_deviation_rad"] <= tolerances.retarget_residual_rad,
        not direct.report["flagged_joints"],
    ]
    case_report: dict[str, Any] = {
        "case_id": case_name,
        "source_seed": inputs.source_seed,
        "target_seed": inputs.target_seed,
        "schema_version": schema_version,
        "json_quat_order": "xyzw" if schema_version == 1 else quat_order,
        "binary": {
            "endian": endian,
            "index_width": index_width,
            "quat_order": quat_order,
            "channel_joint_order": list(channel_order) if channel_order else list(range(source.joint_count)),
            "parsed_metadata": parsed_binary.metadata.to_dict(),
            "resampled_parsed_metadata": resampled_binary_meta.to_dict(),
            "target_parsed_metadata": target_binary_meta.to_dict(),
        },
        "migration": {
            **migration_first.to_dict(),
            "idempotent": migration_idempotent,
            "second_before_sha256": migration_second.before_sha256,
            "second_after_sha256": migration_second.after_sha256,
        },
        "hashes": {
            "source_json_file": raw_file_hash,
            "source_migrated_json_file": migrated_file_hash,
            "source_binary_file": source_binary_file_hash,
            "binary_back_json_file": back_file_hash,
            "resampled_json_file": resampled_json_file_hash,
            "resampled_binary_file": resampled_binary_file_hash,
            "target_json_file": target_json_file_hash,
            "target_binary_file": target_binary_file_hash,
            "retarget_report_file": retarget_report_hash,
            "target_transforms_file": target_transforms_hash,
            "source_semantic": _semantic_hash(baseline),
            "binary_semantic": _semantic_hash(parsed_binary.animation),
            "target_semantic": _semantic_hash(direct.animation),
        },
        "conversion": conversion_reports,
        "resample": resample_report.to_dict(),
        "comparisons": {
            "json_to_binary": source_binary_comparison,
            "binary_to_json": back_comparison,
            "resampled_json_roundtrip": resampled_json_comparison,
            "resampled_binary_roundtrip": resampled_binary_comparison,
            "json_roundtrip_retarget": json_retarget_comparison,
            "binary_roundtrip_retarget": binary_retarget_comparison,
            "target_json_roundtrip": target_json_comparison,
            "target_binary_roundtrip": target_binary_comparison,
        },
        "retarget": {
            "max_residual_rad": direct.report["max_residual_rad"],
            "max_global_deviation_rad": direct.report["max_global_deviation_rad"],
            "max_residual_frame": direct.report["max_residual_frame"],
            "max_residual_joint": direct.report["max_residual_joint"],
            "max_deviation_frame": direct.report["max_deviation_frame"],
            "max_deviation_joint": direct.report["max_deviation_joint"],
            "flagged_joints": direct.report["flagged_joints"],
            "root_translation_scale": direct.report["root_translation_scale"],
            "joints": joint_residuals,
            "report_path": _relative(output, retarget_report_path),
            "target_transforms_path": _relative(output, target_transforms_path),
        },
        "artifacts": {
            "source_json": _relative(output, raw_path),
            "source_migrated_json": _relative(output, migrated_path),
            "source_binary": _relative(output, source_binary_path),
            "binary_back_json": _relative(output, back_path),
            "resampled_json": _relative(output, resampled_json_path),
            "resampled_binary": _relative(output, resampled_binary_path),
            "target_json": _relative(output, target_json_path),
            "target_binary": _relative(output, target_binary_path),
            "retarget_report": _relative(output, retarget_report_path),
            "target_transforms": _relative(output, target_transforms_path),
        },
        "all_checks_ok": all(bool(item) for item in checks),
    }
    case_report_path = case_dir / "case_report.json"
    _write_json(case_report_path, case_report)
    return case_report


def _semantic_hash(animation: Animation) -> str:
    return animation_semantic_sha256(animation)


def _capture_anomaly(
    name: str,
    expected_code: ErrorCode,
    callable_: Callable[[], Any],
    *,
    expected_joint: str | None = None,
    expected_frame: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "expected_code": expected_code.value,
        "expected_joint": expected_joint,
        "expected_frame": expected_frame,
        "actual": None,
        "passed": False,
    }
    try:
        callable_()
    except PipelineError as exc:
        actual = exc.to_dict()
        joint_ok = expected_joint is None or exc.joint == expected_joint
        frame_ok = expected_frame is None or exc.frame == expected_frame
        record["actual"] = actual
        record["passed"] = exc.code == expected_code and joint_ok and frame_ok
    except Exception as exc:
        record["actual"] = {
            "code": "E_UNEXPECTED_EXCEPTION",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    return record


def _build_anomalies(
    output: Path,
    inputs: CaseInputs,
    tolerances: Tolerances,
) -> list[dict[str, Any]]:
    anomaly_dir = output / "anomalies"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    source = inputs.source_skeleton
    target = inputs.target_skeleton
    animation = generate_animation(source, inputs.source_seed * 31 + 7, frame_count=7, fps=12.0)
    valid_document = animation_to_document(animation, schema_version=3, quat_order="wxyz")
    resampled, _ = resample_to_fps(animation, TARGET_FPS)
    direct = retarget_motion(resampled, source, target, tolerances=tolerances)
    records: list[dict[str, Any]] = []

    unsupported = dict(valid_document)
    unsupported["schema_version"] = 99
    unsupported_path = anomaly_dir / "unsupported_schema.json"
    _write_json(unsupported_path, unsupported)
    records.append(
        _capture_anomaly(
            "json_unsupported_schema",
            ErrorCode.E_SCHEMA_VERSION,
            lambda: animation_from_document(unsupported),
        )
    )

    missing = json.loads(json.dumps(valid_document))
    del missing["joint_map"]
    missing_path = anomaly_dir / "missing_joint_map.json"
    _write_json(missing_path, missing)
    records.append(
        _capture_anomaly(
            "json_missing_joint_map",
            ErrorCode.E_FIELD_MISSING,
            lambda: animation_from_document(missing),
        )
    )

    zero_quat = json.loads(json.dumps(valid_document))
    zero_joint_index = 4
    zero_joint_name = animation.joint_names[zero_joint_index]
    zero_quat["keyframes"][2]["rotations"][zero_joint_index] = [0.0, 0.0, 0.0, 0.0]
    zero_path = anomaly_dir / "zero_quaternion.json"
    _write_json(zero_path, zero_quat)
    records.append(
        _capture_anomaly(
            "json_zero_quaternion",
            ErrorCode.E_QUAT_INVALID,
            lambda: animation_from_document(zero_quat),
            expected_joint=zero_joint_name,
            expected_frame=2,
        )
    )

    nan_quat = json.loads(json.dumps(valid_document))
    nan_joint_index = 3
    nan_joint_name = animation.joint_names[nan_joint_index]
    nan_quat["keyframes"][1]["rotations"][nan_joint_index] = [float("nan"), 0.0, 0.0, 1.0]
    nan_path = anomaly_dir / "nan_quaternion.json"
    _write_json(nan_path, nan_quat, allow_nan=True)
    records.append(
        _capture_anomaly(
            "json_nan_quaternion",
            ErrorCode.E_QUAT_INVALID,
            lambda: animation_from_document(nan_quat),
            expected_joint=nan_joint_name,
            expected_frame=1,
        )
    )

    nonmonotonic = json.loads(json.dumps(valid_document))
    nonmonotonic["keyframes"][2]["time"] = nonmonotonic["keyframes"][1]["time"]
    nonmonotonic_path = anomaly_dir / "nonmonotonic_time.json"
    _write_json(nonmonotonic_path, nonmonotonic)
    records.append(
        _capture_anomaly(
            "json_nonmonotonic_time",
            ErrorCode.E_TIME_NONMONOTONIC,
            lambda: animation_from_document(nonmonotonic),
            expected_frame=2,
        )
    )

    valid_binary, _ = animation_to_binary(animation, endian="little", index_width=4, quat_order="wxyz")
    bad_endian = bytearray(valid_binary)
    bad_endian[5] = ord("B")
    bad_endian_path = anomaly_dir / "endian_conflict.bin"
    _write_bytes(bad_endian_path, bytes(bad_endian))
    records.append(
        _capture_anomaly(
            "binary_endian_conflict",
            ErrorCode.E_FORMAT_ENDIAN,
            lambda: parse_binary(bytes(bad_endian)),
        )
    )

    index_overflow = bytearray(valid_binary)
    struct.pack_into("<I", index_overflow, 24, 0xFFFFFFFF)
    index_overflow_path = anomaly_dir / "index_overflow.bin"
    _write_bytes(index_overflow_path, bytes(index_overflow))
    records.append(
        _capture_anomaly(
            "binary_index_overflow",
            ErrorCode.E_INDEX_OVERFLOW,
            lambda: parse_binary(bytes(index_overflow)),
        )
    )

    truncated = valid_binary[:-8]
    truncated_path = anomaly_dir / "truncated_rotations.bin"
    _write_bytes(truncated_path, truncated)
    records.append(
        _capture_anomaly(
            "binary_truncated_rotations",
            ErrorCode.E_QUAT_INVALID,
            lambda: parse_binary(truncated),
        )
    )

    unsupported_binary = bytearray(valid_binary)
    unsupported_binary[4] = 99
    unsupported_binary_path = anomaly_dir / "unsupported_format_version.bin"
    _write_bytes(unsupported_binary_path, bytes(unsupported_binary))
    records.append(
        _capture_anomaly(
            "binary_unsupported_version",
            ErrorCode.E_SCHEMA_VERSION,
            lambda: parse_binary(bytes(unsupported_binary)),
        )
    )

    reduced_target = Skeleton(name=target.name, joints=target.joints[:-1])
    missing_target_joint = target.joints[-1].name
    records.append(
        _capture_anomaly(
            "target_skeleton_missing_joint",
            ErrorCode.E_FIELD_MISSING,
            lambda: retarget_motion(resampled, source, reduced_target, tolerances=tolerances),
            expected_joint=missing_target_joint,
        )
    )

    altered_rotations = direct.animation.rotations.copy()
    alteration_joint_index = 2
    alteration_frame = 1
    altered_rotations[alteration_frame, alteration_joint_index] = quat_multiply(
        altered_rotations[alteration_frame, alteration_joint_index],
        quat_from_axis_angle([0.0, 1.0, 0.0], 0.2),
    )
    altered = Animation(
        joint_names=direct.animation.joint_names,
        times=direct.animation.times.copy(),
        rotations=altered_rotations,
        root_translation=direct.animation.root_translation.copy(),
        interpolation=direct.animation.interpolation,
        fps=direct.animation.fps,
    )
    records.append(
        _capture_anomaly(
            "retarget_consistency_divergence",
            ErrorCode.E_RETARGET_DIVERGE,
            lambda: compare_animations(
                direct.animation,
                altered,
                tolerances=tolerances,
                left_name="expected",
                right_name="altered",
                raise_on_divergence=True,
            ),
            expected_joint=direct.animation.joint_names[alteration_joint_index],
            expected_frame=alteration_frame,
        )
    )
    _write_json(anomaly_dir / "anomaly_report.json", records)
    return records


def _collect_file_hashes(output: Path, *, excluded_names: set[str]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name in excluded_names:
            continue
        relative = path.relative_to(output).as_posix()
        entries.append((relative, sha256_bytes(path.read_bytes())))
    return entries


def _manifest_text(entries: list[tuple[str, str]]) -> tuple[str, str]:
    text = "".join(f"{digest}  {relative}\n" for relative, digest in entries)
    return text, sha256_bytes(text.encode("utf-8"))


def _report_text(report: dict[str, Any]) -> str:
    lines = [
        "Bone retarget verification report",
        f"status: {report['status']}",
        f"cases: {report['case_count']}",
        f"anomalies: {report['anomaly_count']}",
        f"error_coverage: {', '.join(report['error_coverage'])}",
        f"manifest_sha256: {report['manifest_sha256']}",
        "",
        "Cases:",
    ]
    for case in report["cases"]:
        lines.append(
            f"- {case['case_id']}: ok={case['all_checks_ok']} "
            f"frames={case['frame_count']} joints={case['joint_count']} "
            f"max_residual={case['max_residual_rad']:.6e} "
            f"max_deviation={case['max_global_deviation_rad']:.6e} "
            f"flagged={','.join(case['flagged_joints']) or 'none'}"
        )
        for joint in case["joint_residuals"]:
            lines.append(
                f"    {joint['joint']}: residual={joint['max_residual_rad']:.6e} "
                f"deviation={joint['max_global_deviation_rad']:.6e} "
                f"flagged={joint['flagged']}"
            )
    lines.append("")
    lines.append("Anomalies:")
    for anomaly in report["anomalies"]:
        actual_code = anomaly["actual"]["code"] if anomaly.get("actual") else "none"
        lines.append(
            f"- {anomaly['name']}: passed={anomaly['passed']} "
            f"expected={anomaly['expected_code']} actual={actual_code} "
            f"joint={anomaly.get('actual', {}).get('joint') if anomaly.get('actual') else None} "
            f"frame={anomaly.get('actual', {}).get('frame') if anomaly.get('actual') else None}"
        )
    return "\n".join(lines) + "\n"


def run_verification(
    output_dir: str | Path,
    *,
    clean: bool = True,
    tolerances: Tolerances | None = None,
) -> dict[str, Any]:
    tol = tolerances or Tolerances()
    output = Path(output_dir)
    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    pair_inputs: list[CaseInputs] = []
    for source_seed, target_seed in PAIR_SEEDS:
        source = generate_skeleton(source_seed, f"source_{source_seed}")
        target = generate_skeleton(target_seed, f"target_{target_seed}")
        pair_dir = output / "skeletons" / f"pair_{source_seed}_{target_seed}"
        _write_json(pair_dir / "source_skeleton.json", source.to_dict())
        _write_json(pair_dir / "target_skeleton.json", target.to_dict())
        pair_inputs.append(
            CaseInputs(
                source_skeleton=source,
                target_skeleton=target,
                source_seed=source_seed,
                target_seed=target_seed,
            )
        )

    case_reports: list[dict[str, Any]] = []
    case_index = 0
    for inputs in pair_inputs:
        for schema_version, endian, index_width, quat_order in CASE_VARIANTS:
            case_report = _run_case(
                output,
                case_index,
                inputs,
                schema_version,
                endian,
                index_width,
                quat_order,
                tol,
            )
            case_reports.append(case_report)
            case_index += 1

    anomalies = _build_anomalies(output, pair_inputs[0], tol)
    error_coverage = sorted({record["actual"]["code"] for record in anomalies if record.get("actual")})
    expected_codes = sorted(code.value for code in ErrorCode)
    all_errors_covered = error_coverage == expected_codes
    anomalies_ok = all(record["passed"] for record in anomalies)
    cases_ok = all(case["all_checks_ok"] for case in case_reports)

    entries = _collect_file_hashes(output, excluded_names=REPORT_EXCLUDED_NAMES)
    manifest_text, manifest_sha256 = _manifest_text(entries)
    report: dict[str, Any] = {
        "status": "PASS" if cases_ok and anomalies_ok and all_errors_covered else "FAIL",
        "pipeline_version": "1.0.0",
        "target_fps": TARGET_FPS,
        "tolerances": tol.to_dict(),
        "case_count": len(case_reports),
        "anomaly_count": len(anomalies),
        "cases_ok": cases_ok,
        "anomalies_ok": anomalies_ok,
        "all_errors_covered": all_errors_covered,
        "error_coverage": error_coverage,
        "expected_error_codes": expected_codes,
        "cases": [
            {
                "case_id": case["case_id"],
                "schema_version": case["schema_version"],
                "endian": case["binary"]["endian"],
                "index_width": case["binary"]["index_width"],
                "quat_order": case["binary"]["quat_order"],
                "all_checks_ok": case["all_checks_ok"],
                "frame_count": case["resample"]["target_frame_count"],
                "joint_count": len(case["retarget"]["joints"]),
                "max_residual_rad": case["retarget"]["max_residual_rad"],
                "max_global_deviation_rad": case["retarget"]["max_global_deviation_rad"],
                "max_residual_frame": case["retarget"]["max_residual_frame"],
                "max_residual_joint": case["retarget"]["max_residual_joint"],
                "flagged_joints": case["retarget"]["flagged_joints"],
                "joint_residuals": case["retarget"]["joints"],
                "retarget_report": case["retarget"]["report_path"],
                "target_transforms": case["retarget"]["target_transforms_path"],
                "hashes": case["hashes"],
            }
            for case in case_reports
        ],
        "anomalies": anomalies,
        "artifact_count": len(entries),
        "manifest_sha256": manifest_sha256,
    }
    report_json_path = output / "report.json"
    report_json_path.write_bytes(document_json_bytes(report, pretty=True))
    report_text = _report_text(report)
    (output / "report.txt").write_text(report_text, encoding="utf-8")
    (output / "manifest.sha256").write_text(manifest_text, encoding="utf-8")

    all_entries = _collect_file_hashes(output, excluded_names={"hashes.sha256"})
    hashes_text = "".join(f"{digest}  {relative}\n" for relative, digest in all_entries)
    (output / "hashes.sha256").write_text(hashes_text, encoding="utf-8")
    return report
