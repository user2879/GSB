"""Threshold-configurable waste-frame detection with cached evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import (
    DEFECT_JITTER,
    DEFECT_JUMP,
    DEFECT_MISSING,
    DEFECT_ROOT_SLIP,
    DEFAULT_THRESHOLDS,
    ROOT_NAME,
    SCHEMA_DETECTION,
    SKELETON,
    THRESHOLD_RANGES,
)
from .errors import BusinessError, ErrorCode, require
from .io import atomic_write_json, clean_float, sha256_json
from .math3d import quat_angle_deg, relative_rotvec, slerp
from .model import DefectEntry, DetectionResult, Evidence, Motion, Thresholds


def _sha256_quantized_arrays(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        quantized = np.rint(np.asarray(array, dtype=np.float64) * 1_000_000_000.0).astype("<i8")
        digest.update(quantized.tobytes(order="C"))
    return digest.hexdigest()


def validate_thresholds(thresholds: Thresholds) -> None:
    for key, value in thresholds.to_dict().items():
        low, high = THRESHOLD_RANGES[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < low or value > high:
            raise BusinessError(
                ErrorCode.E_THRESHOLD_RANGE,
                "detector threshold is outside its declared range",
                {"key": key, "value": value, "min": low, "max": high},
            )


def thresholds_from_mapping(mapping: dict[str, Any] | None = None) -> Thresholds:
    values = dict(DEFAULT_THRESHOLDS)
    if mapping:
        unknown = set(mapping) - set(values)
        if unknown:
            raise BusinessError(
                ErrorCode.E_THRESHOLD_RANGE,
                "unknown detector threshold key",
                {"keys": sorted(unknown)},
            )
        values.update({key: float(value) for key, value in mapping.items()})
    thresholds = Thresholds(**values)
    validate_thresholds(thresholds)
    return thresholds


def compute_evidence(record: Motion) -> Evidence:
    """Compute measurements once; threshold-only previews reuse this object."""

    n = record.frame_count
    j = record.joint_count
    quats = record.quats
    norms = np.linalg.norm(quats, axis=2)
    invalid = norms <= 1e-12
    valid = ~invalid
    residual = np.zeros((n, j), dtype=np.float64)
    second_diff = np.zeros((n, j), dtype=np.float64)
    step_prev = np.zeros((n, j), dtype=np.float64)
    step_next = np.zeros((n, j), dtype=np.float64)

    def pair_angle(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        dot = np.abs(np.einsum("...i,...i->...", left, right))
        return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

    def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = np.moveaxis(left, -1, 0)
        rw, rx, ry, rz = np.moveaxis(right, -1, 0)
        return np.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            axis=-1,
        )

    def conjugate(value: np.ndarray) -> np.ndarray:
        return value * np.array([1.0, -1.0, -1.0, -1.0])

    def log_relative(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        relative = multiply(conjugate(previous), current)
        relative /= np.maximum(1e-15, np.linalg.norm(relative, axis=-1, keepdims=True))
        relative *= np.where(relative[..., :1] < 0.0, -1.0, 1.0)
        vector = relative[..., 1:]
        length = np.linalg.norm(vector, axis=-1, keepdims=True)
        angle = 2.0 * np.arctan2(length, np.clip(relative[..., :1], -1.0, 1.0))
        return np.where(length > 1e-15, vector * (angle / np.maximum(length, 1e-15)), 0.0)

    # Adjacent angular steps.
    step_prev[1:, :] = np.where(
        valid[:-1, :] & valid[1:, :], pair_angle(quats[:-1, :], quats[1:, :]), 0.0
    )
    step_next[:-1, :] = np.where(
        valid[:-1, :] & valid[1:, :], pair_angle(quats[:-1, :], quats[1:, :]), 0.0
    )

    # The shortest SLERP at alpha=0.5 is exactly normalize(q0 + aligned q1).
    triple_valid = valid[:-2, :] & valid[1:-1, :] & valid[2:, :]
    left = quats[:-2, :]
    right = quats[2:, :]
    aligned_right = np.where(
        np.einsum("...i,...i->...", left, right)[..., None] < 0.0, -right, right
    )
    midpoint = left + aligned_right
    midpoint /= np.maximum(1e-15, np.linalg.norm(midpoint, axis=-1, keepdims=True))
    midpoint_residual = pair_angle(midpoint, quats[1:-1, :])
    residual[1:-1, :] = np.where(triple_valid, midpoint_residual, 0.0)
    first_delta = log_relative(left, quats[1:-1, :])
    second_delta = log_relative(quats[1:-1, :], right)
    second_diff[1:-1, :] = np.where(
        triple_valid,
        np.degrees(np.linalg.norm(second_delta - first_delta, axis=-1)),
        0.0,
    )

    roots = record.roots
    root_residual = np.zeros(n, dtype=np.float64)
    root_step_prev = np.zeros(n, dtype=np.float64)
    for frame in range(1, n):
        root_step_prev[frame] = float(np.linalg.norm(roots[frame] - roots[frame - 1]))
    for frame in range(1, n - 1):
        midpoint = (roots[frame - 1] + roots[frame + 1]) * 0.5
        root_residual[frame] = float(np.linalg.norm(roots[frame] - midpoint))

    evidence_hash = _sha256_quantized_arrays(
        (
            invalid.astype(np.float64),
            residual,
            second_diff,
            step_prev,
            step_next,
            root_residual,
            root_step_prev,
        )
    )
    return Evidence(
        invalid=invalid,
        midpoint_residual_deg=residual,
        angular_second_diff_deg=second_diff,
        step_prev_deg=step_prev,
        step_next_deg=step_next,
        root_midpoint_residual_m=root_residual,
        root_step_prev_m=root_step_prev,
        evidence_hash=evidence_hash,
    )


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(flags)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            runs.append((start, previous))
            start = current
        previous = current
    runs.append((start, previous))
    return runs


def classify_evidence(evidence: Evidence, thresholds: Thresholds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validate_thresholds(thresholds)
    joint_bad = evidence.invalid.copy()
    joint_type = np.full(evidence.invalid.shape, "", dtype=object)
    joint_type[evidence.invalid] = DEFECT_MISSING
    n, j = evidence.invalid.shape

    candidate = (evidence.midpoint_residual_deg >= thresholds.joint_jitter_deg) | (
        evidence.angular_second_diff_deg >= thresholds.joint_jitter_deg
    )
    candidate &= ~evidence.invalid

    for joint in range(j):
        for start, end in _runs(candidate[:, joint]):
            segment_residual = evidence.midpoint_residual_deg[start : end + 1, joint]
            peak_offset = int(np.argmax(segment_residual))
            peak_residual = float(segment_residual[peak_offset])
            if peak_residual >= thresholds.joint_jump_deg:
                peak_frame = start + peak_offset
                joint_bad[peak_frame, joint] = True
                joint_type[peak_frame, joint] = DEFECT_JUMP
                # A single-sample jump produces transition shoulders. They are
                # resolved to the excursion peak and are not independent defects.
                continue
            # The shoulders of a finite jitter block decay to about half the
            # interior residual. Keep the interior only; otherwise both clean
            # boundary frames would be reported as false positives.
            jitter_floor = max(thresholds.joint_jitter_deg, 0.65 * peak_residual)
            for frame in range(start, end + 1):
                residual_value = evidence.midpoint_residual_deg[frame, joint]
                accel_value = evidence.angular_second_diff_deg[frame, joint]
                if residual_value >= jitter_floor and (
                    residual_value >= thresholds.joint_jitter_deg
                    or accel_value >= thresholds.joint_jitter_deg
                ):
                    joint_bad[frame, joint] = True
                    joint_type[frame, joint] = DEFECT_JITTER

    root_candidate = evidence.root_midpoint_residual_m >= thresholds.root_slip_m
    root_bad = np.zeros(n, dtype=bool)
    root_type = np.full(n, "", dtype=object)
    for start, end in _runs(root_candidate):
        residual = evidence.root_midpoint_residual_m[start : end + 1]
        peak_offset = int(np.argmax(residual))
        # Single slips have a symmetric residual shoulder. Long root anomalies
        # are still attributed to their strongest frame, keeping repair anchors
        # and the manifest's instantaneous-slip semantics unambiguous.
        peak_frame = start + peak_offset
        root_bad[peak_frame] = True
        root_type[peak_frame] = DEFECT_ROOT_SLIP

    return joint_bad, joint_type, root_bad, root_type


def _entries_from_masks(
    joint_bad: np.ndarray, joint_type: np.ndarray, root_bad: np.ndarray, root_type: np.ndarray
) -> tuple[DefectEntry, ...]:
    entries: list[DefectEntry] = []
    segment_id = 1
    for frame in range(joint_bad.shape[0]):
        active = [(joint, str(joint_type[frame, joint])) for joint in range(joint_bad.shape[1]) if joint_bad[frame, joint]]
        active.sort(key=lambda item: SKELETON[item[0]])
        for joint, defect_type in active:
            entries.append(DefectEntry(frame, SKELETON[joint], defect_type, segment_id))
        if root_bad[frame]:
            entries.append(DefectEntry(frame, ROOT_NAME, str(root_type[frame]), segment_id))
        if active or root_bad[frame]:
            segment_id += 1
    entries.sort(key=lambda entry: (entry.frame, entry.joint, entry.defect_type))
    return tuple(entries)


def compare_with_truth(entries: tuple[DefectEntry, ...], truth: tuple[DefectEntry, ...]) -> dict[str, Any]:
    detected = {entry.key() for entry in entries}
    expected = {entry.key() for entry in truth}
    matched = detected & expected
    missed = sorted(expected - detected)
    false_positive = sorted(detected - expected)
    by_type: dict[str, dict[str, int]] = {}
    all_types = sorted({item[2] for item in expected | detected})
    for defect_type in all_types:
        expected_keys = {item for item in expected if item[2] == defect_type}
        detected_keys = {item for item in detected if item[2] == defect_type}
        by_type[defect_type] = {
            "truth": len(expected_keys),
            "detected": len(detected_keys),
            "matched": len(expected_keys & detected_keys),
            "missed": len(expected_keys - detected_keys),
            "false_positive": len(detected_keys - expected_keys),
        }
    return {
        "truth_count": len(expected),
        "detected_count": len(detected),
        "matched_count": len(matched),
        "missed_count": len(missed),
        "false_positive_count": len(false_positive),
        "exact_match": not missed and not false_positive,
        "missed": [{"frame": f, "joint": j, "type": t} for f, j, t in missed],
        "false_positive": [{"frame": f, "joint": j, "type": t} for f, j, t in false_positive],
        "by_type": by_type,
        "comparison_key": "frame+joint+defect_type",
    }


def detect(
    record: Motion,
    truth_entries: tuple[DefectEntry, ...],
    thresholds: Thresholds,
    evidence: Evidence | None = None,
) -> DetectionResult:
    evidence = evidence or compute_evidence(record)
    joint_bad, joint_type, root_bad, root_type = classify_evidence(evidence, thresholds)
    entries = _entries_from_masks(joint_bad, joint_type, root_bad, root_type)
    comparison = compare_with_truth(entries, truth_entries)
    detection_hash = sha256_json(
        {
            "thresholds": thresholds.to_dict(),
            "entries": [
                {"frame": entry.frame, "joint": entry.joint, "type": entry.defect_type}
                for entry in entries
            ],
            "evidence_hash": evidence.evidence_hash,
        }
    )
    return DetectionResult(
        thresholds=thresholds,
        entries=entries,
        joint_bad=joint_bad,
        joint_type=joint_type,
        root_bad=root_bad,
        root_type=root_type,
        comparison=comparison,
        detection_hash=detection_hash,
    )


def detection_payload(detection: DetectionResult, evidence: Evidence, record: Motion) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame in range(record.frame_count):
        joint_records = {}
        for joint_index, joint in enumerate(SKELETON):
            joint_records[joint] = {
                "defect": bool(detection.joint_bad[frame, joint_index]),
                "type": str(detection.joint_type[frame, joint_index]) or None,
                "invalid_placeholder": bool(evidence.invalid[frame, joint_index]),
                "step_prev_deg": clean_float(evidence.step_prev_deg[frame, joint_index]),
                "step_next_deg": clean_float(evidence.step_next_deg[frame, joint_index]),
                "midpoint_residual_deg": clean_float(evidence.midpoint_residual_deg[frame, joint_index]),
                "angular_second_diff_deg": clean_float(evidence.angular_second_diff_deg[frame, joint_index]),
            }
        frames.append(
            {
                "frame": frame,
                "joints": joint_records,
                "root": {
                    "defect": bool(detection.root_bad[frame]),
                    "type": str(detection.root_type[frame]) or None,
                    "step_prev_m": clean_float(evidence.root_step_prev_m[frame]),
                    "midpoint_residual_m": clean_float(evidence.root_midpoint_residual_m[frame]),
                },
            }
        )
    body: dict[str, Any] = {
        "schema": SCHEMA_DETECTION,
        "record_sha256": record.content_sha256,
        "thresholds": detection.thresholds.to_dict(),
        "evidence_sha256": evidence.evidence_hash,
        "entries": detection.entries_as_dicts(),
        "comparison": detection.comparison,
        "frames": frames,
        "detection_sha256": detection.detection_hash,
    }
    return body


def write_detection(
    path: Path,
    detection: DetectionResult,
    evidence: Evidence,
    record: Motion,
) -> str:
    return atomic_write_json(path, detection_payload(detection, evidence, record))
