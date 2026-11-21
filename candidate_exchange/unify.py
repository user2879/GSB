from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .config import DetectionParams, MatchConfig, validate_runtime_weight
from .errors import (
    E_CLASS_UNKNOWN,
    E_COORD_INVALID,
    E_IMAGE_INVALID,
    E_NON_FINITE,
    E_SCORE_INVALID,
    E_VERSION_MISMATCH,
    BusinessError,
)
from .formats import ParsedExchange
from .models import CanonicalCandidate, FailureRecord, RawCandidate, UnifiedFrame
from .util import finite_number, quantize_scaled, sha256_bytes, stable_hash


FRAME_NUMBER_RE = re.compile(r"(\d+)\.png$", re.IGNORECASE)


def discover_series_images(images_root: str | Path) -> dict[str, dict[int, Path]]:
    root = Path(images_root)
    result: dict[str, dict[int, Path]] = {}
    if not root.exists():
        return result
    for series_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        frames: dict[int, Path] = {}
        for image_path in sorted(series_dir.glob("*.png")):
            match = FRAME_NUMBER_RE.search(image_path.name)
            if not match:
                continue
            frame_index = int(match.group(1))
            if frame_index in frames:
                raise BusinessError(
                    E_IMAGE_INVALID,
                    f"duplicate frame index {frame_index} in {series_dir}",
                    "detection",
                    series_name=series_dir.name,
                )
            frames[frame_index] = image_path
        if frames:
            result[series_dir.name] = frames
    return result


def _business_failure(error: BusinessError) -> FailureRecord:
    details = dict(error.details)
    exception_type = str(details.pop("exception_type", error.__class__.__name__))
    return FailureRecord(
        series_name=error.series_name or "<unknown>",
        frame_index=error.frame_index,
        stage=error.stage,
        error_code=error.code,
        exception_type=exception_type,
        message=error.message,
        details=details,
    )


def _canonical_from_raw(
    raw: RawCandidate,
    *,
    active_version: str,
    class_map: dict[int, str],
    decimal_places: int,
    image_size: tuple[int, int],
) -> CanonicalCandidate:
    if not all(
        finite_number(value)
        for value in (raw.x1, raw.y1, raw.x2, raw.y2, raw.score)
    ):
        raise BusinessError(
            E_NON_FINITE,
            "candidate contains a non-finite value",
            f"parsing_{raw.source_version}",
            series_name=raw.series_name,
            frame_index=raw.frame_index,
        )
    class_name = class_map.get(raw.class_code)
    if class_name is None:
        raise BusinessError(
            E_CLASS_UNKNOWN,
            f"class code {raw.class_code} is not present in class_map",
            f"parsing_{raw.source_version}",
            series_name=raw.series_name,
            frame_index=raw.frame_index,
            details={"class_code": raw.class_code},
        )
    if raw.class_name is not None and raw.class_name != class_name:
        raise BusinessError(
            E_VERSION_MISMATCH,
            f"class code {raw.class_code} maps to {class_name!r}, not {raw.class_name!r}",
            f"parsing_{raw.source_version}",
            series_name=raw.series_name,
            frame_index=raw.frame_index,
            details={"class_code": raw.class_code, "mapped_name": class_name, "seen_name": raw.class_name},
        )
    if not 0.0 <= raw.score <= 1.0:
        raise BusinessError(
            E_SCORE_INVALID,
            f"score {raw.score!r} is outside [0, 1]",
            f"parsing_{raw.source_version}",
            series_name=raw.series_name,
            frame_index=raw.frame_index,
        )
    x1_q = quantize_scaled(raw.x1, decimal_places)
    y1_q = quantize_scaled(raw.y1, decimal_places)
    x2_q = quantize_scaled(raw.x2, decimal_places)
    y2_q = quantize_scaled(raw.y2, decimal_places)
    width, height = image_size
    scale = 10**decimal_places
    if not (0 <= x1_q < x2_q <= width * scale and 0 <= y1_q < y2_q <= height * scale):
        raise BusinessError(
            E_COORD_INVALID,
            "candidate coordinates are non-monotonic or outside the image",
            f"parsing_{raw.source_version}",
            series_name=raw.series_name,
            frame_index=raw.frame_index,
            details={
                "box": [raw.x1, raw.y1, raw.x2, raw.y2],
                "image_size": [width, height],
            },
        )
    return CanonicalCandidate(
        series_name=raw.series_name,
        frame_index=raw.frame_index,
        class_code=raw.class_code,
        class_name=class_name,
        x1_q=x1_q,
        y1_q=y1_q,
        x2_q=x2_q,
        y2_q=y2_q,
        score_micro=quantize_scaled(raw.score, 6),
        precision_label=raw.precision_label,
        source_version=raw.source_version,
        source_ordinal=raw.source_ordinal,
        missing_fields=raw.missing_fields,
        extra=raw.extra,
    )


def _candidate_compare_key(
    candidate: CanonicalCandidate, decimal_places: int
) -> tuple[Any, ...]:
    return (
        -candidate.score_pm,
        candidate.integer_box(decimal_places),
        candidate.class_code,
        candidate.source_ordinal,
    )


def _reconcile_pairs(
    v1_candidates: list[CanonicalCandidate],
    v2_candidates: list[CanonicalCandidate],
    *,
    series_name: str,
    frame_index: int,
    decimal_places: int,
) -> None:
    if len(v1_candidates) != len(v2_candidates):
        raise BusinessError(
            E_VERSION_MISMATCH,
            f"candidate count differs: v1={len(v1_candidates)} v2={len(v2_candidates)}",
            "unify",
            series_name=series_name,
            frame_index=frame_index,
        )
    v1_sorted = sorted(
        v1_candidates, key=lambda item: _candidate_compare_key(item, decimal_places)
    )
    v2_sorted = sorted(
        v2_candidates, key=lambda item: _candidate_compare_key(item, decimal_places)
    )
    for ordinal, (left, right) in enumerate(zip(v1_sorted, v2_sorted, strict=True)):
        if left.class_code != right.class_code or left.class_name != right.class_name:
            raise BusinessError(
                E_VERSION_MISMATCH,
                f"candidate {ordinal} class differs between versions",
                "unify",
                series_name=series_name,
                frame_index=frame_index,
                details={
                    "v1": [left.class_code, left.class_name],
                    "v2": [right.class_code, right.class_name],
                },
            )
        if left.integer_box(decimal_places) != right.integer_box(decimal_places):
            raise BusinessError(
                E_VERSION_MISMATCH,
                f"candidate {ordinal} coordinates differ between versions",
                "unify",
                series_name=series_name,
                frame_index=frame_index,
                details={
                    "v1_integer_box": list(left.integer_box(decimal_places)),
                    "v2_rounded_box": list(right.integer_box(decimal_places)),
                },
            )
        if left.score_pm != right.score_pm:
            raise BusinessError(
                E_VERSION_MISMATCH,
                f"candidate {ordinal} score differs between versions",
                "unify",
                series_name=series_name,
                frame_index=frame_index,
                details={"v1_score_pm": left.score_pm, "v2_score_pm": right.score_pm},
            )


def unify_series(
    *,
    series_name: str,
    images: dict[int, Path],
    v1: ParsedExchange,
    v2: ParsedExchange,
    detection: DetectionParams,
    match: MatchConfig,
    active_version: str,
    sidecar_frame_fingerprints: dict[tuple[str, int], str] | None = None,
) -> tuple[list[UnifiedFrame], list[FailureRecord]]:
    failures: list[FailureRecord] = []
    sidecar_frame_fingerprints = sidecar_frame_fingerprints or {}
    parse_failures_by_frame: dict[tuple[str, int], list[BusinessError]] = {}
    for parsed in (v1, v2):
        for error in parsed.failures:
            if error.series_name == series_name and error.frame_index is not None:
                parse_failures_by_frame.setdefault(
                    (series_name, error.frame_index), []
                ).append(error)
            elif error.series_name in {None, "<file>", "<unknown>"}:
                failures.append(_business_failure(error))
    candidate_frame_indices = {
        frame_index for series, frame_index in list(v1.frames) + list(v2.frames) if series == series_name
    }
    all_frame_indices = sorted(set(images) | candidate_frame_indices | {
        frame_index
        for series, frame_index in parse_failures_by_frame
        if series == series_name
    })
    frames: list[UnifiedFrame] = []
    for frame_index in all_frame_indices:
        image_path = images.get(frame_index)
        if image_path is None:
            failures.append(
                FailureRecord(
                    series_name=series_name,
                    frame_index=frame_index,
                    stage="detection",
                    error_code=E_IMAGE_INVALID,
                    exception_type="FileNotFoundError",
                    message="candidate frame has no corresponding PNG image",
                )
            )
            continue
        try:
            image_bytes = image_path.read_bytes()
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image_size = image.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            failures.append(
                FailureRecord(
                    series_name=series_name,
                    frame_index=frame_index,
                    stage="detection",
                    error_code=E_IMAGE_INVALID,
                    exception_type=exc.__class__.__name__,
                    message=f"PNG validation failed: {exc}",
                )
            )
            continue
        try:
            validate_runtime_weight(detection.raw, series_name, frame_index)
        except BusinessError as exc:
            failures.append(_business_failure(exc))
            continue
        frame_parse_failures = parse_failures_by_frame.get((series_name, frame_index), [])
        if frame_parse_failures:
            failures.extend(_business_failure(error) for error in frame_parse_failures)
            continue
        raw_v1 = v1.frames.get((series_name, frame_index), [])
        raw_v2 = v2.frames.get((series_name, frame_index), [])
        try:
            canonical_v1 = [
                _canonical_from_raw(
                    item,
                    active_version="v1",
                    class_map=detection.class_map,
                    decimal_places=match.fixed_decimal_places,
                    image_size=image_size,
                )
                for item in raw_v1
            ]
            canonical_v2 = [
                _canonical_from_raw(
                    item,
                    active_version="v2",
                    class_map=detection.class_map,
                    decimal_places=match.fixed_decimal_places,
                    image_size=image_size,
                )
                for item in raw_v2
            ]
            _reconcile_pairs(
                canonical_v1,
                canonical_v2,
                series_name=series_name,
                frame_index=frame_index,
                decimal_places=match.fixed_decimal_places,
            )
        except BusinessError as exc:
            failures.append(_business_failure(exc))
            continue
        selected = canonical_v2 if active_version == "v2" else canonical_v1
        selected = sorted(
            selected,
            key=lambda item: _candidate_compare_key(item, match.fixed_decimal_places),
        )
        candidate_payload = [
            {
                "class_code": item.class_code,
                "class_name": item.class_name,
                "box_q": [item.x1_q, item.y1_q, item.x2_q, item.y2_q],
                "score_micro": item.score_micro,
                "precision_label": item.precision_label,
                "source_ordinal": item.source_ordinal,
                "missing_fields": list(item.missing_fields),
            }
            for item in selected
        ]
        content_fingerprint = stable_hash(
            {
                "image_sha256": sha256_bytes(image_bytes),
                "candidates": candidate_payload,
                "active_version": active_version,
                "sidecar": sidecar_frame_fingerprints.get((series_name, frame_index), "none"),
            }
        )
        frames.append(
            UnifiedFrame(
                series_name=series_name,
                frame_index=frame_index,
                image_path=str(image_path),
                image_sha256=sha256_bytes(image_bytes),
                image_size=image_size,
                candidates=selected,
                content_fingerprint=content_fingerprint,
            )
        )
    return frames, failures


def frame_cache_key(
    frame: UnifiedFrame,
    *,
    detection_fingerprint: str,
    format_fingerprint: str,
    decimal_places: int,
    sidecar_fingerprint: str,
) -> str:
    candidate_payload = [
        {
            "class_code": item.class_code,
            "class_name": item.class_name,
            "box_q": [item.x1_q, item.y1_q, item.x2_q, item.y2_q],
            "score_micro": item.score_micro,
            "precision_label": item.precision_label,
            "grading": item.missing_fields,
        }
        for item in frame.candidates
    ]
    return stable_hash(
        {
            "schema": "frame-cache-key-v1",
            "frame_content_fingerprint": frame.content_fingerprint,
            "detection_parameter_fingerprint": detection_fingerprint,
            "format_version_fingerprint": format_fingerprint,
            "sidecar_metadata_fingerprint": sidecar_fingerprint,
            "fixed_decimal_places": decimal_places,
            "candidate_payload": candidate_payload,
        }
    )
