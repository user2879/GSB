from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import DetectionParams, MatchConfig
from .errors import E_SERIALIZE_INVALID, BusinessError
from .formats import ParsedExchange
from .models import RawCandidate
from .util import (
    atomic_write_json,
    atomic_write_text,
    histogram,
    quantize_int,
    quantize_milli,
    stable_hash,
)
from .v1 import format_v1_row, V1_HEADER, parse_v1
from .v2 import V2_FORMAT_VERSION, parse_v2


def _sort_raw(candidate: RawCandidate) -> tuple[Any, ...]:
    return (
        candidate.series_name,
        candidate.frame_index,
        -quantize_milli(candidate.score),
        quantize_int(candidate.x1),
        quantize_int(candidate.y1),
        quantize_int(candidate.x2),
        quantize_int(candidate.y2),
        candidate.class_code,
        candidate.source_ordinal,
    )


def _grouped_frames(candidates: Iterable[RawCandidate]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[RawCandidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.series_name, candidate.frame_index), []).append(candidate)
    frames: list[dict[str, Any]] = []
    for (series_name, frame_index), items in sorted(groups.items()):
        items = sorted(items, key=_sort_raw)
        frames.append(
            {
                "series_name": series_name,
                "frame_index": frame_index,
                "candidates": [
                    {
                        "series_name": item.series_name,
                        "frame_index": item.frame_index,
                        "class_name": item.class_name,
                        "class_code": item.class_code,
                        "x1": float(item.x1),
                        "y1": float(item.y1),
                        "x2": float(item.x2),
                        "y2": float(item.y2),
                        "score": float(item.score),
                        "precision_label": item.precision_label,
                        "extra": dict(item.extra),
                    }
                    for item in items
                ],
            }
        )
    return frames


def serialize_v2_document(
    candidates: Iterable[RawCandidate], *, precision_label: str
) -> dict[str, Any]:
    return {
        "format_version": V2_FORMAT_VERSION,
        "schema": "candidate-exchange-v2-json",
        "precision": precision_label,
        "frames": _grouped_frames(candidates),
    }


def _v1_to_v2_raw(v1: ParsedExchange, detection: DetectionParams) -> list[RawCandidate]:
    candidates: list[RawCandidate] = []
    for items in v1.frames.values():
        for item in items:
            class_name = detection.class_map.get(item.class_code)
            if class_name is None:
                raise BusinessError(
                    "E_CLASS_UNKNOWN",
                    f"class code {item.class_code} cannot be mapped for v1->v2",
                    "conversion",
                    series_name=item.series_name,
                    frame_index=item.frame_index,
                )
            candidates.append(
                RawCandidate(
                    series_name=item.series_name,
                    frame_index=item.frame_index,
                    class_code=item.class_code,
                    class_name=class_name,
                    x1=float(int(item.x1)),
                    y1=float(int(item.y1)),
                    x2=float(int(item.x2)),
                    y2=float(int(item.y2)),
                    score=int(round(item.score * 1000)) / 1000.0,
                    precision_label="v1_exact",
                    source_version="v2",
                    source_ordinal=item.source_ordinal,
                    extra={
                        "source_format": "v1",
                        "source_score_pm": int(round(item.score * 1000)),
                    },
                )
            )
    return sorted(candidates, key=_sort_raw)


def _v2_to_v1_raw(
    v2: ParsedExchange,
) -> tuple[list[RawCandidate], list[dict[str, Any]]]:
    candidates: list[RawCandidate] = []
    sidecar: list[dict[str, Any]] = []
    for items in v2.frames.values():
        for item in items:
            x1 = quantize_int(item.x1)
            y1 = quantize_int(item.y1)
            x2 = quantize_int(item.x2)
            y2 = quantize_int(item.y2)
            score_pm = quantize_milli(item.score)
            sidecar.append(
                {
                    "series_name": item.series_name,
                    "frame_index": item.frame_index,
                    "class_code": item.class_code,
                    "class_name": item.class_name,
                    "precision_label": item.precision_label,
                    "source_box": [item.x1, item.y1, item.x2, item.y2],
                    "restored_box": [x1, y1, x2, y2],
                    "coordinate_residual": [
                        item.x1 - x1,
                        item.y1 - y1,
                        item.x2 - x2,
                        item.y2 - y2,
                    ],
                    "source_score": item.score,
                    "restored_score_pm": score_pm,
                    "score_residual": item.score - score_pm / 1000.0,
                    "extra_metadata": dict(item.extra),
                }
            )
            candidates.append(
                RawCandidate(
                    series_name=item.series_name,
                    frame_index=item.frame_index,
                    class_code=item.class_code,
                    class_name=item.class_name,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=score_pm / 1000.0,
                    precision_label="rounded_from_v2",
                    source_version="v1",
                    source_ordinal=item.source_ordinal,
                    extra={"sidecar_ref": len(sidecar) - 1},
                )
            )
    candidates.sort(key=_sort_raw)
    sidecar.sort(
        key=lambda item: (
            item["series_name"],
            item["frame_index"],
            -quantize_milli(item["source_score"]),
            quantize_int(item["source_box"][0]),
            quantize_int(item["source_box"][1]),
            quantize_int(item["source_box"][2]),
            quantize_int(item["source_box"][3]),
            item["class_code"],
        )
    )
    return candidates, sidecar


def _flat_by_frame(parsed: ParsedExchange) -> dict[tuple[str, int], list[RawCandidate]]:
    result: dict[tuple[str, int], list[RawCandidate]] = {}
    for key, items in parsed.frames.items():
        result[key] = sorted(items, key=_sort_raw)
    return result


def _compare(
    source: ParsedExchange,
    restored: ParsedExchange,
    *,
    direction: str,
    detection: DetectionParams,
) -> dict[str, Any]:
    source_frames = _flat_by_frame(source)
    restored_frames = _flat_by_frame(restored)
    coordinate_errors: list[float] = []
    score_errors: list[float] = []
    class_expected = 0
    class_restored = 0
    class_mismatches: list[dict[str, Any]] = []
    pairing_failures: list[dict[str, Any]] = []
    precision_labels = Counter()
    for key in sorted(set(source_frames) | set(restored_frames)):
        left = source_frames.get(key, [])
        right = restored_frames.get(key, [])
        if len(left) != len(right):
            pairing_failures.append(
                {
                    "series_name": key[0],
                    "frame_index": key[1],
                    "source_count": len(left),
                    "restored_count": len(right),
                }
            )
            continue
        for source_item, restored_item in zip(left, right, strict=True):
            if source_item.class_name is not None:
                expected_name = source_item.class_name
            else:
                expected_name = detection.class_map.get(source_item.class_code)
            class_expected += 1
            restored_name = (
                restored_item.class_name
                if restored_item.class_name is not None
                else detection.class_map.get(restored_item.class_code)
            )
            if restored_name == expected_name:
                class_restored += 1
            else:
                class_mismatches.append(
                    {
                        "series_name": key[0],
                        "frame_index": key[1],
                        "expected": expected_name,
                        "actual": restored_name,
                    }
                )
            restored_box = [
                quantize_int(restored_item.x1),
                quantize_int(restored_item.y1),
                quantize_int(restored_item.x2),
                quantize_int(restored_item.y2),
            ]
            source_box = [source_item.x1, source_item.y1, source_item.x2, source_item.y2]
            coordinate_errors.extend(
                abs(restored_value - source_value)
                for restored_value, source_value in zip(restored_box, source_box, strict=True)
            )
            score_errors.append(abs(restored_item.score - source_item.score))
            precision_labels[restored_item.precision_label] += 1
    coordinate_hist = histogram(coordinate_errors, places=3)
    score_hist = histogram(score_errors, places=6)
    return {
        "direction": direction,
        "candidate_count": class_expected,
        "serialized_then_parsed": not pairing_failures and not restored.failures,
        "pairing_failures": pairing_failures,
        "parse_failures": [error.to_dict() for error in restored.failures],
        "coordinate_abs_error_histogram": coordinate_hist,
        "score_abs_error_histogram": score_hist,
        "max_coordinate_deviation": max(coordinate_errors, default=0.0),
        "max_score_deviation": max(score_errors, default=0.0),
        "mean_coordinate_deviation": (
            sum(coordinate_errors) / len(coordinate_errors) if coordinate_errors else 0.0
        ),
        "mean_score_deviation": sum(score_errors) / len(score_errors) if score_errors else 0.0,
        "class_mapping": {
            "expected": class_expected,
            "restored": class_restored,
            "complete": class_expected == class_restored,
            "mismatches": class_mismatches,
        },
        "precision_label_counts": dict(sorted(precision_labels.items())),
    }


def run_conversion(
    *,
    v1: ParsedExchange,
    v2: ParsedExchange,
    detection: DetectionParams,
    match: MatchConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    v1_to_v2_path = destination / "candidates.v1_to_v2.json"
    v2_to_v1_path = destination / "candidates.v2_to_v1.txt"
    sidecar_path = destination / "candidates.v2_to_v1.sidecar.json"

    v1_to_v2_raw = _v1_to_v2_raw(v1, detection)
    v2_document = serialize_v2_document(v1_to_v2_raw, precision_label="v1_exact")
    atomic_write_json(v1_to_v2_path, v2_document)

    v2_to_v1_raw, sidecar_entries = _v2_to_v1_raw(v2)
    sidecar_payload = {
        "schema": "candidate-exchange-sidecar-v1",
        "direction": "v2_to_v1",
        "fixed_decimal_places": match.fixed_decimal_places,
        "entries": sidecar_entries,
    }
    sidecar_payload["sidecar_metadata_hash"] = stable_hash(sidecar_entries)
    atomic_write_json(sidecar_path, sidecar_payload)
    lines = [V1_HEADER]
    for item in v2_to_v1_raw:
        lines.append(
            format_v1_row(
                item.series_name,
                item.frame_index,
                item.class_code,
                int(item.x1),
                int(item.y1),
                int(item.x2),
                int(item.y2),
                quantize_milli(item.score),
            )
        )
    atomic_write_text(v2_to_v1_path, "\n".join(lines) + "\n")

    parsed_v2_output = parse_v2(v1_to_v2_path)
    parsed_v1_output = parse_v1(v2_to_v1_path)
    checks = {
        "v1_to_v2": _compare(v1, parsed_v2_output, direction="v1_to_v2", detection=detection),
        "v2_to_v1": _compare(v2, parsed_v1_output, direction="v2_to_v1", detection=detection),
    }
    if parsed_v2_output.failures or parsed_v1_output.failures:
        raise BusinessError(
            E_SERIALIZE_INVALID,
            "conversion output could not be parsed back",
            "serialization",
            details={
                "v1_to_v2_parse_failures": [item.to_dict() for item in parsed_v2_output.failures],
                "v2_to_v1_parse_failures": [item.to_dict() for item in parsed_v1_output.failures],
            },
        )
    return {
        "outputs": {
            "v1_to_v2": str(v1_to_v2_path),
            "v2_to_v1": str(v2_to_v1_path),
            "sidecar": str(sidecar_path),
        },
        "sidecar_metadata_hash": sidecar_payload["sidecar_metadata_hash"],
        "sidecar_entry_count": len(sidecar_entries),
        "checks": checks,
        "histogram_note": (
            "Coordinates are compared in integer-pixel space after serialization and "
            "deserialization; scores are compared as floats after per-mille restoration."
        ),
    }
