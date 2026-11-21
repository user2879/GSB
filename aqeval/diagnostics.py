"""Cross-version consistency, deviation and injection diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .config import DIMENSIONS, Config
from .conversion import convert_one
from .errors import Failure, LevelMapError
from .formats import parse_bytes, write_format_bytes
from .levels import LevelMap
from .models import Degradation, UnifiedResult
from .scoring import grade_total_micro
from .util import micro_to_float, sha256_json


def deviation_report(
    canonical: dict[str, UnifiedResult],
    originals: dict[int, dict[str, UnifiedResult]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for version in sorted(originals):
        for file_id in sorted(originals[version]):
            expected = canonical.get(file_id)
            actual = originals[version][file_id]
            if expected is None:
                continue
            dim_deltas = {
                name: actual.dimensions_micro[name] - expected.dimensions_micro[name]
                for name in DIMENSIONS
            }
            rows.append(
                {
                    "file_id": file_id,
                    "version": version,
                    "total_delta_micro": actual.total_micro - expected.total_micro,
                    "total_delta_permille": round(
                        (actual.total_micro - expected.total_micro) / 1000.0, 6
                    ),
                    "dimension_delta_micro": dim_deltas,
                    "expected_total_permille": expected.total_permille,
                    "actual_total_permille": actual.total_permille,
                    "expected_level": expected.level_name or expected.level_code,
                    "actual_level": actual.level_name or actual.level_code,
                    "level_matches": (expected.level_name or expected.level_code)
                    == (actual.level_name or actual.level_code),
                    "precision_tag": actual.precision_tag,
                }
            )
    rows.sort(key=lambda item: (item["file_id"], item["version"]))
    return rows


def _classify_flip(
    left_version: int,
    right_version: int,
    canonical: UnifiedResult,
    left: UnifiedResult | None,
    right: UnifiedResult | None,
) -> tuple[str, dict[str, Any]]:
    if left is None or right is None:
        return "level_mapping_missing", {"missing_side": "version_roundtrip"}
    if left_version == 1:
        v1_permille = left.total_permille
        residual = canonical.total_micro - v1_permille * 1000
        if residual != 0:
            return "permille_truncation", {
                "permille_residual_micro": residual,
                "v1_total_permille": v1_permille,
            }
    if right_version == 1:
        v1_permille = right.total_permille
        residual = canonical.total_micro - v1_permille * 1000
        if residual != 0:
            return "permille_truncation", {
                "permille_residual_micro": residual,
                "v1_total_permille": v1_permille,
            }
    if right_version == 3 and right.source_fixed_bits is not None and right.source_fixed_bits < 6:
        return "fixed_width", {
            "fixed_decimal_bits": right.source_fixed_bits,
            "total_delta_micro": right.total_micro - canonical.total_micro,
        }
    if left_version == 3 and left.source_fixed_bits is not None and left.source_fixed_bits < 6:
        return "fixed_width", {
            "fixed_decimal_bits": left.source_fixed_bits,
            "total_delta_micro": left.total_micro - canonical.total_micro,
        }
    return "threshold_boundary", {}


def consistency_report(
    canonical: dict[str, UnifiedResult],
    level_map: LevelMap,
    fixed_bits: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[Failure] = []
    flips: list[dict[str, Any]] = []
    degradation_entries: list[dict[str, Any]] = []
    pair_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"compared": 0, "level_flips": 0, "max_total_delta_micro": 0}
    )
    degradation_hits = Counter()
    for file_id in sorted(canonical):
        record = canonical[file_id]
        roundtrips: dict[int, UnifiedResult] = {}
        roundtrip_degradations: dict[int, list[Degradation]] = {}
        roundtrip_failure_codes: dict[int, str] = {}
        for version in (1, 2, 3):
            try:
                converted, degradations, _ = convert_one(record, version, level_map, fixed_bits)
                serialized = write_format_bytes(version, [converted], fixed_bits)
                parsed = parse_bytes(version, serialized, f"<memory:{file_id}:v{version}>", level_map)
            except (LevelMapError, ValueError) as exc:
                error_code = "E_LEVEL_UNMAPPED" if isinstance(exc, LevelMapError) else "E_EVAL_FORMAT"
                roundtrip_failure_codes[version] = error_code
                failures.append(
                    Failure(
                        code=error_code,
                        message=f"consistency roundtrip failed: {exc}",
                        file_id=file_id,
                        version=version,
                        stage="conversion",
                        location=f"memory:{file_id}:v{version}",
                        task_key=f"{file_id}|consistency|v{version}",
                    )
                )
                continue
            if parsed.failures:
                failures.extend(parsed.failures)
                continue
            if not parsed.records:
                continue
            roundtrip = parsed.records[0]
            roundtrips[version] = roundtrip
            roundtrip_degradations[version] = degradations
            for degradation in degradations:
                degradation_hits[degradation.code] += 1
                degradation_entries.append(degradation.to_dict())
            dimension_deltas = {
                name: roundtrip.dimensions_micro[name] - record.dimensions_micro[name]
                for name in DIMENSIONS
            }
            total_delta = roundtrip.total_micro - record.total_micro
            expected_level = record.level_name or record.level_code
            actual_level = roundtrip.level_name or roundtrip.level_code
            regraded_code, regraded_name, _ = grade_total_micro(
                roundtrip.total_micro, level_map
            )
            regraded_level = regraded_name or regraded_code
            rows.append(
                {
                    "file_id": file_id,
                    "version": version,
                    "total_delta_micro": total_delta,
                    "dimension_delta_micro": dimension_deltas,
                    "expected_level": expected_level,
                    "roundtrip_level": actual_level,
                    "regraded_level": regraded_level,
                    "regraded_matches_embedded": regraded_level == actual_level,
                    "level_restored": expected_level == actual_level,
                    "precision_tag": roundtrip.precision_tag,
                    "degradation_codes": sorted(item.code for item in degradations),
                    "bypass_hit_count": sum(
                        1
                        for value in roundtrip.bypass.values()
                        if value not in (None, 0, "", [], {})
                    ),
                }
            )
        for left_version, right_version in ((1, 2), (2, 3)):
            pair_key = f"v{left_version}-v{right_version}"
            pair_summary[pair_key]["compared"] += 1
            left = roundtrips.get(left_version)
            right = roundtrips.get(right_version)
            if left is None or right is None:
                missing_versions = [
                    version
                    for version, value in ((left_version, left), (right_version, right))
                    if value is None
                ]
                failure_codes = [
                    roundtrip_failure_codes.get(version, "roundtrip_missing")
                    for version in missing_versions
                ]
                cause = (
                    "level_mapping_missing"
                    if "E_LEVEL_UNMAPPED" in failure_codes
                    else "roundtrip_missing"
                )
                pair_summary[pair_key]["level_flips"] += 1
                flips.append(
                    {
                        "file_id": file_id,
                        "left_version": left_version,
                        "right_version": right_version,
                        "left_level": (left.level_name or left.level_code) if left else None,
                        "right_level": (right.level_name or right.level_code) if right else None,
                        "left_embedded_level": (left.level_name or left.level_code) if left else None,
                        "right_embedded_level": (right.level_name or right.level_code) if right else None,
                        "total_delta_micro": (
                            (right.total_micro if right else record.total_micro)
                            - (left.total_micro if left else record.total_micro)
                        ),
                        "cause": cause,
                        "evidence": {
                            "missing_versions": missing_versions,
                            "failure_codes": failure_codes,
                            "canonical_level": record.level_name or record.level_code,
                            "canonical_total_micro": record.total_micro,
                        },
                    }
                )
                continue
            for side in (left, right):
                canonical_grade = grade_total_micro(record.total_micro, level_map)
                side_grade = grade_total_micro(side.total_micro, level_map)
                if (side_grade[0], side_grade[1]) != canonical_grade[:2]:
                    pair_summary[pair_key]["level_flips"] += 1
                    break
            left_grade = grade_total_micro(left.total_micro, level_map)
            right_grade = grade_total_micro(right.total_micro, level_map)
            left_level = left_grade[1] or left_grade[0]
            right_level = right_grade[1] or right_grade[0]
            delta = abs(right.total_micro - left.total_micro)
            pair_summary[pair_key]["max_total_delta_micro"] = max(
                pair_summary[pair_key]["max_total_delta_micro"], delta
            )
            if left_level != right_level:
                cause, evidence = _classify_flip(
                    left_version, right_version, record, left, right
                )
                flips.append(
                    {
                        "file_id": file_id,
                        "left_version": left_version,
                        "right_version": right_version,
                        "left_level": left_level,
                        "right_level": right_level,
                        "left_embedded_level": left.level_name or left.level_code,
                        "right_embedded_level": right.level_name or right.level_code,
                        "total_delta_micro": right.total_micro - left.total_micro,
                        "cause": cause,
                        "evidence": evidence,
                    }
                )
    rows.sort(key=lambda item: (item["file_id"], item["version"]))
    flips.sort(key=lambda item: (item["file_id"], item["left_version"], item["right_version"]))
    failures.sort(key=Failure.sort_key)
    return {
        "schema": "aqeval.consistency.v1",
        "roundtrips": rows,
        "pair_summary": {key: pair_summary[key] for key in sorted(pair_summary)},
        "level_flips": flips,
        "degradation_code_hits": dict(sorted(degradation_hits.items())),
        "degradations": sorted(
            degradation_entries,
            key=lambda item: (
                item["file_id"],
                item["source_version"],
                item["target_version"],
                item["code"],
            ),
        ),
        "failures": [failure.to_dict() for failure in failures],
        "canonical_digest": sha256_json(
            {
                file_id: {
                    "total_micro": record.total_micro,
                    "dimensions_micro": record.dimensions_micro,
                    "level_code": record.level_code,
                    "level_name": record.level_name,
                }
                for file_id, record in sorted(canonical.items())
            }
        ),
    }


def degradation_statistics(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = (
            int(entry.get("source_version", 0)),
            int(entry.get("target_version", 0)),
            str(entry.get("code", "")),
        )
        groups[key].append(entry)
    stats: list[dict[str, Any]] = []
    for (source, target, code), values in sorted(groups.items()):
        stats.append(
            {
                "source_version": source,
                "target_version": target,
                "code": code,
                "count": len(values),
                "absolute_residual_micro": sum(
                    abs(int(value.get("residual_micro", 0))) for value in values
                ),
                "files": sorted({str(value.get("file_id", "")) for value in values}),
            }
        )
    return {
        "total_entries": sum(item["count"] for item in stats),
        "groups": stats,
    }


INJECTION_CATALOG = [
    {"id": "INJ-001", "business_code": "DEG_V1_TRUNCATION", "description": "v1 permille truncation"},
    {"id": "INJ-002", "business_code": "DEG_V3_FIXED_WIDTH", "description": "v3 fixed-width loss"},
    {"id": "INJ-003", "business_code": "E_LEVEL_UNMAPPED", "description": "level code/name not mapped"},
    {"id": "INJ-004", "business_code": "BOUNDARY_EXACT_THRESHOLD", "description": "exact scoring threshold"},
    {"id": "INJ-005", "business_code": "CONCURRENCY_STABLE_ORDER", "description": "serial/concurrent equality"},
    {"id": "INJ-006", "business_code": "E_SNAPSHOT_STALE", "description": "stale snapshot rebuild"},
    {"id": "INJ-007", "business_code": "E_EVAL_FORMAT", "description": "malformed exchange record"},
    {"id": "INJ-008", "business_code": "E_AUDIO_FORMAT", "description": "invalid/missing audio"},
]


def injection_hits(
    *,
    failures: list[dict[str, Any]],
    degradations: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    concurrency_match: bool,
    snapshot_stale: bool,
) -> list[dict[str, Any]]:
    failure_codes = Counter(str(item.get("code")) for item in failures)
    degradation_codes = Counter(str(item.get("code")) for item in degradations)
    boundary_files = sorted({str(item.get("file_id", "")) for item in boundaries})
    hits = {
        "DEG_V1_TRUNCATION": degradation_codes["DEG_V1_TRUNCATION"],
        "DEG_V3_FIXED_WIDTH": degradation_codes["DEG_V3_FIXED_WIDTH"],
        "E_LEVEL_UNMAPPED": failure_codes["E_LEVEL_UNMAPPED"],
        "BOUNDARY_EXACT_THRESHOLD": len(boundaries),
        "CONCURRENCY_STABLE_ORDER": 1 if concurrency_match else 0,
        "E_SNAPSHOT_STALE": 1 if snapshot_stale else 0,
        "E_EVAL_FORMAT": failure_codes["E_EVAL_FORMAT"],
        "E_AUDIO_FORMAT": failure_codes["E_AUDIO_FORMAT"] + failure_codes["E_AUDIO_MISSING"],
    }
    result: list[dict[str, Any]] = []
    for item in INJECTION_CATALOG:
        count = hits[item["business_code"]]
        result.append(
            {
                **item,
                "hit": count > 0,
                "hit_count": count,
                "files": boundary_files
                if item["business_code"] == "BOUNDARY_EXACT_THRESHOLD"
                else [],
            }
        )
    return result
