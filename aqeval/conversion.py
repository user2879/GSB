"""Loss-aware bidirectional conversion across exchange versions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compat import fixed_factor, micro_to_fixed
from .config import DIMENSIONS
from .errors import E_LEVEL_UNMAPPED, Failure, LevelMapError
from .levels import LevelMap
from .models import Degradation, UnifiedResult, sorted_degradations
from .util import atomic_write_json, sha256_json


DEG_V1_TRUNCATION = "DEG_V1_TRUNCATION"
DEG_V3_FIXED_WIDTH = "DEG_V3_FIXED_WIDTH"
DEG_FRAME_SUMMARY_UNAVAILABLE = "DEG_FRAME_SUMMARY_UNAVAILABLE"
DEG_FRAME_SUMMARY_DROPPED = "DEG_FRAME_SUMMARY_DROPPED"
DEG_LEVEL_METADATA_LOST = "DEG_LEVEL_METADATA_LOST"


@dataclass
class ConversionOutcome:
    target_version: int
    records: list[UnifiedResult] = field(default_factory=list)
    degradations: list[Degradation] = field(default_factory=list)
    bypass_entries: list[dict[str, Any]] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)

    def bypass_document(self) -> dict[str, Any]:
        return {
            "schema": "aqeval.bypass.v1",
            "target_version": self.target_version,
            "entries": sorted(self.bypass_entries, key=lambda item: item["file_id"]),
        }

    def to_summary(self) -> dict[str, Any]:
        return {
            "target_version": self.target_version,
            "record_count": len(self.records),
            "failure_count": len(self.failures),
            "degradation_count": len(self.degradations),
            "records": [record.to_dict() for record in self.records],
            "degradations": [item.to_dict() for item in self.degradations],
            "bypass": self.bypass_document(),
            "failures": [failure.to_dict() for failure in self.failures],
        }


def _permille_part(micro: int) -> tuple[int, int]:
    return (micro + 500) // 1000, micro % 1000


def _resolve_level(record: UnifiedResult, level_map: LevelMap):
    if record.level_code is not None:
        return level_map.by_code(record.level_code)
    if record.level_name is not None:
        return level_map.by_name(record.level_name)
    raise LevelMapError(f"{record.file_id}: result has neither a level code nor a level name")


def convert_one(
    record: UnifiedResult,
    target_version: int,
    level_map: LevelMap,
    fixed_bits: int,
) -> tuple[UnifiedResult, list[Degradation], dict[str, Any] | None]:
    entry = _resolve_level(record, level_map)
    bypass: dict[str, Any] = {
        "source_version": record.source_version,
        "target_version": target_version,
        "source_precision_tag": record.precision_tag,
        "source_level_code": record.level_code,
        "source_level_name": record.level_name,
        "canonical_total_micro": record.total_micro,
        "canonical_dimensions_micro": dict(record.dimensions_micro),
    }
    losses: list[Degradation] = []
    if target_version == 1:
        total_permille, total_residual = _permille_part(record.total_micro)
        dims_permille: dict[str, int] = {}
        dims_residual: dict[str, int] = {}
        for name in DIMENSIONS:
            dims_permille[name], dims_residual[name] = _permille_part(
                record.dimensions_micro[name]
            )
        bypass.update(
            {
                "total_residual_micro": total_residual,
                "dimension_residual_micro": dims_residual,
                "total_permille": total_permille,
                "dimensions_permille": dims_permille,
                "discarded_level_name": entry.name,
                "discarded_precision_tag": record.precision_tag,
                "discarded_frame_summary_count": len(record.frame_summaries),
            }
        )
        if record.source_version != 1:
            losses.append(
                Degradation(
                    code=DEG_V1_TRUNCATION,
                    file_id=record.file_id,
                    source_version=record.source_version,
                    target_version=1,
                    message="v1 stores only permille integers; sub-permille residual and named level are bypassed",
                    residual_micro=total_residual
                    + sum(abs(value) for value in dims_residual.values()),
                    fields={
                        "total_residual_micro": total_residual,
                        "dimension_residual_micro": dims_residual,
                        "discarded_level_name": entry.name,
                    },
                )
            )
            if record.precision_tag or record.level_name:
                losses.append(
                    Degradation(
                        code=DEG_LEVEL_METADATA_LOST,
                        file_id=record.file_id,
                        source_version=record.source_version,
                        target_version=1,
                        message="v1 cannot express a precision tag or named level; both are retained in bypass metadata",
                        fields={
                            "precision_tag": record.precision_tag,
                            "level_name": record.level_name,
                        },
                    )
                )
        if record.frame_summaries:
            losses.append(
                Degradation(
                    code=DEG_FRAME_SUMMARY_DROPPED,
                    file_id=record.file_id,
                    source_version=record.source_version,
                    target_version=1,
                    message="v1 cannot express frame-level feature summaries; count is retained in bypass metadata",
                    fields={"discarded_frame_summary_count": len(record.frame_summaries)},
                )
            )
        converted = UnifiedResult(
            file_id=record.file_id,
            source_version=1,
            total_micro=total_permille * 1000,
            dimensions_micro={name: dims_permille[name] * 1000 for name in DIMENSIONS},
            level_code=entry.code,
            level_name=entry.name,
            precision_tag="permille",
            frame_summaries=[],
            bypass=bypass,
            location=record.location,
        )
        return converted, losses, {
            "file_id": record.file_id,
            **bypass,
        }
    if target_version == 2:
        precision = record.precision_tag or "float6"
        if record.source_version == 1:
            precision = "permille_to_float6"
        elif record.source_version == 3:
            precision = f"v3_fixed{record.source_fixed_bits}->float6"
        bypass["target_precision_tag"] = precision
        bypass["discarded_frame_summary_count"] = len(record.frame_summaries)
        if record.source_version == 3:
            inherited = record.total_micro % max(1, fixed_factor(record.source_fixed_bits or 6))
            bypass["inherited_fixed_residual_micro"] = inherited
        if record.frame_summaries:
            losses.append(
                Degradation(
                    code=DEG_FRAME_SUMMARY_DROPPED,
                    file_id=record.file_id,
                    source_version=record.source_version,
                    target_version=2,
                    message="v2 JSON cannot express frame-level feature summaries; count is retained in bypass metadata",
                    fields={"discarded_frame_summary_count": len(record.frame_summaries)},
                )
            )
        converted = UnifiedResult(
            file_id=record.file_id,
            source_version=2,
            total_micro=record.total_micro,
            dimensions_micro=dict(record.dimensions_micro),
            level_code=entry.code,
            level_name=entry.name,
            precision_tag=precision,
            frame_summaries=[],
            bypass=bypass,
            location=record.location,
        )
        return converted, losses, {
            "file_id": record.file_id,
            **bypass,
        }
    if target_version == 3:
        factor = fixed_factor(fixed_bits)
        total_fixed = micro_to_fixed(record.total_micro, fixed_bits)
        dims_fixed = {
            name: micro_to_fixed(record.dimensions_micro[name], fixed_bits)
            for name in DIMENSIONS
        }
        total_residual = record.total_micro - total_fixed * factor
        dims_residual = {
            name: record.dimensions_micro[name] - dims_fixed[name] * factor
            for name in DIMENSIONS
        }
        bypass.update(
            {
                "target_fixed_decimal_bits": fixed_bits,
                "fixed_factor_micro": factor,
                "total_fixed": total_fixed,
                "dimensions_fixed": dims_fixed,
                "total_residual_micro": total_residual,
                "dimension_residual_micro": dims_residual,
            }
        )
        if total_residual or any(dims_residual.values()):
            losses.append(
                Degradation(
                    code=DEG_V3_FIXED_WIDTH,
                    file_id=record.file_id,
                    source_version=record.source_version,
                    target_version=3,
                    message="v3 fixed-point width cannot represent all micro-score digits",
                    residual_micro=abs(total_residual)
                    + sum(abs(value) for value in dims_residual.values()),
                    fields={
                        "fixed_decimal_bits": fixed_bits,
                        "total_residual_micro": total_residual,
                        "dimension_residual_micro": dims_residual,
                    },
                )
            )
        frames = list(record.frame_summaries)
        if not frames and record.source_version in (1, 2):
            losses.append(
                Degradation(
                    code=DEG_FRAME_SUMMARY_UNAVAILABLE,
                    file_id=record.file_id,
                    source_version=record.source_version,
                    target_version=3,
                    message="source exchange has no frame-level summaries; v3 frame partition is empty",
                    fields={"frame_count": 0},
                )
            )
        converted = UnifiedResult(
            file_id=record.file_id,
            source_version=3,
            total_micro=total_fixed * factor,
            dimensions_micro={name: dims_fixed[name] * factor for name in DIMENSIONS},
            level_code=entry.code,
            level_name=entry.name,
            precision_tag=f"v3_fixed_{fixed_bits}",
            frame_summaries=frames,
            bypass=bypass,
            location=record.location,
            source_fixed_bits=fixed_bits,
        )
        return converted, losses, {
            "file_id": record.file_id,
            **bypass,
        }
    raise ValueError(f"unsupported target version: {target_version}")


def convert_batch(
    records: list[UnifiedResult],
    target_version: int,
    level_map: LevelMap,
    fixed_bits: int,
) -> ConversionOutcome:
    outcome = ConversionOutcome(target_version=target_version)
    for record in sorted(records, key=lambda item: item.file_id):
        try:
            converted, losses, bypass = convert_one(
                record, target_version, level_map, fixed_bits
            )
        except LevelMapError as exc:
            outcome.failures.append(
                Failure(
                    code=E_LEVEL_UNMAPPED,
                    message=str(exc),
                    file_id=record.file_id,
                    version=record.source_version,
                    stage="conversion",
                    location=record.location,
                    task_key=f"{record.file_id}|conversion|v{target_version}",
                )
            )
            continue
        outcome.records.append(converted)
        outcome.degradations.extend(losses)
        if bypass is not None:
            outcome.bypass_entries.append(bypass)
    outcome.records.sort(key=lambda item: item.file_id)
    outcome.degradations = sorted_degradations(outcome.degradations)
    outcome.failures.sort(key=Failure.sort_key)
    return outcome


def convert_file(
    source_version: int,
    target_version: int,
    source_path: str | Path,
    target_path: str | Path,
    level_map: LevelMap,
    fixed_bits: int,
    bypass_path: str | Path | None = None,
) -> dict[str, Any]:
    from .formats import read_exchange, write_exchange

    parsed = read_exchange(source_version, source_path, level_map)
    converted = convert_batch(parsed.records, target_version, level_map, fixed_bits)
    failures = list(parsed.failures) + converted.failures
    failures.sort(key=Failure.sort_key)
    summary: dict[str, Any] = {
        "source_version": source_version,
        "target_version": target_version,
        "source": str(source_path),
        "target": str(target_path),
        "source_records": len(parsed.records),
        "converted_records": len(converted.records),
        "failures": [failure.to_dict() for failure in failures],
        "degradations": [item.to_dict() for item in converted.degradations],
        "output_sha256": None,
    }
    if converted.records:
        write_exchange(target_version, converted.records, fixed_bits, target_path)
        from .util import sha256_file

        summary["output_sha256"] = sha256_file(target_path)
    elif Path(target_path).exists():
        Path(target_path).unlink()
    if bypass_path is not None:
        atomic_write_json(bypass_path, converted.bypass_document())
        summary["bypass_path"] = str(bypass_path)
        summary["bypass_sha256"] = sha256_json(converted.bypass_document())
    return summary
