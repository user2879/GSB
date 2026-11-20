from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any, Iterable

from .canonical import fingerprint
from .config import LabConfig
from .models import V1Record, V2Record

SIDECAR_SCHEMA_VERSION = 1


def milli_to_float(value: int) -> float:
    return int(value) / 1000.0


def float_to_milli_half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("cannot convert non-finite float to milli")
    with localcontext() as context:
        context.prec = 60
        scaled = Decimal(str(float(value))) * Decimal(1000)
        rounded = scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(rounded)


def residual_milli_decimal(value: float, rounded_milli: int) -> str:
    with localcontext() as context:
        context.prec = 60
        exact = Decimal(str(float(value))) * Decimal(1000)
        residual = exact - Decimal(int(rounded_milli))
        if residual == 0:
            return "0"
        return format(residual.normalize(), "f")


def _conversion_failure(
    message: str,
    *,
    pair_id: str,
    source_version: str,
    target_version: str,
    injection_id: str | None = None,
    code: str = "E_SIDECAR_INVALID",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": code,
        "message": message,
        "stage": "conversion",
        "source_version": source_version,
        "target_version": target_version,
        "pair_id": pair_id,
        "severity": "error",
    }
    if injection_id is not None:
        result["injection_id"] = injection_id
    return result


def sidecar_entry_valid(entry: dict[str, Any], source_record: V1Record) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("milli_reward_score") != source_record.score_milli:
        return False
    if entry.get("milli_reward_delta") != source_record.delta_milli:
        return False
    expected_fingerprint = fingerprint(source_record.content_dict(), prefix="v1:")
    stored_fingerprint = entry.get("v1_content_fingerprint")
    return stored_fingerprint == expected_fingerprint


@dataclass
class ConversionOutcome:
    records: list[Any]
    sidecar: dict[str, Any]
    losses: dict[str, Any]
    failures: list[dict[str, Any]]
    restored_pair_ids: list[str]


def convert_v1_to_v2(
    records: Iterable[V1Record],
    config: LabConfig,
    *,
    sidecar: dict[str, Any] | None = None,
    force_missing_sidecar_pair: str | None = None,
    force_missing_sidecar_injection_id: str | None = None,
) -> ConversionOutcome:
    converted: list[V2Record] = []
    failures: list[dict[str, Any]] = []
    restored_pair_ids: list[str] = []
    loss_rows: list[dict[str, Any]] = []
    sidecar_records = (sidecar or {}).get("records", {})
    if not isinstance(sidecar_records, dict):
        sidecar_records = {}
    precision_label = (
        "milli_exact" if "milli_exact" in config.precision_labels else config.precision_labels[0]
    )
    for record in sorted(records, key=lambda item: item.pair_id):
        if record.pair_id == force_missing_sidecar_pair:
            failures.append(
                _conversion_failure(
                    "required sidecar metadata is missing for restore",
                    pair_id=record.pair_id,
                    source_version="v1",
                    target_version="v2",
                    injection_id=force_missing_sidecar_injection_id,
                    code="E_SIDECAR_MISSING",
                )
            )
            continue
        entry = sidecar_records.get(record.pair_id)
        if entry is not None and sidecar_entry_valid(entry, record):
            label = entry.get("reward_label")
            restored_precision = entry.get("precision_label")
            try:
                original_score = float(entry["reward_score"])
                original_delta = float(entry["reward_delta"])
            except (KeyError, TypeError, ValueError):
                failures.append(
                    _conversion_failure(
                        "sidecar metadata cannot be restored",
                        pair_id=record.pair_id,
                        source_version="v1",
                        target_version="v2",
                    )
                )
                continue
            if (
                label not in config.reward_labels
                or restored_precision not in config.precision_labels
                or not math.isfinite(original_score)
                or not math.isfinite(original_delta)
            ):
                failures.append(
                    _conversion_failure(
                        "sidecar metadata contains an invalid label or non-finite value",
                        pair_id=record.pair_id,
                        source_version="v1",
                        target_version="v2",
                    )
                )
                continue
            converted.append(
                V2Record(
                    pair_id=record.pair_id,
                    reward_score=float(original_score),
                    reward_delta=float(original_delta),
                    reward_label=str(label),
                    precision_label=str(restored_precision),
                )
            )
            restored_pair_ids.append(record.pair_id)
            loss_rows.append(
                {
                    "pair_id": record.pair_id,
                    "mode": "sidecar_restore",
                    "score_loss": 0.0,
                    "delta_loss": 0.0,
                }
            )
        else:
            score = milli_to_float(record.score_milli)
            delta = milli_to_float(record.delta_milli)
            converted.append(
                V2Record(
                    pair_id=record.pair_id,
                    reward_score=score,
                    reward_delta=delta,
                    reward_label=config.v1_default_label,
                    precision_label=precision_label,
                )
            )
            loss_rows.append(
                {
                    "pair_id": record.pair_id,
                    "mode": "exact_milli",
                    "score_loss": 0.0,
                    "delta_loss": 0.0,
                }
            )
    losses = {
        "direction": "v1_to_v2",
        "record_count": len(converted),
        "restored_from_sidecar": len(restored_pair_ids),
        "exact_milli_conversions": len(converted) - len(restored_pair_ids),
        "max_abs_score_loss": 0.0,
        "max_abs_delta_loss": 0.0,
        "total_abs_score_loss": 0.0,
        "total_abs_delta_loss": 0.0,
        "rows": sorted(loss_rows, key=lambda row: row["pair_id"]),
    }
    return ConversionOutcome(
        records=converted,
        sidecar={},
        losses=losses,
        failures=failures,
        restored_pair_ids=restored_pair_ids,
    )


def convert_v2_to_v1(
    records: Iterable[V2Record],
    config: LabConfig,
    *,
    force_missing_sidecar_pair: str | None = None,
    force_missing_sidecar_injection_id: str | None = None,
) -> ConversionOutcome:
    converted: list[V1Record] = []
    failures: list[dict[str, Any]] = []
    sidecar_records: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.pair_id):
        if record.pair_id == force_missing_sidecar_pair:
            failures.append(
                _conversion_failure(
                    "required sidecar metadata is missing for downgrade",
                    pair_id=record.pair_id,
                    source_version="v2",
                    target_version="v1",
                    injection_id=force_missing_sidecar_injection_id,
                    code="E_SIDECAR_MISSING",
                )
            )
            continue
        try:
            score_milli = float_to_milli_half_up(record.reward_score)
            delta_milli = float_to_milli_half_up(record.reward_delta)
        except ValueError:
            failures.append(
                _conversion_failure(
                    "non-finite v2 reward cannot be downgraded",
                    pair_id=record.pair_id,
                    source_version="v2",
                    target_version="v1",
                )
            )
            continue
        v1_record = V1Record(record.pair_id, score_milli, delta_milli)
        converted.append(v1_record)
        score_residual = residual_milli_decimal(record.reward_score, score_milli)
        delta_residual = residual_milli_decimal(record.reward_delta, delta_milli)
        sidecar_records[record.pair_id] = {
            "pair_id": record.pair_id,
            "reward_score": float(record.reward_score),
            "reward_delta": float(record.reward_delta),
            "milli_reward_score": score_milli,
            "milli_reward_delta": delta_milli,
            "score_residual_milli": score_residual,
            "delta_residual_milli": delta_residual,
            "reward_label": record.reward_label,
            "precision_label": record.precision_label,
            "source_version": "v2",
            "target_version": "v1",
            "v1_content_fingerprint": fingerprint(v1_record.content_dict(), prefix="v1:"),
            "v2_content_fingerprint": fingerprint(record.content_dict(), prefix="v2:"),
        }
        rows.append(
            {
                "pair_id": record.pair_id,
                "score_residual_milli": score_residual,
                "delta_residual_milli": delta_residual,
                "reward_label": record.reward_label,
                "precision_label": record.precision_label,
            }
        )
    score_residuals = [abs(Decimal(row["score_residual_milli"])) for row in rows]
    delta_residuals = [abs(Decimal(row["delta_residual_milli"])) for row in rows]
    max_score = max(score_residuals, default=Decimal(0))
    max_delta = max(delta_residuals, default=Decimal(0))
    total_score = sum(score_residuals, Decimal(0))
    total_delta = sum(delta_residuals, Decimal(0))
    losses = {
        "direction": "v2_to_v1",
        "record_count": len(converted),
        "restored_from_sidecar": 0,
        "exact_milli_conversions": len(converted),
        "max_abs_score_residual_milli": str(max_score),
        "max_abs_delta_residual_milli": str(max_delta),
        "total_abs_score_residual_milli": str(total_score),
        "total_abs_delta_residual_milli": str(total_delta),
        "rows": sorted(rows, key=lambda row: row["pair_id"]),
    }
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "direction": "v2_to_v1",
        "rules": {
            "rounding": "ROUND_HALF_UP",
            "residual_unit": "milli",
            "labels_preserved": True,
        },
        "records": {key: sidecar_records[key] for key in sorted(sidecar_records)},
    }
    return ConversionOutcome(
        records=converted,
        sidecar=sidecar,
        losses=losses,
        failures=failures,
        restored_pair_ids=[],
    )


def conversion_key(
    source_record: V1Record | V2Record,
    *,
    source_version: str,
    target_version: str,
    config: LabConfig,
) -> str:
    return fingerprint(
        {
            "source": source_record.content_dict(),
            "source_version": source_version,
            "target_version": target_version,
            "params": config.stage_params("conversion"),
            "converter_schema": 1,
        },
        prefix="conversion:",
    )
