from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from .fixedpoint import (
    ALLOWED_FRACTIONAL_BITS,
    ALLOWED_QUANTIZATION,
    ALLOWED_ROUNDING,
)
from .models import AppConfig, Session

ERROR_CODE = "E_INPUT_INVALID"


def issue(path: str, message: str) -> dict[str, str]:
    return {"code": ERROR_CODE, "path": path, "message": message}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_decimal(value: Any) -> bool:
    try:
        return Decimal(value).is_finite()
    except (InvalidOperation, ValueError, TypeError):
        return False


def _nonnegative(value: Any) -> bool:
    if not _finite_decimal(value):
        return False
    return Decimal(value) >= 0


def validate_config(config: AppConfig) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if config.schema_version != 1:
        issues.append(issue("schema_version", "only schema_version 1 is supported"))
    if not config.channel_weights:
        issues.append(issue("channel_weights", "at least one channel is required"))
    for name, weight in config.channel_weights.items():
        if not name or not name.strip():
            issues.append(issue("channel_weights", "channel name must be non-empty"))
        if not _finite_decimal(weight):
            issues.append(
                issue(f"channel_weights.{name}", "weight must be finite")
            )
        elif Decimal(weight) < 0:
            issues.append(
                issue(f"channel_weights.{name}", "weight must be non-negative")
            )

    fixed = config.fixed_point
    if fixed.fractional_bits not in ALLOWED_FRACTIONAL_BITS:
        issues.append(
            issue(
                "fixed_point.fractional_bits",
                f"must be one of {list(ALLOWED_FRACTIONAL_BITS)}",
            )
        )
    if fixed.quantization not in ALLOWED_QUANTIZATION:
        issues.append(
            issue(
                "fixed_point.quantization",
                f"must be one of {list(ALLOWED_QUANTIZATION)}",
            )
        )
    if fixed.rounding not in ALLOWED_ROUNDING:
        issues.append(
            issue(
                "fixed_point.rounding",
                f"must be one of {list(ALLOWED_ROUNDING)}",
            )
        )

    reference = config.reference
    if reference.fractional_bits < fixed.fractional_bits:
        issues.append(
            issue(
                "reference.fractional_bits",
                "reference precision must be at least target precision",
            )
        )
    if reference.fractional_bits not in (20, 24, 32):
        issues.append(
            issue("reference.fractional_bits", "must be one of [20, 24, 32]")
        )
    if reference.quantization not in ALLOWED_QUANTIZATION:
        issues.append(
            issue(
                "reference.quantization",
                f"must be one of {list(ALLOWED_QUANTIZATION)}",
            )
        )
    if reference.rounding not in ALLOWED_ROUNDING:
        issues.append(
            issue(
                "reference.rounding",
                f"must be one of {list(ALLOWED_ROUNDING)}",
            )
        )

    if not _is_int(config.topic.switch_threshold_bp) or not (
        0 <= config.topic.switch_threshold_bp <= 10000
    ):
        issues.append(
            issue(
                "topic.switch_threshold_bp",
                "must be an integer in [0, 10000]",
            )
        )
    if not _nonnegative(config.confidence.turn_decay) or not (
        Decimal(0) <= config.confidence.turn_decay <= Decimal(1)
    ):
        issues.append(
            issue("confidence.turn_decay", "must be a finite number in [0, 1]")
        )
    if not _nonnegative(config.confidence.cross_topic_penalty) or not (
        config.confidence.cross_topic_penalty <= Decimal(1)
    ):
        issues.append(
            issue(
                "confidence.cross_topic_penalty",
                "must be a finite number in [0, 1]",
            )
        )
    if not _nonnegative(config.confidence.answer_threshold) or not (
        config.confidence.answer_threshold <= Decimal(1)
    ):
        issues.append(
            issue("confidence.answer_threshold", "must be a finite number in [0, 1]")
        )
    if not _is_int(config.limits.max_workers) or not (
        1 <= config.limits.max_workers <= 32
    ):
        issues.append(
            issue("limits.max_workers", "must be an integer in [1, 32]")
        )
    if not _is_int(config.limits.evidence_budget) or (
        config.limits.evidence_budget <= 0
    ):
        issues.append(
            issue("limits.evidence_budget", "must be a positive integer")
        )
    if not _nonnegative(config.diagnostics.tolerance):
        issues.append(
            issue("diagnostics.tolerance", "must be a finite non-negative number")
        )
    return issues


def _validate_channels(
    turn: Any, index: int, config: AppConfig, evidence_ids: set[str]
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    prefix = f"turns[{index}]"
    if not _is_int(turn.turn_id):
        issues.append(issue(f"{prefix}.turn_id", "turn_id must be an integer"))
    if not isinstance(turn.query, str) or not turn.query.strip():
        issues.append(issue(f"{prefix}.query", "query must be non-empty"))
    if not turn.channels:
        issues.append(issue(f"{prefix}.channels", "at least one channel is required"))
    for channel, declared in turn.channels.items():
        if channel not in config.channel_weights:
            issues.append(
                issue(
                    f"{prefix}.channels.{channel}",
                    "channel has no configured weight",
                )
            )
        if len(declared) == 0:
            issues.append(
                issue(f"{prefix}.channels.{channel}", "must declare evidence IDs")
            )
        for evidence_id in declared:
            if evidence_id not in evidence_ids:
                issues.append(
                    issue(
                        f"{prefix}.channels.{channel}",
                        f"unknown evidence_id {evidence_id!r}",
                    )
                )
    return issues


def validate_session(
    config: AppConfig, session: Session
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not session.session_id.strip():
        issues.append(issue("session_id", "session_id must be non-empty"))

    evidence_ids: set[str] = set()
    if not session.evidence:
        issues.append(issue("evidence", "evidence library must not be empty"))
    for index, evidence in enumerate(session.evidence):
        prefix = f"evidence[{index}]"
        if not evidence.evidence_id.strip():
            issues.append(issue(f"{prefix}.evidence_id", "must be non-empty"))
        elif evidence.evidence_id in evidence_ids:
            issues.append(
                issue(
                    f"{prefix}.evidence_id",
                    f"duplicate global evidence_id {evidence.evidence_id!r}",
                )
            )
        evidence_ids.add(evidence.evidence_id)
        if not evidence.text.strip():
            issues.append(issue(f"{prefix}.text", "must be non-empty"))
        if not _nonnegative(evidence.weight):
            issues.append(
                issue(f"{prefix}.weight", "must be finite and non-negative")
            )

    if not session.turns:
        issues.append(issue("turns", "at least one turn is required"))
    expected = 1
    for index, turn in enumerate(session.turns):
        if _is_int(turn.turn_id) and turn.turn_id != expected:
            issues.append(
                issue(
                    f"turns[{index}].turn_id",
                    f"expected contiguous turn_id {expected}, got {turn.turn_id}",
                )
            )
        expected += 1
        issues.extend(_validate_channels(turn, index, config, evidence_ids))
    return issues


def validate_commands(commands: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    expected = 1
    for index, command in enumerate(commands):
        prefix = f"commands[{index}]"
        seq = command.get("seq")
        if not _is_int(seq):
            issues.append(issue(f"{prefix}.seq", "seq must be an integer"))
        elif seq != expected:
            issues.append(
                issue(
                    f"{prefix}.seq",
                    f"command sequence must be contiguous; expected {expected}",
                )
            )
            expected = seq + 1
        else:
            expected += 1
        if not isinstance(command.get("op"), str) or not command.get("op"):
            issues.append(issue(f"{prefix}.op", "op must be a non-empty string"))
    return issues


def valid_finite_decimal(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and Decimal(value).is_finite()
    except (InvalidOperation, ValueError, TypeError):
        return False
