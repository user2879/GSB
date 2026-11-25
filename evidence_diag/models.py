from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .fixedpoint import FixedDomain
from .util import decimal_text


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise InvalidOperation("boolean is not a decimal")
    return Decimal(str(value))


@dataclass(frozen=True)
class FixedPointConfig:
    fractional_bits: int = 12
    quantization: int = 4
    rounding: str = "half_up"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "FixedPointConfig":
        value = dict(raw or {})
        return cls(
            fractional_bits=value.get("fractional_bits", 12),
            quantization=value.get("quantization", 4),
            rounding=value.get("rounding", "half_up"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractional_bits": self.fractional_bits,
            "quantization": self.quantization,
            "rounding": self.rounding,
        }

    def domain(self) -> FixedDomain:
        return FixedDomain(
            fractional_bits=self.fractional_bits,
            quantization=self.quantization,
            rounding=self.rounding,
        )


@dataclass(frozen=True)
class ReferenceFixedPointConfig:
    fractional_bits: int = 24
    quantization: int = 1
    rounding: str = "half_up"

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any] | None
    ) -> "ReferenceFixedPointConfig":
        value = dict(raw or {})
        return cls(
            fractional_bits=value.get("fractional_bits", 24),
            quantization=value.get("quantization", 1),
            rounding=value.get("rounding", "half_up"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractional_bits": self.fractional_bits,
            "quantization": self.quantization,
            "rounding": self.rounding,
        }

    def domain(self) -> FixedDomain:
        return FixedDomain(
            fractional_bits=self.fractional_bits,
            quantization=self.quantization,
            rounding=self.rounding,
        )


@dataclass(frozen=True)
class TopicConfig:
    switch_threshold_bp: int = 4200

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TopicConfig":
        value = dict(raw or {})
        return cls(switch_threshold_bp=value.get("switch_threshold_bp", 4200))

    def to_dict(self) -> dict[str, Any]:
        return {"switch_threshold_bp": self.switch_threshold_bp}


@dataclass(frozen=True)
class ConfidenceConfig:
    turn_decay: Decimal = Decimal("0.93")
    cross_topic_penalty: Decimal = Decimal("0.15")
    answer_threshold: Decimal = Decimal("0.55")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ConfidenceConfig":
        value = dict(raw or {})
        return cls(
            turn_decay=_decimal(value.get("turn_decay", "0.93")),
            cross_topic_penalty=_decimal(value.get("cross_topic_penalty", "0.15")),
            answer_threshold=_decimal(value.get("answer_threshold", "0.55")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_decay": decimal_text(self.turn_decay),
            "cross_topic_penalty": decimal_text(self.cross_topic_penalty),
            "answer_threshold": decimal_text(self.answer_threshold),
        }


@dataclass(frozen=True)
class LimitsConfig:
    max_workers: int = 4
    evidence_budget: int = 64

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "LimitsConfig":
        value = dict(raw or {})
        return cls(
            max_workers=value.get("max_workers", 4),
            evidence_budget=value.get("evidence_budget", 64),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "evidence_budget": self.evidence_budget,
        }


@dataclass(frozen=True)
class DiagnosticsConfig:
    tolerance: Decimal = Decimal("0.002")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "DiagnosticsConfig":
        value = dict(raw or {})
        return cls(tolerance=_decimal(value.get("tolerance", "0.002")))

    def to_dict(self) -> dict[str, Any]:
        return {"tolerance": decimal_text(self.tolerance)}


@dataclass(frozen=True)
class AppConfig:
    channel_weights: Mapping[str, Decimal]
    fixed_point: FixedPointConfig = field(default_factory=FixedPointConfig)
    topic: TopicConfig = field(default_factory=TopicConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    reference: ReferenceFixedPointConfig = field(
        default_factory=ReferenceFixedPointConfig
    )
    schema_version: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AppConfig":
        return cls(
            channel_weights={
                str(name): _decimal(weight)
                for name, weight in dict(raw.get("channel_weights", {})).items()
            },
            fixed_point=FixedPointConfig.from_dict(raw.get("fixed_point")),
            topic=TopicConfig.from_dict(raw.get("topic")),
            confidence=ConfidenceConfig.from_dict(raw.get("confidence")),
            limits=LimitsConfig.from_dict(raw.get("limits")),
            diagnostics=DiagnosticsConfig.from_dict(raw.get("diagnostics")),
            reference=ReferenceFixedPointConfig.from_dict(raw.get("reference")),
            schema_version=raw.get("schema_version", 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "channel_weights": {
                name: decimal_text(weight)
                for name, weight in sorted(self.channel_weights.items())
            },
            "fixed_point": self.fixed_point.to_dict(),
            "topic": self.topic.to_dict(),
            "confidence": self.confidence.to_dict(),
            "limits": self.limits.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "reference": self.reference.to_dict(),
        }

    def with_fixed_point(self, fixed: FixedPointConfig) -> "AppConfig":
        return AppConfig(
            channel_weights=dict(self.channel_weights),
            fixed_point=fixed,
            topic=self.topic,
            confidence=self.confidence,
            limits=self.limits,
            diagnostics=self.diagnostics,
            reference=self.reference,
            schema_version=self.schema_version,
        )

    def with_limits(self, limits: LimitsConfig) -> "AppConfig":
        return AppConfig(
            channel_weights=dict(self.channel_weights),
            fixed_point=self.fixed_point,
            topic=self.topic,
            confidence=self.confidence,
            limits=limits,
            diagnostics=self.diagnostics,
            reference=self.reference,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    weight: Decimal

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Evidence":
        return cls(
            evidence_id=str(raw.get("evidence_id", "")),
            text=str(raw.get("text", "")),
            weight=_decimal(raw.get("weight", "0")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "weight": decimal_text(self.weight),
        }


@dataclass(frozen=True)
class Turn:
    turn_id: int
    query: str
    channels: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Turn":
        channels_raw = dict(raw.get("channels", {}))
        channels: dict[str, tuple[str, ...]] = {}
        for channel, evidence_ids in channels_raw.items():
            if isinstance(evidence_ids, list):
                channels[str(channel)] = tuple(str(item) for item in evidence_ids)
            else:
                channels[str(channel)] = tuple()
        return cls(
            turn_id=raw.get("turn_id"),  # type: ignore[arg-type]
            query=str(raw.get("query", "")),
            channels=channels,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "channels": {
                channel: list(ids)
                for channel, ids in sorted(self.channels.items())
            },
        }


@dataclass(frozen=True)
class Session:
    session_id: str
    turns: tuple[Turn, ...]
    evidence: tuple[Evidence, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Session":
        return cls(
            session_id=str(raw.get("session_id", "")),
            turns=tuple(Turn.from_dict(item) for item in raw.get("turns", [])),
            evidence=tuple(
                Evidence.from_dict(item) for item in raw.get("evidence", [])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def evidence_map(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}
