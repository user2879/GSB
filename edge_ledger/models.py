from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


QUANTIZATION_STEPS = {
    "Q100": 10_000,
    "Q1000": 1_000,
    "Q10000": 100,
}
ALLOWED_DECIMAL_PLACES = (2, 3, 4, 6)


@dataclass(frozen=True)
class SlotRule:
    name: str
    required: bool = True
    allowed_values: tuple[str, ...] = ()
    min_length: int | None = None
    max_length: int | None = None
    default: str | None = None
    source_revision: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "allowed_values": list(self.allowed_values),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "default": self.default,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True)
class InputSlot:
    name: str
    value: str
    priority: int = 0
    input_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "priority": self.priority,
            "input_index": self.input_index,
        }


@dataclass(frozen=True)
class ResolvedSlot:
    name: str
    value: str
    priority: int
    source_index: int
    suppressed: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "priority": self.priority,
            "source_index": self.source_index,
            "suppressed": list(self.suppressed),
        }


@dataclass
class SampleInput:
    sample_id: str
    template_id: str
    template_revision: int
    answer_template_id: str
    answer_template_revision: int
    source_shard: str
    slot_fills: list[InputSlot]
    batch_id: str
    batch_file: str
    input_index: int = 0
    injection_ids: tuple[str, ...] = ()
    quality_override: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def content_document(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "template_id": self.template_id,
            "template_revision": self.template_revision,
            "answer_template_id": self.answer_template_id,
            "answer_template_revision": self.answer_template_revision,
            "source_shard": self.source_shard,
            "slot_fills": [fill.as_dict() for fill in self.slot_fills],
            "quality_override": self.quality_override,
            "injection_ids": list(self.injection_ids),
        }


@dataclass(frozen=True)
class ValidationProducts:
    normalized_templates: tuple[dict[str, Any], ...]
    normalized_samples: tuple[dict[str, Any], ...]
    templates_fingerprint: str
    samples_fingerprint: str
    config_hash: str
    global_version: str
    template_revision_fingerprints: dict[str, str]
    sample_fingerprints: dict[str, str]


@dataclass(frozen=True)
class QualityBreakdown:
    raw_score: int
    quantized_score: int
    quantum_step: int
    residual: int
    components: dict[str, int]
    decimal_places: int
    quantization: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_score": self.raw_score,
            "quantized_score": self.quantized_score,
            "quantum_step": self.quantum_step,
            "residual": self.residual,
            "components": self.components,
            "decimal_places": self.decimal_places,
            "quantization": self.quantization,
        }

