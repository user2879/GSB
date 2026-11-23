from __future__ import annotations

from dataclasses import dataclass

from .config import TrainingConfig
from .state import Sample


@dataclass(frozen=True)
class CVRVisibility:
    sample_id: str
    click_label: int
    conversion_label: int
    conversion_delay_ms: int | None
    visible_at_ms: int | None
    training_event_time_ms: int
    cutoff_ms: int
    visible: bool
    effective_target: int
    reason: str
    weight: int

    def to_dict(self) -> dict:
        return {
            "click_label": self.click_label,
            "conversion_label": self.conversion_label,
            "conversion_delay_ms": self.conversion_delay_ms,
            "visible_at_ms": self.visible_at_ms,
            "training_event_time_ms": self.training_event_time_ms,
            "cutoff_ms": self.cutoff_ms,
            "visible": self.visible,
            "effective_target": self.effective_target,
            "reason": self.reason,
            "cvr_sample_weight": self.weight,
        }


def evaluate_cvr_visibility(
    sample: Sample,
    training_event_time_ms: int,
    cutoff_ms: int,
    training: TrainingConfig,
    scale: int,
) -> CVRVisibility:
    if sample.click == 0:
        reason = "NOT_CLICKED"
        visible = False
        effective = 0
    elif sample.conversion == 0:
        reason = "NO_CONVERSION"
        visible = False
        effective = 0
    elif sample.conversion_delay_ms is None:
        reason = "MISSING_DELIVERY"
        visible = False
        effective = 0
    elif sample.conversion_delay_ms > cutoff_ms:
        reason = "DELAY_EXCEEDS_CUTOFF"
        visible = False
        effective = 0
    elif sample.event_time_ms + sample.conversion_delay_ms <= training_event_time_ms:
        reason = "VISIBLE"
        visible = True
        effective = 1
    else:
        reason = "LABEL_PENDING"
        visible = False
        effective = 0

    if effective == 1:
        weight_real = training.positive_weight * training.conversion_weight
    else:
        weight_real = training.negative_weight
    weight = int(round(weight_real * scale))

    visible_at = (
        sample.event_time_ms + sample.conversion_delay_ms
        if sample.conversion_delay_ms is not None
        else None
    )
    return CVRVisibility(
        sample_id=sample.sample_id,
        click_label=sample.click,
        conversion_label=sample.conversion,
        conversion_delay_ms=sample.conversion_delay_ms,
        visible_at_ms=visible_at,
        training_event_time_ms=training_event_time_ms,
        cutoff_ms=cutoff_ms,
        visible=visible,
        effective_target=effective,
        reason=reason,
        weight=max(1, weight),
    )
