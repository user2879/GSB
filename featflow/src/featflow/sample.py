"""Training-sample generation with label effective delay."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .common import CODE_SAMPLE_PENDING, Failure
from .compute import FeatureComputer
from .models import Event, LabelDef


@dataclass(frozen=True)
class Sample:
    sample_id: str
    label_name: str
    trigger_event_id: str
    decision_ts: int
    entity_kind: str
    entity_id: str
    status: str                 # ready | pending
    label: int                  # 0/1 when ready, -1 when pending
    features: dict[str, int]
    missing_features: list[str]
    failure_code: str

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "label_name": self.label_name,
            "trigger_event_id": self.trigger_event_id,
            "decision_ts": self.decision_ts,
            "entity_kind": self.entity_kind,
            "entity_id": self.entity_id,
            "status": self.status,
            "label": self.label,
            "features": self.features,
            "missing_features": self.missing_features,
            "failure_code": self.failure_code,
        }


def _compare(decision_value: int, threshold_scaled: int, op: str) -> bool:
    if op == "ge":
        return decision_value >= threshold_scaled
    if op == "gt":
        return decision_value > threshold_scaled
    if op == "le":
        return decision_value <= threshold_scaled
    if op == "lt":
        return decision_value < threshold_scaled
    return decision_value == threshold_scaled


class SampleGenerator:
    def __init__(self, labels: list[LabelDef], computer: FeatureComputer,
                 scale: int, quant_step: str) -> None:
        self.labels = labels
        self.computer = computer
        factor = Decimal(10) ** scale
        self.thresholds = {
            lb.name: int((Decimal(lb.threshold_raw) / Decimal(quant_step)
                          ).quantize(Decimal("1")) * factor)
            for lb in labels
        }

    def evaluate(self, event: Event, point_seq: int,
                 feature_side: dict[str, str],
                 aborted: set[str],
                 event_point_seq: int | None = None
                 ) -> tuple[list[Sample], list[Failure]]:
        samples: list[Sample] = []
        failures: list[Failure] = []
        for lb in self.labels:
            if event.behavior not in lb.trigger_behaviors:
                continue
            decision_ts = event.ts_ms + lb.delay_ms
            sample_id = f"{event.event_id}|{lb.name}"
            side = feature_side[lb.decision_feature]
            entity_id = event.user_id if side == "user" else event.item_id

            features: dict[str, int] = {}
            missing: list[str] = []
            for fname in lb.observed_features:
                fside = feature_side[fname]
                fentity = event.user_id if fside == "user" else event.item_id
                # A visible snapshot (including an empty-window sentinel) is
                # always preferred; abort only matters when nothing is visible.
                obs = self.computer.latest_visible(
                    fname, fentity, decision_ts, event_point_seq)
                if obs is None and fname in aborted:
                    missing.append(fname)
                elif obs is None:
                    missing.append(fname)
                else:
                    features[fname] = obs[2]

            if missing:
                samples.append(Sample(
                    sample_id, lb.name, event.event_id, decision_ts,
                    side, entity_id, "pending", -1, features, sorted(missing),
                    CODE_SAMPLE_PENDING))
                failures.append(Failure(
                    "sample", sample_id, CODE_SAMPLE_PENDING,
                    f"unavailable features: {','.join(sorted(missing))}"))
                continue

            decision_value = features[lb.decision_feature]
            label = 1 if _compare(decision_value, self.thresholds[lb.name],
                                  lb.positive_op) else 0
            samples.append(Sample(
                sample_id, lb.name, event.event_id, decision_ts,
                side, entity_id, "ready", label, features, [], ""))
        return samples, failures
