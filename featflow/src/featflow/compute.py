"""Fixed-point feature evaluation.

Base features aggregate raw event context inside the millisecond window.
Derived features aggregate upstream *fixed-point* feature observations inside
the same window.  All arithmetic is exact (Decimal over exact float text) and
all outputs are fixed-point integers, so results are reproducible byte for byte.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .common import (
    AGG_COUNT,
    AGG_LAST,
    AGG_MEAN,
    AGG_SUM,
    FixedPoint,
)
from .models import FeatureDef
from .storage import EventStore


@dataclass(frozen=True)
class FeatResult:
    feature: str
    side: str
    entity_id: str
    point_seq: int
    anchor_ts: int
    value: int                    # fixed-point integer
    empty: bool                   # empty window / empty source
    empty_source: str             # "none" | "no_event" | "no_source_value" | "no_upstream_obs"
    n_inputs: int                 # number of accumulated inputs (charged points bookkeeping)


class FeatureComputer:
    def __init__(self, store: EventStore, fp: FixedPoint, default_empty_raw: str) -> None:
        self.store = store
        self.fp = fp
        self.scale_factor = Decimal(10) ** fp.scale
        self.default_empty = fp.empty_value(default_empty_raw)
        # feature -> entity_id -> sorted list of (anchor_ts, point_seq, value)
        self.observations: dict[str, dict[str, list[tuple[int, int, int]]]] = {}

    # ------------------------------------------------------------- observation
    def record_observation(self, feature: str, entity_id: str, anchor_ts: int,
                           point_seq: int, value: int, empty: bool = False) -> None:
        # Empty-window outputs are retained for sample visibility/snapshots but
        # tagged so derived features do not treat them as real upstream inputs.
        per_entity = self.observations.setdefault(feature, {}).setdefault(entity_id, [])
        per_entity.append((anchor_ts, point_seq, value, empty))

    def get_observation(self, feature: str, entity_id: str, anchor_ts: int,
                        window_ms: int, point_seq: int | None = None
                        ) -> list[tuple[int, int, int]]:
        per_entity = self.observations.get(feature, {}).get(entity_id, [])
        lower = anchor_ts - window_ms
        visible = [
            obs for obs in per_entity
            if lower <= obs[0] <= anchor_ts
            and (point_seq is None or obs[1] <= point_seq)
        ]
        return visible

    def latest_visible(self, feature: str, entity_id: str,
                       decision_ts: int,
                       max_point_seq: int | None = None
                       ) -> tuple[int, int, int, bool] | None:
        per_entity = self.observations.get(feature, {}).get(entity_id, [])
        visible = [
            obs for obs in per_entity
            if obs[0] <= decision_ts
            and (max_point_seq is None or obs[1] <= max_point_seq)
        ]
        if not visible:
            return None
        # largest timestamp, tie broken by largest point_seq (arrival order)
        return max(visible, key=lambda obs: (obs[0], obs[1]))

    # ----------------------------------------------------------------- helpers
    def _empty(self, feat: FeatureDef, point_seq: int, anchor_ts: int,
               entity_id: str, source: str) -> FeatResult:
        if feat.empty_value is None:
            value = self.default_empty
        else:
            value = self.fp.empty_value(feat.empty_value)
        return FeatResult(feat.name, feat.side, entity_id, point_seq,
                          anchor_ts, value, True, source, 0)

    def _aggregate_raw(self, feat: FeatureDef, ids: list[str],
                       values: Any, point_seq: int, anchor_ts: int,
                       entity_id: str) -> FeatResult:
        if not ids:
            return self._empty(feat, point_seq, anchor_ts, entity_id, "no_event")
        if feat.agg == AGG_COUNT:
            present = ids if values is None else [
                eid for eid, v in zip(ids, values) if not _is_nan(v)
            ]
            value = self.fp.quantize(Decimal(len(present)))
            return FeatResult(feat.name, feat.side, entity_id, point_seq,
                              anchor_ts, value, not present,
                              "no_source_value" if not present else "none", len(present))
        if values is None:
            return self._empty(feat, point_seq, anchor_ts, entity_id, "no_source_value")
        present = [(eid, v) for eid, v in zip(ids, values) if not _is_nan(v)]
        if not present:
            return self._empty(feat, point_seq, anchor_ts, entity_id, "no_source_value")
        # Fixed accumulation order: event_id lexicographic (ids already sorted).
        exact = [Decimal(repr(float(v))) for _, v in present]
        if feat.agg == AGG_SUM:
            raw = sum(exact, Decimal(0))
            value = self.fp.quantize(raw)
        elif feat.agg == AGG_LAST:
            value = self.fp.quantize(exact[-1])
        else:  # mean in fixed point: scaled average then re-quantized
            total_scaled = sum((self.fp.quantize(x) for x in exact), 0)
            avg_scaled = (Decimal(total_scaled) / Decimal(len(exact))).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            value = int(avg_scaled)
        return FeatResult(feat.name, feat.side, entity_id, point_seq,
                          anchor_ts, value, False, "none", len(present))

    def _aggregate_derived(self, feat: FeatureDef, point_seq: int,
                           anchor_ts: int, entity_id: str,
                           max_point_seq: int | None = None) -> FeatResult:
        merged: list[tuple[int, int, int]] = []
        for up in feat.upstream:
            merged.extend(self.get_observation(
                up, entity_id, anchor_ts, feat.window_ms, max_point_seq))
        if not merged:
            return self._empty(feat, point_seq, anchor_ts, entity_id, "no_upstream_obs")
        # Drop empty upstream observations: a base feature with no events in
        # its window cannot feed a derived aggregate.
        merged = [obs for obs in merged if not obs[3]]
        if not merged:
            return self._empty(feat, point_seq, anchor_ts, entity_id, "no_upstream_obs")
        # Stable merge order: (anchor_ts, point_seq, upstream feature name, value).
        merged.sort(key=lambda obs: (obs[0], obs[1], obs[2]))
        values = [obs[2] for obs in merged]
        if feat.agg == AGG_COUNT:
            value = self.fp.quantize(Decimal(len(values)))
        elif feat.agg == AGG_SUM:
            value = self.fp.quantize(Decimal(sum(values)) / self.scale_factor)
        elif feat.agg == AGG_LAST:
            value = values[-1]
        else:
            avg = (Decimal(sum(values)) / Decimal(len(values))).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            value = int(avg)
        return FeatResult(feat.name, feat.side, entity_id, point_seq,
                          anchor_ts, value, False, "none", len(values))

    # ------------------------------------------------------------------ public
    def evaluate_base(self, feat: FeatureDef, entity_id: str,
                      point_seq: int, anchor_ts: int) -> FeatResult:
        ids, values = self.store.select_window(
            feat.side, entity_id, anchor_ts, feat.window_ms, feat.source_field
        )
        return self._aggregate_raw(feat, ids, values, point_seq, anchor_ts, entity_id)

    def evaluate_derived(self, feat: FeatureDef, entity_id: str,
                         point_seq: int, anchor_ts: int) -> FeatResult:
        return self._aggregate_derived(feat, point_seq, anchor_ts, entity_id,
                                       point_seq)


def _is_nan(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
