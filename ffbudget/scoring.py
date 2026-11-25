"""Stage 4a: deterministic fixed-point slot scoring.

All arithmetic stays in signed 64-bit integers (Q16.16). A slot contribution
is weight * stale_penalty * sum(vector). Unavailable slots use the fixed
default vector and a dedicated default penalty; the vector source is carried
on every artifact for audit.
"""
from __future__ import annotations

import numpy as np

from .fingerprint import cache_key, content_fingerprint
from .fixedpoint import (WIDTH_TIER, slot_contribution, to_q16,
                         vector_sum_q16)
from .models import SlotResolution, SlotSpec
from .view import STATE_FRESH, STATE_UNAVAILABLE


def penalty_q16(resolution: SlotResolution, config: dict) -> np.int64:
    staleness_cfg = config["staleness"]
    if resolution.state == STATE_UNAVAILABLE:
        return np.int64(staleness_cfg["penalty_unavailable_q16"])
    if resolution.state == STATE_FRESH:
        return np.int64(staleness_cfg["penalty_fresh_q16"])
    staleness = int(resolution.staleness_ms or 0)
    raw = np.int64(staleness_cfg["penalty_fresh_q16"]) - \
        np.int64(np.int64(staleness_cfg["stale_penalty_per_ms_q16"])
                 * np.int64(staleness))
    floor = np.int64(staleness_cfg["stale_floor_q16"])
    return np.int64(max(raw, floor))


def score_slot(resolution: SlotResolution, slot: SlotSpec,
               config: dict, staleness_fp: str) -> dict:
    weight_q16 = to_q16(slot.weight)
    penalty = penalty_q16(resolution, config)
    value_q16 = vector_sum_q16(resolution.vector)
    contribution = slot_contribution(weight_q16, penalty, value_q16)
    content_fp = content_fingerprint(list(resolution.vector))
    authority_fp = resolution.used_digest or "default"
    key = cache_key(["score", slot.key, content_fp, authority_fp,
                     staleness_fp, WIDTH_TIER])
    return {
        "slot_key": slot.key,
        "group_key": resolution.group_key,
        "state": resolution.state,
        "staleness_ms": resolution.staleness_ms,
        "used_digest": resolution.used_digest,
        "used_rev_seq": resolution.used_rev_seq,
        "used_batch_seq": resolution.used_batch_seq,
        "vector_source": resolution.vector_source,
        "weight_q16": int(weight_q16),
        "penalty_q16": int(penalty),
        "value_q16": int(value_q16),
        "contribution_q16": int(contribution),
        "score_cache_key": key,
        "content_fingerprint": content_fp,
        "authority_fingerprint": authority_fp,
        "width_tier": WIDTH_TIER,
    }
