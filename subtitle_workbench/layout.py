"""Deterministic lane-avoidance layout solver.

Ordering rule (total order, no ties): reading priority (lower number wins),
then start frame, then end frame, then item id. Lane rule: within the declared
lane band of the cue type (dialogue anchored bottom, AD anchored top) take the
lowest free lane on every shared frame; when the declared band is full the
fixed yielding rule moves the loser to the opposite band's overflow lanes and
an arbitration record is written. No cue is ever silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .errors import DragBoundsError, TimeInvalidError, TrackConflictError
from .models import (FRAME_HEIGHT, FRAME_WIDTH, SubtitleItem, content_hash)
from .wrapping import FONT_TIERS, wrap_text

# --- declared geometry (1920x1080) --------------------------------------
SIDE_MARGIN = 120
TOP_MARGIN = 90
BOTTOM_MARGIN = 140
LANE_HEIGHT = 132
LANE_GAP = 16
DIALOGUE_LANES = 4
AD_LANES = 2
OVERFLOW_LANES = 2
TOTAL_LANES = DIALOGUE_LANES + AD_LANES + OVERFLOW_LANES

# AD cues get a longer minimum on-screen duration (frames @30fps).
AD_MIN_DURATION = 90        # 3.0 s
DIALOGUE_MIN_DURATION = 45  # 1.5 s


@dataclass(frozen=True)
class LayoutConfig:
    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT
    side_margin: int = SIDE_MARGIN
    top_margin: int = TOP_MARGIN
    bottom_margin: int = BOTTOM_MARGIN
    lane_height: int = LANE_HEIGHT
    lane_gap: int = LANE_GAP
    dialogue_lanes: int = DIALOGUE_LANES
    ad_lanes: int = AD_LANES
    overflow_lanes: int = OVERFLOW_LANES
    max_lines: int = 2
    font_tiers: tuple[int, ...] = FONT_TIERS
    ad_min_duration: int = AD_MIN_DURATION
    dialogue_min_duration: int = DIALOGUE_MIN_DURATION

    @property
    def total_lanes(self) -> int:
        return self.dialogue_lanes + self.ad_lanes + self.overflow_lanes

    def lane_y(self, lane_index: int, line_count: int = 1) -> int:
        """Top Y of a lane slot; AD top-anchored, dialogue bottom-anchored.

        Slot pitch is the single-line lane height so all lanes fit the 1080
        frame; multi-line cues extend inside their slot and are clamped by
        :meth:`cue_y` so they can never leave the declared safe area.
        """
        block_height = self.lane_height
        if lane_index < self.ad_lanes:
            return self.top_margin + lane_index * (self.lane_height +
                                                    self.lane_gap)
        # dialogue / overflow lanes anchored at the bottom safe edge
        reverse = self.total_lanes - 1 - lane_index
        return (self.height - self.bottom_margin - block_height
                - reverse * (self.lane_height + self.lane_gap))

    def block_height(self) -> int:
        """Single-line slot pitch (geometry is independent of line count)."""
        return self.lane_height

    def cue_y(self, lane_index: int, line_count: int) -> int:
        """Top Y of the rendered cue; bottom-aligned then safe-area clamped."""
        slot_top = self.lane_y(lane_index)
        if lane_index < self.ad_lanes:
            return max(self.top_margin, slot_top)
        rendered_height = line_count * self.lane_height
        candidate = slot_top + self.lane_height - rendered_height
        return max(self.top_margin, candidate)


@dataclass(frozen=True)
class Arbitration:
    item_id: str
    winner_id: str
    reason: str
    declared_lane: int
    assigned_lane: int
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "winner_id": self.winner_id,
            "reason": self.reason,
            "declared_lane": self.declared_lane,
            "assigned_lane": self.assigned_lane,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


@dataclass(frozen=True)
class LaidOutCue:
    item: SubtitleItem
    lane_index: int
    x: int
    y: int
    width: int
    side_margin: int
    wrap: Any  # WrapResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "lane_index": self.lane_index,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "side_margin": self.side_margin,
            "wrap": self.wrap.to_dict(),
        }


@dataclass(frozen=True)
class LayoutResult:
    cues: tuple[LaidOutCue, ...]
    arbitrations: tuple[Arbitration, ...]
    config: LayoutConfig
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "config": self.config.__dict__,
            "cues": [cue.to_dict() for cue in self.cues],
            "arbitrations": [arb.to_dict() for arb in self.arbitrations],
        }
        payload["content_hash"] = content_hash(payload)
        return payload


def ordering_key(item: SubtitleItem) -> tuple:
    return (item.priority, item.start_frame, item.end_frame, item.item_id)


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _declared_band(item: SubtitleItem, config: LayoutConfig) -> range:
    if item.is_ad:
        return range(0, config.ad_lanes)
    return range(config.ad_lanes, config.ad_lanes + config.dialogue_lanes)


def _overflow_band(item: SubtitleItem, config: LayoutConfig) -> range:
    base = config.ad_lanes + config.dialogue_lanes
    if item.is_ad:
        return range(base, base + config.overflow_lanes)
    return range(base, base + config.overflow_lanes)


def _ad_yields(item: SubtitleItem, occupant: SubtitleItem) -> bool:
    """Fixed yielding rule for cross-type contention on overflow.

    Dialogue keeps its band; AD yields to any dialogue. Within the same type
    the later-placed cue (lower in the total order) yields.
    """
    if item.is_ad and not occupant.is_ad:
        return True
    if not item.is_ad and occupant.is_ad:
        return False
    return ordering_key(item) > ordering_key(occupant)


def solve_layout(items: Iterable[SubtitleItem],
                 config: LayoutConfig | None = None) -> LayoutResult:
    config = config or LayoutConfig()
    ordered = sorted(items, key=ordering_key)
    total_frames_span = max((it.end_frame for it in ordered), default=0)
    # occupants[lane] = list of (start, end, item)
    occupants: list[list[tuple[int, int, SubtitleItem]]] = [
        [] for _ in range(config.total_lanes)
    ]
    laid: list[LaidOutCue] = []
    arbitrations: list[Arbitration] = []

    def lane_free(lane: int, start: int, end: int) -> SubtitleItem | None:
        for occ_start, occ_end, occupant in occupants[lane]:
            if overlaps(start, end, occ_start, occ_end):
                return occupant
        return None

    for item in ordered:
        band = _declared_band(item, config)
        chosen: int | None = None
        winner: SubtitleItem | None = None
        for lane in band:
            if lane_free(lane, item.start_frame, item.end_frame) is None:
                chosen = lane
                break
        if chosen is None:
            # Declared band full: apply the fixed yielding rule, then overflow.
            blockers = [
                occupant for lane in band
                for _, _, occupant in occupants[lane]
                if overlaps(item.start_frame, item.end_frame,
                            occupant.start_frame, occupant.end_frame)
            ]
            declared_lane = band[0]
            non_yielding = [occ for occ in blockers
                            if not _ad_yields(item, occ)]
            if non_yielding:
                winner = sorted(non_yielding, key=ordering_key)[0]
            for lane in _overflow_band(item, config):
                if lane_free(lane, item.start_frame, item.end_frame) is None:
                    chosen = lane
                    break
            if chosen is None:
                # Whole physical safe area exhausted: hard failure, no drop.
                raise TrackConflictError(
                    "no lane available inside the safe area",
                    conflict_item_id=(winner.item_id if winner else
                                      blockers[0].item_id))
            reason = ("ad_yields_to_dialogue"
                      if item.is_ad and winner is not None and not winner.is_ad
                      else "declared_band_full_overflow")
            arbitrations.append(Arbitration(
                item_id=item.item_id,
                winner_id=winner.item_id if winner else blockers[0].item_id,
                reason=reason,
                declared_lane=declared_lane,
                assigned_lane=chosen,
                start_frame=item.start_frame,
                end_frame=item.end_frame,
            ))

        wrap = wrap_text(item.text, safe_width=config.width,
                         side_margin=config.side_margin,
                         max_lines=config.max_lines,
                         font_tiers=config.font_tiers)
        y_pos = config.cue_y(chosen, len(wrap.lines))
        laid.append(LaidOutCue(
            item=item, lane_index=chosen, x=config.side_margin, y=y_pos,
            width=config.width - 2 * config.side_margin,
            side_margin=config.side_margin, wrap=wrap,
        ))
        occupants[chosen].append((item.start_frame, item.end_frame, item))

    cues = tuple(sorted(laid, key=lambda cue: cue.item.item_id))
    result = LayoutResult(cues=cues, arbitrations=tuple(arbitrations),
                          config=config, content_hash="")
    return LayoutResult(cues=cues, arbitrations=tuple(arbitrations),
                        config=config,
                        content_hash=content_hash(_result_payload(result)))


def _result_payload(result: LayoutResult) -> dict[str, Any]:
    return {
        "config": result.config.__dict__,
        "cues": [cue.to_dict() for cue in result.cues],
        "arbitrations": [arb.to_dict() for arb in result.arbitrations],
    }


def validate_drag(item: SubtitleItem, target_lane: int,
                  new_start: int, new_end: int,
                  committed: Iterable[SubtitleItem],
                  config: LayoutConfig | None = None) -> None:
    """Deterministic validation used by both UI drags and headless tests.

    Raises :class:`DragBoundsError`, :class:`TimeInvalidError` or
    :class:`TrackConflictError`; callers keep the previous state untouched.
    """
    config = config or LayoutConfig()
    if new_start >= new_end or new_start < 0 or new_end <= 0:
        raise TimeInvalidError(
            f"invalid interval [{new_start}, {new_end})",
            conflict_item_id=item.item_id)
    if not 0 <= target_lane < config.total_lanes:
        raise DragBoundsError(
            f"lane {target_lane} outside safe area "
            f"(0..{config.total_lanes - 1})",
            conflict_item_id=item.item_id)

    candidate = SubtitleItem(item.item_id, new_start, new_end, item.text,
                             item.speaker, item.language, item.priority,
                             item.is_ad)
    for other in sorted(committed, key=ordering_key):
        if other.item_id == item.item_id:
            continue
        if not overlaps(new_start, new_end, other.start_frame, other.end_frame):
            continue
        # Same lane clash: only allowed when the dragged cue wins the rule.
        if _hard_conflict(candidate, other, target_lane):
            raise TrackConflictError(
                f"unresolvable overlap with {other.item_id} on lane "
                f"{target_lane}",
                conflict_item_id=other.item_id)


def _hard_conflict(candidate: SubtitleItem, occupant: SubtitleItem,
                   target_lane: int) -> bool:
    """Conflict only when the dragged cue cannot lawfully take that lane."""
    same_band = (candidate.is_ad == occupant.is_ad)
    if not same_band:
        # Cross-type on a shared lane: AD must yield to higher-priority
        # dialogue; dialogue never yields to AD on an explicit drag.
        if candidate.is_ad and not occupant.is_ad:
            return occupant.priority <= candidate.priority
        return not candidate.is_ad and occupant.is_ad
    return ordering_key(candidate) > ordering_key(occupant)
