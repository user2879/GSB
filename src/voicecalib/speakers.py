"""说话人归属追踪与轨迹重建。

归属依据校准轮的基频证据分量与配置锚点距离（含相邻档位容差），
同分按说话人标识字典序决断；轨迹按最短片段与容许空档轮数断开。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CalibConfig
from .calibration import CalibrationRow
from .evidence import StrengthArtifact
from .hashing import short_key


@dataclass
class AttributionRow:
    round: int
    speaker: str
    candidates: list[str]
    distance_hz: dict[str, float]
    segment_id: str | None
    break_reason: str | None
    calib_round: int
    f0_hz: float
    evidence_component: str
    cache_key: str

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "speaker": self.speaker,
            "candidates": self.candidates,
            "distance_hz": self.distance_hz,
            "segment_id": self.segment_id,
            "break_reason": self.break_reason,
            "calib_round": self.calib_round,
            "f0_hz": self.f0_hz,
            "evidence_component": self.evidence_component,
            "cache_key": self.cache_key,
        }


@dataclass
class Trajectory:
    segment_id: str
    speaker: str
    rounds: list[int]
    start_round: int
    end_round: int
    break_reason: str | None
    short_segment: bool

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "speaker": self.speaker,
            "rounds": self.rounds,
            "start_round": self.start_round,
            "end_round": self.end_round,
            "break_reason": self.break_reason,
            "short_segment": self.short_segment,
        }


def attribute_round(round_index: int, strength: StrengthArtifact,
                    calib: CalibrationRow, config: CalibConfig) -> AttributionRow:
    f0_hz = float(strength.feature_vector[0])
    candidates = sorted(config.speaker_anchors_hz.keys())
    distance: dict[str, float] = {}
    for speaker in candidates:  # 字典序固定
        distance[speaker] = abs(f0_hz - config.speaker_anchors_hz[speaker])
    best_distance = min(distance.values())
    # 容差内视为同分，同分集合按字典序取第一个。
    tied = sorted(
        speaker for speaker, value in distance.items()
        if value <= best_distance + config.adjacent_tolerance
    )
    chosen = tied[0]
    cache_key = short_key(
        "attribution",
        calib.cache_key,
        f"{f0_hz:.6f}",
        config.adjacent_tolerance,
        dict(sorted(config.speaker_anchors_hz.items())),
    )
    return AttributionRow(
        round=round_index,
        speaker=chosen,
        candidates=tied,
        distance_hz=distance,
        segment_id=None,
        break_reason=None,
        calib_round=calib.round,
        f0_hz=f0_hz,
        evidence_component="features.f0_mean_hz",
        cache_key=cache_key,
    )


def build_trajectories(attributions: list[AttributionRow],
                       config: CalibConfig) -> tuple[list[AttributionRow], list[Trajectory]]:
    """串联归属轨迹：空档超 max_gap_rounds 断开；短片段标注 SHORT_SEGMENT。"""
    by_round = {row.round: row for row in attributions}
    trajectories: list[Trajectory] = []
    segment_index = 0
    current_speaker: str | None = None
    current_rounds: list[int] = []
    last_round: int | None = None

    def flush(reason: str | None) -> None:
        nonlocal segment_index, current_speaker, current_rounds, last_round
        if not current_rounds:
            return
        segment_index += 1
        segment_id = f"SEG-{segment_index:03d}"
        short = len(current_rounds) < config.min_segment_rounds
        break_reason = "SHORT_SEGMENT" if short else reason
        trajectory = Trajectory(
            segment_id=segment_id,
            speaker=current_speaker,
            rounds=list(current_rounds),
            start_round=current_rounds[0],
            end_round=current_rounds[-1],
            break_reason=break_reason,
            short_segment=short,
        )
        trajectories.append(trajectory)
        for rnd in current_rounds:
            by_round[rnd].segment_id = segment_id
        if break_reason:
            by_round[current_rounds[-1]].break_reason = break_reason
        current_speaker = None
        current_rounds = []
        last_round = None

    for row in attributions:  # 已按轮次升序
        if current_speaker is None:
            current_speaker = row.speaker
            current_rounds = [row.round]
            last_round = row.round
            continue
        gap = row.round - last_round - 1
        if row.speaker == current_speaker and gap <= config.max_gap_rounds:
            current_rounds.append(row.round)
            last_round = row.round
            continue
        reason = "GAP_EXCEEDED" if gap > config.max_gap_rounds else "SPEAKER_SWITCH"
        flush(reason)
        current_speaker = row.speaker
        current_rounds = [row.round]
        last_round = row.round
    flush(None)
    return attributions, trajectories


def trajectory_session_key(config: CalibConfig) -> str:
    return short_key(
        "trajectory-session",
        config.min_segment_rounds,
        config.max_gap_rounds,
        config.adjacent_tolerance,
        dict(sorted(config.speaker_anchors_hz.items())),
    )
