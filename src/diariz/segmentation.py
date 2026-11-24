"""Frame-level speaker segmentation with overlapping window arbitration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .clustering import ClusteringResult
from .config import DiarConfig, SAMPLE_RATE
from .features import WindowInfo
from .quantize import q16_to_float


@dataclass
class FrameVote:
    frame_index: int
    start_sample: int
    end_sample: int
    winner_window: int
    speaker: str
    margin_q16: int
    participating_windows: Tuple[int, ...]
    tie_broken: bool


@dataclass
class Segment:
    index: int
    speaker: str
    start_ms: float
    end_ms: float
    start_sample: int
    end_sample: int
    avg_similarity: float
    arbitration_windows: Tuple[int, ...]
    boundary_basis: str
    merged_from: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class SegmentationResult:
    frame_votes: List[FrameVote]
    segments: List[Segment]
    dropped_short: List[Dict]
    merged_gaps: List[Dict]


def _frame_bounds(cfg: DiarConfig, num_frames: int) -> List[Tuple[int, int]]:
    frame_len = int(round(cfg.frame_ms * SAMPLE_RATE / 1000.0))
    hop = int(round(cfg.frame_hop_ms * SAMPLE_RATE / 1000.0))
    return [(i * hop, i * hop + frame_len) for i in range(num_frames)]


def vote_frames(windows: Sequence[WindowInfo], frame_count: int,
                clustering: ClusteringResult, cfg: DiarConfig) -> List[FrameVote]:
    bounds = _frame_bounds(cfg, frame_count)
    cover: Dict[int, List[int]] = {f: [] for f in range(frame_count)}
    for window in windows:
        for frame in window.frame_indices:
            if frame < frame_count:
                cover[frame].append(window.index)

    votes: List[FrameVote] = []
    for frame in range(frame_count):
        owners = sorted(cover[frame])
        if not owners:
            continue
        # Highest similarity margin wins; exact ties resolve by window index.
        ranked = sorted(owners,
                        key=lambda wid: (-clustering.window_margin_q16[wid], wid))
        winner = ranked[0]
        tie = any(clustering.window_margin_q16[wid]
                  == clustering.window_margin_q16[winner] for wid in ranked[1:])
        start, end = bounds[frame]
        votes.append(FrameVote(
            frame_index=frame, start_sample=start, end_sample=end,
            winner_window=winner, speaker=clustering.assignments[winner],
            margin_q16=clustering.window_margin_q16[winner],
            participating_windows=tuple(owners), tie_broken=tie,
        ))
    return votes


def _boundary_position(rule: str, left_vote: FrameVote,
                       right_vote: FrameVote, left_win: WindowInfo,
                       right_win: WindowInfo) -> Tuple[int, str]:
    if rule == "frame_edge":
        return right_vote.start_sample, "frame_edge:right_frame_start"
    if rule == "window_midpoint":
        midpoint = (left_win.end_sample + right_win.start_sample) // 2
        return midpoint, "window_midpoint:gap_midpoint"
    # margin_weighted: weighted average of the two frame starts.
    lm = max(0, left_vote.margin_q16)
    rm = max(0, right_vote.margin_q16)
    denom = lm + rm
    if denom == 0:
        point = (left_vote.end_sample + right_vote.start_sample) // 2
        basis = "margin_weighted:zero_margin_fallback"
    else:
        point = (left_vote.end_sample * rm + right_vote.start_sample * lm) // denom
        basis = f"margin_weighted:lm={lm},rm={rm}"
    return point, basis


def _runs(votes: List[FrameVote]) -> List[List[FrameVote]]:
    runs: List[List[FrameVote]] = []
    for vote in votes:
        if runs and runs[-1][-1].speaker == vote.speaker \
                and vote.frame_index == runs[-1][-1].frame_index + 1:
            runs[-1].append(vote)
        else:
            runs.append([vote])
    return runs


def build_segments(votes: List[FrameVote], windows: Sequence[WindowInfo],
                   clustering: ClusteringResult, cfg: DiarConfig,
                   total_samples: int) -> SegmentationResult:
    window_by_index = {w.index: w for w in windows}
    raw_runs = _runs(votes)
    min_segment_samples = cfg.min_segment_ms * SAMPLE_RATE // 1000
    min_silence_samples = cfg.min_silence_ms * SAMPLE_RATE // 1000

    kept: List[List[FrameVote]] = []
    dropped: List[Dict] = []
    for run in raw_runs:
        length = run[-1].end_sample - run[0].start_sample
        if length < min_segment_samples:
            dropped.append({
                "speaker": run[0].speaker,
                "start_ms": round(run[0].start_sample * 1000.0 / SAMPLE_RATE, 3),
                "end_ms": round(run[-1].end_sample * 1000.0 / SAMPLE_RATE, 3),
                "frames": [v.frame_index for v in run],
            })
            continue
        kept.append(run)

    merged_gaps: List[Dict] = []
    merged: List[List[FrameVote]] = []
    for run in kept:
        if merged and merged[-1][-1].speaker == run[0].speaker:
            gap = run[0].start_sample - merged[-1][-1].end_sample
            if gap <= min_silence_samples:
                merged_gaps.append({
                    "speaker": run[0].speaker,
                    "gap_ms": round(gap * 1000.0 / SAMPLE_RATE, 3),
                    "left_frame": merged[-1][-1].frame_index,
                    "right_frame": run[0].frame_index,
                })
                merged[-1].extend(run)
                continue
        merged.append(list(run))

    segments: List[Segment] = []
    for index, run in enumerate(merged):
        first_vote, last_vote = run[0], run[-1]
        start_sample = first_vote.start_sample
        end_sample = min(total_samples, last_vote.end_sample)
        basis_parts = ["run_frame_edges"]
        if segments:
            prev = merged[index - 1]
            left_vote, right_vote = prev[-1], first_vote
            point, basis = _boundary_position(
                cfg.boundary_rule, left_vote, right_vote,
                window_by_index[left_vote.winner_window],
                window_by_index[right_vote.winner_window])
            start_sample = point
            basis_parts = [basis]
        participating = tuple(sorted({
            wid for vote in run for wid in vote.participating_windows
        }))
        sims = [q16_to_float(clustering.window_similarity[v.winner_window])
                for v in run]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        segments.append(Segment(
            index=index, speaker=first_vote.speaker,
            start_ms=round(start_sample * 1000.0 / SAMPLE_RATE, 3),
            end_ms=round(end_sample * 1000.0 / SAMPLE_RATE, 3),
            start_sample=start_sample, end_sample=end_sample,
            avg_similarity=round(avg_sim, 6),
            arbitration_windows=participating,
            boundary_basis=";".join(basis_parts),
            merged_from=tuple(v.frame_index for v in run),
        ))
    return SegmentationResult(frame_votes=votes, segments=segments,
                              dropped_short=dropped, merged_gaps=merged_gaps)
