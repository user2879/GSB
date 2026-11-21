"""Stable hotspot ranking and caller/callee context localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .folding import FoldResult


@dataclass(frozen=True)
class FrameRank:
    rank: int                 # 1-based
    frame: str
    self_weight: int
    cumulative_weight: int


@dataclass(frozen=True)
class ContextEdge:
    """An aggregated direct caller -> frame or frame -> callee relation."""

    counterpart: str
    edge_weight: int          # folded sample weights crossing this edge
    call_count: int           # number of folded stack paths crossing it


@dataclass(frozen=True)
class FrameContext:
    frame: str
    callers: Tuple[ContextEdge, ...]
    callees: Tuple[ContextEdge, ...]


def rank_frames(result: FoldResult, top_k: int) -> List[FrameRank]:
    """Rank frames by (-self, -cumulative, frame name ascending).

    The name tie-break makes the ordering total and therefore independent of
    insertion order and of the weight distribution.
    """

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    ordered = sorted(
        result.frame_metrics,
        key=lambda metric: (
            -metric.self_weight,
            -metric.cumulative_weight,
            metric.frame,
        ),
    )
    return [
        FrameRank(
            rank=index,
            frame=metric.frame,
            self_weight=metric.self_weight,
            cumulative_weight=metric.cumulative_weight,
        )
        for index, metric in enumerate(ordered[:top_k], start=1)
    ]


def _edge_sort_key(edge: ContextEdge) -> Tuple[object, ...]:
    # Higher aggregated weight first, then more paths, then name ascending.
    return (-edge.edge_weight, -edge.call_count, edge.counterpart)


def build_contexts(result: FoldResult,
                   ranked: Sequence[FrameRank]) -> Dict[str, FrameContext]:
    """Aggregate direct caller/callee weights for every ranked frame.

    Edges are read from folded stack paths only, so the result is independent
    of original sample ordering.
    """

    wanted = {entry.frame for entry in ranked}
    caller_weight: Dict[str, Dict[str, int]] = {}
    caller_paths: Dict[str, Dict[str, int]] = {}
    callee_weight: Dict[str, Dict[str, int]] = {}
    callee_paths: Dict[str, Dict[str, int]] = {}

    for stack in result.stacks:
        frames = stack.frames
        weight = stack.weight
        for index, frame in enumerate(frames):
            if frame not in wanted:
                continue
            if index > 0:
                caller = frames[index - 1]
                caller_weight.setdefault(frame, {})
                caller_weight[frame][caller] = \
                    caller_weight[frame].get(caller, 0) + weight
                caller_paths.setdefault(frame, {})
                caller_paths[frame][caller] = \
                    caller_paths[frame].get(caller, 0) + 1
            if index < len(frames) - 1:
                callee = frames[index + 1]
                callee_weight.setdefault(frame, {})
                callee_weight[frame][callee] = \
                    callee_weight[frame].get(callee, 0) + weight
                callee_paths.setdefault(frame, {})
                callee_paths[frame][callee] = \
                    callee_paths[frame].get(callee, 0) + 1

    contexts: Dict[str, FrameContext] = {}
    for frame in wanted:
        callers = tuple(
            sorted(
                (
                    ContextEdge(
                        counterpart=name,
                        edge_weight=caller_weight.get(frame, {})
                        .get(name, 0),
                        call_count=caller_paths.get(frame, {})
                        .get(name, 0),
                    )
                    for name in caller_weight.get(frame, {})
                ),
                key=_edge_sort_key,
            )
        )
        callees = tuple(
            sorted(
                (
                    ContextEdge(
                        counterpart=name,
                        edge_weight=callee_weight.get(frame, {})
                        .get(name, 0),
                        call_count=callee_paths.get(frame, {})
                        .get(name, 0),
                    )
                    for name in callee_weight.get(frame, {})
                ),
                key=_edge_sort_key,
            )
        )
        contexts[frame] = FrameContext(
            frame=frame, callers=callers, callees=callees)
    return contexts
