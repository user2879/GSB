from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .model import LoopOrder


MERGE_SCHEDULES: dict[LoopOrder, str] = {
    "ijk": "k-chunk-forward-sequential",
    "ikj": "k-chunk-reverse-sequential",
    "jik": "k-chunk-balanced-tree",
}


def merge_partial_arrays(
    partials: Sequence[np.ndarray],
    loop_order: LoopOrder,
) -> np.ndarray:
    if not partials:
        raise ValueError("At least one partial matrix is required.")
    if len(partials) == 1:
        return partials[0].copy()
    if loop_order in ("ijk", "ikj"):
        indices = range(len(partials)) if loop_order == "ijk" else range(len(partials) - 1, -1, -1)
        iterator = iter(indices)
        accumulated = partials[next(iterator)].copy()
        for index in iterator:
            accumulated = np.float32(accumulated + partials[index])
        return accumulated

    nodes = [partial.copy() for partial in partials]
    while len(nodes) > 1:
        next_nodes: list[np.ndarray] = []
        for position in range(0, len(nodes), 2):
            if position + 1 >= len(nodes):
                next_nodes.append(nodes[position])
            else:
                next_nodes.append(np.float32(nodes[position] + nodes[position + 1]))
        nodes = next_nodes
    return nodes[0]


def merge_rounding_for_element(
    partial_values: Sequence[np.float32],
    loop_order: LoopOrder,
) -> tuple[np.float32, tuple[dict[str, Any], ...]]:
    """Scalar replay used by diagnostics; mirrors merge_partial_arrays exactly."""
    schedule = MERGE_SCHEDULES[loop_order]
    events: list[dict[str, Any]] = []
    if not partial_values:
        raise ValueError("At least one partial value is required.")
    if len(partial_values) == 1:
        return np.float32(partial_values[0]), ()

    if loop_order in ("ijk", "ikj"):
        indices = list(range(len(partial_values))) if loop_order == "ijk" else list(
            reversed(range(len(partial_values)))
        )
        accumulated = np.float32(partial_values[indices[0]])
        accumulated_chunks = (indices[0],)
        for level, index in enumerate(indices[1:], start=1):
            right = np.float32(partial_values[index])
            merged = np.float32(accumulated + right)
            rounding = float(merged) - (float(accumulated) + float(right))
            events.append(
                {
                    "stage": "chunk_merge",
                    "merge_schedule": schedule,
                    "merge_level": level,
                    "left_chunks": list(accumulated_chunks),
                    "right_chunks": [index],
                    "left_value": float(accumulated),
                    "right_value": float(right),
                    "result_value": float(merged),
                    "local_merge_rounding": rounding,
                }
            )
            accumulated = merged
            accumulated_chunks = accumulated_chunks + (index,)
        return accumulated, tuple(events)

    nodes: list[tuple[tuple[int, ...], np.float32]] = [
        ((index,), np.float32(value)) for index, value in enumerate(partial_values)
    ]
    level = 0
    while len(nodes) > 1:
        next_nodes: list[tuple[tuple[int, ...], np.float32]] = []
        for position in range(0, len(nodes), 2):
            left_chunks, left = nodes[position]
            if position + 1 >= len(nodes):
                next_nodes.append((left_chunks, left))
                continue
            right_chunks, right = nodes[position + 1]
            merged = np.float32(left + right)
            rounding = float(merged) - (float(left) + float(right))
            events.append(
                {
                    "stage": "chunk_merge",
                    "merge_schedule": schedule,
                    "merge_level": level,
                    "left_chunks": list(left_chunks),
                    "right_chunks": list(right_chunks),
                    "left_value": float(left),
                    "right_value": float(right),
                    "result_value": float(merged),
                    "local_merge_rounding": rounding,
                }
            )
            next_nodes.append((left_chunks + right_chunks, merged))
        nodes = next_nodes
        level += 1
    return nodes[0][1], tuple(events)
