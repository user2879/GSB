from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .errors import (
    E_LABEL_LENGTH_MISMATCH,
    E_MAX_ITERATIONS_EXCEEDED,
    E_NONFINITE_SCORE,
    E_PARAM_SHAPE_MISMATCH,
    WorkbenchError,
)


@dataclass(frozen=True)
class DecodeResult:
    labels: tuple[str, ...]
    path_score: int
    best_scores: tuple[tuple[int, ...], ...]
    backpointers: tuple[tuple[int, ...], ...]


def viterbi_decode(
    emissions: Sequence[Sequence[int]],
    label_names: Sequence[str],
    transition_weights: Sequence[Sequence[int]],
    start_weights: Sequence[int],
    *,
    expected_length: int | None = None,
    max_iterations: int = 512,
) -> DecodeResult:
    if expected_length is not None and len(emissions) != expected_length:
        raise WorkbenchError(
            E_LABEL_LENGTH_MISMATCH,
            "decode",
            f"label sequence length {expected_length} != emission length {len(emissions)}",
        )
    if not emissions:
        raise WorkbenchError(E_LABEL_LENGTH_MISMATCH, "decode", "cannot decode an empty sequence")
    label_count = len(label_names)
    if len(transition_weights) != label_count or any(len(row) != label_count for row in transition_weights):
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "decode", "transition matrix shape does not match labels")
    if len(start_weights) != label_count:
        raise WorkbenchError(E_PARAM_SHAPE_MISMATCH, "decode", "start weights shape does not match labels")
    for position, row in enumerate(emissions):
        if len(row) != label_count:
            raise WorkbenchError(
                E_PARAM_SHAPE_MISMATCH,
                "decode",
                f"emission row {position} has {len(row)} labels, expected {label_count}",
            )
        for value in row:
            if not isinstance(value, int) or not math.isfinite(value):
                raise WorkbenchError(
                    E_NONFINITE_SCORE,
                    "decode",
                    f"non-finite emission score at position {position}",
                )

    positions = len(emissions)
    if positions > int(max_iterations):
        raise WorkbenchError(
            E_MAX_ITERATIONS_EXCEEDED,
            "decode",
            f"forward iterations {positions} exceed max_iterations {max_iterations}",
        )

    scores: list[list[int]] = []
    backpointers: list[list[int]] = []
    first = [int(start_weights[label]) + int(emissions[0][label]) for label in range(label_count)]
    scores.append(first)
    backpointers.append([-1] * label_count)

    for position in range(1, positions):
        previous_scores = scores[position - 1]
        current_scores: list[int] = []
        current_backpointers: list[int] = []
        for current_label in range(label_count):
            best_previous = 0
            best_score = previous_scores[0] + int(transition_weights[0][current_label])
            for previous_label in range(1, label_count):
                candidate = previous_scores[previous_label] + int(
                    transition_weights[previous_label][current_label]
                )
                if candidate > best_score or (candidate == best_score and previous_label < best_previous):
                    best_score = candidate
                    best_previous = previous_label
            current_scores.append(best_score + int(emissions[position][current_label]))
            current_backpointers.append(best_previous)
        scores.append(current_scores)
        backpointers.append(current_backpointers)

    final_scores = scores[-1]
    final_label = 0
    final_score = final_scores[0]
    for label in range(1, label_count):
        if final_scores[label] > final_score:
            final_score = final_scores[label]
            final_label = label

    path = [final_label]
    for position in range(positions - 1, 0, -1):
        path.append(backpointers[position][path[-1]])
    path.reverse()
    selected_labels = tuple(label_names[index] for index in path)
    return DecodeResult(
        labels=selected_labels,
        path_score=int(final_score),
        best_scores=tuple(tuple(row) for row in scores),
        backpointers=tuple(tuple(row) for row in backpointers),
    )


def verify_path_score(
    labels: Sequence[str],
    label_names: Sequence[str],
    emissions: Sequence[Sequence[int]],
    transition_weights: Sequence[Sequence[int]],
    start_weights: Sequence[int],
) -> int:
    label_to_index = {label: index for index, label in enumerate(label_names)}
    indices = [label_to_index[label] for label in labels]
    total = int(start_weights[indices[0]]) + int(emissions[0][indices[0]])
    for position in range(1, len(indices)):
        total += int(transition_weights[indices[position - 1]][indices[position]])
        total += int(emissions[position][indices[position]])
    return total
