"""Deterministic log-domain BIESO Viterbi decoder."""

from __future__ import annotations

from typing import Any

from .labels import LABELS


def viterbi_decode(emission: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    scores = emission["scores"]
    length = len(scores)
    label_count = len(LABELS)
    transitions = package["transition_weights"]["matrix"]
    start_weights = package["transition_weights"]["start"]
    end_weights = package["transition_weights"]["end"]

    forward: list[list[float]] = [[0.0] * label_count for _ in range(length)]
    candidates: list[list[float]] = [[0.0] * label_count for _ in range(length)]
    backpointers: list[list[int]] = [[-1] * label_count for _ in range(length)]
    predecessor_ties: list[dict[str, Any]] = []

    for position in range(length):
        for label_index in range(label_count):
            emission_score = float(scores[position][label_index])
            if position == 0:
                value = float(start_weights[label_index]) + emission_score
                forward[position][label_index] = value
                candidates[position][label_index] = value
                continue

            best_value: float | None = None
            best_previous = -1
            tied_previous: list[int] = []
            for previous_index in range(label_count):
                transition_score = float(transitions[previous_index][label_index])
                candidate = forward[position - 1][previous_index] + transition_score
                if best_value is None or candidate > best_value:
                    best_value = candidate
                    best_previous = previous_index
                    tied_previous = [previous_index]
                elif candidate == best_value:
                    tied_previous.append(previous_index)

            assert best_value is not None
            if len(tied_previous) > 1:
                predecessor_ties.append(
                    {
                        "position": position,
                        "target_label": LABELS[label_index],
                        "candidate_predecessors": [LABELS[index] for index in tied_previous],
                        "selected_predecessor": LABELS[best_previous],
                        "decision": "predecessor_label_code_ascending",
                    }
                )
            value = best_value + emission_score
            forward[position][label_index] = value
            candidates[position][label_index] = value
            backpointers[position][label_index] = best_previous

    tie_positions: list[dict[str, Any]] = []
    for position in range(length):
        best_score: float | None = None
        tied_labels: list[int] = []
        for label_index in range(label_count):
            value = candidates[position][label_index]
            if best_score is None or value > best_score:
                best_score = value
                tied_labels = [label_index]
            elif value == best_score:
                tied_labels.append(label_index)
        if len(tied_labels) > 1:
            selected = tied_labels[0]
            tie_positions.append(
                {
                    "position": position,
                    "tied_labels": [LABELS[index] for index in tied_labels],
                    "selected_label": LABELS[selected],
                    "decision": "label_code_ascending_then_position_ascending",
                }
            )

    best_total: float | None = None
    last_label = -1
    final_tied: list[int] = []
    last_position = length - 1
    for label_index in range(label_count):
        total = forward[last_position][label_index] + float(end_weights[label_index])
        if best_total is None or total > best_total:
            best_total = total
            last_label = label_index
            final_tied = [label_index]
        elif total == best_total:
            final_tied.append(label_index)
    if len(final_tied) > 1:
        tie_positions.append(
            {
                "position": last_position,
                "tied_labels": [LABELS[index] for index in final_tied],
                "selected_label": LABELS[final_tied[0]],
                "decision": "final_label_code_ascending",
            }
        )

    label_codes = [0] * length
    label_codes[last_position] = last_label
    for position in range(length - 1, 0, -1):
        previous = backpointers[position][label_codes[position]]
        label_codes[position - 1] = previous
    decoded_labels = [LABELS[index] for index in label_codes]

    return {
        "sample_id": emission["sample_id"],
        "labels": decoded_labels,
        "label_codes": label_codes,
        "path_total": best_total,
        "candidate_scores": candidates,
        "tie_positions": tie_positions,
        "predecessor_ties": predecessor_ties,
        "accumulation_order": "position_asc,target_label_code_asc,predecessor_label_code_asc",
    }
