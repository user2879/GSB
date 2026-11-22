"""NumPy-backed read-only analytics for the diagnostic UI and reports.

Nothing in this module participates in scheduling decisions: every binding and
error bound is computed with pure Python integers. NumPy only summarizes the
already-decided integer results (histograms, means, gap tables).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .models import Decision, NodeScore


def score_matrix(decision: Decision) -> np.ndarray:
    """Return an int64 matrix (nodes x plugins) of raw plugin scores."""
    if not decision.node_scores:
        return np.zeros((0, 0), dtype=np.int64)
    return np.array(
        [[plugin.raw_score for plugin in score.plugins] for score in decision.node_scores],
        dtype=np.int64,
    )


def plugin_averages(decisions: Sequence[Decision]) -> Dict[str, float]:
    """Mean raw plugin score across every scored node and pod."""
    blocks = [score_matrix(decision) for decision in decisions if decision.feasible]
    blocks = [block for block in blocks if block.size]
    if not blocks:
        return {}
    matrix = np.vstack(blocks)
    from .constants import PLUGIN_ORDER

    return {
        plugin: float(matrix[:, index].mean())
        for index, plugin in enumerate(PLUGIN_ORDER)
    }


def gap_table(decisions: Sequence[Decision]) -> List[Dict[str, int]]:
    """Deterministic summary of winner/runner-up gaps and combined error."""
    table: List[Dict[str, int]] = []
    for decision in decisions:
        if decision.runner_up is None:
            continue
        table.append(
            {
                "pod_id": decision.pod_id,
                "winner": decision.bound_node,
                "runner_up": decision.runner_up,
                "gap_mp": int(decision.score_gap_mp or 0),
                "combined_error_mp": int(decision.combined_error_mp or 0),
                "error_influenced": int(decision.error_influenced),
            }
        )
    return table


def error_bound_histogram(node_scores: Sequence[NodeScore]) -> Dict[int, int]:
    histogram: Dict[int, int] = {}
    for score in node_scores:
        histogram[score.error_bound_points] = histogram.get(score.error_bound_points, 0) + 1
    return dict(sorted(histogram.items()))
