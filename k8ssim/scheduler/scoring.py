"""Weighted composition, propagated error bounds and stable tie breaking."""
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

from ..constants import PLUGIN_ORDER, WEIGHT_BASIS
from ..models import Decision, NodeScore, PluginScore, Rejection
from .plugins import PluginOutput

TIE_UNIQUE = "unique"
TIE_EXACT = "exact_tie"
TIE_ERROR_BOUND = "error_bound_tie"
UNSCHEDULABLE = "unschedulable"


def compose_node_score(
    node_name: str,
    outputs: Mapping[str, PluginOutput],
) -> NodeScore:
    """Combine plugin outputs for one node using integer arithmetic only."""
    plugins: List[PluginScore] = []
    total_mp = 0
    propagated_error_mp = 0
    for plugin in PLUGIN_ORDER:
        output = outputs[plugin]
        total_mp += output.contribution_mp
        # The plugin error is expressed in micro-points of one plugin point.
        # Weighted propagation scales it by |weight| / WEIGHT_BASIS with an
        # integer ceil so the propagated bound stays rigorous.
        propagated_error_mp += -(-(abs(output.weight) * output.quant_error_mp)
                                 // WEIGHT_BASIS)
        plugins.append(
            PluginScore(
                plugin=output.plugin,
                raw_score=output.raw_score,
                weight=output.weight,
                contribution_mp=output.contribution_mp,
                truncation=output.truncation,
                quant_error_mp=output.quant_error_mp,
            )
        )

    # Contribution micro-points use MICRO per plugin point, and weights are
    # basis points. The displayed 0..100 total therefore divides by WEIGHT_BASIS
    # in one final floor operation; its remainder is the composition truncation
    # error, strictly below one weight-basis unit expressed in micro-points.
    total_points = total_mp // WEIGHT_BASIS
    comp_trunc_error_mp = total_mp % WEIGHT_BASIS
    error_bound_mp = propagated_error_mp + comp_trunc_error_mp
    error_bound_points = -(-error_bound_mp // WEIGHT_BASIS)
    return NodeScore(
        node=node_name,
        plugins=tuple(plugins),
        total_mp=total_mp,
        total_points=total_points,
        comp_trunc_error_mp=comp_trunc_error_mp,
        error_bound_mp=error_bound_mp,
        error_bound_points=error_bound_points,
    )


def _sort_key(node_score: NodeScore):
    # Deterministic ordering: higher total first, then plugin vector, then name.
    return (
        -node_score.total_mp,
        tuple(-value for value in node_score.vector()),
        node_score.node,
    )


def select_binding(
    candidates: Sequence[str],
    node_scores: Sequence[NodeScore],
    pod_id: str,
    parameter_version: int,
    rejections: Sequence[Rejection],
) -> Decision:
    """Apply the stable, error-aware binding rule."""
    ordered: List[NodeScore] = sorted(node_scores, key=_sort_key)
    if not candidates:
        return Decision(
            pod_id=pod_id,
            feasible=False,
            candidates=tuple(),
            rejections=tuple(rejections),
            node_scores=tuple(),
            bound_node=None,
            tie_reason=UNSCHEDULABLE,
            error_influenced=False,
            runner_up=None,
            score_gap_mp=None,
            combined_error_mp=None,
            parameter_version=parameter_version,
        )

    winner = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    if runner_up is None:
        tie_reason = TIE_UNIQUE
        error_influenced = False
        gap: Optional[int] = None
        combined: Optional[int] = None
    else:
        gap = winner.total_mp - runner_up.total_mp
        combined = winner.error_bound_mp + runner_up.error_bound_mp
        if gap == 0:
            tie_reason = TIE_EXACT
            error_influenced = False
        elif gap <= combined:
            tie_reason = TIE_ERROR_BOUND
            error_influenced = True
        else:
            tie_reason = TIE_UNIQUE
            error_influenced = False

    return Decision(
        pod_id=pod_id,
        feasible=True,
        candidates=tuple(candidates),
        rejections=tuple(rejections),
        node_scores=tuple(ordered),
        bound_node=winner.node,
        tie_reason=tie_reason,
        error_influenced=error_influenced,
        runner_up=runner_up.node if runner_up else None,
        score_gap_mp=gap,
        combined_error_mp=combined,
        parameter_version=parameter_version,
    )
