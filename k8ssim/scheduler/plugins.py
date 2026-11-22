"""Four integer fixed-point score plugins.

Each plugin returns an unscaled numerator and a positive divisor; a single
floor division (with session scale factor ``S``) converts them to the 0..100
integer score. Keeping exactly one division per plugin makes the worst-case
quantization error exact rather than estimated:

    raw      = floor(numerator * S / (divisor * S))  with clamping to [0, 100]
    error_mp = ceil(divisor * 100 / numerator scale) ... see ``quantize``

All arithmetic uses Python integers (arbitrary precision); NumPy is never used
for scheduling decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

from ..constants import (
    PLUGIN_LABEL_LOCALITY,
    PLUGIN_NODE_AFFINITY,
    PLUGIN_POD_ANTI_AFFINITY,
    PLUGIN_RESOURCE_BALANCE,
    PLUGIN_ORDER,
    SCORE_MAX,
    SCORE_MIN,
    TRUNCATION,
)
from ..models import Node, Pod
from .filter import NodeCapacity, Placement
from .match import matches_requirement

MICRO = 1_000_000  # micro-points per plugin point


@dataclass(frozen=True)
class PluginOutput:
    plugin: str
    numerator: int
    divisor: int
    raw_score: int
    weight: int
    contribution_mp: int
    quant_error_mp: int
    truncation: str = TRUNCATION


def quantize(numerator: int, divisor: int, scale_factor: int) -> tuple[int, int]:
    """Return (raw_score 0..100, exact quantization error in micro-points).

    The real-valued score is ``100 * numerator / divisor`` and the integer
    result is ``floor(real * S) / S``. Using the division remainder the exact
    absolute error is
    ``(100 * numerator * S mod (divisor * S)) / (divisor * S)`` points,
    converted to micro-points with an integer ceil so the bound stays rigorous.
    A single floor division is performed, with ``S`` applied before dividing.
    """
    if divisor <= 0:
        raise ValueError("plugin divisor must be positive")
    numerator = max(0, min(numerator, divisor))
    scaled_denominator = divisor * scale_factor
    scaled_numerator = SCORE_MAX * numerator * scale_factor
    raw = scaled_numerator // scaled_denominator
    raw = max(SCORE_MIN, min(SCORE_MAX, raw))
    remainder = scaled_numerator % scaled_denominator
    if remainder == 0:
        error_mp = 0
    else:
        # ceil(MICRO * remainder / scaled_denominator)
        error_mp = -(-(MICRO * remainder) // scaled_denominator)
    return raw, error_mp


# --------------------------------------------------------------------------- #
# Plugin 1: node affinity soft constraints
# --------------------------------------------------------------------------- #


def node_affinity_soft(pod: Pod, node: Node, scale_factor: int, weight: int) -> PluginOutput:
    terms = pod.node_affinity.preferred
    if not terms:
        divisor = 2
        numerator = 1
    else:
        total_weight = sum(term.weight for term in terms)
        matched = sum(
            term.weight
            for term in terms
            if matches_requirement(node.labels, term.requirement)
        )
        numerator, divisor = matched, total_weight
    raw, error_mp = quantize(numerator, divisor, scale_factor)
    return _build(PLUGIN_NODE_AFFINITY, numerator, divisor, raw, weight, error_mp)


# --------------------------------------------------------------------------- #
# Plugin 2: pod anti-affinity soft constraints (spread reward)
# --------------------------------------------------------------------------- #


def pod_anti_affinity_soft(
    pod: Pod,
    node: Node,
    placements: Sequence[Placement],
    node_lookup: Mapping[str, Node],
    scale_factor: int,
    weight: int,
) -> PluginOutput:
    anti = pod.pod_anti_affinity
    terms = anti.preferred
    if not terms:
        numerator, divisor = 1, 2
    else:
        total_weight = sum(term.weight for term in terms)
        conflict_weight = 0
        topology_key = anti.preferred_topology_key
        own_domain = node.labels.get(topology_key)
        for term in terms:
            for placed in placements:
                placed_node = node_lookup[placed.node]
                if topology_key == "kubernetes.io/hostname":
                    same_domain = placed.node == node.name
                else:
                    same_domain = placed_node.labels.get(topology_key) == own_domain
                if same_domain and matches_requirement(placed.labels, term.requirement):
                    conflict_weight += term.weight
                    break
        numerator = max(0, total_weight - conflict_weight)
        divisor = total_weight
    raw, error_mp = quantize(numerator, divisor, scale_factor)
    return _build(PLUGIN_POD_ANTI_AFFINITY, numerator, divisor, raw, weight, error_mp)


# --------------------------------------------------------------------------- #
# Plugin 3: resource balance (favor nodes retaining balanced slack)
# --------------------------------------------------------------------------- #


def resource_balance(
    node: Node,
    capacity: NodeCapacity,
    pod: Pod,
    scale_factor: int,
    weight: int,
) -> PluginOutput:
    cpu_avail = capacity.cpu_available
    mem_avail = capacity.memory_available
    # Hypothetical post-schedule slack; both resources must remain non-negative
    # because the Filter stage already guaranteed feasibility.
    cpu_slack = max(0, cpu_avail - pod.cpu_request)
    mem_slack = max(0, mem_avail - pod.memory_request)
    # Harmonic-mean-like balance expressed with integer arithmetic only:
    # balance = 2 * cpu_slack * mem_slack / (cpu_slack + mem_slack), normalized
    # by the node's smaller total capacity to stay within [0, 1].
    balance_den = cpu_slack + mem_slack
    if balance_den == 0:
        numerator, divisor = 0, 1
    else:
        balance_num = 2 * cpu_slack * mem_slack
        limit = min(node.cpu, node.memory)
        numerator = balance_num
        divisor = balance_den * limit
    raw, error_mp = quantize(numerator, divisor, scale_factor)
    return _build(PLUGIN_RESOURCE_BALANCE, numerator, divisor, raw, weight, error_mp)


# --------------------------------------------------------------------------- #
# Plugin 4: label locality (pod/node label key intersection ratio)
# --------------------------------------------------------------------------- #


def label_locality(
    pod: Pod, node: Node, scale_factor: int, weight: int
) -> PluginOutput:
    pod_keys = set(pod.labels)
    if not pod_keys:
        numerator, divisor = 1, 2
    else:
        shared = sum(
            1 for key in pod_keys
            if key in node.labels and node.labels[key] == pod.labels[key]
        )
        numerator, divisor = shared, len(pod_keys)
    raw, error_mp = quantize(numerator, divisor, scale_factor)
    return _build(PLUGIN_LABEL_LOCALITY, numerator, divisor, raw, weight, error_mp)


def _build(
    plugin: str,
    numerator: int,
    divisor: int,
    raw: int,
    weight: int,
    error_mp: int,
) -> PluginOutput:
    return PluginOutput(
        plugin=plugin,
        numerator=numerator,
        divisor=divisor,
        raw_score=raw,
        weight=weight,
        contribution_mp=raw * weight,
        quant_error_mp=error_mp,
        truncation=TRUNCATION,
    )


def score_node(
    pod: Pod,
    node: Node,
    capacity: NodeCapacity,
    placements: Sequence[Placement],
    node_lookup: Mapping[str, Node],
    weights: Mapping[str, int],
    scale_factor: int,
) -> Dict[str, PluginOutput]:
    """Run all four plugins; the dict is returned keyed by plugin id."""
    outputs = (
        node_affinity_soft(pod, node, scale_factor, weights[PLUGIN_NODE_AFFINITY]),
        pod_anti_affinity_soft(
            pod, node, placements, node_lookup, scale_factor,
            weights[PLUGIN_POD_ANTI_AFFINITY],
        ),
        resource_balance(node, capacity, pod, scale_factor, weights[PLUGIN_RESOURCE_BALANCE]),
        label_locality(pod, node, scale_factor, weights[PLUGIN_LABEL_LOCALITY]),
    )
    return {output.plugin: output for output in outputs}


PLUGIN_IDS = PLUGIN_ORDER
