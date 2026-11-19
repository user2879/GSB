from __future__ import annotations

from typing import Sequence

import numpy as np

from .concurrency import BoundedOrderedEvaluator


def trace_source_layers(
    source_vertex: int,
    operator_index: int,
    neighbors: Sequence[Sequence[int]],
    deviations: np.ndarray,
    max_layers: int,
    tolerance: float,
) -> dict[str, object]:
    if source_vertex < 0 or source_vertex >= len(neighbors):
        raise IndexError("propagation source is outside the vertex array")
    visited = {source_vertex}
    frontier = [source_vertex]
    layers: list[dict[str, object]] = []
    affected: set[int] = set()
    for layer_index in range(max_layers + 1):
        if not frontier:
            break
        ordered_vertices = sorted(frontier)
        values = deviations[np.asarray(ordered_vertices, dtype=np.int64)]
        maximum = float(np.max(values)) if len(values) else 0.0
        layers.append(
            {
                "layer": layer_index,
                "vertices": ordered_vertices,
                "vertex_count": len(ordered_vertices),
                "max_abs_error": maximum,
                "mean_abs_error": float(np.mean(values)) if len(values) else 0.0,
                "exceeds_tolerance": bool(maximum > tolerance),
            }
        )
        affected.update(ordered_vertices)
        next_frontier: list[int] = []
        for vertex in ordered_vertices:
            for neighbor in neighbors[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    first_below = next(
        (
            int(layer["layer"])
            for layer in layers[1:]
            if float(layer["max_abs_error"]) <= tolerance
        ),
        None,
    )
    later_nonzero = any(
        float(layer["max_abs_error"]) > 0.0 for layer in layers[1:]
    )
    if not later_nonzero:
        decay = "source_only"
    elif first_below is not None:
        decay = "decays_below_tolerance"
    else:
        decay = "persists_above_tolerance"
    return {
        "operator_index": operator_index,
        "source_vertex": source_vertex,
        "layers": layers,
        "affected_vertices": sorted(affected),
        "reached_layer_count": len(layers) - 1,
        "truncated": len(visited) > len(affected),
        "decay_conclusion": decay,
    }


def parallel_trace_sources(
    operator_index: int,
    source_vertices: Sequence[int],
    neighbors: Sequence[Sequence[int]],
    deviations: np.ndarray,
    trace_radius: int,
    tolerance: float,
    concurrency: int,
    memory_budget_nodes: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    unique_sources = sorted(set(int(vertex) for vertex in source_vertices))
    evaluator = BoundedOrderedEvaluator[dict[str, object]](
        concurrency=concurrency,
        memory_budget_nodes=memory_budget_nodes,
    )
    task_cost = max(1, int(deviations.shape[0]))
    tasks = [
        (
            source,
            lambda source=source: trace_source_layers(
                source,
                operator_index,
                neighbors,
                deviations,
                trace_radius,
                tolerance,
            ),
            task_cost,
        )
        for source in unique_sources
    ]
    traces = evaluator.map_indexed(tasks)
    union: set[int] = set()
    max_layer = 0
    conclusions: dict[str, int] = {}
    for trace in traces:
        union.update(int(vertex) for vertex in trace["affected_vertices"])  # type: ignore[arg-type]
        max_layer = max(max_layer, int(trace["reached_layer_count"]))
        conclusion = str(trace["decay_conclusion"])
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
    aggregate = {
        "operator_index": operator_index,
        "source_count": len(unique_sources),
        "source_vertices": unique_sources,
        "trace_radius": trace_radius,
        "affected_vertex_count": len(union),
        "affected_vertices": sorted(union),
        "max_reached_layer": max_layer,
        "decay_counts": dict(sorted(conclusions.items())),
        "evaluation": evaluator.metrics.to_dict(),
    }
    return traces, aggregate
