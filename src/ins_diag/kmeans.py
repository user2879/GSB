from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import EffectiveConfig
from .fixedpoint import FixedPoint, rounded_grid_division
from .util import fingerprint


def _assign_one(
    window_index: int,
    vector: tuple[int, ...],
    centroids: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    best_cluster = 0
    best_distance: int | None = None
    for cluster_id, centroid in enumerate(centroids):
        distance = 0
        for dimension_index, value in enumerate(vector):
            difference = value - centroid[dimension_index]
            distance += difference * difference
        if best_distance is None or distance < best_distance:
            best_cluster = cluster_id
            best_distance = distance
    assert best_distance is not None
    return {
        "window_index": window_index,
        "cluster_id": best_cluster,
        "distance_units": best_distance,
        "tie_break": "lowest_centroid_id_on_equal_distance",
    }


def _assign_all(
    observations: list[tuple[int, tuple[int, ...]]],
    centroids: list[tuple[int, ...]],
    executor: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    centroid_tuple = tuple(centroids)
    futures = []
    for window_index, vector in observations:
        key = (window_index, "kmeans-assign")
        future = executor.submit(_assign_one, window_index, vector, centroid_tuple)
        futures.append((key, future))
    results = [(key, future.result()) for key, future in futures]
    results.sort(key=lambda item: item[0])
    return [result for _, result in results]


def _squared_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    total = 0
    for dimension_index in range(len(a)):
        difference = a[dimension_index] - b[dimension_index]
        total += difference * difference
    return total


def run_fixed_kmeans(
    observations: dict[int, list[float]],
    config: EffectiveConfig,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    fixed = FixedPoint.from_config(config.fixed_point)
    ordered_indices = sorted(observations)
    if config.families > len(ordered_indices):
        raise ValueError(
            f"family count {config.families} exceeds valid window count {len(ordered_indices)}"
        )

    quantized: list[tuple[int, tuple[int, ...]]] = []
    quantization_residual_units = 0
    for window_index in ordered_indices:
        vector = tuple(fixed.quantize_vector(observations[window_index]))
        for original, quantized_value in zip(observations[window_index], vector):
            quantization_residual_units += fixed.residual_units(float(original), quantized_value)
        quantized.append((window_index, vector))

    count = len(quantized)
    initial_indices: list[int] = []
    for cluster_id in range(config.families):
        offset = (cluster_id * count) // config.families
        initial_indices.append(offset)
    centroids: list[tuple[int, ...]] = [quantized[offset][1] for offset in initial_indices]
    rounds: list[dict[str, Any]] = []
    converged = False
    final_assignments: list[dict[str, Any]] = []

    for iteration in range(1, config.max_iterations + 1):
        assignments = _assign_all(quantized, centroids, executor)
        final_assignments = assignments
        groups: dict[int, list[int]] = {cluster_id: [] for cluster_id in range(config.families)}
        for assignment in assignments:
            groups[int(assignment["cluster_id"])].append(int(assignment["window_index"]))
        for cluster_id in groups:
            groups[cluster_id].sort()

        by_window = {window_index: vector for window_index, vector in quantized}
        new_centroids: list[tuple[int, ...]] = []
        round_residual = 0
        for cluster_id in range(config.families):
            members = groups[cluster_id]
            if not members:
                new_centroids.append(centroids[cluster_id])
                continue
            totals = [0] * len(centroids[cluster_id])
            for window_index in members:
                for dimension_index, value in enumerate(by_window[window_index]):
                    totals[dimension_index] += value
            updated: list[int] = []
            for total in totals:
                value = rounded_grid_division(total, len(members), fixed.quantization_units)
                updated.append(value)
                round_residual += abs(total - value * len(members))
            new_centroids.append(tuple(updated))

        empty_actions: list[dict[str, Any]] = []
        reserved_windows: set[int] = set()
        for cluster_id in range(config.families):
            if groups[cluster_id]:
                continue
            candidate: tuple[int, int, int, int] | None = None
            for assignment in assignments:
                window_index = int(assignment["window_index"])
                if window_index in reserved_windows:
                    continue
                distance = int(assignment["distance_units"])
                source_cluster = int(assignment["cluster_id"])
                candidate_key = (-distance, window_index, source_cluster, window_index)
                if candidate is None or candidate_key < candidate:
                    candidate = candidate_key
            assert candidate is not None
            _, window_index, source_cluster, _ = candidate
            reserved_windows.add(window_index)
            new_centroids[cluster_id] = by_window[window_index]
            empty_actions.append(
                {
                    "cluster_id": cluster_id,
                    "window_index": window_index,
                    "source_cluster_id": source_cluster,
                    "distance_units": -candidate[0],
                    "basis": "farthest_window_from_its_assigned_centroid",
                    "tie_break": "lowest_window_index",
                }
            )

        centroid_changed = new_centroids != centroids
        round_payload = {
            "iteration": iteration,
            "centroids_units": [list(centroid) for centroid in new_centroids],
            "assignment_counts": [
                len(groups[cluster_id]) for cluster_id in range(config.families)
            ],
            "quantization_residual_units": round_residual,
            "empty_cluster_actions": empty_actions,
            "centroid_changed": centroid_changed,
        }
        rounds.append(round_payload)
        centroids = new_centroids
        if not centroid_changed:
            converged = True
            break

    assignments = _assign_all(quantized, centroids, executor)
    final_assignments = assignments
    sse_units = sum(int(item["distance_units"]) for item in assignments)
    assignments_by_index = {
        int(item["window_index"]): item for item in final_assignments
    }
    assignment_rows: list[dict[str, Any]] = []
    members_by_cluster: dict[int, list[int]] = {
        cluster_id: [] for cluster_id in range(config.families)
    }
    for window_index in ordered_indices:
        assignment = assignments_by_index[window_index]
        assignment_rows.append(
            {
                "window_index": window_index,
                "cluster_id": int(assignment["cluster_id"]),
                "distance_units": int(assignment["distance_units"]),
            }
        )
        members_by_cluster[int(assignment["cluster_id"])].append(window_index)

    cluster_sse_units = {cluster_id: 0 for cluster_id in range(config.families)}
    for row in assignment_rows:
        cluster_sse_units[int(row["cluster_id"])] += int(row["distance_units"])

    payload = {
        "algorithm": "fixed_point_kmeans",
        "fixed_point": fixed.to_dict(),
        "initial_centroid_rule": "equal_stride_window_index",
        "tie_break": "lowest_centroid_id_then_lowest_window_index",
        "initial_centroid_window_indices": initial_indices,
        "observations": [
            {
                "window_index": window_index,
                "quantized_units": list(vector),
                "quantized_values": fixed.dequantize_vector(vector),
            }
            for window_index, vector in quantized
        ],
        "input_quantization_residual_units": quantization_residual_units,
        "centroids_units": [list(centroid) for centroid in centroids],
        "centroids": [fixed.dequantize_vector(centroid) for centroid in centroids],
        "assignments": assignment_rows,
        "cluster_members": {
            str(cluster_id): members for cluster_id, members in members_by_cluster.items()
        },
        "cluster_sse_units": {
            str(cluster_id): value for cluster_id, value in cluster_sse_units.items()
        },
        "sse_units": sse_units,
        "iterations": len(rounds),
        "converged": converged,
        "rounds": rounds,
    }
    payload["result_fingerprint"] = fingerprint(
        {
            "fixed_point": payload["fixed_point"],
            "initial": initial_indices,
            "centroids": payload["centroids_units"],
            "assignments": payload["assignments"],
            "sse_units": payload["sse_units"],
        }
    )
    return payload

