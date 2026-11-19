from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CompressionCandidate, SimConfig
from .error_tracking import aggregate_fixed_point, aggregate_float64
from .protocol import FRAME_HEADER_SIZE
from .quantization import QuantParams, compute_common_quant_params, packet_size, quantize
from .topology import Topology


@dataclass
class DecisionResult:
    decision: dict[str, Any]
    selected: CompressionCandidate
    quant_params: dict[str, QuantParams]
    tasks: list[dict[str, Any]]
    link_budgets: dict[str, int]


def _candidate_error(
    gradients: dict[int, dict[str, np.ndarray]],
    tensor_names: tuple[str, ...],
    candidate: CompressionCandidate,
    mode: str,
) -> tuple[float, float, dict[str, QuantParams], dict[str, dict[str, float]]]:
    max_abs_overall = 0.0
    max_rel_overall = 0.0
    params_by_tensor: dict[str, QuantParams] = {}
    per_tensor: dict[str, dict[str, float]] = {}
    for tensor_name in tensor_names:
        arrays = {
            worker_id: gradients[worker_id][tensor_name]
            for worker_id in sorted(gradients)
        }
        params = compute_common_quant_params(
            tensor_name,
            (arrays[worker_id] for worker_id in sorted(arrays)),
            candidate.bits,
            candidate.scheme,
        )
        params_by_tensor[tensor_name] = params
        q_by_origin = {
            worker_id: quantize(arrays[worker_id], params) for worker_id in sorted(arrays)
        }
        truth, _ = aggregate_float64(arrays, mode)
        fixed_values, _, _ = aggregate_fixed_point(q_by_origin, params, mode)
        error = np.abs(fixed_values.astype(np.float64) - truth.astype(np.float64))
        max_abs = float(np.max(error)) if error.size else 0.0
        denominator = np.maximum(np.abs(truth.astype(np.float64)), 1e-12)
        max_rel = float(np.max(error / denominator)) if error.size else 0.0
        per_tensor[tensor_name] = {
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
        }
        max_abs_overall = max(max_abs_overall, max_abs)
        max_rel_overall = max(max_rel_overall, max_rel)
    return max_abs_overall, max_rel_overall, params_by_tensor, per_tensor


def _plan_tasks(
    *,
    topology: Topology,
    window: int,
    ordered_tensors: list[str],
    packet_bytes: dict[str, int],
    bits: int,
) -> list[dict[str, Any]]:
    workers = topology.workers
    tasks: list[dict[str, Any]] = []
    if workers <= 1:
        return tasks
    tensor_rank = {name: rank for rank, name in enumerate(ordered_tensors)}
    if topology.kind == "full":
        for origin in range(workers):
            for tensor_name in ordered_tensors:
                for dst in range(workers):
                    if dst == origin:
                        continue
                    tasks.append(
                        {
                            "window": window,
                            "origin_worker": origin,
                            "src_holder": origin,
                            "dst_worker": dst,
                            "route_round": 0,
                            "tensor": tensor_name,
                            "tensor_rank": tensor_rank[tensor_name],
                            "bits": bits,
                            "packet_bytes": packet_bytes[tensor_name],
                            "wire_bytes": FRAME_HEADER_SIZE + packet_bytes[tensor_name],
                            "stable_key": [window, origin, tensor_name, 0, dst],
                        }
                    )
    else:
        for route_round in range(workers - 1):
            for origin in range(workers):
                holder = (origin + route_round) % workers
                dst = (holder + 1) % workers
                for tensor_name in ordered_tensors:
                    tasks.append(
                        {
                            "window": window,
                            "origin_worker": origin,
                            "src_holder": holder,
                            "dst_worker": dst,
                            "route_round": route_round,
                            "tensor": tensor_name,
                            "tensor_rank": tensor_rank[tensor_name],
                            "bits": bits,
                            "packet_bytes": packet_bytes[tensor_name],
                            "wire_bytes": FRAME_HEADER_SIZE + packet_bytes[tensor_name],
                            "stable_key": [window, origin, tensor_name, route_round, dst],
                        }
                    )
    return tasks


def plan_window_tasks(
    *,
    topology: Topology,
    window: int,
    ordered_tensors: list[str],
    packet_bytes: dict[str, int],
    bits: int,
) -> list[dict[str, Any]]:
    return _plan_tasks(
        topology=topology,
        window=window,
        ordered_tensors=ordered_tensors,
        packet_bytes=packet_bytes,
        bits=bits,
    )


def _link_bytes(tasks: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for task in tasks:
        link = f"{task['src_holder']}->{task['dst_worker']}"
        totals[link] = totals.get(link, 0) + int(task["wire_bytes"])
    return totals


def _choose_order(
    config: SimConfig,
    tensor_names: tuple[str, ...],
    packet_bytes: dict[str, int],
) -> tuple[str, str, list[str]]:
    requested = config.aggregation.order
    if requested == "auto":
        sizes = [packet_bytes[name] for name in tensor_names]
        if len(sizes) > 1 and max(sizes) >= 1.5 * max(1, min(sizes)):
            order_mode = "bytes_desc"
            reason = "auto selected bytes_desc because tensor packet sizes differ by at least 1.5x"
        else:
            order_mode = "name_asc"
            reason = "auto selected name_asc because tensor packet sizes are close"
    else:
        order_mode = requested
        reason = f"configuration requested {requested}"
    if order_mode == "bytes_desc":
        ordered = sorted(tensor_names, key=lambda name: (-packet_bytes[name], name))
    else:
        ordered = sorted(tensor_names)
    return order_mode, reason, ordered


def decide_window(
    *,
    config: SimConfig,
    topology: Topology,
    window: int,
    gradients: dict[int, dict[str, np.ndarray]],
    budget_overrides: dict[tuple[int, int], int] | None = None,
) -> DecisionResult:
    tensor_names = config.tensor_names()
    if tuple(sorted(gradients)) != tuple(range(config.workers)):
        raise ValueError("gradient worker set does not match config")
    source_gradient_bytes = int(
        sum(array.nbytes for worker_values in gradients.values() for array in worker_values.values())
    )
    link_budgets = topology.budget_map(budget_overrides)
    window_budget = config.compression.max_bytes_per_window

    evaluations: list[dict[str, Any]] = []
    evaluation_details: list[
        tuple[CompressionCandidate, dict[str, QuantParams], dict[str, int], dict[str, int], int, int]
    ] = []
    for candidate in config.compression.candidates:
        max_abs, max_rel, params_by_tensor, per_tensor = _candidate_error(
            gradients, tensor_names, candidate, config.aggregation.mode
        )
        sizes = {
            name: packet_size(name, gradients[0][name].shape, candidate.bits)
            for name in tensor_names
        }
        order_mode, _, ordered_tensors = _choose_order(config, tensor_names, sizes)
        tasks = _plan_tasks(
            topology=topology,
            window=window,
            ordered_tensors=ordered_tensors,
            packet_bytes=sizes,
            bits=candidate.bits,
        )
        link_totals = _link_bytes(tasks)
        max_link_bytes = max(link_totals.values(), default=0)
        total_bytes = sum(link_totals.values())
        per_link_fit = all(
            link_totals.get(link, 0) <= budget for link, budget in link_budgets.items()
        )
        fits_budget = per_link_fit and total_bytes <= window_budget
        tolerance_ok = max_abs <= config.communication_error_tolerance
        error_ratio = max_abs / config.communication_error_tolerance
        byte_ratio = total_bytes / max(1, window_budget)
        penalty = (0.0 if fits_budget else 100.0) + (0.0 if tolerance_ok else 10.0)
        score = penalty + 0.8 * min(error_ratio, 100.0) + 0.2 * min(byte_ratio, 100.0)
        evaluations.append(
            {
                "scheme": candidate.scheme,
                "bits": candidate.bits,
                "candidate": candidate.label,
                "packet_bytes_by_tensor": sizes,
                "total_compressed_bytes": total_bytes,
                "max_link_bytes": max_link_bytes,
                "window_budget_bytes": window_budget,
                "source_gradient_bytes": source_gradient_bytes,
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "error_tolerance": config.communication_error_tolerance,
                "tolerance_ok": tolerance_ok,
                "fits_budget": fits_budget,
                "per_link_fit": per_link_fit,
                "link_bytes": dict(sorted(link_totals.items())),
                "score": score,
                "per_tensor_error": per_tensor,
                "order_mode": order_mode,
                "ordered_tensors": ordered_tensors,
            }
        )
        evaluation_details.append(
            (candidate, params_by_tensor, sizes, link_totals, total_bytes, max_link_bytes)
        )

    if not evaluations:
        raise ValueError("no compression candidates were configured")
    candidate_by_label = {candidate.label: candidate for candidate in config.compression.candidates}
    def selection_key(index: int) -> tuple[object, ...]:
        evaluation = evaluations[index]
        scheme_rank = candidate_by_label[str(evaluation["candidate"])].scheme_rank
        if evaluation["fits_budget"]:
            return (
                0,
                0 if evaluation["tolerance_ok"] else 1,
                float(evaluation["score"]),
                int(evaluation["total_compressed_bytes"]),
                -int(evaluation["bits"]),
                scheme_rank,
            )
        return (
            1,
            int(evaluation["total_compressed_bytes"]),
            0 if evaluation["tolerance_ok"] else 1,
            float(evaluation["max_abs_error"]),
            -int(evaluation["bits"]),
            scheme_rank,
        )

    selected_index = min(range(len(evaluations)), key=selection_key)
    selected_record = evaluations[selected_index]
    selected = candidate_by_label[str(selected_record["candidate"])]
    _, params_by_tensor, sizes, link_totals, total_bytes, max_link_bytes = (
        evaluation_details[selected_index]
    )
    order_mode, order_reason, ordered_tensors = _choose_order(config, tensor_names, sizes)
    tasks = _plan_tasks(
        topology=topology,
        window=window,
        ordered_tensors=ordered_tensors,
        packet_bytes=sizes,
        bits=selected.bits,
    )
    link_totals = _link_bytes(tasks)
    max_link_bytes = max(link_totals.values(), default=0)
    total_bytes = sum(link_totals.values())
    per_link_fit = all(
        link_totals.get(link, 0) <= budget for link, budget in link_budgets.items()
    )
    fits_budget = per_link_fit and total_bytes <= window_budget
    selected_reason = (
        f"selected {selected.label}: score={selected_record['score']:.6f}, "
        f"max_abs_error={selected_record['max_abs_error']:.9g} "
        f"(tolerance={config.communication_error_tolerance}), "
        f"wire_bytes={total_bytes}, max_link_bytes={max_link_bytes}, "
        f"fits_budget={fits_budget}"
    )
    for task in tasks:
        task["candidate_scores"] = evaluations
        task["selection_reason"] = selected_reason

    min_wire_bytes = min((int(task["wire_bytes"]) for task in tasks), default=0)
    used_link_budgets = {
        link: link_budgets.get(link, 0) for link in sorted(link_totals)
    }
    budgeted_links = (
        sum(1 for budget in used_link_budgets.values() if min_wire_bytes and budget >= min_wire_bytes)
        if tasks
        else 0
    )
    total_capacity = sum(used_link_budgets.values())
    capacity_links = (
        max(1, total_capacity // min_wire_bytes)
        if tasks and min_wire_bytes and total_capacity >= min_wire_bytes
        else 0
    )
    if tasks:
        bandwidth_concurrency_limit = max(
            1,
            min(
                config.concurrency,
                max(1, budgeted_links),
                max(1, capacity_links),
                max(1, config.compression.max_bytes_per_window // min_wire_bytes),
            ),
        )
    else:
        bandwidth_concurrency_limit = 1
    decision = {
        "window": window,
        "source_gradient_bytes": source_gradient_bytes,
        "window_budget_bytes": window_budget,
        "link_budgets": dict(sorted(link_budgets.items())),
        "candidate_scores": evaluations,
        "selected_candidate": {"scheme": selected.scheme, "bits": selected.bits, "label": selected.label},
        "selected_score": selected_record["score"],
        "selected_reason": selected_reason,
        "selected_packet_bytes_by_tensor": sizes,
        "aggregation_order": order_mode,
        "aggregation_order_reason": order_reason,
        "ordered_tensors": [
            {
                "tensor": name,
                "packet_bytes": sizes[name],
                "tie_break": "tensor_name_lexicographic",
            }
            for name in ordered_tensors
        ],
        "total_compressed_bytes": total_bytes,
        "max_link_bytes": max_link_bytes,
        "fits_budget": fits_budget,
        "selected_quant_params": {
            name: params.as_dict() for name, params in sorted(params_by_tensor.items())
        },
        "planned_send_count": len(tasks),
        "bandwidth_concurrency_limit": bandwidth_concurrency_limit,
        "bandwidth_concurrency_reason": (
            "min(configured concurrency, links whose budget fits one frame, "
            "link capacity in frames, window budget in frames)"
        ),
    }
    return DecisionResult(
        decision=decision,
        selected=selected,
        quant_params=params_by_tensor,
        tasks=tasks,
        link_budgets=link_budgets,
    )
