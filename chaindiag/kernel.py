from __future__ import annotations

import math
from typing import Any

from .fixedpoint import FixedContext, apply_ratio, round_rational, stable_int


def scale_to_fixed(values: list[int], scale_num: int, scale_den: int, ctx: FixedContext, phase: str) -> tuple[list[int], list[dict[str, Any]]]:
    output: list[int] = []
    steps: list[dict[str, Any]] = []
    for dim, value in enumerate(values):
        scaled, direction, tie = apply_ratio(value, scale_num, scale_den, ctx.tie_rule)
        output.append(scaled)
        steps.append({"phase": phase, "dim": dim, "direction": direction, "tie": tie, "before": value, "after": scaled})
    return output, steps


def embedding_lookup(*, token_id: int, dim: int, seed: int, ctx: FixedContext,
                    scale_num: int, scale_den: int, int_bits: int) -> dict[str, Any]:
    raw = [stable_int(seed, 0xE1B, int(token_id), index) % 17 - 8 for index in range(dim)]
    fixed_base = [value * ctx.scale for value in raw]
    scaled, steps = scale_to_fixed(fixed_base, scale_num, scale_den, ctx, "embedding.scale")
    quantized, quant_steps, residual, ties = ctx.quantize(scaled, int_bits, "embedding.quantize")
    return {
        "vector": quantized,
        "residual": residual,
        "rounding_steps": steps + quant_steps,
        "ties": ties,
        "details": {"token_value": token_id, "raw_vector": raw, "tree_path": ["embedding"], "scale": ctx.scale},
    }


def segment_pool(*, member_records: list[dict[str, Any]], ctx: FixedContext,
                 scale_num: int, scale_den: int, int_bits: int) -> dict[str, Any]:
    dim = len(member_records[0]["vector"])
    ordered_sum = [0 for _ in range(dim)]
    for record in member_records:  # member_records already in token seq ascending.
        vector = record["vector"]
        for index, value in enumerate(vector):
            ordered_sum[index] += int(value)

    scaled, steps = scale_to_fixed(ordered_sum, scale_num, scale_den, ctx, "segment_pool.scale")
    quantized, quant_steps, residual, ties = ctx.quantize(scaled, int_bits, "segment_pool.quantize")
    return {
        "vector": quantized,
        "residual": residual,
        "rounding_steps": steps + quant_steps,
        "ties": ties,
        "details": {
            "valid_token_count": len(member_records),
            "member_uids": [record["token_uid"] for record in member_records],
            "ordered_sum": ordered_sum,
        },
    }


def _attention_weights(*, query_token: int, member_tokens: list[int], member_uids: list[str],
                       head: int, dim: int, seed: int, temperature_permille: int,
                       ctx: FixedContext, query_vector: list[int], member_vectors: list[list[int]]) -> tuple[list[int], list[dict[str, Any]], list[int]]:
    raw_scores: list[int] = []
    steps: list[dict[str, Any]] = []
    for position, member_vector in enumerate(member_vectors):
        dot = 0
        for index in range(dim):  # dimension order ascending.
            dot += int(query_vector[index]) * int(member_vector[index])
        raw_score, direction, _ = round_rational(dot, ctx.scale * ctx.scale, ctx.tie_rule)
        temp_score, temp_direction, _ = apply_ratio(raw_score, temperature_permille, 1000, ctx.tie_rule)
        bias = stable_int(seed, 0xA77, head, int(query_token), int(member_tokens[position])) % 7 - 3
        score = temp_score + bias
        raw_scores.append(score)
        steps.append(
            {
                "phase": "attention.score",
                "head": head,
                "member_position": position,
                "member_uid": member_uids[position],
                "direction": direction if direction else temp_direction,
                "dot": dot,
                "score": score,
            }
        )

    minimum = min(raw_scores)
    bases = [score - minimum + 1 for score in raw_scores]
    total = sum(bases)
    floors: list[int] = []
    remainders: list[int] = []
    for position, base in enumerate(bases):
        numerator = base * 1000
        floor_value, remainder = divmod(numerator, total)
        floors.append(floor_value)
        remainders.append(remainder)
        _, direction, _ = round_rational(numerator, total, ctx.tie_rule)
        steps.append(
            {
                "phase": "attention.weight_allocation",
                "head": head,
                "member_position": position,
                "member_uid": member_uids[position],
                "direction": direction,
                "floor_permille": floor_value,
                "remainder": remainder,
            }
        )

    deficit = 1000 - sum(floors)
    order = sorted(range(len(bases)), key=lambda pos: (-remainders[pos], pos))
    weights = list(floors)
    for position in order[:deficit]:
        weights[position] += 1
    return weights, steps, raw_scores


def multi_head_attention(*, query_token: int, query_uid: str, member_tokens: list[int], member_uids: list[str],
                         embedding_records: list[dict[str, Any]], pool_record: dict[str, Any] | None,
                         heads: int, head_dim: int, seed: int, temperature_permille: int,
                         ctx: FixedContext, scale_num: int, scale_den: int, int_bits: int) -> dict[str, Any]:
    query_vector = embedding_records[member_uids.index(query_uid)]["vector"] if query_uid in member_uids else embedding_records[0]["vector"]
    member_vectors = [record["vector"] for record in embedding_records]
    dimension = heads * head_dim
    output = [0 for _ in range(dimension)]
    all_steps: list[dict[str, Any]] = []
    all_weights: dict[str, list[int]] = {}
    all_scores: dict[str, list[int]] = {}

    pool_vector = pool_record["vector"] if pool_record is not None else [0] * dimension
    pool_count = max(1, int((pool_record or {}).get("details", {}).get("valid_token_count", len(member_uids))))
    gate_average = []
    for dim_index in range(dimension):
        average, direction, tie = round_rational(int(pool_vector[dim_index]), pool_count, ctx.tie_rule)
        gate_average.append(average)
        all_steps.append({"phase": "attention.pool_average", "dim": dim_index, "direction": direction, "tie": tie, "after": average})

    for head in range(heads):  # head order ascending.
        weights, weight_steps, scores = _attention_weights(
            query_token=query_token,
            member_tokens=member_tokens,
            member_uids=member_uids,
            head=head,
            dim=head_dim,
            seed=seed,
            temperature_permille=temperature_permille,
            ctx=ctx,
            query_vector=[query_vector[head * head_dim + j] for j in range(head_dim)],
            member_vectors=[[vector[head * head_dim + j] for j in range(head_dim)] for vector in member_vectors],
        )
        all_steps.extend(weight_steps)
        all_weights[str(head)] = weights
        all_scores[str(head)] = scores

        for dim_in_head in range(head_dim):  # dimension order ascending inside each head.
            dim = head * head_dim + dim_in_head
            gate_value, gate_direction, gate_tie = round_rational(gate_average[dim], 8, ctx.tie_rule)
            all_steps.append({"phase": "attention.gate", "head": head, "dim": dim, "direction": gate_direction, "tie": gate_tie, "after": gate_value})
            accumulator = 0
            for position, weight in enumerate(weights):
                value = int(member_vectors[position][dim]) + gate_value
                accumulator += int(weight) * value
            averaged, direction, tie = round_rational(accumulator, 1000, ctx.tie_rule)
            scaled, scale_direction, scale_tie = apply_ratio(averaged, scale_num, scale_den, ctx.tie_rule)
            output[dim] = scaled
            all_steps.append(
                {
                    "phase": "attention.weighted_average",
                    "head": head,
                    "dim": dim,
                    "direction": direction,
                    "tie": tie or scale_tie,
                    "scale_direction": scale_direction,
                    "accumulator": accumulator,
                    "after": scaled,
                }
            )

    quantized, quant_steps, residual, ties = ctx.quantize(output, int_bits, "attention.quantize")
    all_steps.extend(quant_steps)
    return {
        "vector": quantized,
        "residual": residual,
        "rounding_steps": all_steps,
        "ties": ties,
        "details": {
            "heads": heads,
            "head_dim": head_dim,
            "temperature_permille": temperature_permille,
            "weights_permille": all_weights,
            "scores": all_scores,
            "member_uids": member_uids,
        },
    }


def integer_sqrt_round(value: int, tie_rule: str) -> tuple[int, int, bool]:
    if value <= 0:
        return 0, 0, False
    root = math.isqrt(value)
    lower_gap = value - root * root
    upper_gap = (root + 1) * (root + 1) - value
    exact_half = lower_gap == upper_gap
    if upper_gap < lower_gap or (exact_half and tie_rule == "half_away_from_zero"):
        return root + 1, 1 if upper_gap < lower_gap else 0, exact_half
    return root, -1 if lower_gap else 0, exact_half


def layer_normalize(*, values: list[int], ctx: FixedContext, gamma_int: int | None = None,
                    beta_int: int = 0, int_bits: int = 16,
                    op_scale_num: int = 1, op_scale_den: int = 1) -> dict[str, Any]:
    dimension = len(values)
    total = 0
    for value in values:  # token order is handled by caller; dimensions ascend here.
        total += int(value)
    mean_q, mean_direction, mean_tie = round_rational(total, dimension, ctx.tie_rule)

    deviations = [int(value) - mean_q for value in values]
    square_sum = 0
    for deviation in deviations:
        square_sum += deviation * deviation
    variance_q, variance_direction, _ = round_rational(square_sum, dimension * ctx.scale, ctx.tie_rule)
    epsilon_q = 1
    std_target = max(0, variance_q + epsilon_q) * ctx.scale
    std_q, sqrt_direction, sqrt_tie = integer_sqrt_round(std_target, ctx.tie_rule)

    gamma = ctx.scale if gamma_int is None else int(gamma_int)
    pre_quant: list[int] = []
    steps: list[dict[str, Any]] = [
        {"phase": "layer_norm.mean", "direction": mean_direction, "tie": mean_tie, "after": mean_q},
        {"phase": "layer_norm.variance", "direction": variance_direction, "after": variance_q},
        {"phase": "layer_norm.sqrt", "direction": sqrt_direction, "tie": sqrt_tie, "after": std_q},
    ]
    for dim, deviation in enumerate(deviations):
        if std_q == 0:
            normalized = 0
            direction = 0
            tie = False
        else:
            normalized, direction, tie = round_rational(deviation * ctx.scale, std_q, ctx.tie_rule)
        scaled, gamma_direction, gamma_tie = apply_ratio(normalized, gamma, ctx.scale, ctx.tie_rule)
        scaled, op_direction, op_tie = apply_ratio(scaled, op_scale_num, op_scale_den, ctx.tie_rule)
        result = scaled + int(beta_int)
        pre_quant.append(result)
        steps.append(
            {
                "phase": "layer_norm.normalize",
                "dim": dim,
                "direction": direction,
                "gamma_direction": gamma_direction,
                "op_scale_direction": op_direction,
                "tie": tie or gamma_tie or op_tie,
                "before": deviation,
                "after": result,
            }
        )

    quantized, quant_steps, residual, quant_ties = ctx.quantize(pre_quant, int_bits, "layer_norm.quantize")
    steps.extend(quant_steps)
    half_step_boundary = sorted(set(quant_ties))
    return {
        "vector": quantized,
        "residual": residual,
        "rounding_steps": steps,
        "ties": half_step_boundary,
        "boundary_token": bool(half_step_boundary),
        "details": {
            "dimension": dimension,
            "sum": total,
            "mean_int": mean_q,
            "deviations": deviations,
            "square_sum": square_sum,
            "variance_int": variance_q,
            "std_int": std_q,
            "epsilon_int": epsilon_q,
            "gamma_int": gamma,
            "beta_int": beta_int,
            "half_step_dims": half_step_boundary,
        },
    }
