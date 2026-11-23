from __future__ import annotations

import hashlib
import math
import copy
from dataclasses import dataclass
from typing import Any

from .canonical import fingerprint
from .config import FeatureConfig, TrainingConfig
from .features import FeatureSnapshot
from .fixed import (
    clamp_fixed,
    fraction_scale,
    quantize_real,
    sigmoid_fixed,
    trace_rounding,
)
from .state import Sample
from .visibility import CVRVisibility


MODEL_SCHEMA = "shared-hidden-dual-tower-v1"
PARAM_ORDER = ("shared_w", "shared_b", "ctr_w", "ctr_b", "cvr_w", "cvr_b")


@dataclass(frozen=True)
class ModelDims:
    shared_features: tuple[str, ...]
    ctr_features: tuple[str, ...]
    cvr_features: tuple[str, ...]
    hidden_dim: int

    def payload(self) -> dict[str, Any]:
        return {
            "shared_features": list(self.shared_features),
            "ctr_features": list(self.ctr_features),
            "cvr_features": list(self.cvr_features),
            "hidden_dim": self.hidden_dim,
        }


@dataclass(frozen=True)
class Schedule:
    selected: tuple[tuple[str, str], ...]
    pending: tuple[tuple[str, str], ...]
    batches: tuple[tuple[tuple[str, str], ...], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "selected": [list(key) for key in self.selected],
            "pending": [list(key) for key in self.pending],
            "batches": [[list(key) for key in batch] for batch in self.batches],
        }


def build_dims(config: FeatureConfig) -> ModelDims:
    return ModelDims(
        shared_features=tuple(config.shared_features),
        ctr_features=tuple(config.tower_features["ctr"]),
        cvr_features=tuple(config.tower_features["cvr"]),
        hidden_dim=config.hidden_dim,
    )


def build_schedule(samples: list[Sample], training: TrainingConfig) -> Schedule:
    keys = [sample.key for sample in sorted(samples, key=lambda item: item.key)]
    selected = tuple(keys[: training.global_train_sample_budget])
    pending = tuple(keys[training.global_train_sample_budget :])
    effective_batch = min(training.batch_size, training.per_batch_sample_quota)
    batches = tuple(
        selected[index : index + effective_batch]
        for index in range(0, len(selected), effective_batch)
    )
    return Schedule(selected=selected, pending=pending, batches=batches)


def _hash_unit(seed: int, tag: str, i: int, j: int) -> float:
    material = f"{seed}|{tag}|{i}|{j}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(1 << 64)


def _initial_q(seed: int, tag: str, i: int, j: int, fractional_bits: int, tier: str) -> int:
    unit = _hash_unit(seed, tag, i, j)
    value = (unit * 2.0 - 1.0) * 0.25
    return quantize_real(value, fractional_bits, tier).value


def initialize_parameters(
    dims: ModelDims,
    seed: int,
    fractional_bits: int,
    tier: str,
) -> dict[str, Any]:
    hidden = dims.hidden_dim
    shared_w = [
        [
            _initial_q(seed, "shared_w", i, j, fractional_bits, tier)
            for j in range(len(dims.shared_features))
        ]
        for i in range(hidden)
    ]
    ctr_dim = hidden + len(dims.ctr_features)
    cvr_dim = hidden + len(dims.cvr_features)
    ctr_w = [
        _initial_q(seed, "ctr_w", 0, j, fractional_bits, tier) for j in range(ctr_dim)
    ]
    cvr_w = [
        _initial_q(seed, "cvr_w", 0, j, fractional_bits, tier) for j in range(cvr_dim)
    ]
    return {
        "shared_w": shared_w,
        "shared_b": [0 for _ in range(hidden)],
        "ctr_w": ctr_w,
        "ctr_b": [0],
        "cvr_w": cvr_w,
        "cvr_b": [0],
    }


def dequantize_parameters(params: dict[str, Any], fractional_bits: int) -> dict[str, Any]:
    scale = fraction_scale(fractional_bits)
    result: dict[str, Any] = {}
    for key in PARAM_ORDER:
        values = params[key]
        if key in ("shared_w",):
            result[key] = [[value / scale for value in row] for row in values]
        else:
            result[key] = [value / scale for value in values]
    return result


def _vector(row: dict[str, Any], names: tuple[str, ...], *, fixed: bool) -> list[Any]:
    if fixed:
        return [int(row.get(name, 0)) for name in names]
    return [float(row.get(name, 0.0)) for name in names]


def _fixed_forward(
    params: dict[str, Any],
    x_shared: list[int],
    x_ctr: list[int],
    x_cvr: list[int],
    fractional_bits: int,
    tier: str,
    *,
    trace: list[dict[str, Any]] | None,
    context: dict[str, Any] | None = None,
) -> tuple[list[int], int, int, int, int]:
    scale = fraction_scale(fractional_bits)
    context = context or {}
    hidden: list[int] = []
    hidden_pre: list[int] = []
    for node, weights in enumerate(params["shared_w"]):
        numerator = sum(weight * value for weight, value in zip(weights, x_shared))
        if trace is None:
            value = (2 * abs(numerator) + scale) // (2 * scale)
            value = -value if numerator < 0 and value != 0 else value
        else:
            value, record = trace_rounding(
                "shared_forward",
                "dot_rescale",
                numerator,
                scale,
                parameter=f"shared_w[{node}]",
                extra=context,
            )
            trace.append(record)
        value += params["shared_b"][node]
        value, clipped, direction = clamp_fixed(value, tier)
        if clipped and trace is not None:
            trace.append(
                {
                    "stage": "shared_forward",
                    "operation": "tier_clip",
                    "parameter": f"shared_h[{node}]",
                    "rounded": value,
                    "residual": 0,
                    "direction": direction,
                    **context,
                }
            )
        hidden_pre.append(value)
        hidden.append(max(0, value))

    def tower_forward(tower: str, specific: list[int]) -> tuple[int, int]:
        weights = params[f"{tower}_w"]
        activations = hidden + specific
        numerator = sum(weight * value for weight, value in zip(weights, activations))
        if trace is None:
            value = (2 * abs(numerator) + scale) // (2 * scale)
            value = -value if numerator < 0 and value != 0 else value
        else:
            value, record = trace_rounding(
                f"{tower}_forward",
                "dot_rescale",
                numerator,
                scale,
                parameter=f"{tower}_w",
                extra=context,
            )
            trace.append(record)
        value += params[f"{tower}_b"][0]
        value, clipped, direction = clamp_fixed(value, tier)
        logistic_low = -8 * scale
        logistic_high = 8 * scale
        if value < logistic_low or value > logistic_high:
            value = min(logistic_high, max(logistic_low, value))
            if trace is not None:
                trace.append(
                    {
                        "stage": f"{tower}_forward",
                        "operation": "logistic_domain_clip",
                        "parameter": f"{tower}_preactivation",
                        "rounded": value,
                        "residual": 0,
                        "direction": "CLIPPED_LOW" if value == logistic_low else "CLIPPED_HIGH",
                        **context,
                    }
                )
        score, _, _ = sigmoid_fixed(value, fractional_bits, tier)
        return value, score

    ctr_pre, ctr_score = tower_forward("ctr", x_ctr)
    cvr_pre, cvr_score = tower_forward("cvr", x_cvr)
    return hidden, ctr_pre, ctr_score, cvr_pre, cvr_score


def _float_sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _float_forward(
    params: dict[str, Any],
    x_shared: list[float],
    x_ctr: list[float],
    x_cvr: list[float],
) -> tuple[list[float], float, float, float, float]:
    hidden_pre = [
        sum(weight * value for weight, value in zip(weights, x_shared)) + params["shared_b"][node]
        for node, weights in enumerate(params["shared_w"])
    ]
    hidden = [max(0.0, value) for value in hidden_pre]
    ctr_pre = sum(weight * value for weight, value in zip(params["ctr_w"], hidden + x_ctr)) + params[
        "ctr_b"
    ][0]
    cvr_pre = sum(weight * value for weight, value in zip(params["cvr_w"], hidden + x_cvr)) + params[
        "cvr_b"
    ][0]
    ctr_score = _float_sigmoid(max(-8.0, min(8.0, ctr_pre)))
    cvr_score = _float_sigmoid(max(-8.0, min(8.0, cvr_pre)))
    return hidden, ctr_pre, ctr_score, cvr_pre, cvr_score


def _loss(score: float, target: int, weight: float = 1.0) -> float:
    clipped = max(1e-12, min(1.0 - 1e-12, score))
    numerator = clipped if target == 1 else 1.0 - clipped
    return -math.log(numerator) * weight


def _accumulate_gradients(
    *,
    gradient: dict[str, Any],
    prefix: str,
    delta: int,
    activations: list[int],
    fractional_bits: int,
    trace: list[dict[str, Any]] | None,
    context: dict[str, Any],
) -> None:
    scale = fraction_scale(fractional_bits)
    for index, activation in enumerate(activations):
        numerator = delta * activation
        if trace is None:
            value = (2 * abs(numerator) + scale) // (2 * scale)
            value = -value if numerator < 0 and value != 0 else value
        else:
            value, record = trace_rounding(
                "tower_gradient",
                "gradient_rescale",
                numerator,
                scale,
                parameter=f"{prefix}_w[{index}]",
                extra=context,
            )
            trace.append(record)
        gradient[f"{prefix}_w"][index] += value
    gradient[f"{prefix}_b"][0] += delta


def _average_gradient(sum_value: int, batch_size: int, trace: list[dict[str, Any]] | None, parameter: str, context: dict[str, Any]) -> int:
    if trace is None:
        return (2 * abs(sum_value) + batch_size) // (2 * batch_size) * (-1 if sum_value < 0 else 1)
    value, record = trace_rounding(
        "batch_gradient",
        "batch_average",
        sum_value,
        batch_size,
        parameter=parameter,
        extra=context,
    )
    trace.append(record)
    return value


def _update_parameter(
    value: int,
    gradient: int,
    lr_q: int,
    fractional_bits: int,
    tier: str,
    trace: list[dict[str, Any]] | None,
    parameter: str,
    context: dict[str, Any],
) -> int:
    scale = fraction_scale(fractional_bits)
    numerator = lr_q * gradient
    if trace is None:
        delta = (2 * abs(numerator) + scale) // (2 * scale)
        delta = -delta if numerator < 0 and delta != 0 else delta
    else:
        delta, record = trace_rounding(
            "parameter_update",
            "lr_gradient_rescale",
            numerator,
            scale,
            parameter=parameter,
            extra=context,
        )
        trace.append(record)
    updated, clipped, direction = clamp_fixed(value - delta, tier)
    if clipped and trace is not None:
        trace.append(
            {
                "stage": "parameter_update",
                "operation": "tier_clip",
                "parameter": parameter,
                "rounded": updated,
                "residual": 0,
                "direction": direction,
                **context,
            }
        )
    return updated


def _zero_gradient(dims: ModelDims) -> dict[str, Any]:
    return {
        "shared_w": [[0 for _ in dims.shared_features] for _ in range(dims.hidden_dim)],
        "shared_b": [0 for _ in range(dims.hidden_dim)],
        "ctr_w": [0 for _ in range(dims.hidden_dim + len(dims.ctr_features))],
        "ctr_b": [0],
        "cvr_w": [0 for _ in range(dims.hidden_dim + len(dims.cvr_features))],
        "cvr_b": [0],
    }


def _apply_updates(
    params: dict[str, Any],
    gradient: dict[str, Any],
    batch_size: int,
    lr_q: int,
    fractional_bits: int,
    tier: str,
    trace: list[dict[str, Any]] | None,
    context: dict[str, Any],
) -> None:
    for i in range(len(params["shared_w"])):
        for j in range(len(params["shared_w"][i])):
            avg = _average_gradient(
                gradient["shared_w"][i][j],
                batch_size,
                trace,
                f"shared_w[{i}][{j}]",
                context,
            )
            params["shared_w"][i][j] = _update_parameter(
                params["shared_w"][i][j],
                avg,
                lr_q,
                fractional_bits,
                tier,
                trace,
                f"shared_w[{i}][{j}]",
                context,
            )
    for i in range(len(params["shared_b"])):
        avg = _average_gradient(
            gradient["shared_b"][i], batch_size, trace, f"shared_b[{i}]", context
        )
        params["shared_b"][i] = _update_parameter(
            params["shared_b"][i],
            avg,
            lr_q,
            fractional_bits,
            tier,
            trace,
            f"shared_b[{i}]",
            context,
        )
    for tower in ("ctr", "cvr"):
        for index in range(len(params[f"{tower}_w"])):
            avg = _average_gradient(
                gradient[f"{tower}_w"][index],
                batch_size,
                trace,
                f"{tower}_w[{index}]",
                context,
            )
            params[f"{tower}_w"][index] = _update_parameter(
                params[f"{tower}_w"][index],
                avg,
                lr_q,
                fractional_bits,
                tier,
                trace,
                f"{tower}_w[{index}]",
                context,
            )
        avg = _average_gradient(
            gradient[f"{tower}_b"][0], batch_size, trace, f"{tower}_b[0]", context
        )
        params[f"{tower}_b"][0] = _update_parameter(
            params[f"{tower}_b"][0],
            avg,
            lr_q,
            fractional_bits,
            tier,
            trace,
            f"{tower}_b[0]",
            context,
        )


def _labels_payload(
    samples: list[Sample],
    visibilities: dict[tuple[str, str], CVRVisibility],
) -> dict[str, Any]:
    return {
        "samples": [
            {
                "sample": sample.sample_id,
                "click": sample.click,
                "conversion": sample.conversion,
                "delay_ms": sample.conversion_delay_ms,
                "visibility": visibilities[sample.key].to_dict(),
            }
            for sample in sorted(samples, key=lambda item: item.key)
        ]
    }


def train_models(
    config: FeatureConfig,
    training: TrainingConfig,
    samples: list[Sample],
    features: FeatureSnapshot,
    visibilities: dict[tuple[str, str], CVRVisibility],
    quantization_tier: str,
    fractional_bits: int,
) -> dict[str, Any]:
    dims = build_dims(config)
    schedule = build_schedule(samples, training)
    scale = fraction_scale(fractional_bits)
    lr_q = quantize_real(training.learning_rate, fractional_bits, quantization_tier).value
    if lr_q <= 0:
        lr_q = 1
    fixed_params = initialize_parameters(dims, config.seed ^ training.seed, fractional_bits, quantization_tier)
    ref_params = dequantize_parameters(fixed_params, fractional_bits)
    initial_fingerprint = fingerprint(
        {
            "schema": MODEL_SCHEMA,
            "dims": dims.payload(),
            "initial_fixed": fixed_params,
            "tier": quantization_tier,
            "fractional_bits": fractional_bits,
            "seed": config.seed ^ training.seed,
        }
    )
    label_payload = _labels_payload(samples, visibilities)
    labels_key = fingerprint(label_payload)
    sample_map = {sample.key: sample for sample in samples}
    arithmetic_trace: list[dict[str, Any]] = []
    loss_trace: list[dict[str, Any]] = []
    reference_loss_trace: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    batch_serial = 0

    for epoch in range(1, training.epochs + 1):
        for batch_index, batch in enumerate(schedule.batches, start=1):
            gradient = _zero_gradient(dims)
            ref_scratch: list[dict[str, Any]] = []
            ctr_loss_sum = 0.0
            cvr_loss_sum = 0.0
            ref_ctr_loss_sum = 0.0
            ref_cvr_loss_sum = 0.0
            ctr_grad_batch: list[dict[str, Any]] = []
            cvr_grad_batch: list[dict[str, Any]] = []
            context = {"epoch": epoch, "batch": batch_index}
            for key in batch:
                sample = sample_map[key]
                x_shared = _vector(features.fixed_vectors[key], dims.shared_features, fixed=True)
                x_ctr = _vector(features.fixed_vectors[key], dims.ctr_features, fixed=True)
                x_cvr = _vector(features.fixed_vectors[key], dims.cvr_features, fixed=True)
                context_sample = {**context, "sample": sample.sample_id}
                hidden, ctr_pre, ctr_score, cvr_pre, cvr_score = _fixed_forward(
                    fixed_params,
                    x_shared,
                    x_ctr,
                    x_cvr,
                    fractional_bits,
                    quantization_tier,
                    trace=arithmetic_trace,
                    context=context_sample,
                )
                visibility = visibilities[key]
                ctr_target = sample.click
                cvr_target = visibility.effective_target
                ctr_loss = _loss(ctr_score / scale, ctr_target)
                cvr_loss = _loss(cvr_score / scale, cvr_target, visibility.weight / scale)
                ctr_loss_sum += ctr_loss
                cvr_loss_sum += cvr_loss

                ctr_delta_raw = ctr_score - ctr_target * scale
                ctr_delta, ctr_delta_trace = trace_rounding(
                    "inference_weighting",
                    "ctr_delta",
                    ctr_delta_raw,
                    1,
                    sample=sample.sample_id,
                    extra=context,
                )
                arithmetic_trace.append(ctr_delta_trace)
                cvr_weighted_num = (cvr_score - cvr_target * scale) * visibility.weight
                cvr_delta, cvr_delta_trace = trace_rounding(
                    "inference_weighting",
                    "cvr_weighted_delta",
                    cvr_weighted_num,
                    scale,
                    sample=sample.sample_id,
                    extra=context,
                )
                arithmetic_trace.append(cvr_delta_trace)
                ctr_activations = hidden + x_ctr
                cvr_activations = hidden + x_cvr
                _accumulate_gradients(
                    gradient=gradient,
                    prefix="ctr",
                    delta=ctr_delta,
                    activations=ctr_activations,
                    fractional_bits=fractional_bits,
                    trace=arithmetic_trace,
                    context=context_sample,
                )
                _accumulate_gradients(
                    gradient=gradient,
                    prefix="cvr",
                    delta=cvr_delta,
                    activations=cvr_activations,
                    fractional_bits=fractional_bits,
                    trace=arithmetic_trace,
                    context=context_sample,
                )
                ctr_grad_batch.append(
                    {
                        "sample": sample.sample_id,
                        "target": ctr_target,
                        "score_q": ctr_score,
                        "score": ctr_score / scale,
                        "loss": ctr_loss,
                        "delta_q": ctr_delta,
                    }
                )
                cvr_grad_batch.append(
                    {
                        "sample": sample.sample_id,
                        "target": cvr_target,
                        "visible": visibility.visible,
                        "visibility_reason": visibility.reason,
                        "weight_q": visibility.weight,
                        "score_q": cvr_score,
                        "score": cvr_score / scale,
                        "loss": cvr_loss,
                        "delta_q": cvr_delta,
                    }
                )
                for hidden_index in range(dims.hidden_dim):
                    numerator = (
                        ctr_delta * fixed_params["ctr_w"][hidden_index]
                        + cvr_delta * fixed_params["cvr_w"][hidden_index]
                    )
                    dhidden, record = trace_rounding(
                        "shared_gradient",
                        "dhidden_rescale",
                        numerator,
                        scale,
                        sample=sample.sample_id,
                        parameter=f"shared_h[{hidden_index}]",
                        extra=context,
                    )
                    arithmetic_trace.append(record)
                    if hidden[hidden_index] <= 0:
                        dhidden = 0
                    for shared_index, shared_value in enumerate(x_shared):
                        grad_num = dhidden * shared_value
                        grad, grad_record = trace_rounding(
                            "shared_gradient",
                            "gradient_rescale",
                            grad_num,
                            scale,
                            sample=sample.sample_id,
                            parameter=f"shared_w[{hidden_index}][{shared_index}]",
                            extra=context,
                        )
                        arithmetic_trace.append(grad_record)
                        gradient["shared_w"][hidden_index][shared_index] += grad
                    gradient["shared_b"][hidden_index] += dhidden

                rx_shared = _vector(features.raw_vectors[key], dims.shared_features, fixed=False)
                rx_ctr = _vector(features.raw_vectors[key], dims.ctr_features, fixed=False)
                rx_cvr = _vector(features.raw_vectors[key], dims.cvr_features, fixed=False)
                rhidden, rctr_pre, rctr_score, rcvr_pre, rcvr_score = _float_forward(
                    ref_params, rx_shared, rx_ctr, rx_cvr
                )
                ref_ctr_loss_sum += _loss(rctr_score, ctr_target)
                ref_cvr_loss_sum += _loss(rcvr_score, cvr_target, visibility.weight / scale)
                rctr_delta = rctr_score - ctr_target
                rcvr_delta = (rcvr_score - cvr_target) * (visibility.weight / scale)
                ctr_grad_batch[-1]["reference_score"] = rctr_score
                ctr_grad_batch[-1]["reference_loss"] = _loss(rctr_score, ctr_target)
                cvr_grad_batch[-1]["reference_score"] = rcvr_score
                cvr_grad_batch[-1]["reference_loss"] = _loss(
                    rcvr_score, cvr_target, visibility.weight / scale
                )
                ref_scratch.append(
                    {
                        "hidden": rhidden,
                        "rctr_delta": rctr_delta,
                        "rcvr_delta": rcvr_delta,
                        "rx_shared": rx_shared,
                        "rx_ctr": rx_ctr,
                        "rx_cvr": rx_cvr,
                    }
                )

            _apply_updates(
                fixed_params,
                gradient,
                len(batch),
                lr_q,
                fractional_bits,
                quantization_tier,
                arithmetic_trace,
                context,
            )

            ref_grad = {
                "shared_w": [[0.0 for _ in dims.shared_features] for _ in range(dims.hidden_dim)],
                "shared_b": [0.0 for _ in range(dims.hidden_dim)],
                "ctr_w": [0.0 for _ in range(dims.hidden_dim + len(dims.ctr_features))],
                "ctr_b": [0.0],
                "cvr_w": [0.0 for _ in range(dims.hidden_dim + len(dims.cvr_features))],
                "cvr_b": [0.0],
            }
            for item in ref_scratch:
                rhidden = item["hidden"]
                rctr_delta = item["rctr_delta"]
                rcvr_delta = item["rcvr_delta"]
                for index, value in enumerate(rhidden + item["rx_ctr"]):
                    ref_grad["ctr_w"][index] += rctr_delta * value
                ref_grad["ctr_b"][0] += rctr_delta
                for index, value in enumerate(rhidden + item["rx_cvr"]):
                    ref_grad["cvr_w"][index] += rcvr_delta * value
                ref_grad["cvr_b"][0] += rcvr_delta
                for hidden_index in range(dims.hidden_dim):
                    dhidden = rctr_delta * ref_params["ctr_w"][hidden_index] + rcvr_delta * ref_params[
                        "cvr_w"
                    ][hidden_index]
                    if rhidden[hidden_index] <= 0:
                        dhidden = 0.0
                    for shared_index, value in enumerate(item["rx_shared"]):
                        ref_grad["shared_w"][hidden_index][shared_index] += dhidden * value
                    ref_grad["shared_b"][hidden_index] += dhidden
            for i in range(len(ref_params["shared_w"])):
                for j in range(len(ref_params["shared_w"][i])):
                    ref_params["shared_w"][i][j] -= training.learning_rate * ref_grad["shared_w"][i][j] / len(batch)
            for i in range(len(ref_params["shared_b"])):
                ref_params["shared_b"][i] -= training.learning_rate * ref_grad["shared_b"][i] / len(batch)
            for tower in ("ctr", "cvr"):
                for index in range(len(ref_params[f"{tower}_w"])):
                    ref_params[f"{tower}_w"][index] -= (
                        training.learning_rate * ref_grad[f"{tower}_w"][index] / len(batch)
                    )
                ref_params[f"{tower}_b"][0] -= training.learning_rate * ref_grad[f"{tower}_b"][
                    0
                ] / len(batch)

            loss_trace.append(
                {
                    "epoch": epoch,
                    "batch": batch_index,
                    "samples": [sample_map[key].sample_id for key in batch],
                    "ctr_loss": ctr_loss_sum / len(batch),
                    "cvr_loss": cvr_loss_sum / len(batch),
                    "ctr_details": ctr_grad_batch,
                    "cvr_details": cvr_grad_batch,
                }
            )
            reference_loss_trace.append(
                {
                    "epoch": epoch,
                    "batch": batch_index,
                    "ctr_loss": ref_ctr_loss_sum / len(batch),
                    "cvr_loss": ref_cvr_loss_sum / len(batch),
                }
            )
            batch_serial += 1
            if batch_serial % training.snapshot_commit_interval_batches == 0:
                checkpoints.append(
                    {
                        "batch_serial": batch_serial,
                        "epoch": epoch,
                        "batch": batch_index,
                        "parameter_fingerprint": fingerprint(fixed_params),
                        "fixed_parameters": copy.deepcopy(fixed_params),
                        "reference_parameters": copy.deepcopy(ref_params),
                    }
                )

    hidden_fixed: dict[str, list[int]] = {}
    hidden_reference: dict[str, list[float]] = {}
    ctr_fixed: dict[str, dict[str, Any]] = {}
    ctr_reference: dict[str, float] = {}
    cvr_fixed: dict[str, dict[str, Any]] = {}
    cvr_reference: dict[str, float] = {}
    for sample in sorted(samples, key=lambda item: item.key):
        x_shared = _vector(features.fixed_vectors[sample.key], dims.shared_features, fixed=True)
        x_ctr = _vector(features.fixed_vectors[sample.key], dims.ctr_features, fixed=True)
        x_cvr = _vector(features.fixed_vectors[sample.key], dims.cvr_features, fixed=True)
        hidden, ctr_pre, ctr_score, cvr_pre, cvr_score = _fixed_forward(
            fixed_params,
            x_shared,
            x_ctr,
            x_cvr,
            fractional_bits,
            quantization_tier,
            trace=None,
        )
        rx_shared = _vector(features.raw_vectors[sample.key], dims.shared_features, fixed=False)
        rx_ctr = _vector(features.raw_vectors[sample.key], dims.ctr_features, fixed=False)
        rx_cvr = _vector(features.raw_vectors[sample.key], dims.cvr_features, fixed=False)
        rhidden, rctr_pre, rctr_score, rcvr_pre, rcvr_score = _float_forward(
            ref_params, rx_shared, rx_ctr, rx_cvr
        )
        hidden_fixed[sample.sample_id] = hidden
        hidden_reference[sample.sample_id] = rhidden
        ctr_fixed[sample.sample_id] = {
            "preactivation_q": ctr_pre,
            "score_q": ctr_score,
            "score": ctr_score / scale,
        }
        cvr_fixed[sample.sample_id] = {
            "preactivation_q": cvr_pre,
            "score_q": cvr_score,
            "score": cvr_score / scale,
        }
        ctr_reference[sample.sample_id] = rctr_score
        cvr_reference[sample.sample_id] = rcvr_score

    return {
        "schema": MODEL_SCHEMA,
        "dims": dims.payload(),
        "tier": quantization_tier,
        "fractional_bits": fractional_bits,
        "scale": scale,
        "learning_rate_q": lr_q,
        "initial_parameter_fingerprint": initial_fingerprint,
        "label_key": labels_key,
        "labels": label_payload,
        "schedule": schedule.payload(),
        "fixed_parameters": fixed_params,
        "reference_parameters": ref_params,
        "hidden_fixed": hidden_fixed,
        "hidden_reference": hidden_reference,
        "ctr_fixed": ctr_fixed,
        "ctr_reference": ctr_reference,
        "cvr_fixed": cvr_fixed,
        "cvr_reference": cvr_reference,
        "loss_trace": loss_trace,
        "reference_loss_trace": reference_loss_trace,
        "arithmetic_trace": arithmetic_trace,
        "checkpoints": checkpoints,
        "selected_samples": [sample_map[key].sample_id for key in schedule.selected],
        "pending_samples": [sample_map[key].sample_id for key in schedule.pending],
    }
