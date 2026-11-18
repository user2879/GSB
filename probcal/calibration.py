from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .util import (
    SCHEME_ORDER,
    fixed_probability_to_float,
    quantize_probability,
    scheme_rank,
    sigmoid,
)


def build_calibration_stage(
    data_artifact: dict[str, Any],
    model_artifact: dict[str, Any],
    calibrator_config: dict[str, Any],
    fixed: dict[str, int],
    calibration_key: str,
    max_workers: int,
) -> dict[str, Any]:
    tasks: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for model in model_artifact["models"]:
        for scheme in SCHEME_ORDER:
            scheme_config = calibrator_config[scheme]
            if not scheme_config["enabled"]:
                continue
            stable_key = (
                model["model_name"],
                scheme_rank(scheme),
                int(model["fold_id"]),
                model["params_hash"],
            )
            tasks.append(
                (
                    stable_key,
                    {
                        "model": model,
                        "scheme": scheme,
                        "scheme_config": scheme_config,
                    },
                )
            )
    tasks.sort(key=lambda item: item[0])
    if max_workers <= 1 or len(tasks) <= 1:
        results = [
            _calibrate_task(task, data_artifact, fixed) for _, task in tasks
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="calibration"
        ) as pool:
            futures = {
                pool.submit(_calibrate_task, task, data_artifact, fixed): stable_key
                for stable_key, task in tasks
            }
            by_key = {stable_key: future.result() for future, stable_key in futures.items()}
        results = [by_key[stable_key] for stable_key, _ in tasks]
    results.sort(
        key=lambda item: (
            item["model_name"],
            scheme_rank(item["scheme"]),
            item["fold_id"],
            item["params_hash"],
        )
    )
    return {
        "calibration_key": calibration_key,
        "model_key": model_artifact["model_key"],
        "data_key": data_artifact["data_key"],
        "fixed_point": fixed,
        "fit_scope": {
            "calibration_fit": "training fold only",
            "validation_labels_read_during_fit": False,
        },
        "enabled_schemes": [
            scheme for scheme in SCHEME_ORDER if calibrator_config[scheme]["enabled"]
        ],
        "results": results,
        "execution": {
            "task_order_policy": "sorted by (model_name, scheme, fold_id, params_hash)",
        },
    }


def _calibrate_task(
    task: dict[str, Any], data_artifact: dict[str, Any], fixed: dict[str, int]
) -> dict[str, Any]:
    model = task["model"]
    scheme = task["scheme"]
    config = task["scheme_config"]
    train_probability_by_id = {
        sample_id: probability
        for sample_id, probability in zip(
            model["raw_probabilities"]["train"]["sample_ids"],
            model["raw_probabilities"]["train"]["probabilities_fixed"],
            strict=True,
        )
    }
    train_ids = list(data_artifact["train"]["sample_ids"])
    train_labels = list(data_artifact["train"]["labels"])
    train_pairs = [
        (sample_id, train_probability_by_id[sample_id], int(label))
        for sample_id, label in zip(train_ids, train_labels, strict=True)
    ]
    validation_probability_by_id = {
        sample_id: probability
        for sample_id, probability in zip(
            model["raw_probabilities"]["validation"]["sample_ids"],
            model["raw_probabilities"]["validation"]["probabilities_fixed"],
            strict=True,
        )
    }
    validation_ids = list(data_artifact["validation"]["sample_ids"])
    fitted = _fit_scheme(scheme, config, train_pairs, fixed)
    records = []
    for sample_id in validation_ids:
        raw_fixed = int(validation_probability_by_id[sample_id])
        calibrated_fixed = _map_probability(scheme, fitted["mapping"], raw_fixed, fixed)
        records.append(
            {
                "sample_id": int(sample_id),
                "raw_probability_fixed": raw_fixed,
                "calibrated_probability_fixed": calibrated_fixed,
            }
        )
    inversions = _find_inversions(
        records, scheme, fitted.get("bin_count"), fixed["scale"]
    )
    return {
        "model_name": model["model_name"],
        "model_id": model["model_id"],
        "model_type": model["model_type"],
        "params_hash": model["params_hash"],
        "fold_id": int(model["fold_id"]),
        "scheme": scheme,
        "stable_key": [
            model["model_name"],
            scheme,
            int(model["fold_id"]),
            model["params_hash"],
        ],
        "scheme_config": config,
        "fitted_parameters": fitted["parameters"],
        "bin_details": fitted.get("bins", []),
        "bin_count": fitted.get("bin_count"),
        "probabilities": records,
        "inversions": inversions,
        "fit_audit": {
            "train_sample_count": len(train_pairs),
            "validation_label_count_read": 0,
        },
    }


def _fit_scheme(
    scheme: str,
    config: dict[str, Any],
    train_pairs: list[tuple[int, int, int]],
    fixed: dict[str, int],
) -> dict[str, Any]:
    if scheme == "raw":
        return {"parameters": {"type": "identity"}, "mapping": {"kind": "identity"}}
    if scheme == "platt":
        return _fit_platt(config, train_pairs, fixed)
    if scheme == "isotonic":
        return _fit_isotonic(config, train_pairs, fixed)
    if scheme == "histogram":
        return _fit_histogram(config, train_pairs, fixed)
    raise ValueError(f"unknown calibration scheme: {scheme}")


def _fit_platt(
    config: dict[str, Any], train_pairs: list[tuple[int, int, int]], fixed: dict[str, int]
) -> dict[str, Any]:
    scale = fixed["scale"]
    regularization = float(config.get("regularization", 0.0))
    iterations = int(config.get("iterations", 200))
    learning_rate = float(config.get("learning_rate", 0.2))
    a = 1.0
    b = 0.0
    for _ in range(iterations):
        gradient_a = 0.0
        gradient_b = 0.0
        for _, probability_fixed, label in train_pairs:
            probability = fixed_probability_to_float(probability_fixed, scale)
            logit = _logit(probability)
            prediction = sigmoid(a * logit + b)
            residual = prediction - label
            gradient_a += residual * logit
            gradient_b += residual
        count = float(len(train_pairs))
        gradient_a = gradient_a / count + regularization * (a - 1.0)
        gradient_b = gradient_b / count
        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b
    return {
        "parameters": {"type": "platt", "a": round(a, 12), "b": round(b, 12)},
        "mapping": {"kind": "platt", "a": a, "b": b},
    }


def _fit_isotonic(
    config: dict[str, Any], train_pairs: list[tuple[int, int, int]], fixed: dict[str, int]
) -> dict[str, Any]:
    bins, values = _binned_rates(config, train_pairs, fixed, enforce_monotonic=True)
    return {
        "parameters": {
            "type": "isotonic",
            "bins": int(config["bins"]),
            "regularization": float(config.get("regularization", 0.0)),
            "method": "equal_width_bins_then_weighted_pava",
        },
        "mapping": {"kind": "binned", "values": values, "bins": int(config["bins"])},
        "bins": bins,
        "bin_count": int(config["bins"]),
    }


def _fit_histogram(
    config: dict[str, Any], train_pairs: list[tuple[int, int, int]], fixed: dict[str, int]
) -> dict[str, Any]:
    bins, values = _binned_rates(config, train_pairs, fixed, enforce_monotonic=False)
    return {
        "parameters": {
            "type": "histogram",
            "bins": int(config["bins"]),
            "regularization": float(config.get("regularization", 0.0)),
        },
        "mapping": {"kind": "binned", "values": values, "bins": int(config["bins"])},
        "bins": bins,
        "bin_count": int(config["bins"]),
    }


def _binned_rates(
    config: dict[str, Any],
    train_pairs: list[tuple[int, int, int]],
    fixed: dict[str, int],
    enforce_monotonic: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    bin_count = int(config["bins"])
    scale = fixed["scale"]
    regularization = float(config.get("regularization", 0.0))
    counts = [0] * bin_count
    positives = [0] * bin_count
    for _, probability_fixed, label in train_pairs:
        index = min(bin_count - 1, int(probability_fixed) * bin_count // scale)
        counts[index] += 1
        positives[index] += int(label)
    total_count = len(train_pairs)
    total_positive = sum(positives)
    fallback = (total_positive + regularization) / (total_count + 2.0 * regularization)
    raw_values: list[float | None] = []
    for count, positive in zip(counts, positives, strict=True):
        if count == 0:
            raw_values.append(None)
        else:
            raw_values.append((positive + regularization) / (count + 2.0 * regularization))
    if enforce_monotonic:
        values = _pava_with_empty_fill(raw_values, counts, fallback)
    else:
        values = [fallback if value is None else value for value in raw_values]
    bin_values_fixed = [
        quantize_probability(value, fixed["quantization_levels"], scale) for value in values
    ]
    bins = []
    for index, value_fixed in enumerate(bin_values_fixed):
        lower = index * scale // bin_count
        upper = (index + 1) * scale // bin_count
        bins.append(
            {
                "bin": index,
                "lower_fixed_inclusive": lower,
                "upper_fixed_exclusive": upper if index < bin_count - 1 else scale,
                "count": counts[index],
                "positive_count": positives[index],
                "value_fixed": value_fixed,
            }
        )
    return bins, bin_values_fixed


def _pava_with_empty_fill(
    raw_values: list[float | None], counts: list[int], fallback: float
) -> list[float]:
    blocks: list[dict[str, float]] = []
    for index, value in enumerate(raw_values):
        if value is None:
            continue
        blocks.append(
            {
                "start": index,
                "end": index,
                "weight": float(counts[index]),
                "mean": float(value),
            }
        )
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            mean = (left["mean"] * left["weight"] + right["mean"] * right["weight"]) / weight
            blocks.append(
                {
                    "start": left["start"],
                    "end": right["end"],
                    "weight": weight,
                    "mean": mean,
                }
            )
    values: list[float | None] = [None] * len(raw_values)
    for block in blocks:
        for index in range(int(block["start"]), int(block["end"]) + 1):
            values[index] = block["mean"]
    populated = [index for index, value in enumerate(values) if value is not None]
    if not populated:
        return [fallback] * len(values)
    first = populated[0]
    leading_value = float(values[first])
    for index in range(first):
        values[index] = leading_value
    last = leading_value
    for index in range(first, len(values)):
        if values[index] is None:
            values[index] = last
        else:
            last = float(values[index])
    return [float(value) for value in values]


def _map_probability(
    scheme: str, mapping: dict[str, Any], probability_fixed: int, fixed: dict[str, int]
) -> int:
    scale = fixed["scale"]
    if scheme == "raw":
        return min(scale, max(0, int(probability_fixed)))
    probability = fixed_probability_to_float(probability_fixed, scale)
    if scheme == "platt":
        value = sigmoid(float(mapping["a"]) * _logit(probability) + float(mapping["b"]))
    elif scheme in ("isotonic", "histogram"):
        bins = int(mapping["bins"])
        index = min(bins - 1, int(probability_fixed) * bins // scale)
        value = float(mapping["values"][index]) / scale
    else:
        raise ValueError(f"unknown calibration scheme: {scheme}")
    return quantize_probability(value, fixed["quantization_levels"], scale)


def _find_inversions(
    records: list[dict[str, int]], scheme: str, bin_count: int | None, scale: int
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda record: (record["raw_probability_fixed"], record["sample_id"]),
    )
    inversions = []
    for left, right in zip(ordered, ordered[1:]):
        if right["raw_probability_fixed"] < left["raw_probability_fixed"]:
            continue
        if right["calibrated_probability_fixed"] >= left["calibrated_probability_fixed"]:
            continue
        bin_number = None
        if bin_count:
            bin_number = int(right["raw_probability_fixed"]) * bin_count // scale
            bin_number = min(bin_count - 1, bin_number)
        inversions.append(
            {
                "scheme": scheme,
                "left_sample_id": left["sample_id"],
                "right_sample_id": right["sample_id"],
                "left_raw_probability_fixed": left["raw_probability_fixed"],
                "right_raw_probability_fixed": right["raw_probability_fixed"],
                "left_calibrated_probability_fixed": left["calibrated_probability_fixed"],
                "right_calibrated_probability_fixed": right["calibrated_probability_fixed"],
                "right_bin": bin_number,
            }
        )
    return inversions


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    return math.log(clipped / (1.0 - clipped))
