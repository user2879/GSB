from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .util import fingerprint, round_half_up_ratio


class DataStageError(Exception):
    pass


def build_data_stage(
    spec: dict[str, Any],
    config: dict[str, Any],
    overrides: dict[str, Any],
    data_key: str,
) -> dict[str, Any]:
    rows = generate_rows(spec, config, overrides)
    split = choose_split(rows, spec, config)
    preprocessing = fit_preprocessing(rows, split["train_ids"], config)
    train = encode_rows(rows, split["train_ids"], preprocessing, config)
    validation = encode_rows(rows, split["validation_ids"], preprocessing, config)
    row_digest = fingerprint(
        [
            {"sample_id": row["sample_id"], "values": row["values"], "target": row["target"]}
            for row in rows
        ]
    )
    return {
        "data_key": data_key,
        "data_fingerprint": row_digest,
        "seed": spec["seed"],
        "n_samples": spec["n_samples"],
        "sample_ids": [row["sample_id"] for row in rows],
        "rows": rows,
        "split": {
            "train_ratio": config["split"]["train_ratio"],
            "strategy": "deterministic_stratified_shuffle",
            "train_ids": split["train_ids"],
            "validation_ids": split["validation_ids"],
        },
        "preprocessing": preprocessing,
        "train": train,
        "validation": validation,
        "fit_scope": {
            "imputation": "training fold only",
            "category_encoding": "training fold only",
        },
    }


def generate_rows(
    spec: dict[str, Any], config: dict[str, Any], overrides: dict[str, Any]
) -> list[dict[str, Any]]:
    seed = int(spec["seed"])
    n_samples = int(spec["n_samples"])
    rng = np.random.Generator(np.random.PCG64(seed))
    descriptors = spec["columns"]
    by_name = {descriptor["name"]: descriptor for descriptor in descriptors}
    values_by_name: dict[str, list[Any]] = {}
    target_descriptor: dict[str, Any] | None = None

    for descriptor in descriptors:
        name = descriptor["name"]
        kind = descriptor["type"]
        if kind == "target":
            target_descriptor = descriptor
            continue
        if kind == "identifier":
            values_by_name[name] = list(range(1, n_samples + 1))
            continue
        if kind == "numeric":
            if "redundant_of" in descriptor:
                source = descriptor["redundant_of"]
                source_values = values_by_name.get(source)
                if source_values is None:
                    raise DataStageError(f"redundant source is unavailable: {source}")
                scale = float(descriptor.get("scale", 1.0))
                offset = float(descriptor.get("offset", 0.0))
                noise_std = float(descriptor.get("noise_std", 0.0))
                values = []
                for source_value in source_values:
                    if source_value is None:
                        values.append(None)
                    else:
                        noise = float(rng.normal(0.0, noise_std)) if noise_std else 0.0
                        values.append(float(source_value) * scale + offset + noise)
            else:
                generator = descriptor["generator"]
                if generator == "normal":
                    raw = rng.normal(
                        float(descriptor["mean"]), float(descriptor["std"]), n_samples
                    )
                elif generator == "uniform":
                    raw = rng.uniform(
                        float(descriptor["low"]), float(descriptor["high"]), n_samples
                    )
                elif generator == "lognormal":
                    raw = np.exp(
                        rng.normal(
                            float(descriptor["mean"]),
                            float(descriptor["sigma"]),
                            n_samples,
                        )
                    )
                else:
                    raise DataStageError(f"unsupported numeric generator: {generator}")
                values = [float(value) for value in raw]
            digits = int(descriptor.get("digits", 6))
            values_by_name[name] = [
                None if value is None else round(float(value), digits) for value in values
            ]
            continue
        if kind == "categorical":
            if "redundant_of" in descriptor:
                source = descriptor["redundant_of"]
                source_values = values_by_name.get(source)
                if source_values is None:
                    raise DataStageError(f"redundant source is unavailable: {source}")
                values_by_name[name] = list(source_values)
            else:
                categories = list(descriptor["values"])
                weights = descriptor.get("weights")
                probabilities = None
                if weights is not None:
                    total = float(sum(float(weight) for weight in weights))
                    probabilities = [float(weight) / total for weight in weights]
                indices = rng.choice(
                    len(categories), size=n_samples, replace=True, p=probabilities
                )
                values_by_name[name] = [categories[int(index)] for index in indices]
            continue
        raise DataStageError(f"unsupported descriptor type: {kind}")

    if target_descriptor is None:
        raise DataStageError("target descriptor is missing")

    target_values = generate_target(values_by_name, target_descriptor, rng)
    target_name = target_descriptor["name"]
    values_by_name[target_name] = target_values

    for descriptor in descriptors:
        name = descriptor["name"]
        kind = descriptor["type"]
        if kind in ("identifier", "target"):
            continue
        missing_rate = float(descriptor.get("missing_rate", 0.0))
        if missing_rate <= 0.0:
            continue
        mask = rng.random(n_samples) < missing_rate
        values_by_name[name] = [
            None if bool(mask[index]) else values_by_name[name][index]
            for index in range(n_samples)
        ]

    identifiers = [
        descriptor["name"]
        for descriptor in descriptors
        if descriptor["type"] == "identifier"
    ]
    if len(identifiers) != 1:
        raise DataStageError("exactly one identifier column is required")
    identifier = identifiers[0]
    rows = []
    for index in range(n_samples):
        rows.append(
            {
                "sample_id": int(values_by_name[identifier][index]),
                "values": {
                    descriptor["name"]: values_by_name[descriptor["name"]][index]
                    for descriptor in descriptors
                    if descriptor["type"] not in ("identifier", "target")
                },
                "target": int(values_by_name[target_name][index]),
            }
        )

    apply_overrides(rows, overrides, by_name)
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def generate_target(
    values_by_name: dict[str, list[Any]],
    descriptor: dict[str, Any],
    rng: np.random.Generator,
) -> list[int]:
    n_samples = len(next(iter(values_by_name.values())))
    drivers = descriptor.get("driver_weights", {})
    risk = np.zeros(n_samples, dtype=np.float64)
    for column, raw_weight in drivers.items():
        values = values_by_name[column]
        numeric_values = []
        category_codes: dict[str, int] = {}
        for value in values:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_values.append(float(value))
            elif isinstance(value, str):
                if value not in category_codes:
                    category_codes[value] = len(category_codes)
                numeric_values.append(float(category_codes[value]))
            else:
                numeric_values.append(0.0)
        array = np.asarray(numeric_values, dtype=np.float64)
        mean = float(array.mean())
        std = float(array.std())
        normalized = (array - mean) / std if std > 1e-12 else np.zeros_like(array)
        risk += float(raw_weight) * normalized
    noise_std = float(descriptor.get("noise_std", 1.0))
    if noise_std:
        risk += rng.normal(0.0, noise_std, n_samples)

    positive_rate = float(descriptor["positive_rate"])
    positive_count = int(math.floor(n_samples * positive_rate + 0.5))
    positive_count = min(n_samples - 1, max(1, positive_count))
    order = sorted(range(n_samples), key=lambda index: (-float(risk[index]), index))
    positive = set(order[:positive_count])
    return [1 if index in positive else 0 for index in range(n_samples)]


def apply_overrides(
    rows: list[dict[str, Any]],
    overrides: dict[str, Any],
    descriptor_map: dict[str, dict[str, Any]],
) -> None:
    samples = overrides.get("samples", {}) if isinstance(overrides, dict) else {}
    by_id = {int(row["sample_id"]): row for row in rows}
    target_names = [
        name for name, descriptor in descriptor_map.items() if descriptor["type"] == "target"
    ]
    target_name = target_names[0]
    changed_sources: set[str] = set()
    for raw_id, override in samples.items():
        sample_id = int(raw_id)
        row = by_id[sample_id]
        for column, value in override.get("values", {}).items():
            if column == target_name:
                row["target"] = int(value)
            else:
                row["values"][column] = value
                changed_sources.add(column)
        if "target" in override:
            row["target"] = int(override["target"])

    for descriptor in descriptor_map.values():
        name = descriptor["name"]
        if descriptor["type"] != "numeric" or "redundant_of" not in descriptor:
            continue
        source = descriptor["redundant_of"]
        if source not in changed_sources:
            continue
        for raw_id, override in samples.items():
            if name in override.get("values", {}):
                continue
            row = by_id[int(raw_id)]
            source_value = row["values"].get(source)
            if source_value is None:
                row["values"][name] = None
            else:
                scale = float(descriptor.get("scale", 1.0))
                offset = float(descriptor.get("offset", 0.0))
                digits = int(descriptor.get("digits", 6))
                row["values"][name] = round(float(source_value) * scale + offset, digits)


def choose_split(
    rows: list[dict[str, Any]], spec: dict[str, Any], config: dict[str, Any]
) -> dict[str, list[int]]:
    ratio = float(config["split"]["train_ratio"])
    rng = np.random.Generator(np.random.PCG64(int(spec["seed"]) ^ 0x9E3779B9))
    train_ids: list[int] = []
    validation_ids: list[int] = []
    for label in (0, 1):
        ids = [row["sample_id"] for row in rows if row["target"] == label]
        ids.sort()
        if not ids:
            continue
        permutation = rng.permutation(len(ids))
        permuted = [ids[int(index)] for index in permutation]
        requested = round_half_up_ratio(len(ids) * int(round(ratio * 1_000_000)), 1_000_000)
        requested = min(len(ids), max(1, requested))
        if len(ids) > 1:
            requested = min(len(ids) - 1, requested)
        train_ids.extend(permuted[:requested])
        validation_ids.extend(permuted[requested:])
    train_ids.sort()
    validation_ids.sort()
    if not train_ids or not validation_ids:
        raise DataStageError("split produced an empty train or validation fold")
    return {"train_ids": train_ids, "validation_ids": validation_ids}


def fit_preprocessing(
    rows: list[dict[str, Any]], train_ids: list[int], config: dict[str, Any]
) -> dict[str, Any]:
    train_id_set = set(train_ids)
    train_rows = {row["sample_id"]: row for row in rows if row["sample_id"] in train_id_set}
    numeric_medians: dict[str, float] = {}
    categorical_modes: dict[str, str] = {}
    categorical_values: dict[str, list[str]] = {}
    for name in config["features"]["numeric"]:
        values = sorted(
            float(train_rows[sample_id]["values"][name])
            for sample_id in train_ids
            if train_rows[sample_id]["values"].get(name) is not None
        )
        if not values:
            raise DataStageError(f"training fold has no numeric values for {name}")
        midpoint = len(values) // 2
        if len(values) % 2:
            median = values[midpoint]
        else:
            median = (values[midpoint - 1] + values[midpoint]) / 2.0
        numeric_medians[name] = float(median)
    for name in config["features"]["categorical"]:
        counts = Counter(
            str(train_rows[sample_id]["values"][name])
            for sample_id in train_ids
            if train_rows[sample_id]["values"].get(name) is not None
        )
        if not counts:
            raise DataStageError(f"training fold has no categorical values for {name}")
        mode = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        categorical_modes[name] = mode
        categorical_values[name] = sorted(counts)
    unknown_marker = "__UNKNOWN__"
    while any(unknown_marker in values for values in categorical_values.values()):
        unknown_marker += "_"
    return {
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
        "categorical_values": categorical_values,
        "unknown_marker": unknown_marker,
        "source": "train",
    }


def encode_rows(
    rows: list[dict[str, Any]],
    sample_ids: list[int],
    preprocessing: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    numeric_names = list(config["features"]["numeric"])
    categorical_names = list(config["features"]["categorical"])
    categories = {
        name: list(preprocessing["categorical_values"][name]) for name in categorical_names
    }
    unknown = preprocessing["unknown_marker"]
    category_index = {
        name: {value: index for index, value in enumerate(categories[name])}
        for name in categorical_names
    }
    feature_names = list(numeric_names)
    for name in categorical_names:
        feature_names.extend([f"{name}={value}" for value in categories[name]])
        feature_names.append(f"{name}={unknown}")
    matrix: list[list[float]] = []
    labels: list[int] = []
    typed_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        row = by_id[sample_id]
        vector: list[float] = []
        for name in numeric_names:
            value = row["values"].get(name)
            if value is None:
                value = preprocessing["numeric_medians"][name]
            vector.append(float(value))
        for name in categorical_names:
            value = row["values"].get(name)
            if value is None:
                value = preprocessing["categorical_modes"][name]
            value = str(value)
            index = category_index[name].get(value, -1)
            vector.extend([0.0] * len(categories[name]))
            vector.append(0.0)
            if index >= 0:
                vector[-(len(categories[name]) + 1) + index] = 1.0
            else:
                vector[-1] = 1.0
        matrix.append(vector)
        labels.append(int(row["target"]))
        typed_rows.append(
            {
                "sample_id": sample_id,
                "values": dict(row["values"]),
                "target": int(row["target"]),
            }
        )
    return {
        "sample_ids": list(sample_ids),
        "labels": labels,
        "features": matrix,
        "feature_names": feature_names,
        "rows": typed_rows,
    }
