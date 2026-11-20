"""Deterministic residual tree training and tree traversal utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import PreparedData
from .errors import BusinessFailure, InjectedTreeFailure
from .fingerprint import fingerprint, rounded, rounded_tree


def _seed_for_tree(seed: int, tree_index: int, tag: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), 7919, int(tree_index), int(tag)])
    return np.random.default_rng(sequence)


def tree_cache_key(
    config: ExperimentConfig,
    data: PreparedData,
    binning: dict[str, Any],
    tree_index: int,
    fault_stage: str | None = None,
) -> str:
    return fingerprint(
        {
            "stage": "tree/v1",
            "tree_index": tree_index,
            "data_fingerprint": data.data_fingerprint,
            "binning_fingerprint": binning["fingerprint"],
            "features": list(config.features),
            "max_depth": config.max_depth,
            "row_sample_ratio": config.row_sample_ratio,
            "col_sample_ratio": config.col_sample_ratio,
            "min_samples_leaf": config.min_samples_leaf,
            "seed": config.seed,
            "precision_decimals": config.precision_decimals,
            "ensemble_kind": config.ensemble_kind,
            "fault_stage": fault_stage or "",
        }
    )


def _mean(values: list[float]) -> float:
    return math.fsum(values) / float(len(values))


def _sse(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.fsum((value - mean) * (value - mean) for value in values)


def _tree_failure(
    tree_index: int,
    key: str,
    stage: str,
    code: str,
    exception: BaseException,
    decimals: int,
) -> dict[str, Any]:
    failure = BusinessFailure(
        code=code,
        field=f"tree.{tree_index}",
        message=str(exception),
        stage=stage,
        tree_index=tree_index,
        exception_type=type(exception).__name__,
    ).as_dict()
    artifact: dict[str, Any] = {
        "schema": "tree/v1",
        "tree_index": tree_index,
        "key": key,
        "status": "failed",
        "failure": failure,
        "structure": None,
        "structure_summary": {
            "sample_count": 0,
            "node_count": 0,
            "leaf_count": 0,
            "split_count": 0,
            "max_depth": 0,
            "split_features": [],
        },
        "sampled_row_ids": [],
        "sampled_features": [],
        "structure_fingerprint": None,
        "artifact_fingerprint": None,
    }
    artifact["artifact_fingerprint"] = fingerprint({"schema": "tree-artifact/v1", "body": artifact})
    return rounded_tree(artifact, decimals)


def train_tree(
    config: ExperimentConfig,
    data: PreparedData,
    binning: dict[str, Any],
    tree_index: int,
    fault_stage: str | None = None,
) -> dict[str, Any]:
    decimals = config.precision_decimals
    key = tree_cache_key(config, data, binning, tree_index, fault_stage)
    stage = "sampling"
    try:
        if fault_stage == "sampling":
            raise InjectedTreeFailure("sampling")
        row_rng = _seed_for_tree(config.seed, tree_index, 0xA11)
        col_rng = _seed_for_tree(config.seed, tree_index, 0xC01)
        train_indices = list(data.train_indices)
        sample_size = max(
            config.min_samples_leaf,
            min(len(train_indices), int(math.ceil(len(train_indices) * config.row_sample_ratio))),
        )
        sample_size = min(len(train_indices), sample_size)
        if sample_size == len(train_indices):
            sampled_rows = train_indices
        else:
            sampled_rows = sorted(int(value) for value in row_rng.choice(train_indices, size=sample_size, replace=False))
        feature_count = len(data.feature_names)
        feature_sample_size = max(1, int(math.ceil(feature_count * config.col_sample_ratio)))
        if feature_sample_size == feature_count:
            sampled_feature_indices = list(range(feature_count))
        else:
            sampled_feature_indices = sorted(
                int(value) for value in col_rng.choice(feature_count, size=feature_sample_size, replace=False)
            )
        sampled_features = [data.feature_names[index] for index in sampled_feature_indices]

        residual_by_row = {
            row_index: float(data.target[row_index]) - data.base_probability for row_index in sampled_rows
        }
        bin_columns = {
            feature: binning["columns"][feature]["bin_indices"] for feature in sampled_features
        }
        feature_positions = {feature: data.feature_names.index(feature) for feature in sampled_features}
        stage = "split"
        if fault_stage == "split":
            raise InjectedTreeFailure("split")

        node_counter = [0]

        def make_leaf(rows: list[int], depth: int) -> dict[str, Any]:
            node_id = node_counter[0]
            node_counter[0] += 1
            values = [residual_by_row[row] for row in rows]
            return {
                "node_id": node_id,
                "depth": depth,
                "is_leaf": True,
                "samples": len(rows),
                "value": rounded(_mean(values), decimals),
            }

        def choose_split(rows: list[int]) -> tuple[float, str, int, list[int], list[int]] | None:
            if len(rows) < 2 * config.min_samples_leaf:
                return None
            parent_values = [residual_by_row[row] for row in rows]
            parent_sse = _sse(parent_values)
            best_gain = 0.0
            best_key: tuple[int, int] | None = None
            best: tuple[str, int, list[int], list[int]] | None = None
            for feature in sampled_features:
                feature_index = feature_positions[feature]
                bins = bin_columns[feature]
                unique_bins = sorted({int(bins[row]) for row in rows})
                if len(unique_bins) < 2:
                    continue
                for threshold in unique_bins[:-1]:
                    left_rows = [row for row in rows if int(bins[row]) <= threshold]
                    right_rows = [row for row in rows if int(bins[row]) > threshold]
                    if len(left_rows) < config.min_samples_leaf or len(right_rows) < config.min_samples_leaf:
                        continue
                    left_values = [residual_by_row[row] for row in left_rows]
                    right_values = [residual_by_row[row] for row in right_rows]
                    left_sse = _sse(left_values)
                    right_sse = _sse(right_values)
                    gain = parent_sse - left_sse - right_sse
                    if gain < 0.0 and gain > -1e-12:
                        gain = 0.0
                    candidate_key = (feature_index, threshold)
                    if (
                        gain > best_gain + 1e-15
                        or (
                            abs(gain - best_gain) <= 1e-15
                            and (best_key is None or candidate_key < best_key)
                        )
                    ):
                        if gain <= 0.0:
                            continue
                        best_gain = gain
                        best_key = candidate_key
                        best = (feature, threshold, left_rows, right_rows)
            if best is None or best_gain <= 0.0:
                return None
            feature, threshold, left_rows, right_rows = best
            return best_gain, feature, threshold, left_rows, right_rows

        def build_node(rows: list[int], depth: int) -> dict[str, Any]:
            rows = sorted(rows)
            if depth >= config.max_depth:
                return make_leaf(rows, depth)
            split = choose_split(rows)
            if split is None:
                return make_leaf(rows, depth)
            gain, feature, threshold, left_rows, right_rows = split
            node_id = node_counter[0]
            node_counter[0] += 1
            return {
                "node_id": node_id,
                "depth": depth,
                "is_leaf": False,
                "samples": len(rows),
                "feature": feature,
                "feature_index": feature_positions[feature],
                "threshold_bin": int(threshold),
                "gain": rounded(gain, decimals),
                "left": build_node(left_rows, depth + 1),
                "right": build_node(right_rows, depth + 1),
            }

        structure = build_node(sampled_rows, 0)
        stage = "leaf"
        if fault_stage == "leaf":
            raise InjectedTreeFailure("leaf")

        split_features: set[str] = set()
        leaf_count = 0
        node_count = 0
        max_depth = 0

        def summarize(node: dict[str, Any]) -> None:
            nonlocal leaf_count, node_count, max_depth
            node_count += 1
            max_depth = max(max_depth, int(node["depth"]))
            if node["is_leaf"]:
                leaf_count += 1
            else:
                split_features.add(str(node["feature"]))
                summarize(node["left"])
                summarize(node["right"])

        summarize(structure)
        structure_summary = {
            "sample_count": len(sampled_rows),
            "node_count": node_count,
            "leaf_count": leaf_count,
            "split_count": node_count - leaf_count,
            "max_depth": max_depth,
            "split_features": sorted(split_features),
            "row_sample_ratio": config.row_sample_ratio,
            "col_sample_ratio": config.col_sample_ratio,
            "min_samples_leaf": config.min_samples_leaf,
            "ensemble_kind": config.ensemble_kind,
        }
        rounded_structure = rounded_tree(structure, decimals)
        structure_fingerprint = fingerprint(
            {
                "schema": "tree-structure/v1",
                "tree_index": tree_index,
                "sampled_row_ids": [data.row_ids[row] for row in sampled_rows],
                "sampled_features": sampled_features,
                "structure": rounded_structure,
                "summary": structure_summary,
            }
        )
        artifact: dict[str, Any] = {
            "schema": "tree/v1",
            "tree_index": tree_index,
            "key": key,
            "status": "ok",
            "structure": rounded_structure,
            "structure_summary": structure_summary,
            "sampled_row_ids": [data.row_ids[row] for row in sampled_rows],
            "sampled_features": sampled_features,
            "structure_fingerprint": structure_fingerprint,
            "base_probability": rounded(data.base_probability, decimals),
            "failure": None,
        }
        artifact["artifact_fingerprint"] = fingerprint({"schema": "tree-artifact/v1", "body": artifact})
        return rounded_tree(artifact, decimals)
    except Exception as exception:  # Deliberately isolate one tree from the rest of its siblings.
        stage_codes = {
            "sampling": "TREE_SAMPLING_FAILURE",
            "split": "TREE_SPLIT_FAILURE",
            "leaf": "TREE_LEAF_FAILURE",
        }
        return _tree_failure(
            tree_index=tree_index,
            key=key,
            stage=stage,
            code=stage_codes.get(stage, "TREE_UNEXPECTED_FAILURE"),
            exception=exception,
            decimals=decimals,
        )


def follow_tree(
    tree: dict[str, Any],
    bins_by_feature: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node = tree["structure"]
    path: list[dict[str, Any]] = []
    while not node["is_leaf"]:
        path.append(node)
        feature = str(node["feature"])
        if bins_by_feature[feature] <= int(node["threshold_bin"]):
            node = node["left"]
        else:
            node = node["right"]
    return node, path


def leaf_value(tree: dict[str, Any], bins_by_feature: dict[str, int]) -> float:
    leaf, _ = follow_tree(tree, bins_by_feature)
    return float(leaf["value"])


def path_contributions(
    tree: dict[str, Any],
    bins_by_feature: dict[str, int],
    feature_names: list[str],
    weight: float,
    decimals: int,
) -> tuple[float, dict[str, float]]:
    leaf, path = follow_tree(tree, bins_by_feature)
    value = float(leaf["value"])
    contributions = {feature: 0.0 for feature in feature_names}
    if not path:
        contributions["__bias__"] = rounded(value * weight, decimals)
    else:
        gains = [float(node["gain"]) for node in path]
        total_gain = math.fsum(gains)
        if total_gain <= 0.0:
            shares = [1.0 / len(path)] * len(path)
        else:
            shares = [gain / total_gain for gain in gains]
        for node, share in zip(path, shares):
            feature = str(node["feature"])
            contributions[feature] = rounded(
                contributions.get(feature, 0.0) + value * weight * share,
                decimals,
            )
    return rounded(value * weight, decimals), contributions
