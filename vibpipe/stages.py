from __future__ import annotations

from typing import Any

from .config import LoadedConfig
from .fixedpoint import (
    convert_q,
    decimal_to_q,
    fixed_ln,
    mul_div,
    quantize_decimal_gear,
)
from .hashing import fingerprint
from .imputation import build_imputation_index
from .samples import Sample
from .concurrency import ConcurrencyMonitor, parallel_map_ordered


class StageComputeError(RuntimeError):
    def __init__(self, stage: str, message: str, seq: int | None = None):
        super().__init__(message)
        self.stage = stage
        self.seq = seq


def input_scale(config: LoadedConfig) -> int:
    return int(config.raw.get("input_scale", 10 ** config.decimal_width))


def make_source_product(
    config: LoadedConfig,
    samples: list[Sample],
    gear: str | None = None,
    max_workers: int = 1,
    monitor: ConcurrencyMonitor | None = None,
) -> dict[str, Any]:
    scale = input_scale(config)
    imputations = build_imputation_index(config, samples, scale)
    def build_row(sample: Sample) -> dict[str, Any]:
        values: list[int] = []
        flags: list[bool] = []
        sources: list[dict[str, Any] | None] = []
        for channel_index, decimal_value in enumerate(sample.values):
            if sample.missing[channel_index] or decimal_value is None:
                source = imputations[(sample.seq, channel_index)]
                values.append(int(source["mean_q"]))
                flags.append(True)
                sources.append(source)
            else:
                values.append(decimal_to_q(decimal_value, scale))
                flags.append(False)
                sources.append(None)
        return {
            "seq": sample.seq,
            "batch": sample.batch,
            "task_key": [sample.batch, sample.seq, "source"],
            "values": values,
            "imputed": flags,
            "imputation_sources": sources,
        }

    rows = parallel_map_ordered(samples, build_row, max_workers, monitor)
    product = {
        "stage": "source",
        "type": "source",
        "scale": scale,
        "gear": gear or config.gear,
        "features": [
            {"name": name, "id": f"source:{name}", "channel_index": i}
            for i, name in enumerate(config.channels)
        ],
        "rows": rows,
    }
    product["product_fingerprint"] = fingerprint(_fingerprintable(product))
    return product


def feature_ids(product: dict[str, Any]) -> list[str]:
    return [feature["id"] for feature in product["features"]]


def compute_stage(
    stage_cfg: dict[str, Any],
    upstream: dict[str, Any],
    config: LoadedConfig,
    gear: str,
    max_workers: int = 1,
    monitor: ConcurrencyMonitor | None = None,
) -> dict[str, Any]:
    stage_type = stage_cfg["type"]
    if stage_type == "standardize":
        product = _standardize(stage_cfg, upstream, max_workers, monitor)
    elif stage_type == "bucket":
        product = _bucket(stage_cfg, upstream, max_workers, monitor)
    elif stage_type == "log":
        product = _log(stage_cfg, upstream, max_workers, monitor)
    elif stage_type == "interaction":
        product = _interaction(stage_cfg, upstream, max_workers, monitor)
    elif stage_type == "clip":
        product = _clip(stage_cfg, upstream, max_workers, monitor)
    else:
        raise StageComputeError(stage_cfg.get("name", "unknown"), f"unknown stage type {stage_type}")

    product["gear"] = gear
    product["rows"] = sorted(product["rows"], key=lambda row: row["seq"])
    for row in product["rows"]:
        row["values"] = [
            quantize_decimal_gear(int(value), int(product["scale"]), gear)
            for value in row["values"]
        ]
        row["task_key"] = [int(row["batch"]), int(row["seq"]), str(product["stage"])]
    product["product_fingerprint"] = fingerprint(_fingerprintable(product))
    return product


def _fingerprintable(product: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in product.items()
        if key not in {"cache_key", "product_fingerprint", "upstream_fingerprint"}
    }


def _row_map(product: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {row["seq"]: row for row in product["rows"]}


def _require_param(params: dict[str, Any], name: str, stage: str) -> Any:
    if name not in params:
        raise StageComputeError(stage, f"missing integer parameter {name}")
    return params[name]


def _standardize(
    stage_cfg: dict[str, Any],
    upstream: dict[str, Any],
    max_workers: int,
    monitor: ConcurrencyMonitor | None,
) -> dict[str, Any]:
    name = stage_cfg["name"]
    scale = int(stage_cfg["scale"])
    params = stage_cfg.get("params", {})
    mean_q = [int(x) for x in _require_param(params, "mean_q", name)]
    inv_std_q = [int(x) for x in _require_param(params, "inv_std_q", name)]
    if len(mean_q) != len(upstream["features"]) or len(inv_std_q) != len(mean_q):
        raise StageComputeError(name, "mean_q and inv_std_q must match channel count")
    def row_worker(up_row: dict[str, Any]) -> dict[str, Any]:
        values = []
        for i, raw_q in enumerate(up_row["values"]):
            x_q = convert_q(int(raw_q), int(upstream["scale"]), scale)
            diff_q = x_q - mean_q[i]
            values.append(mul_div(diff_q, inv_std_q[i], scale))
        return _inherit_row(up_row, values, up_row.get("imputed", []))

    rows = parallel_map_ordered(upstream["rows"], row_worker, max_workers, monitor)
    features = [
        {"name": f"z_{f['name']}", "id": f"{name}:z_{f['name']}", "channel_index": i}
        for i, f in enumerate(upstream["features"])
    ]
    return {"stage": name, "type": "standardize", "scale": scale, "features": features, "rows": rows}


def _bucket(
    stage_cfg: dict[str, Any],
    upstream: dict[str, Any],
    max_workers: int,
    monitor: ConcurrencyMonitor | None,
) -> dict[str, Any]:
    name = stage_cfg["name"]
    scale = int(stage_cfg["scale"])
    params = stage_cfg.get("params", {})
    edges_q = [int(x) for x in _require_param(params, "edges_q", name)]
    centers_raw = params.get("centers_q")
    if centers_raw is None:
        centers_q = [i * scale for i in range(len(edges_q) + 1)]
    else:
        centers_q = [int(x) for x in centers_raw]
    if len(centers_q) != len(edges_q) + 1:
        raise StageComputeError(name, "centers_q must have len(edges_q)+1")
    if any(edges_q[i] >= edges_q[i + 1] for i in range(len(edges_q) - 1)):
        raise StageComputeError(name, "edges_q must be strictly increasing")

    def row_worker(up_row: dict[str, Any]) -> dict[str, Any]:
        values = []
        for raw_q in up_row["values"]:
            x_q = convert_q(int(raw_q), int(upstream["scale"]), scale)
            index = 0
            while index < len(edges_q) and x_q >= edges_q[index]:
                index += 1
            values.append(centers_q[index])
        return _inherit_row(up_row, values, up_row.get("imputed", []))

    rows = parallel_map_ordered(upstream["rows"], row_worker, max_workers, monitor)
    features = [
        {"name": f"bucket_{f['name']}", "id": f"{name}:bucket_{f['name']}", "channel_index": i}
        for i, f in enumerate(upstream["features"])
    ]
    return {"stage": name, "type": "bucket", "scale": scale, "features": features, "rows": rows}


def _log(
    stage_cfg: dict[str, Any],
    upstream: dict[str, Any],
    max_workers: int,
    monitor: ConcurrencyMonitor | None,
) -> dict[str, Any]:
    name = stage_cfg["name"]
    scale = int(stage_cfg["scale"])
    params = stage_cfg.get("params", {})
    epsilon_q = int(_require_param(params, "epsilon_q", name))
    iterations = int(params.get("iterations", 12))
    def row_worker(up_row: dict[str, Any]) -> dict[str, Any]:
        values = []
        for raw_q in up_row["values"]:
            x_q = convert_q(int(raw_q), int(upstream["scale"]), scale)
            argument_q = abs(x_q) + epsilon_q
            if argument_q <= 0:
                raise StageComputeError(name, "log argument must be positive", up_row["seq"])
            values.append(fixed_ln(argument_q, scale, iterations))
        return _inherit_row(up_row, values, up_row.get("imputed", []))

    rows = parallel_map_ordered(upstream["rows"], row_worker, max_workers, monitor)
    features = [
        {"name": f"log_{f['name']}", "id": f"{name}:log_{f['name']}", "channel_index": i}
        for i, f in enumerate(upstream["features"])
    ]
    return {"stage": name, "type": "log", "scale": scale, "features": features, "rows": rows}


def _interaction(
    stage_cfg: dict[str, Any],
    upstream: dict[str, Any],
    max_workers: int,
    monitor: ConcurrencyMonitor | None,
) -> dict[str, Any]:
    name = stage_cfg["name"]
    scale = int(stage_cfg["scale"])
    params = stage_cfg.get("params", {})
    pairs = _require_param(params, "pairs", name)
    pair_names = params.get("output_names")
    if not isinstance(pairs, list) or not pairs:
        raise StageComputeError(name, "pairs must be a non-empty list")
    features = []
    normalized_pairs: list[tuple[int, int]] = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise StageComputeError(name, "each pair must be [a,b]")
        a, b = int(pair[0]), int(pair[1])
        if a < 0 or b < 0 or a >= len(upstream["features"]) or b >= len(upstream["features"]):
            raise StageComputeError(name, f"pair index out of range: {pair}")
        normalized_pairs.append((a, b))
        local_name = pair_names[i] if pair_names and i < len(pair_names) else f"pair_{a}_{b}"
        features.append({"name": local_name, "id": f"{name}:{local_name}", "pair": [a, b]})

    def row_worker(up_row: dict[str, Any]) -> dict[str, Any]:
        values = []
        flags = []
        up_flags = up_row.get("imputed", [False] * len(up_row["values"]))
        for a, b in normalized_pairs:
            av = convert_q(int(up_row["values"][a]), int(upstream["scale"]), scale)
            bv = convert_q(int(up_row["values"][b]), int(upstream["scale"]), scale)
            values.append(mul_div(av, bv, scale))
            flags.append(bool(up_flags[a] or up_flags[b]))
        row = _inherit_row(up_row, values, flags)
        return row

    rows = parallel_map_ordered(upstream["rows"], row_worker, max_workers, monitor)
    return {"stage": name, "type": "interaction", "scale": scale, "features": features, "rows": rows}


def _clip(
    stage_cfg: dict[str, Any],
    upstream: dict[str, Any],
    max_workers: int,
    monitor: ConcurrencyMonitor | None,
) -> dict[str, Any]:
    name = stage_cfg["name"]
    scale = int(stage_cfg["scale"])
    params = stage_cfg.get("params", {})
    min_q = int(_require_param(params, "min_q", name))
    max_q = int(_require_param(params, "max_q", name))
    if min_q > max_q:
        raise StageComputeError(name, "min_q must not exceed max_q")
    def row_worker(up_row: dict[str, Any]) -> dict[str, Any]:
        values = []
        for raw_q in up_row["values"]:
            x_q = convert_q(int(raw_q), int(upstream["scale"]), scale)
            values.append(min(max(x_q, min_q), max_q))
        return _inherit_row(up_row, values, up_row.get("imputed", []))

    rows = parallel_map_ordered(upstream["rows"], row_worker, max_workers, monitor)
    features = [
        {"name": f"clip_{f['name']}", "id": f"{name}:{f['name']}", "source_feature": f["id"]}
        for f in upstream["features"]
    ]
    return {"stage": name, "type": "clip", "scale": scale, "features": features, "rows": rows}


def _inherit_row(up_row: dict[str, Any], values: list[int], imputed: list[bool]) -> dict[str, Any]:
    return {
        "seq": up_row["seq"],
        "batch": up_row["batch"],
        "values": values,
        "imputed": list(imputed),
    }


def stage_parameter_fingerprint(stage_cfg: dict[str, Any]) -> str:
    payload = {
        "type": stage_cfg.get("type"),
        "scale": stage_cfg.get("scale"),
        "params": stage_cfg.get("params", {}),
        "upstream": stage_cfg.get("upstream", stage_cfg.get("upstream_stage")),
    }
    return fingerprint(payload)
