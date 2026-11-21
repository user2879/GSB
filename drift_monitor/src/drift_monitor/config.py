"""YAML configuration and data-description loading with field-level validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ConfigInvalidError
from .fixedpoint import DEFAULT_TIERS, QuantTier

NUMERIC_METRICS = ("mean_shift", "quantile", "hist_density")
CATEGORICAL_METRICS = ("freq_diff", "chi_square")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str  # numeric | categorical | high_cardinality
    metric: str
    drift_scale: float = 0.0


@dataclass(frozen=True)
class SizeTier:
    name: str
    samples: int
    feature_multiplier: int


@dataclass(frozen=True)
class DataDesc:
    seed: int
    missing_rate: float
    base_columns: tuple[ColumnSpec, ...]
    size_tiers: tuple[SizeTier, ...]


@dataclass(frozen=True)
class MonitorConfig:
    data: DataDesc
    window_count: int
    significance_threshold: float
    tier_names: tuple[str, ...]
    tiers: tuple[QuantTier, ...]
    concurrency: int
    snapshot_commit_interval: int
    output_dir: str
    active_tier: str = "fine"
    fail_keys: tuple[str, ...] = field(default_factory=tuple)
    metric_overrides: dict[str, str] = field(default_factory=dict)
    window_cap: int | None = None
    enable_approx_quantile: bool | None = None

    def with_updates(self, **changes) -> "MonitorConfig":
        data = changes.pop("data", self.data)
        return MonitorConfig(
            data=data,
            window_count=changes.pop("window_count", self.window_count),
            significance_threshold=changes.pop(
                "significance_threshold", self.significance_threshold
            ),
            tier_names=changes.pop("tier_names", self.tier_names),
            tiers=changes.pop("tiers", self.tiers),
            concurrency=changes.pop("concurrency", self.concurrency),
            snapshot_commit_interval=changes.pop(
                "snapshot_commit_interval", self.snapshot_commit_interval
            ),
            output_dir=changes.pop("output_dir", self.output_dir),
            active_tier=changes.pop("active_tier", self.active_tier),
            fail_keys=changes.pop("fail_keys", self.fail_keys),
            metric_overrides=changes.pop("metric_overrides", dict(self.metric_overrides)),
            window_cap=changes.pop("window_cap", self.window_cap),
            enable_approx_quantile=changes.pop(
                "enable_approx_quantile", self.enable_approx_quantile
            ),
        )


def _require(cond: bool, field_path: str, message: str) -> None:
    if not cond:
        raise ConfigInvalidError(message, field=field_path)


def _parse_columns(raw) -> tuple[ColumnSpec, ...]:
    _require(isinstance(raw, list) and raw, "data.columns", "columns must be a non-empty list")
    columns = []
    seen = set()
    for idx, item in enumerate(raw):
        path = f"data.columns[{idx}]"
        _require(isinstance(item, dict), path, "column entry must be a mapping")
        name = item.get("name")
        kind = item.get("kind")
        metric = item.get("metric")
        _require(isinstance(name, str) and name, f"{path}.name", "column name must be a non-empty string")
        _require(name not in seen, f"{path}.name", f"duplicate column name: {name}")
        seen.add(name)
        _require(kind in ("numeric", "categorical", "high_cardinality"), f"{path}.kind",
                 "kind must be numeric, categorical or high_cardinality")
        allowed = NUMERIC_METRICS if kind == "numeric" else CATEGORICAL_METRICS
        _require(metric in allowed, f"{path}.metric",
                 f"metric {metric!r} not allowed for {kind}; choose from {allowed}")
        drift_scale = float(item.get("drift_scale", 0.0))
        _require(drift_scale >= 0, f"{path}.drift_scale", "drift_scale must be >= 0")
        columns.append(ColumnSpec(name, kind, metric, drift_scale))
    return tuple(columns)


def _parse_size_tiers(raw) -> tuple[SizeTier, ...]:
    _require(isinstance(raw, list) and len(raw) >= 1, "data.size_tiers",
             "size_tiers must be a non-empty list")
    tiers = []
    prev_samples = None
    prev_mult = None
    for idx, item in enumerate(raw):
        path = f"data.size_tiers[{idx}]"
        _require(isinstance(item, dict), path, "size tier must be a mapping")
        name = item.get("name", f"tier_{idx}")
        samples = item.get("samples")
        mult = item.get("feature_multiplier", 1)
        _require(isinstance(samples, int) and samples > 0, f"{path}.samples",
                 "samples must be a positive integer")
        _require(isinstance(mult, int) and mult >= 1, f"{path}.feature_multiplier",
                 "feature_multiplier must be a positive integer")
        if prev_samples is not None:
            _require(samples > prev_samples, f"{path}.samples",
                     "size tiers must be strictly increasing in samples")
            _require(mult >= prev_mult, f"{path}.feature_multiplier",
                     "feature_multiplier must be non-decreasing across tiers")
        prev_samples, prev_mult = samples, mult
        tiers.append(SizeTier(str(name), samples, mult))
    return tuple(tiers)


def _parse_tiers(raw) -> tuple[tuple[str, ...], tuple[QuantTier, ...]]:
    if raw is None:
        names = tuple(t.name for t in DEFAULT_TIERS)
        return names, DEFAULT_TIERS
    _require(isinstance(raw, list) and len(raw) >= 2, "quantization.tiers",
             "at least two quantization tiers are required")
    names, tiers = [], []
    prev_decimals = None
    for idx, item in enumerate(raw):
        path = f"quantization.tiers[{idx}]"
        _require(isinstance(item, dict), path, "tier must be a mapping")
        name = item.get("name")
        decimals = item.get("decimals")
        clamp = item.get("clamp", 10 ** (max(decimals or 0, 1) + 4))
        _require(isinstance(name, str) and name, f"{path}.name", "tier name required")
        _require(name not in names, f"{path}.name", f"duplicate tier name: {name}")
        _require(isinstance(decimals, int) and 0 <= decimals <= 12, f"{path}.decimals",
                 "decimals must be an integer in [0, 12]")
        _require(isinstance(clamp, int) and clamp > 0, f"{path}.clamp",
                 "clamp must be a positive integer")
        if prev_decimals is not None:
            _require(decimals > prev_decimals, f"{path}.decimals",
                     "tier decimals must be strictly increasing")
        prev_decimals = decimals
        names.append(name)
        tiers.append(QuantTier(name, decimals, clamp))
    return tuple(names), tuple(tiers)


def load_config(path: str | Path) -> MonitorConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigInvalidError(f"config file not found: {p}", field="config_path")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigInvalidError(f"YAML parse error: {exc}", field="config_path") from exc
    _require(isinstance(raw, dict), "root", "top-level config must be a mapping")

    data_raw = raw.get("data")
    _require(isinstance(data_raw, dict), "data", "section data is required")
    seed = data_raw.get("seed", 42)
    _require(isinstance(seed, int) and 0 <= seed < 2 ** 32, "data.seed",
             "seed must be a non-negative 32-bit integer")
    missing_rate = float(data_raw.get("missing_rate", 0.05))
    _require(0.0 <= missing_rate < 1.0, "data.missing_rate",
             "missing_rate must be in [0, 1)")
    target = data_raw.get("target_column")
    columns = _parse_columns(data_raw.get("columns"))
    feature_names = {c.name for c in columns}
    if target is not None:
        _require(isinstance(target, str) and target, "data.target_column",
                 "target_column must be a string when set")
        _require(target not in feature_names, "data.target_column",
                 "target column must not appear in feature columns")
    size_tiers = _parse_size_tiers(data_raw.get("size_tiers"))
    data = DataDesc(seed=seed, missing_rate=missing_rate,
                    base_columns=columns, size_tiers=size_tiers)

    algo = raw.get("algorithm", {})
    window_count = algo.get("window_count", 4)
    _require(isinstance(window_count, int) and window_count >= 1,
             "algorithm.window_count", "window_count must be a positive integer")
    threshold = float(algo.get("significance_threshold", 1.0))
    _require(threshold > 0, "algorithm.significance_threshold",
             "significance_threshold must be > 0")

    tier_names, tiers = _parse_tiers(raw.get("quantization", {}).get("tiers"))
    active_tier = raw.get("quantization", {}).get("active_tier", tier_names[-1])
    _require(active_tier in tier_names, "quantization.active_tier",
             f"active_tier must be one of {list(tier_names)}")

    runtime = raw.get("runtime", {})
    concurrency = runtime.get("concurrency", 1)
    _require(isinstance(concurrency, int) and concurrency >= 1,
             "runtime.concurrency", "concurrency must be a positive integer")
    commit_interval = runtime.get("snapshot_commit_interval", 1)
    _require(isinstance(commit_interval, int) and commit_interval >= 1,
             "runtime.snapshot_commit_interval",
             "snapshot_commit_interval must be a positive integer")
    output_dir = runtime.get("output_dir", "output")

    fail_keys = tuple(runtime.get("fail_keys", ()) or ())
    return MonitorConfig(
        data=data,
        window_count=window_count,
        significance_threshold=threshold,
        tier_names=tier_names,
        tiers=tiers,
        concurrency=concurrency,
        snapshot_commit_interval=commit_interval,
        output_dir=str(output_dir),
        active_tier=active_tier,
        fail_keys=fail_keys,
    )
