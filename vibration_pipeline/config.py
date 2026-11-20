from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import (
    E_ALGO_PARAM_INVALID,
    E_BAND_INVALID,
    E_CONCURRENCY_INVALID,
    E_CONFIG_INVALID,
    E_QUANT_INVALID,
    E_SEGMENT_LEN_INVALID,
    E_THRESHOLD_INVALID,
)
from .util import hash_object, stable_float


ALGORITHM_NAMES = ("crest_factor", "robust_mahalanobis", "spectral_residual")
FEATURE_MODES = ("time_domain", "band_energy", "combined")


@dataclass(frozen=True)
class ConfigIssue:
    code: str
    detail: str
    field_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "field": self.field_name}


@dataclass(frozen=True)
class FeatureConfig:
    mode: str
    band_edges_hz: tuple[float, ...]
    band_count: int
    valid: bool


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    enabled: bool
    candidates: tuple[dict[str, Any], ...]
    candidate_index: int
    params: dict[str, Any]
    normalization: dict[str, Any]
    weight: float
    params_hash: str
    normalization_hash: str

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "candidate_index": self.candidate_index,
            "candidate_count": len(self.candidates),
            "selected_params": self.params,
            "params_hash": self.params_hash,
            "normalization": self.normalization,
            "normalization_hash": self.normalization_hash,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class QuantizationConfig:
    decimal_places: int
    quantum: int
    tie_policy: str

    @property
    def scale(self) -> int:
        return 10 ** self.decimal_places

    @property
    def fingerprint(self) -> str:
        return hash_object(
            {
                "decimal_places": self.decimal_places,
                "quantum": self.quantum,
                "tie_policy": self.tie_policy,
            },
            namespace="quantization-v1",
        )


@dataclass
class PipelineConfig:
    raw: dict[str, Any]
    known_devices: tuple[str, ...]
    segment_length: int
    score_min: float
    score_max: float
    threshold: float
    features: FeatureConfig
    algorithms: dict[str, AlgorithmSpec]
    quantization: QuantizationConfig
    concurrency: int
    commit_interval: int
    snapshot_schema_version: int
    config_hash: str
    issues: list[ConfigIssue] = field(default_factory=list)

    @property
    def enabled_algorithms(self) -> tuple[str, ...]:
        return tuple(name for name in ALGORITHM_NAMES if self.algorithms.get(name, None) and self.algorithms[name].enabled)

    @property
    def score_range(self) -> tuple[float, float]:
        return self.score_min, self.score_max

    def algorithm_fingerprint(self, name: str) -> str:
        return self.algorithms[name].params_hash

    def logical_definition(self) -> dict[str, Any]:
        return {
            "known_devices": list(self.known_devices),
            "segment_length": self.segment_length,
            "score_range": [self.score_min, self.score_max],
            "threshold": self.threshold,
            "features": {
                "mode": self.features.mode,
                "band_edges_hz": list(self.features.band_edges_hz),
                "band_count": self.features.band_count,
            },
            "algorithms": {name: self.algorithms[name].definition() for name in sorted(self.algorithms)},
            "quantization": {
                "decimal_places": self.quantization.decimal_places,
                "quantum": self.quantization.quantum,
                "tie_policy": self.quantization.tie_policy,
            },
            "concurrency": self.concurrency,
            "snapshot_commit_interval_segments": self.commit_interval,
            "snapshot_schema_version": self.snapshot_schema_version,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_bands(edges: tuple[float, ...], declared_count: int) -> tuple[bool, str]:
    if len(edges) < 2:
        return False, "band_edges_hz must contain at least two boundaries"
    if any(not isinstance(value, (int, float)) or not float("-inf") < float(value) < float("inf") for value in edges):
        return False, "band boundaries must be finite numbers"
    if float(edges[0]) < 0:
        return False, "first band boundary must be >= 0"
    if any(float(left) >= float(right) for left, right in zip(edges, edges[1:])):
        return False, "band boundaries must be strictly increasing"
    if declared_count != len(edges) - 1:
        return False, f"band_count={declared_count} does not match {len(edges) - 1} bands"
    return True, ""


def _parse_algorithms(raw: dict[str, Any], issues: list[ConfigIssue]) -> dict[str, AlgorithmSpec]:
    block = raw.get("algorithms", {})
    if not isinstance(block, dict):
        issues.append(ConfigIssue(E_CONFIG_INVALID, "algorithms must be a mapping", "algorithms"))
        block = {}
    else:
        unknown_names = sorted(set(block) - set(ALGORITHM_NAMES))
        for unknown_name in unknown_names:
            issues.append(
                ConfigIssue(
                    E_CONFIG_INVALID,
                    f"unknown algorithm {unknown_name!r}; supported algorithms are {ALGORITHM_NAMES}",
                    f"algorithms.{unknown_name}",
                )
            )

    parsed: dict[str, AlgorithmSpec] = {}
    for name in ALGORITHM_NAMES:
        if name not in block:
            continue
        value = block[name]
        if not isinstance(value, dict):
            issues.append(ConfigIssue(E_CONFIG_INVALID, f"algorithm {name} must be a mapping", f"algorithms.{name}"))
            continue
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            issues.append(ConfigIssue(E_CONFIG_INVALID, f"algorithm {name}.enabled must be boolean", f"algorithms.{name}.enabled"))
            enabled = True

        defaults: dict[str, Any]
        if name == "robust_mahalanobis":
            defaults = {"robust_scale": 1.4826, "scale_floor": 1e-9}
        elif name == "spectral_residual":
            defaults = {"baseline_method": "median", "epsilon": 1e-12}
        else:
            defaults = {"window_length": 3}

        candidates_value = value.get("candidates")
        if candidates_value is None:
            candidates_value = [value.get("params", defaults)]
        if not isinstance(candidates_value, list) or not candidates_value:
            issues.append(ConfigIssue(E_CONFIG_INVALID, f"algorithm {name}.candidates must be a non-empty list", f"algorithms.{name}.candidates"))
            candidates_value = [defaults]
        candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates_value):
            if not isinstance(candidate, dict):
                issues.append(
                    ConfigIssue(E_CONFIG_INVALID, f"algorithm {name} candidate {index} must be a mapping", f"algorithms.{name}.candidates")
                )
                candidate = {}
            merged = dict(defaults)
            merged.update(candidate)
            candidates.append(merged)

        candidate_index = value.get("candidate_index", 0)
        if not _is_int(candidate_index) or not 0 <= candidate_index < len(candidates):
            issues.append(
                ConfigIssue(
                    E_CONFIG_INVALID,
                    f"algorithm {name}.candidate_index must be in [0, {len(candidates) - 1}]",
                    f"algorithms.{name}.candidate_index",
                )
            )
            candidate_index = 0
        params = dict(candidates[candidate_index])

        valid_params = True
        if name == "robust_mahalanobis":
            scale = params.get("robust_scale")
            floor = params.get("scale_floor")
            if not _is_number(scale) or float(scale) <= 0 or not _is_number(floor) or float(floor) <= 0:
                issues.append(ConfigIssue(E_ALGO_PARAM_INVALID, "robust_scale and scale_floor must be > 0", f"algorithms.{name}"))
                valid_params = False
        elif name == "spectral_residual":
            method = params.get("baseline_method")
            epsilon = params.get("epsilon")
            if method not in ("median", "mean"):
                issues.append(ConfigIssue(E_ALGO_PARAM_INVALID, "baseline_method must be median or mean", f"algorithms.{name}"))
                valid_params = False
            if not _is_number(epsilon) or float(epsilon) <= 0:
                issues.append(ConfigIssue(E_ALGO_PARAM_INVALID, "epsilon must be > 0", f"algorithms.{name}"))
                valid_params = False
        else:
            window = params.get("window_length")
            if not _is_int(window) or window <= 0:
                issues.append(ConfigIssue(E_ALGO_PARAM_INVALID, "window_length must be a positive integer", f"algorithms.{name}"))
                valid_params = False

        normalization_raw = value.get("normalization", {"method": "log1p_clip", "reference": 6.0})
        if not isinstance(normalization_raw, dict):
            issues.append(ConfigIssue(E_CONFIG_INVALID, f"algorithm {name}.normalization must be a mapping", f"algorithms.{name}.normalization"))
            normalization_raw = {"method": "log1p_clip", "reference": 6.0}
        method = normalization_raw.get("method", "log1p_clip")
        reference = normalization_raw.get("reference", 1.0)
        norm_min = normalization_raw.get("min", 0.0)
        norm_max = normalization_raw.get("max", 1.0)
        norm_valid = method in ("identity_clip", "log1p_clip", "minmax_clip")
        if not norm_valid:
            issues.append(ConfigIssue(E_CONFIG_INVALID, f"unknown normalization method {method}", f"algorithms.{name}.normalization.method"))
        if method in ("identity_clip", "log1p_clip") and (not _is_number(reference) or float(reference) <= 0):
            issues.append(ConfigIssue(E_CONFIG_INVALID, "normalization.reference must be > 0", f"algorithms.{name}.normalization.reference"))
            norm_valid = False
        if method == "minmax_clip" and (
            not _is_number(norm_min) or not _is_number(norm_max) or float(norm_min) >= float(norm_max)
        ):
            issues.append(ConfigIssue(E_CONFIG_INVALID, "normalization min/max are invalid", f"algorithms.{name}.normalization"))
            norm_valid = False
        normalization = {
            "method": method if norm_valid else "identity_clip",
            "reference": stable_float(reference) if _is_number(reference) and float(reference) > 0 else 1.0,
            "min": stable_float(norm_min) if _is_number(norm_min) else 0.0,
            "max": stable_float(norm_max) if _is_number(norm_max) else 1.0,
        }

        weight = value.get("weight", 1.0)
        if not _is_number(weight) or not float(weight) >= 0:
            issues.append(ConfigIssue(E_CONFIG_INVALID, f"algorithm {name}.weight must be >= 0", f"algorithms.{name}.weight"))
            weight = 1.0
        weight = stable_float(weight)

        candidates = [dict(candidate) for candidate in candidates]
        for candidate in candidates:
            for key, candidate_value in list(candidate.items()):
                if _is_number(candidate_value):
                    candidate[key] = stable_float(candidate_value)
        params = candidates[candidate_index]
        enabled = enabled and valid_params
        parsed[name] = AlgorithmSpec(
            name=name,
            enabled=enabled,
            candidates=tuple(candidates),
            candidate_index=candidate_index,
            params=params,
            normalization=normalization,
            weight=weight,
            params_hash=hash_object({"name": name, "params": params, "formula": "v1"}, namespace="algorithm-params"),
            normalization_hash=hash_object(normalization, namespace="algorithm-normalization"),
        )
    return parsed


def parse_config(raw: dict[str, Any]) -> PipelineConfig:
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    issues: list[ConfigIssue] = []

    known_value = raw.get("known_devices", [])
    if not isinstance(known_value, list) or any(not isinstance(item, str) or not item for item in known_value):
        issues.append(ConfigIssue(E_CONFIG_INVALID, "known_devices must be a list of non-empty strings", "known_devices"))
        known_devices: tuple[str, ...] = ()
    else:
        known_devices = tuple(sorted(set(known_value)))

    segment_value = raw.get("segment_length")
    if not _is_int(segment_value) or segment_value <= 0:
        issues.append(ConfigIssue(E_SEGMENT_LEN_INVALID, "segment_length must be a positive integer", "segment_length"))
        segment_length = 0
    else:
        segment_length = segment_value

    score_range = raw.get("score_range", [0.0, 1.0])
    score_valid = (
        isinstance(score_range, list)
        and len(score_range) == 2
        and all(_is_number(item) for item in score_range)
        and float(score_range[0]) < float(score_range[1])
    )
    if not score_valid:
        issues.append(ConfigIssue(E_THRESHOLD_INVALID, "score_range must be [min, max] with min < max", "score_range"))
        score_min, score_max = 0.0, 1.0
    else:
        score_min, score_max = stable_float(score_range[0]), stable_float(score_range[1])

    threshold = raw.get("threshold", score_min)
    if not _is_number(threshold) or not score_min <= float(threshold) <= score_max:
        issues.append(
            ConfigIssue(E_THRESHOLD_INVALID, "threshold must be finite and inside score_range", "threshold")
        )
        threshold = stable_float((score_min + score_max) / 2.0)
    else:
        threshold = stable_float(threshold)

    features_raw = raw.get("features", {})
    if not isinstance(features_raw, dict):
        issues.append(ConfigIssue(E_CONFIG_INVALID, "features must be a mapping", "features"))
        features_raw = {}
    mode = features_raw.get("mode", "combined")
    if mode not in FEATURE_MODES:
        issues.append(ConfigIssue(E_CONFIG_INVALID, f"features.mode must be one of {FEATURE_MODES}", "features.mode"))
        mode = "combined"
    edges_value = features_raw.get("band_edges_hz", features_raw.get("band_edges", []))
    if not isinstance(edges_value, list):
        issues.append(ConfigIssue(E_BAND_INVALID, "features.band_edges_hz must be a list", "features.band_edges_hz"))
        edges_value = []
    try:
        edges = tuple(stable_float(float(value)) for value in edges_value)
    except (TypeError, ValueError):
        issues.append(ConfigIssue(E_BAND_INVALID, "features.band_edges_hz must contain numbers", "features.band_edges_hz"))
        edges = ()
    declared_count = features_raw.get("band_count", max(0, len(edges) - 1))
    if not _is_int(declared_count):
        issues.append(ConfigIssue(E_BAND_INVALID, "features.band_count must be an integer", "features.band_count"))
        declared_count = max(0, len(edges) - 1)
    bands_valid, band_detail = _validate_bands(edges, declared_count)
    if not bands_valid:
        issues.append(ConfigIssue(E_BAND_INVALID, band_detail, "features.band_edges_hz"))
    features = FeatureConfig(mode=mode, band_edges_hz=edges, band_count=declared_count, valid=bands_valid)

    algorithms = _parse_algorithms(raw, issues)
    enabled = [spec for spec in algorithms.values() if spec.enabled]
    if not enabled:
        issues.append(ConfigIssue(E_CONFIG_INVALID, "at least one valid algorithm must be enabled", "algorithms"))
    if enabled and sum(spec.weight for spec in enabled) <= 0:
        issues.append(ConfigIssue(E_CONFIG_INVALID, "at least one enabled algorithm must have a positive weight", "algorithms"))
        for name, spec in list(algorithms.items()):
            if spec.enabled:
                algorithms[name] = AlgorithmSpec(
                    **{**spec.__dict__, "weight": 1.0}
                )

    quant_raw = raw.get("quantization", {})
    if not isinstance(quant_raw, dict):
        issues.append(ConfigIssue(E_QUANT_INVALID, "quantization must be a mapping", "quantization"))
        quant_raw = {}
    decimal_places = quant_raw.get("decimal_places", 3)
    quantum = quant_raw.get("quantum", 1)
    tie_policy = quant_raw.get("tie_policy", "anomaly")
    quant_valid = _is_int(decimal_places) and 0 <= decimal_places <= 9
    quant_valid = quant_valid and _is_int(quantum) and 1 <= quantum <= 10 ** int(decimal_places if _is_int(decimal_places) else 3)
    quant_valid = quant_valid and tie_policy in ("anomaly", "normal")
    if not quant_valid:
        issues.append(
            ConfigIssue(
                E_QUANT_INVALID,
                "decimal_places must be 0..9, quantum must be 1..10**decimal_places, tie_policy must be anomaly/normal",
                "quantization",
            )
        )
        if not (_is_int(decimal_places) and 0 <= decimal_places <= 9):
            decimal_places = 3
        if not (_is_int(quantum) and 1 <= quantum <= 10 ** decimal_places):
            quantum = 1
        if tie_policy not in ("anomaly", "normal"):
            tie_policy = "anomaly"
    quantization = QuantizationConfig(int(decimal_places), int(quantum), str(tie_policy))

    concurrency = raw.get("concurrency", 1)
    if not _is_int(concurrency) or concurrency <= 0:
        issues.append(ConfigIssue(E_CONCURRENCY_INVALID, "concurrency must be a positive integer", "concurrency"))
        concurrency = 1

    commit_interval = raw.get("snapshot_commit_interval_segments", 1)
    if not _is_int(commit_interval) or commit_interval <= 0:
        issues.append(
            ConfigIssue(E_CONFIG_INVALID, "snapshot_commit_interval_segments must be a positive integer", "snapshot_commit_interval_segments")
        )
        commit_interval = 1

    snapshot = raw.get("snapshot", {})
    if not isinstance(snapshot, dict):
        issues.append(ConfigIssue(E_CONFIG_INVALID, "snapshot must be a mapping", "snapshot"))
        snapshot = {}
    schema_version = snapshot.get("schema_version", 1)
    if not _is_int(schema_version) or schema_version <= 0:
        issues.append(ConfigIssue(E_CONFIG_INVALID, "snapshot.schema_version must be a positive integer", "snapshot.schema_version"))
        schema_version = 1

    config = PipelineConfig(
        raw=raw,
        known_devices=known_devices,
        segment_length=segment_length,
        score_min=score_min,
        score_max=score_max,
        threshold=threshold,
        features=features,
        algorithms=algorithms,
        quantization=quantization,
        concurrency=concurrency,
        commit_interval=commit_interval,
        snapshot_schema_version=schema_version,
        config_hash="",
        issues=issues,
    )
    config.config_hash = hash_object(config.logical_definition(), namespace="pipeline-config-v1")
    return config
