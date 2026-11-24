"""YAML configuration loading, dotted-path overrides and validation."""

from pathlib import Path

import yaml

from .errors import E_WINDOW_INVALID, PipelineError
from .hashing import stable_hash


DEFAULT_CONFIG = {
    "devices": ["DEV-A", "DEV-B", "DEV-C", "DEV-BAD-FINITE"],
    "window": {"length": 256, "hop": 128},
    "spectrum": {"order": 2, "stat_window": 4},
    "features": {
        "sample_rate": 1024,
        "bands": [
            [0.0, 0.1],
            [0.1, 0.25],
            [0.25, 0.5],
        ],
    },
    "missing": {"fill": "linear"},
    "resample": {"handle": "linear"},
    "detectors": {
        "maha": {"scale_epsilon": 1e-6, "scale_multiplier": 1.0},
        "residual": {"floor": 1e-6, "power": 1.0},
        "crest": {"epsilon": 1e-6},
    },
    "normalization": {"method": "minmax", "epsilon": 1e-12},
    "quantization": {"frac_bits": 8, "ref_extra_bit": 1},
    "fusion": {"weights": [0.4, 0.35, 0.25]},
    "score": {"min": 0, "max": 1000, "threshold": 500, "tie_rule": "anomaly"},
    "scheduler": {"max_workers": 4, "point_budget": 1000000, "serial": False},
}


def _deep_merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(text):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def apply_overrides(cfg, overrides):
    for raw in overrides or []:
        if "=" not in raw:
            raise ValueError(f"override must be key=value, got {raw!r}")
        path, value = raw.split("=", 1)
        keys = path.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"unknown config path: {path}")
            node = node[key]
        last = keys[-1]
        if last not in node:
            raise KeyError(f"unknown config path: {path}")
        if isinstance(node[last], list):
            parsed = [_coerce_scalar(part) for part in value.split(",")]
        else:
            parsed = _coerce_scalar(value.strip())
        node[last] = parsed
    return cfg


def load_config(path=None, overrides=None):
    cfg = {k: dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v
           for k, v in DEFAULT_CONFIG.items()}
    if path is not None:
        path = Path(path)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                user_cfg = yaml.safe_load(fh) or {}
            cfg = _deep_merge(cfg, user_cfg)
    cfg = apply_overrides(cfg, overrides)
    validate_window(cfg)
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg):
    # scheduler settings (workers, per-batch budget, serial flag) are runtime
    # controls, not data/config semantics: exclude them so resuming with a new
    # budget is not mistaken for a configuration change.
    payload = {k: v for k, v in cfg.items()
               if k not in ("config_hash", "scheduler")}
    return stable_hash(payload)


def validate_window(cfg):
    length = cfg["window"]["length"]
    hop = cfg["window"]["hop"]
    if not isinstance(length, int) or not isinstance(hop, int):
        raise PipelineError(E_WINDOW_INVALID,
                            f"window length/hop must be integers, got {length}/{hop}")
    if length <= 0 or hop <= 0:
        raise PipelineError(E_WINDOW_INVALID,
                            f"window length/hop must be positive, got {length}/{hop}")
    if hop > length:
        raise PipelineError(E_WINDOW_INVALID,
                            f"window hop {hop} must not exceed length {length}")
    order = cfg["spectrum"]["order"]
    if not isinstance(order, int) or order < 2:
        raise PipelineError(E_WINDOW_INVALID,
                            f"spectrum order must be an integer >= 2, got {order}")
    stat_window = cfg["spectrum"]["stat_window"]
    if not isinstance(stat_window, int) or stat_window <= 0:
        raise PipelineError(E_WINDOW_INVALID,
                            f"stat_window must be a positive integer, got {stat_window}")
    return True


def as_plain_dict(cfg):
    return {k: v for k, v in cfg.items()}
