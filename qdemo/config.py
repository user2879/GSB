"""Quantization + distillation configuration loading and validation."""
from __future__ import annotations

import copy

from .common import (
    ALLOWED_BITS,
    SCHEMES,
    BusinessError,
    E_CONFIG_INVALID,
    fingerprint,
    read_json,
)

DEFAULT_LAYER_PARAMS = {
    "bits": 4,
    "group_size": 16,
    "scheme": "asymmetric",
    "per_channel": True,
}

DEFAULTS = {
    "experiment": "qdemo",
    "weights_path": "data/weights.npz",
    "calib_path": "data/calib.npz",
    "layers": {"default": dict(DEFAULT_LAYER_PARAMS), "overrides": {}},
    "whitelist": [],
    "distill": {"temperature": 2.0, "soft_label_weight": 0.5},
    "interaction": {
        "budget_ms": 200,
        "concurrency": 4,
        "snapshot_interval": 1,
        "compute_delay_ms_per_layer": 0,
    },
    "rounding": {"scale_sig_digits": 12},
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _fail(msg):
    raise BusinessError(E_CONFIG_INVALID, msg)


def validate_layer_params(params: dict, layer_label: str) -> None:
    bits = params.get("bits")
    if bits not in ALLOWED_BITS:
        _fail(f"{layer_label}: bits {bits!r} not in {ALLOWED_BITS}")
    gs = params.get("group_size")
    if not isinstance(gs, int) or gs <= 0:
        _fail(f"{layer_label}: group_size must be a positive int, got {gs!r}")
    if params.get("scheme") not in SCHEMES:
        _fail(f"{layer_label}: scheme must be one of {SCHEMES}")
    if not isinstance(params.get("per_channel"), bool):
        _fail(f"{layer_label}: per_channel must be bool")


def validate_config(cfg: dict, weight_pack=None) -> None:
    for key in ("weights_path", "calib_path"):
        if not isinstance(cfg.get(key), str):
            _fail(f"missing or invalid {key}")
    layers = cfg.get("layers", {})
    validate_layer_params(layers.get("default", {}), "layers.default")
    overrides = layers.get("overrides", {})
    if not isinstance(overrides, dict):
        _fail("layers.overrides must be an object")
    whitelist = cfg.get("whitelist", [])
    if not isinstance(whitelist, list) or not all(isinstance(x, str) for x in whitelist):
        _fail("whitelist must be a list of layer names")
    distill = cfg.get("distill", {})
    t = distill.get("temperature")
    a = distill.get("soft_label_weight")
    if not isinstance(t, (int, float)) or t <= 0:
        _fail("distill.temperature must be > 0")
    if not isinstance(a, (int, float)) or not (0.0 <= a <= 1.0):
        _fail("distill.soft_label_weight must be in [0, 1]")
    inter = cfg.get("interaction", {})
    if not isinstance(inter.get("budget_ms"), (int, float)) or inter["budget_ms"] <= 0:
        _fail("interaction.budget_ms must be > 0")
    if not isinstance(inter.get("concurrency"), int) or inter["concurrency"] < 1:
        _fail("interaction.concurrency must be an int >= 1")
    if not isinstance(inter.get("snapshot_interval"), int) or inter["snapshot_interval"] < 1:
        _fail("interaction.snapshot_interval must be an int >= 1")
    if not isinstance(inter.get("compute_delay_ms_per_layer"), (int, float)) or inter["compute_delay_ms_per_layer"] < 0:
        _fail("interaction.compute_delay_ms_per_layer must be >= 0")
    sd = cfg.get("rounding", {}).get("scale_sig_digits")
    if not isinstance(sd, int) or not (1 <= sd <= 17):
        _fail("rounding.scale_sig_digits must be an int in [1, 17]")

    if weight_pack is not None:
        known = set(weight_pack.layer_names)
        for name in list(overrides) + list(whitelist):
            if name not in known:
                _fail(f"unknown layer referenced in config: {name}")
        for name in weight_pack.layer_names:
            if name in whitelist:
                continue
            params = layer_params(cfg, name)
            gs = params["group_size"]
            w = weight_pack.arrays[name]
            dim = w.shape[1] if params["per_channel"] else w.size
            if dim % gs != 0:
                _fail(
                    f"layer {name}: group_size {gs} does not divide "
                    f"{'input dim' if params['per_channel'] else 'tensor size'} {dim}"
                )


def layer_params(cfg: dict, layer_name: str) -> dict:
    params = dict(cfg["layers"]["default"])
    params.update(cfg["layers"].get("overrides", {}).get(layer_name, {}))
    return params


def load_config(path: str, weight_pack=None) -> dict:
    raw = read_json(path)
    cfg = _merge(DEFAULTS, raw)
    validate_config(cfg, weight_pack)
    return cfg


def config_fingerprint(cfg: dict) -> str:
    return fingerprint(cfg)
