from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from .errors import InputInvalid


KNOWN_TEMPLATE_VERSIONS = {"dot-matrix-v1"}
ALLOWED_FRACTION_BITS = [8, 16, 24, 32]
ALLOWED_QUANT_LEVELS = [100, 1000, 10000]


def default_config() -> dict[str, Any]:
    return {
        "schema": "vertical-ocr-config/v1",
        "layout": {"width": 720, "height": 960, "mode": "L"},
        "binarize_threshold": 176,
        "column_projection_window": 11,
        "projection_window": 7,
        "sliding_window": {"width": 24, "height": 24},
        "char_min_height": 18,
        "char_max_height": 88,
        "width_ratio_min": 0.35,
        "width_ratio_max": 3.2,
        "split_penalty": 0.22,
        "fusion_weights": [0.50, 0.30, 0.20],
        "fixed_point": {
            "fraction_bits": 16,
            "quantization_level": 1000,
        },
        "template_version": "dot-matrix-v1",
        "max_workers": 3,
        "match_point_budget": 1_000_000,
        "keypoint_selector": "authoritative",
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def canonical_hash(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise InputInvalid("config must be a JSON object", field="config")
    cfg = deepcopy(default_config())
    cfg.update({k: v for k, v in config.items() if k != "fixed_point"})
    if isinstance(config.get("fixed_point"), dict):
        fp = deepcopy(cfg["fixed_point"])
        fp.update(config["fixed_point"])
        cfg["fixed_point"] = fp

    layout = cfg.get("layout")
    if not isinstance(layout, dict) or not _is_int(layout.get("width")) or not _is_int(layout.get("height")):
        raise InputInvalid("layout.width and layout.height must be integers", field="layout")
    if layout["width"] <= 0 or layout["height"] <= 0:
        raise InputInvalid("layout dimensions must be positive", field="layout")
    if layout.get("mode") != "L":
        raise InputInvalid("only grayscale mode L is supported", field="layout.mode")

    sw = cfg.get("sliding_window")
    if not isinstance(sw, dict) or not _is_int(sw.get("width")) or not _is_int(sw.get("height")):
        raise InputInvalid("sliding_window.width/height must be integers", field="sliding_window")
    if sw["width"] <= 0 or sw["height"] <= 0:
        raise InputInvalid("sliding_window dimensions must be positive", field="sliding_window")
    if layout["width"] % sw["width"] != 0:
        raise InputInvalid("sliding_window.width must divide layout.width", field="sliding_window.width")
    if layout["height"] % sw["height"] != 0:
        raise InputInvalid("sliding_window.height must divide layout.height", field="sliding_window.height")

    for key in ("column_projection_window", "projection_window"):
        if not _is_int(cfg.get(key)) or cfg[key] <= 0:
            raise InputInvalid(f"{key} must be a positive integer", field=key)

    if not _is_int(cfg.get("char_min_height")) or not _is_int(cfg.get("char_max_height")):
        raise InputInvalid("character height bounds must be integers", field="char_height")
    if cfg["char_min_height"] <= 0 or cfg["char_max_height"] <= 0:
        raise InputInvalid("character height bounds must be positive", field="char_height")
    if cfg["char_min_height"] > cfg["char_max_height"]:
        raise InputInvalid("char_min_height must be <= char_max_height", field="char_min_height")

    for key in ("width_ratio_min", "width_ratio_max", "split_penalty", "binarize_threshold"):
        if key == "binarize_threshold":
            if not _is_int(cfg.get(key)) or not (0 <= cfg[key] <= 255):
                raise InputInvalid("binarize_threshold must be an integer in [0,255]", field=key)
        elif not _finite_nonnegative_number(cfg.get(key)):
            raise InputInvalid(f"{key} must be a finite non-negative number", field=key)
    if cfg["width_ratio_min"] > cfg["width_ratio_max"]:
        raise InputInvalid("width_ratio_min must be <= width_ratio_max", field="width_ratio_min")

    weights = cfg.get("fusion_weights")
    if not isinstance(weights, list) or not weights:
        raise InputInvalid("fusion_weights must be a non-empty list", field="fusion_weights")
    if not all(_finite_nonnegative_number(w) for w in weights):
        raise InputInvalid("fusion_weights must contain finite non-negative numbers", field="fusion_weights")
    if sum(float(w) for w in weights) <= 0:
        raise InputInvalid("fusion_weights must have a positive sum", field="fusion_weights")

    fp = cfg.get("fixed_point")
    if not isinstance(fp, dict):
        raise InputInvalid("fixed_point must be an object", field="fixed_point")
    if fp.get("fraction_bits") not in ALLOWED_FRACTION_BITS:
        raise InputInvalid(
            f"fraction_bits must be one of {ALLOWED_FRACTION_BITS}", field="fixed_point.fraction_bits"
        )
    if fp.get("quantization_level") not in ALLOWED_QUANT_LEVELS:
        raise InputInvalid(
            f"quantization_level must be one of {ALLOWED_QUANT_LEVELS}",
            field="fixed_point.quantization_level",
        )

    if not _is_int(cfg.get("match_point_budget")) or cfg["match_point_budget"] <= 0:
        raise InputInvalid("match_point_budget must be a positive integer", field="match_point_budget")
    if not _is_int(cfg.get("max_workers")) or cfg["max_workers"] <= 0:
        raise InputInvalid("max_workers must be a positive integer", field="max_workers")
    if cfg.get("template_version") not in KNOWN_TEMPLATE_VERSIONS:
        from .codes import TEMPLATE_VERSION_UNKNOWN
        from .errors import BusinessError

        raise BusinessError(
            TEMPLATE_VERSION_UNKNOWN,
            f"unknown template version: {cfg.get('template_version')}",
            field="template_version",
            details={"known": sorted(KNOWN_TEMPLATE_VERSIONS)},
        )
    return cfg


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return validate_config({})
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InputInvalid(f"cannot parse config {p}: {exc}", field=str(p)) from exc
    return validate_config(data)


def stage_parameter_fingerprints(cfg: dict[str, Any]) -> dict[str, str]:
    """Smallest relevant parameter sets enable stage-level precise reuse."""
    fp = cfg["fixed_point"]
    return {
        "column": canonical_hash({
            "threshold": cfg["binarize_threshold"],
            "column_window": cfg["column_projection_window"],
            "layout": cfg["layout"],
        }),
        "projection": canonical_hash({
            "threshold": cfg["binarize_threshold"],
            "projection_window": cfg["projection_window"],
            "min_height": cfg["char_min_height"],
            "width_ratio_min": cfg["width_ratio_min"],
            "width_ratio_max": cfg["width_ratio_max"],
        }),
        "split": canonical_hash({
            "max_height": cfg["char_max_height"],
            "split_penalty": cfg["split_penalty"],
            "projection_window": cfg["projection_window"],
        }),
        "regression": canonical_hash({
            "threshold": cfg["binarize_threshold"],
            "sliding_window": cfg["sliding_window"],
            "min_height": cfg["char_min_height"],
            "max_height": cfg["char_max_height"],
        }),
        "matching": canonical_hash({
            "metric": "fixed_point_weighted_xor_similarity",
            "weights": cfg["fusion_weights"],
            "fraction_bits": fp["fraction_bits"],
            "quantization_level": fp["quantization_level"],
            "template_version": cfg["template_version"],
        }),
        "reading": canonical_hash({
            "keypoint_selector": cfg["keypoint_selector"],
            "quantization_level": fp["quantization_level"],
        }),
    }


COMMAND_SCHEMA = {
    "replace_image": {"image", "seed"},
    "add_column": {"image", "column_id", "x_center", "chars"},
    "remove_column": {"image", "column_id"},
    "add_block": {"image", "column_id", "block_id", "char", "y"},
    "remove_block": {"image", "block_id"},
    "set_keypoint_version": {"image", "block_id", "version_no", "version_name", "annotator", "annotated_ms", "x", "y"},
    "set_config": {"patch"},
}


def load_command_stream(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    commands: list[dict[str, Any]] = []
    try:
        if p.suffix == ".jsonl":
            for lineno, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                cmd = json.loads(line)
                cmd["_lineno"] = lineno
                commands.append(cmd)
        else:
            data = json.loads(text)
            commands = list(data.get("commands", [])) if isinstance(data, dict) else list(data)
    except Exception as exc:
        raise InputInvalid(f"cannot parse command stream {p}: {exc}", field=str(p)) from exc

    for index, cmd in enumerate(commands):
        if not isinstance(cmd, dict) or "op" not in cmd:
            raise InputInvalid("command must be an object with op", field=f"commands[{index}]")
        op = cmd["op"]
        if op not in COMMAND_SCHEMA:
            raise InputInvalid(f"unknown command op: {op}", field=f"commands[{index}].op")
        required = COMMAND_SCHEMA[op]
        missing = [key for key in required if key not in cmd and key not in {"seed", "x_center", "chars", "y"}]
        if missing:
            raise InputInvalid(f"command {op} missing {missing}", field=f"commands[{index}]")
        if op == "set_config" and not isinstance(cmd.get("patch"), dict):
            raise InputInvalid("set_config requires patch object", field=f"commands[{index}].patch")
    return commands
