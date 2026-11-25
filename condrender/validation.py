from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .errors import Codes, Failure
from .fixedpoint import ALLOWED_FRAC_BITS, PROFILE_FRAC_BITS
from .pngcodec import decode_png_rgb


STOPWORDS = {"a", "an", "the", "over", "under", "with", "on", "and", "of", "to"}


def is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_yaml_file(path: Path, failures: list[Failure]) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(Failure(Codes.FILE_NOT_FOUND, "required YAML file is missing", path=str(path)))
    except yaml.YAMLError as exc:
        failures.append(Failure(Codes.CONFIG_INVALID, f"YAML parse error: {exc}", path=str(path)))
    return None


def load_json_file(path: Path, failures: list[Failure]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(Failure(Codes.FILE_NOT_FOUND, "required JSON file is missing", path=str(path)))
    except json.JSONDecodeError as exc:
        failures.append(Failure(Codes.CONFIG_INVALID, f"JSON parse error: {exc}", path=str(path)))
    return None


def tokenize_prompt(text: str) -> list[dict[str, int | str]]:
    hits: list[dict[str, int | str]] = []
    token_index = 0
    cursor = 0
    for token in text.split():
        start = text.find(token, cursor)
        end = start + len(token)
        normalized = "".join(ch for ch in token.lower() if ch.isalnum())
        hits.append({
            "token": token,
            "normalized": normalized,
            "token_index": token_index,
            "char_start": start,
            "char_end": end,
        })
        cursor = end
        token_index += 1
    return hits


def validate_prompt_text(text: Any, vocab: Iterable[str], *, path: str | None = None,
                         frame: int | None = None) -> list[Failure]:
    failures: list[Failure] = []
    if not isinstance(text, str) or not text.strip():
        failures.append(Failure(Codes.VOCAB_EMPTY_PROMPT, "prompt must be a non-empty string",
                                path=path, frame=frame))
        return failures
    allowed = set(vocab)
    unknown: list[str] = []
    for token in tokenize_prompt(text):
        term = str(token["normalized"])
        if term in STOPWORDS:
            continue
        if term not in allowed:
            unknown.append(term)
    for term in unknown:
        failures.append(Failure(
            Codes.VOCAB_UNKNOWN_TERM, f"term {term!r} is outside the controlled vocabulary",
            path=path, frame=frame, details={"term": term},
        ))
    return failures


def _require_positive_int(failures: list[Failure], value: Any, field: str,
                          path: str | None = None) -> None:
    if not is_plain_int(value) or value <= 0:
        failures.append(Failure(
            Codes.POSITIVE_INT, f"{field} must be a positive integer", path=path,
            details={"field": field, "actual": value},
        ))


def validate_config(config: Any, path: str | None = None) -> list[Failure]:
    failures: list[Failure] = []
    if not isinstance(config, dict):
        failures.append(Failure(Codes.CONFIG_INVALID, "render config must be a mapping", path=path))
        return failures

    seed = config.get("seed")
    seed_min = config.get("seed_min", 0)
    seed_max = config.get("seed_max", 2**32 - 1)
    if not is_plain_int(seed) or not (is_plain_int(seed_min) and is_plain_int(seed_max)) \
            or not (seed_min <= seed <= seed_max):
        failures.append(Failure(
            Codes.SEED_RANGE, "seed must be an integer within the declared range", path=path,
            details={"seed": seed, "range": [seed_min, seed_max]},
        ))

    for field in ("frames", "width", "height", "max_workers"):
        _require_positive_int(failures, config.get(field), field, path)
    _require_positive_int(failures, config.get("denoise_steps"), "denoise_steps", path)
    _require_positive_int(failures, config.get("tile_rows"), "tile_rows", path)

    height = config.get("height")
    tile_rows = config.get("tile_rows")
    if is_plain_int(height) and is_plain_int(tile_rows) and tile_rows > 0:
        if tile_rows > height:
            failures.append(Failure(
                Codes.TILE_ROW_RANGE, "tile_rows must not exceed frame height in pixels",
                path=path, details={"tile_rows": tile_rows, "height": height},
            ))

    fixed = config.get("fixed_point", {})
    profile = fixed.get("quant_profile") if isinstance(fixed, dict) else None
    frac_bits = fixed.get("frac_bits") if isinstance(fixed, dict) else None
    if profile not in PROFILE_FRAC_BITS:
        failures.append(Failure(
            Codes.QUANT_ENUM, "quant_profile is not in the allowed enum", path=path,
            details={"actual": profile, "allowed": sorted(PROFILE_FRAC_BITS)},
        ))
    if frac_bits not in ALLOWED_FRAC_BITS:
        failures.append(Failure(
            Codes.FRAC_BITS_ENUM, "frac_bits is not in the allowed enum", path=path,
            details={"actual": frac_bits, "allowed": list(ALLOWED_FRAC_BITS)},
        ))
    if profile in PROFILE_FRAC_BITS and frac_bits in ALLOWED_FRAC_BITS \
            and PROFILE_FRAC_BITS[profile] != frac_bits:
        failures.append(Failure(
            Codes.QUANT_ENUM, "quant_profile does not match frac_bits", path=path,
            details={"profile": profile, "frac_bits": frac_bits},
        ))

    noise = config.get("noise", {})
    if not isinstance(noise, dict) or not isinstance(noise.get("allowed_versions"), list) \
            or not noise.get("allowed_versions"):
        failures.append(Failure(Codes.NOISE_VERSION, "noise.allowed_versions must be a non-empty list",
                                path=path))

    budget = config.get("interaction_budget_ms")
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget < 0:
        failures.append(Failure(Codes.CONFIG_INVALID, "interaction_budget_ms must be non-negative",
                                path=path, details={"actual": budget}))

    grid = config.get("motion_grid", {})
    width = config.get("width")
    if isinstance(grid, dict) and is_plain_int(width) and is_plain_int(height):
        bw, bh = grid.get("block_width"), grid.get("block_height")
        cols, rows = grid.get("cols"), grid.get("rows")
        if not all(is_plain_int(v) and v > 0 for v in (bw, bh, cols, rows)):
            failures.append(Failure(Codes.MOTION_GRID, "motion grid fields must be positive integers",
                                    path=path))
        elif bw * cols != width or bh * rows != height:
            failures.append(Failure(
                Codes.MOTION_GRID, "motion block grid is not compatible with frame dimensions",
                path=path,
                details={"frame": [width, height], "block": [bw, bh], "grid": [cols, rows]},
            ))
    else:
        failures.append(Failure(Codes.CONFIG_INVALID, "motion_grid/frame dimensions are invalid", path=path))
    return failures


def validate_motion_table(motion: Any, config: dict[str, Any], path: str | None = None) -> list[Failure]:
    failures: list[Failure] = []
    if not isinstance(motion, dict):
        return [Failure(Codes.MOTION_GRID, "motion table must be a mapping", path=path)]
    grid = config["motion_grid"]
    expected_shape = (
        int(config["frames"]), int(grid["rows"]), int(grid["cols"]), 2
    )
    vectors = motion.get("vectors")
    arr = np.asarray(vectors, dtype=np.int64) if vectors is not None else None
    if arr is None or arr.shape != expected_shape:
        actual = None if arr is None else list(arr.shape)
        code = Codes.MOTION_BLOCK_MISMATCH if arr is not None and arr.ndim >= 3 \
            and arr.shape[0] == expected_shape[0] else Codes.MOTION_GRID
        failures.append(Failure(
            code, "motion vector block count/shape is incompatible with frame size",
            path=path, details={"actual": actual, "expected": list(expected_shape)},
        ))
        return failures
    if motion.get("frame_width") != config["width"] or motion.get("frame_height") != config["height"]:
        failures.append(Failure(
            Codes.MOTION_GRID, "motion table frame dimensions do not match render config", path=path,
            details={
                "actual": [motion.get("frame_width"), motion.get("frame_height")],
                "expected": [config["width"], config["height"]],
            },
        ))
    return failures


def validate_noise_table(noise: Any, config: dict[str, Any], path: str | None = None) -> list[Failure]:
    failures: list[Failure] = []
    if not isinstance(noise, dict):
        return [Failure(Codes.NOISE_VERSION_MISSING, "noise table is not a mapping", path=path)]
    if "version" not in noise or noise.get("version") in (None, ""):
        failures.append(Failure(Codes.NOISE_VERSION_MISSING, "noise table version is missing", path=path))
    else:
        allowed = config.get("noise", {}).get("allowed_versions", [])
        if noise["version"] not in allowed:
            failures.append(Failure(
                Codes.NOISE_VERSION, "noise table version is not in the declared set", path=path,
                details={"actual": noise["version"], "allowed": allowed},
            ))
    entries = noise.get("entries")
    latent_size = config["width"] * config["height"] * 3
    steps = config.get("max_noise_steps", config.get("denoise_steps"))
    try:
        arr = np.asarray(entries, dtype=np.int64)
        if arr.ndim != 2 or arr.shape[0] < int(config["denoise_steps"]) or arr.shape[1] != latent_size:
            failures.append(Failure(
                Codes.COND_DIM_MISMATCH, "noise table dimensions do not match config/latent shape",
                path=path, details={"actual": list(arr.shape),
                                    "expected_min": [steps, latent_size]},
            ))
    except (TypeError, ValueError):
        failures.append(Failure(Codes.COND_DIM_MISMATCH, "noise entries are not integer arrays", path=path))
    return failures


def validate_dimensions(config: dict[str, Any], vocab_terms: list[str]) -> list[Failure]:
    failures: list[Failure] = []
    dims = config.get("condition_dims", {})
    expected_text = len(vocab_terms)
    image_cfg = config.get("condition_image", {})
    expected_image = int(image_cfg.get("block_cols", 0)) * int(image_cfg.get("block_rows", 0)) * 5
    grid = config.get("motion_grid", {})
    expected_motion = int(grid.get("rows", 0)) * int(grid.get("cols", 0)) * 2
    expected_motion_row = int(grid.get("cols", 0)) * 2
    expected_fused = expected_text + expected_image + expected_motion_row
    expected = {
        "text": expected_text,
        "image": expected_image,
        "motion": expected_motion,
        "motion_row": expected_motion_row,
        "fused": expected_fused,
    }
    for key, value in expected.items():
        if dims.get(key) != value:
            failures.append(Failure(
                Codes.COND_DIM_MISMATCH, f"condition_dims.{key} does not match deterministic layout",
                details={"field": key, "actual": dims.get(key), "expected": value},
            ))
    return failures


def write_failure_list(failures: list[Failure], output_path: str | Path,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    failure_dicts = [
        failure.to_dict() if hasattr(failure, "to_dict") else dict(failure)
        for failure in failures
    ]
    payload = {
        "count": len(failure_dicts),
        "failures": failure_dicts,
    }
    if extra:
        payload.update(extra)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
                      encoding="utf-8")
    return payload


def validate_png_bytes(data: bytes, config: dict[str, Any], path: str | None = None) -> list[Failure]:
    image_cfg = config["condition_image"]
    try:
        decode_png_rgb(data, image_cfg["width"], image_cfg["height"], source_path=path)
        return []
    except Exception as exc:
        failure = getattr(exc, "failure", None)
        if isinstance(failure, Failure):
            return [failure]
        return [Failure(Codes.PNG_HEADER, str(exc), path=path)]
