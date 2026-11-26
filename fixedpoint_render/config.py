"""Render configuration model and schema validation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

from .errors import (
    E_BUDGET_POSITIVE, E_CONFIG_FIELD, E_CONFIG_PARSE, E_DIM_MISMATCH,
    E_QBITS_RANGE, E_QMODE_ENUM, E_WEIGHT_VALUE, ErrorBundle, RenderError,
)
from .fixedpoint import ALLOWED_FRAC_BITS, ALLOWED_QMODES, parse_decimal_q

CONDITION_TYPES = ("image", "motion", "text")  # sorted by condition type name
# Note: alphabetical order of condition type names is image < motion < text.


@dataclass(frozen=True)
class RenderConfig:
    vocab: tuple[str, ...]
    width: int
    height: int
    pixel_mode: str                 # "RGB" or "L"
    channels: int
    tile_rows: int                  # number of block rows
    condition_dim: int              # F, identical for all three encoders
    latent_dim: int                 # L
    frac_bits: int                  # fixed point fractional width
    qmode: str                      # trunc | nearest | floor
    weights: dict[str, str]         # condition type -> decimal string
    guidance: str                   # decimal string
    per_frame_budget: int           # b: max denoise steps for any frame
    global_step_budget: int         # B: total steps across all frames
    noise_seed: int
    max_workers: int
    snapshot_versions: int = 1      # schema/snapshot version marker

    def weight_q(self, cond_type: str) -> int:
        return parse_decimal_q(self.weights[cond_type], self.frac_bits, self.qmode)

    def guidance_q(self) -> int:
        return parse_decimal_q(self.guidance, self.frac_bits, self.qmode)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["vocab"] = list(self.vocab)
        return data

    def canonical_json(self) -> str:
        # max_workers is an execution-only parameter and must not influence any
        # content fingerprint, cache key or numeric result.
        data = self.to_dict()
        data.pop("max_workers", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)


REQUIRED_FIELDS = {
    "vocab": list, "width": int, "height": int, "pixel_mode": str,
    "tile_rows": int, "condition_dim": int, "latent_dim": int,
    "frac_bits": int, "qmode": str, "weights": dict, "guidance": str,
    "per_frame_budget": int, "global_step_budget": int, "noise_seed": int,
    "max_workers": int,
}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_config(path: str) -> RenderConfig:
    location = os.path.basename(path)
    errors: list[RenderError] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise ErrorBundle([RenderError(
            "E_FILE_MISSING", location, "config", "config file not found")])
    except (json.JSONDecodeError, OSError) as exc:
        raise ErrorBundle([RenderError(E_CONFIG_PARSE, location, "config", str(exc))])

    if not isinstance(raw, dict):
        raise ErrorBundle([RenderError(E_CONFIG_PARSE, location, "config",
                                       "top-level config must be an object")])

    for field, typ in REQUIRED_FIELDS.items():
        if field not in raw:
            errors.append(RenderError(E_CONFIG_FIELD, location, field,
                                      f"missing required field {field}"))
        elif not isinstance(raw[field], typ):
            errors.append(RenderError(E_CONFIG_FIELD, location, field,
                                      f"expected {typ.__name__}"))
    if errors:
        raise ErrorBundle(errors)

    vocab = raw["vocab"]
    if not vocab or not all(isinstance(t, str) and t for t in vocab):
        errors.append(RenderError(E_CONFIG_FIELD, location, "vocab",
                                  "vocab must be a non-empty list of strings"))
    elif len(vocab) != len(set(vocab)):
        errors.append(RenderError(E_CONFIG_FIELD, location, "vocab",
                                  "vocab terms must be unique"))

    for dim_field in ("width", "height", "tile_rows", "condition_dim",
                      "latent_dim"):
        if not _is_int(raw[dim_field]) or raw[dim_field] <= 0:
            errors.append(RenderError(E_CONFIG_FIELD, location, dim_field,
                                      "must be a positive integer"))
    if not errors:
        if raw["height"] % raw["tile_rows"] != 0:
            errors.append(RenderError(E_CONFIG_FIELD, location, "tile_rows",
                                      "height must be divisible by tile_rows"))
        if raw["pixel_mode"] not in ("RGB", "L"):
            errors.append(RenderError(E_PNG_MODE, location, "pixel_mode",
                                      "pixel_mode must be RGB or L"))
        else:
            expected_ch = 3 if raw["pixel_mode"] == "RGB" else 1
            if "channels" in raw and raw["channels"] != expected_ch:
                errors.append(RenderError(E_DIM_MISMATCH, location, "channels",
                                          f"channels must be {expected_ch}"))

    errors.extend(_validate_q(raw, location))
    errors.extend(_validate_weights_budget(raw, location))

    if not _is_int(raw["max_workers"]) or raw["max_workers"] < 1:
        errors.append(RenderError(E_CONFIG_FIELD, location, "max_workers",
                                  "must be an integer >= 1"))
    if not _is_int(raw["noise_seed"]):
        errors.append(RenderError(E_CONFIG_FIELD, location, "noise_seed",
                                  "noise_seed must be an integer"))

    if errors:
        raise ErrorBundle(errors)

    channels = 3 if raw["pixel_mode"] == "RGB" else 1
    return RenderConfig(
        vocab=tuple(vocab),
        width=raw["width"], height=raw["height"], pixel_mode=raw["pixel_mode"],
        channels=channels, tile_rows=raw["tile_rows"],
        condition_dim=raw["condition_dim"], latent_dim=raw["latent_dim"],
        frac_bits=raw["frac_bits"], qmode=raw["qmode"],
        weights={k: str(v) for k, v in raw["weights"].items()},
        guidance=str(raw["guidance"]),
        per_frame_budget=raw["per_frame_budget"],
        global_step_budget=raw["global_step_budget"],
        noise_seed=raw["noise_seed"], max_workers=raw["max_workers"],
        snapshot_versions=int(raw.get("snapshot_versions", 1)),
    )


def _validate_q(raw: dict, location: str) -> list[RenderError]:
    errs: list[RenderError] = []
    if raw["frac_bits"] not in ALLOWED_FRAC_BITS:
        errs.append(RenderError(E_QBITS_RANGE, location, "frac_bits",
                                f"not in {ALLOWED_FRAC_BITS}"))
    if raw["qmode"] not in ALLOWED_QMODES:
        errs.append(RenderError(E_QMODE_ENUM, location, "qmode",
                                f"not in {ALLOWED_QMODES}"))
    return errs


def _validate_weights_budget(raw: dict, location: str) -> list[RenderError]:
    errs: list[RenderError] = []
    weights = raw["weights"]
    if set(weights.keys()) != set(CONDITION_TYPES):
        errs.append(RenderError(E_CONFIG_FIELD, location, "weights",
                                f"keys must be exactly {CONDITION_TYPES}"))
    else:
        # Use a safe rounding mode for parsing when the configured qmode is
        # itself invalid; qmode validation already reports that separately.
        parse_mode = raw["qmode"] if raw["qmode"] in ALLOWED_QMODES else "trunc"
        for key in CONDITION_TYPES:
            try:
                qv = parse_decimal_q(str(weights[key]), raw["frac_bits"],
                                     parse_mode)
            except ValueError as exc:
                errs.append(RenderError(E_WEIGHT_VALUE, location,
                                        f"weights.{key}", str(exc)))
                continue
            if qv < 0:
                errs.append(RenderError(E_WEIGHT_VALUE, location,
                                        f"weights.{key}", "weight must be >= 0"))
        try:
            if parse_decimal_q(str(raw["guidance"]), raw["frac_bits"],
                               parse_mode) < 0:
                errs.append(RenderError(E_WEIGHT_VALUE, location, "guidance",
                                        "guidance must be >= 0"))
        except ValueError as exc:
            errs.append(RenderError(E_WEIGHT_VALUE, location, "guidance", str(exc)))

    for field in ("per_frame_budget", "global_step_budget"):
        if not _is_int(raw[field]) or raw[field] <= 0:
            errs.append(RenderError(E_BUDGET_POSITIVE, location, field,
                                    "budget must be a positive integer"))
    if not errs and raw["global_step_budget"] < raw["per_frame_budget"]:
        errs.append(RenderError(E_BUDGET_POSITIVE, location,
                                "global_step_budget",
                                "global budget must be >= per-frame budget"))
    return errs
