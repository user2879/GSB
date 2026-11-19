"""Render configuration model with strict, write-free validation.

Validation never touches the snapshot store, so a rejected configuration cannot
mutate any existing artifact or version (requirement 5).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    E_CONFIG_INVALID,
    E_GUIDE_NOT_FINITE,
    E_NOISE_TABLE_MISMATCH,
    E_PATCH_OUT_OF_RANGE,
    E_PATCH_SHAPE_MISMATCH,
    E_SEED_OUT_OF_RANGE,
    E_STEP_ZERO,
    WorkbenchError,
)
from .hashing import canonical_bytes, fingerprint_json, sha256_hex

SEED_MIN = 0
SEED_MAX = 2**32 - 1
LATENT_H = 8
LATENT_W = 8
LATENT_C = 4
UPSCALE = 4
IMAGE_SIZE = LATENT_H * UPSCALE  # 32
NOISE_ENTRY_LEN = LATENT_C
MAX_WORKERS_CAP = 64

REQUIRED_FIELDS = (
    "vocab_file",
    "prompt",
    "seed",
    "steps",
    "guide_scale",
    "noise_table",
    "latent_shape",
    "max_workers",
)


@dataclass(frozen=True)
class Patch:
    step: int
    top: int
    left: int
    height: int
    width: int
    values: tuple  # nested tuple of floats, shape (height, width, channels)

    def array(self) -> "np.ndarray":  # pragma: no cover - typing hint only
        import numpy as np

        return np.asarray(self.values, dtype=np.float64)


@dataclass(frozen=True)
class Config:
    prompt: str
    seed: int
    steps: int
    guide_scale: float
    noise_table: tuple[tuple[float, ...], ...]
    latent_shape: tuple[int, int, int]
    max_workers: int
    vocab_file: str = ""
    initial_image: str | None = None
    image_replacement_steps: int = 0
    patches: tuple[Patch, ...] = field(default_factory=tuple)
    source_path: str | None = None

    # ---- fingerprinting -------------------------------------------------
    def normalized(self) -> dict[str, Any]:
        """Canonical, JSON-only representation used for the config fingerprint."""
        return {
            "prompt": self.prompt,
            "seed": self.seed,
            "steps": self.steps,
            "guide_scale": self.guide_scale,
            "noise_table": [list(row) for row in self.noise_table],
            "latent_shape": list(self.latent_shape),
            "max_workers": self.max_workers,
            "vocab_file": self.vocab_file,
            "initial_image": self.initial_image,
            "image_replacement_steps": self.image_replacement_steps,
            "patches": [
                {
                    "step": p.step,
                    "top": p.top,
                    "left": p.left,
                    "height": p.height,
                    "width": p.width,
                    "values": p.values,
                }
                for p in self.patches
            ],
        }

    def fingerprint(self) -> str:
        return fingerprint_json(
            {"schema": "dbench.config.v1", "config": self.normalized()}
        )

    def effective_fingerprint(self) -> str:
        """Fingerprint excluding the concurrency setting.

        Concurrency is an execution scheduling parameter and must never change
        computed artifacts (requirement 8).
        """
        data = self.normalized()
        # Concurrency is pure scheduling and patches are step-local inputs;
        # neither may invalidate unrelated upstream stages. Patches enter only
        # the fingerprint of the specific denoise step they target.
        data.pop("max_workers")
        data.pop("patches")
        # Replacement count is DAG routing, not an encoder/decoder input.
        data.pop("image_replacement_steps")
        return fingerprint_json({"schema": "dbench.config.effective.v1", "config": data})

    def patches_for_step(self, step: int) -> tuple[Patch, ...]:
        return tuple(p for p in self.patches if p.step == step)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fail(code: str, message: str, context: dict | None = None) -> None:
    raise WorkbenchError(code, message, context)


def _validate_patch(raw: Any, index: int, steps: int, shape: tuple[int, int, int]) -> Patch:
    where = f"patches[{index}]"
    if not isinstance(raw, dict):
        _fail(E_CONFIG_INVALID, f"{where} must be an object")
    required = ("step", "top", "left", "height", "width", "values")
    for key in required:
        if key not in raw:
            _fail(E_CONFIG_INVALID, f"{where} missing field '{key}'")
    for key in ("step", "top", "left", "height", "width"):
        if not _is_int(raw[key]):
            _fail(E_CONFIG_INVALID, f"{where}.{key} must be an integer")
    step, top, left, height, width = (
        raw["step"],
        raw["top"],
        raw["left"],
        raw["height"],
        raw["width"],
    )
    if not (1 <= step <= steps):
        _fail(E_CONFIG_INVALID, f"{where}.step must be in [1, steps]", {"step": step})
    if height <= 0 or width <= 0:
        _fail(E_CONFIG_INVALID, f"{where} height/width must be positive")
    channels, lh, lw = shape
    # Range check before shape check so the two failure modes stay distinct.
    if top < 0 or left < 0 or top + height > lh or left + width > lw:
        _fail(
            E_PATCH_OUT_OF_RANGE,
            f"{where} rectangle is outside the latent grid",
            {"top": top, "left": left, "height": height, "width": width,
             "latent_height": lh, "latent_width": lw},
        )
    values = raw["values"]
    if not isinstance(values, list):
        _fail(E_PATCH_SHAPE_MISMATCH, f"{where}.values must be a nested array")
    expected_shape = (height, width, channels)
    try:
        import numpy as np

        arr = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        _fail(E_PATCH_SHAPE_MISMATCH, f"{where}.values is not numeric: {exc}")
    if tuple(arr.shape) != expected_shape:
        _fail(
            E_PATCH_SHAPE_MISMATCH,
            f"{where}.values shape {tuple(arr.shape)} != region {expected_shape}",
            {"actual": list(arr.shape), "expected": list(expected_shape)},
        )
    nested = tuple(tuple(tuple(float(x) for x in col) for col in row) for row in arr)
    return Patch(step, top, left, height, width, nested)


def load_config(path: str) -> Config:
    import json

    if not os.path.isfile(path):
        _fail(E_CONFIG_INVALID, f"config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        _fail(E_CONFIG_INVALID, f"config is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        _fail(E_CONFIG_INVALID, "config root must be a JSON object")
    return parse_config(raw, source_path=path)


def parse_config(raw: dict[str, Any], source_path: str | None = None) -> Config:
    # Structural presence / type checks first -> E_CONFIG_INVALID.
    for key in REQUIRED_FIELDS:
        if key not in raw:
            _fail(E_CONFIG_INVALID, f"missing required field '{key}'")

    if not isinstance(raw["vocab_file"], str) or not raw["vocab_file"]:
        _fail(E_CONFIG_INVALID, "vocab_file must be a non-empty string path")
    if not isinstance(raw["prompt"], str):
        _fail(E_CONFIG_INVALID, "prompt must be a string")
    if not _is_int(raw["seed"]):
        _fail(E_CONFIG_INVALID, "seed must be an integer")
    if not _is_int(raw["steps"]):
        _fail(E_CONFIG_INVALID, "steps must be an integer")
    if not _is_real(raw["guide_scale"]):
        _fail(E_CONFIG_INVALID, "guide_scale must be a number")
    if not _is_int(raw["max_workers"]):
        _fail(E_CONFIG_INVALID, "max_workers must be an integer")
    if not (
        isinstance(raw["latent_shape"], list)
        and len(raw["latent_shape"]) == 3
        and all(_is_int(v) for v in raw["latent_shape"])
    ):
        _fail(E_CONFIG_INVALID, "latent_shape must be [channels, height, width] ints")
    if not isinstance(raw["noise_table"], list):
        _fail(E_CONFIG_INVALID, "noise_table must be an array")

    # Domain-specific codes (order mirrors requirement 5).
    if not (SEED_MIN <= raw["seed"] <= SEED_MAX):
        _fail(E_SEED_OUT_OF_RANGE, "seed out of uint32 range",
              {"seed": raw["seed"], "min": SEED_MIN, "max": SEED_MAX})
    if raw["steps"] == 0:
        _fail(E_STEP_ZERO, "steps must be >= 1", {"steps": 0})
    if raw["steps"] < 0:
        _fail(E_CONFIG_INVALID, "steps must be positive")
    if not math.isfinite(float(raw["guide_scale"])):
        _fail(E_GUIDE_NOT_FINITE, "guide_scale must be finite",
              {"guide_scale": str(raw["guide_scale"])})
    if not (1 <= raw["max_workers"] <= MAX_WORKERS_CAP):
        _fail(E_CONFIG_INVALID, f"max_workers must be in [1, {MAX_WORKERS_CAP}]")

    shape = tuple(int(v) for v in raw["latent_shape"])
    steps = int(raw["steps"])

    # Noise table: one fixed-length finite row per denoise step.
    table: list[tuple[float, ...]] = []
    for idx, row in enumerate(raw["noise_table"]):
        if not isinstance(row, list) or not all(_is_real(v) for v in row):
            _fail(E_CONFIG_INVALID, f"noise_table[{idx}] must be a numeric array")
        if len(row) != NOISE_ENTRY_LEN:
            _fail(
                E_NOISE_TABLE_MISMATCH,
                f"noise_table[{idx}] length {len(row)} != {NOISE_ENTRY_LEN}",
                {"index": idx, "actual": len(row), "expected": NOISE_ENTRY_LEN},
            )
        if not all(math.isfinite(float(v)) for v in row):
            _fail(E_CONFIG_INVALID, f"noise_table[{idx}] contains non-finite values")
        table.append(tuple(float(v) for v in row))
    if len(table) != steps:
        _fail(
            E_NOISE_TABLE_MISMATCH,
            f"noise_table length {len(table)} != steps {steps}",
            {"actual": len(table), "expected": steps},
        )

    # Optional image-to-image branch.
    initial_image = raw.get("initial_image")
    if initial_image is not None and not isinstance(initial_image, str):
        _fail(E_CONFIG_INVALID, "initial_image must be a string path or null")
    repl = raw.get("image_replacement_steps", 0)
    if not _is_int(repl):
        _fail(E_CONFIG_INVALID, "image_replacement_steps must be an integer")
    if repl < 0:
        _fail(E_CONFIG_INVALID, "image_replacement_steps must be >= 0")
    if initial_image and repl == 0:
        _fail(E_CONFIG_INVALID,
              "image_replacement_steps must be >= 1 when initial_image is set")
    if repl > steps:
        _fail(E_CONFIG_INVALID, "image_replacement_steps cannot exceed steps")

    patches: list[Patch] = []
    raw_patches = raw.get("patches", [])
    if not isinstance(raw_patches, list):
        _fail(E_CONFIG_INVALID, "patches must be an array")
    for idx, item in enumerate(raw_patches):
        patches.append(_validate_patch(item, idx, steps, shape))

    return Config(
        prompt=str(raw["prompt"]),
        seed=int(raw["seed"]),
        steps=steps,
        guide_scale=float(raw["guide_scale"]),
        noise_table=tuple(table),
        latent_shape=shape,
        max_workers=int(raw["max_workers"]),
        vocab_file=str(raw["vocab_file"]),
        initial_image=initial_image,
        image_replacement_steps=int(repl),
        patches=tuple(patches),
        source_path=source_path,
    )


def config_identity_digest(config: Config) -> str:
    """Stable digest joining the effective config fingerprint."""
    return sha256_hex(b"dbench.config.identity.v1\0" + canonical_bytes(
        config.effective_fingerprint()
    ))
