"""Pure deterministic computations for every pipeline stage.

Every function takes explicit inputs and returns arrays plus a deterministic
parameter record. No global random state and no wall-clock data are used.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from PIL import Image

from .config import Config, LATENT_C, LATENT_H, LATENT_W, UPSCALE
from .errors import (
    E_IMAGE_DECODE_FAILED,
    E_IMAGE_SIZE_MISMATCH,
    E_NON_FINITE_LATENT,
    WorkbenchError,
)
from .hashing import fingerprint_array, fingerprint_json, sha256_hex
from .rng import derive_rng

# --------------------------------------------------------------------------
# Stage 1: prompt encoding (restricted vocabulary + hashed bag-of-words)
# --------------------------------------------------------------------------

def load_vocab(path: str) -> tuple[str, ...]:
    if not os.path.isfile(path):
        raise WorkbenchError("E_CONFIG_INVALID", f"vocab file not found: {path}")
    words: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token:
                words.append(token)
    if not words:
        raise WorkbenchError("E_CONFIG_INVALID", f"vocab file is empty: {path}")
    return tuple(words)


def vocab_fingerprint(vocab: tuple[str, ...]) -> str:
    return fingerprint_json({"schema": "dbench.vocab.v1", "vocab": list(vocab)})


def _tokenize(prompt: str) -> list[str]:
    return [token for token in prompt.lower().split() if token]


def _hashed_index(token: str, vocab_size: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % vocab_size


def encode_prompt(prompt: str, vocab: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    vocab_size = len(vocab)
    index = {word: i for i, word in enumerate(vocab)}
    counts = np.zeros(vocab_size, dtype=np.float64)
    oov_hashed = np.zeros(vocab_size, dtype=np.float64)
    in_vocab_tokens: list[str] = []
    oov_tokens: list[str] = []
    for token in _tokenize(prompt):
        if token in index:
            counts[index[token]] += 1.0
            in_vocab_tokens.append(token)
        else:
            oov_hashed[_hashed_index(token, vocab_size)] += 1.0
            oov_tokens.append(token)
    raw = counts + oov_hashed
    norm = float(np.linalg.norm(raw, ord=1))
    embedding = raw / norm if norm > 0.0 else raw.copy()
    params = {
        "schema": "dbench.stage.encode.v1",
        "vocab_fingerprint": vocab_fingerprint(vocab),
        "vocab_size": vocab_size,
        "prompt": prompt,
        "prompt_sha256": sha256_hex(prompt.encode("utf-8")),
        "in_vocab_tokens": in_vocab_tokens,
        "oov_tokens": oov_tokens,
        "l1_norm": norm,
    }
    return embedding.astype(np.float64, copy=False), params


# --------------------------------------------------------------------------
# Stage 2: latent initialization from a prompt/seed/config-derived RNG
# --------------------------------------------------------------------------

def initialize_latent(config: Config, prompt_fingerprint: str,
                      latent_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict]:
    rng = derive_rng(
        "dbench.init-latent.v1",
        prompt_fingerprint,
        config.seed,
        config.effective_fingerprint(),
        list(latent_shape),
    )
    latent = rng.standard_normal(latent_shape)
    params = {
        "schema": "dbench.stage.init.v1",
        "rng_domain": "dbench.init-latent.v1",
        "seed": config.seed,
        "config_fingerprint": config.effective_fingerprint(),
        "prompt_fingerprint": prompt_fingerprint,
        "latent_shape": list(latent_shape),
    }
    return latent, params


# --------------------------------------------------------------------------
# Stage 3 (img2img only): encode an initial PNG into a latent
# --------------------------------------------------------------------------

def encode_initial_image(path: str, config_fingerprint: str,
                         expected_size: int) -> tuple[np.ndarray, dict]:
    if not os.path.isfile(path):
        raise WorkbenchError(E_IMAGE_DECODE_FAILED, f"initial image not found: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if (width, height) != (expected_size, expected_size):
                raise WorkbenchError(
                    E_IMAGE_SIZE_MISMATCH,
                    "initial image dimensions do not match configuration",
                    {"path": path, "actual": [height, width],
                     "expected": [expected_size, expected_size]},
                )
            rgb = image.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.float64)
    except WorkbenchError:
        raise
    except Exception as exc:  # corrupt / unreadable PNG
        raise WorkbenchError(E_IMAGE_DECODE_FAILED,
                             f"cannot decode image {path}: {exc}") from exc

    # Average-pool 32x32 RGB -> 8x8 RGB, then project to 4 latent channels
    # using a config-derived deterministic matrix (no learned weights).
    pooled = pixels.reshape(
        LATENT_H, UPSCALE, LATENT_W, UPSCALE, 3
    ).mean(axis=(1, 3))
    rng = derive_rng("dbench.image-encode.proj.v1", config_fingerprint)
    proj = rng.standard_normal((3, LATENT_C))
    latent = np.moveaxis(pooled @ proj, -1, 0)
    params = {
        "schema": "dbench.stage.image_encode.v1",
        "path": os.path.abspath(path),
        "expected_size": expected_size,
        "pixel_fingerprint": fingerprint_array(pixels),
        "config_fingerprint": config_fingerprint,
    }
    return latent, params


# --------------------------------------------------------------------------
# Stages 4..: denoise steps
# --------------------------------------------------------------------------

def _guidance_matrix(config: Config, step: int, latent_shape: tuple[int, int, int]) -> np.ndarray:
    rng = derive_rng("dbench.guidance.v1", config.effective_fingerprint(), step,
                     list(latent_shape))
    channels = latent_shape[0]
    return rng.standard_normal((channels, channels))


def _step_noise(config: Config, step: int, latent_shape: tuple[int, int, int]) -> np.ndarray:
    rng = derive_rng("dbench.step-noise.v1", config.seed,
                     config.effective_fingerprint(), step)
    return rng.standard_normal(latent_shape)


def denoise_step(step: int, previous: np.ndarray, embedding: np.ndarray,
                 config: Config) -> tuple[np.ndarray, dict]:
    latent_shape = tuple(int(v) for v in config.latent_shape)
    noise = _step_noise(config, step, latent_shape)
    scales = np.asarray(config.noise_table[step - 1], dtype=np.float64).reshape(
        (latent_shape[0], 1, 1)
    )
    guidance = _guidance_matrix(config, step, latent_shape)
    # Conditional guidance: per-pixel channel mixing driven by the normalized
    # bag-of-words embedding, scaled by guide_scale.
    cond = (guidance @ embedding[:latent_shape[0]].reshape(-1)).reshape(
        (latent_shape[0], 1, 1)
    )
    strength = 1.0 / float(step + 1)
    updated = (
        previous
        + strength * noise * scales
        + config.guide_scale * strength * cond
    )
    params = {
        "schema": "dbench.stage.denoise.v1",
        "step": step,
        "rng_domain": "dbench.step-noise.v1",
        "noise_table_entry": list(config.noise_table[step - 1]),
        "guide_scale": config.guide_scale,
        "strength": strength,
        "embedding_fingerprint": fingerprint_array(embedding),
        "noise_fingerprint": fingerprint_array(noise),
    }
    return updated.astype(np.float64, copy=False), params


def assert_finite(latent: np.ndarray, stage: str, step: int | None,
                  injection_id: str | None = None) -> None:
    if not np.isfinite(latent).all():
        raise WorkbenchError(
            E_NON_FINITE_LATENT,
            "non-finite latent element detected",
            {"stage": stage, "step": step, "injection_id": injection_id,
             "nonfinite_count": int((~np.isfinite(latent)).sum())},
        )


def apply_patch(latent: np.ndarray, patch) -> np.ndarray:
    channels, _, _ = latent.shape
    result = latent.copy()
    values = np.asarray(patch.values, dtype=np.float64)
    result[:, patch.top : patch.top + patch.height,
                 patch.left : patch.left + patch.width] = np.moveaxis(values, 2, 0)
    return result


# --------------------------------------------------------------------------
# Final stage: linear-projection decode + per-pixel clip -> PNG
# --------------------------------------------------------------------------

def decode_latent(latent: np.ndarray, config: Config) -> tuple[np.ndarray, dict]:
    channels = latent.shape[0]
    rng = derive_rng("dbench.decode.proj.v1", config.effective_fingerprint())
    weight = rng.standard_normal((channels, 3))
    bias = rng.uniform((3,), low=-10.0, high=10.0)
    moved = np.moveaxis(latent, 0, -1)  # H x W x C
    projected = moved @ weight + bias
    upscaled = np.repeat(np.repeat(projected, UPSCALE, axis=0), UPSCALE, axis=1)
    pixels = np.clip(np.rint(upscaled), 0.0, 255.0).astype(np.uint8)
    params = {
        "schema": "dbench.stage.decode.v1",
        "weight_fingerprint": fingerprint_array(weight),
        "bias_fingerprint": fingerprint_array(bias),
        "upscale": UPSCALE,
        "pixel_fingerprint": fingerprint_array(pixels),
    }
    return pixels, params


def write_png(pixels: np.ndarray, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    image = Image.fromarray(pixels)
    temp = path + ".tmp.png"
    image.save(temp, format="PNG", optimize=False, compress_level=1)
    os.replace(temp, path)
