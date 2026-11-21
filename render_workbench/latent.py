"""Latent initialization, deterministic denoise steps and image encoding."""

import numpy as np

from .hashing import array_fingerprint, random_state


def latent_shape(tier, channels):
    return (tier["height"], tier["width"], channels)


def condition_matrix(tier, embedding_dim, channels, config_fp):
    """Tier-specific deterministic projection from embedding to condition."""
    rng = random_state("condition-matrix", tier["id"], tier["width"],
                       tier["height"], embedding_dim, channels, config_fp)
    return rng.standard_normal((embedding_dim, channels))


def condition_field(embedding, matrix, shape):
    """Broadcast embedding projected to a per-pixel condition latent field."""
    projected = embedding @ matrix  # shape (channels,)
    return np.broadcast_to(
        projected.reshape(1, 1, shape[2]), shape
    ).astype(np.float64, copy=True)


def initialize_latent(prompt_fp, seed, config_fp, tier, channels):
    """Draw the initial latent from an isolated, input-derived RandomState."""
    shape = latent_shape(tier, channels)
    rng = random_state("init-latent", prompt_fp, int(seed), config_fp,
                       tier["id"], tier["width"], tier["height"], channels)
    latent = rng.standard_normal(shape).astype(np.float64, copy=False)
    return latent


def noise_field(step_index, tier, channels, config_fp):
    """Deterministic per-step noise pattern shared by both render paths."""
    shape = latent_shape(tier, channels)
    rng = random_state("noise-field", config_fp, tier["id"], step_index,
                       tier["width"], tier["height"], channels)
    return rng.standard_normal(shape).astype(np.float64, copy=False)


def denoise_step(prev_latent, step_index, embedding, tier, embedding_dim,
                 channels, guidance_scale, config_fp):
    """Apply one denoise step with a single fixed evaluation order."""
    matrix = condition_matrix(tier, embedding_dim, channels, config_fp)
    condition = condition_field(embedding, matrix, prev_latent.shape)
    noise = noise_field(step_index, tier, channels, config_fp)
    noise_scale = tier["noise"][step_index]
    guidance_table_scale = tier["guidance"][step_index]

    latent = prev_latent.copy()
    latent = latent + noise_scale * noise
    latent = latent + (guidance_scale * guidance_table_scale) * condition
    # Fixed deterministic relaxation, identical on every run and every path.
    latent = latent / np.float64(1.0 + 0.1 * (step_index + 1))
    return latent.astype(np.float64, copy=False)


def encode_init_image(pixels, tier, channels, prompt_fp, init_image_fp,
                      config_fp):
    """Encode an RGB8 init image into a latent via a deterministic projection.

    The image is resized (nearest neighbour) to the tier resolution, scaled to
    [-1, 1], then linearly projected to ``channels`` latent channels with an
    isolated RNG. This latent replaces the early txt2img denoise steps on the
    img2img path.
    """
    resized = _resize_nearest(pixels, tier["width"], tier["height"])
    normalized = resized.astype(np.float64) / np.float64(255.0)
    normalized = normalized * np.float64(2.0) - np.float64(1.0)
    rng = random_state("image-encoder", config_fp, tier["id"], tier["width"],
                       tier["height"], channels, prompt_fp, init_image_fp)
    weights = rng.standard_normal((3, channels))
    latent = normalized @ weights
    return latent.astype(np.float64, copy=False)


def _resize_nearest(pixels, target_width, target_height):
    source_height, source_width = pixels.shape[:2]
    row_idx = np.floor(
        np.arange(target_height, dtype=np.float64)
        * (source_height / target_height)
    ).astype(np.int64)
    col_idx = np.floor(
        np.arange(target_width, dtype=np.float64)
        * (source_width / target_width)
    ).astype(np.int64)
    row_idx = np.minimum(row_idx, source_height - 1)
    col_idx = np.minimum(col_idx, source_width - 1)
    return pixels[np.ix_(row_idx, col_idx)]


def apply_patch(latent, patch):
    """Return a copy of ``latent`` with the rectangular patch overwritten."""
    patched = latent.copy()
    values = np.asarray(patch["values"], dtype=np.float64)
    block = values.reshape(patch["height"], patch["width"], latent.shape[2])
    y0, y1 = patch["y"], patch["y"] + patch["height"]
    x0, x1 = patch["x"], patch["x"] + patch["width"]
    patched[y0:y1, x0:x1, :] = block
    return patched


def patch_is_noop(latent, patch, tolerance):
    """A patch below tolerance changes nothing and must trigger no recompute."""
    values = np.asarray(patch["values"], dtype=np.float64)
    block = values.reshape(patch["height"], patch["width"], latent.shape[2])
    y0, y1 = patch["y"], patch["y"] + patch["height"]
    x0, x1 = patch["x"], patch["x"] + patch["width"]
    current = latent[y0:y1, x0:x1, :]
    return bool(np.all(np.abs(current - block) <= float(tolerance)))


def fingerprint_latent(latent):
    return array_fingerprint(latent)
