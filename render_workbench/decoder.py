"""Per-frame linear projection decoder producing deterministic PNG output."""

import numpy as np

from .hashing import array_fingerprint, bytes_fingerprint, random_state
from .png_codec import encode_png


def decoder_matrix(tier, channels, config_fp):
    """Tier-specific deterministic latent -> RGB projection."""
    rng = random_state("decoder-matrix", config_fp, tier["id"], tier["width"],
                       tier["height"], channels)
    return rng.standard_normal((channels, 3))


def decoder_bias(tier, config_fp):
    rng = random_state("decoder-bias", config_fp, tier["id"], tier["width"],
                       tier["height"])
    return rng.standard_normal(3)


def decode_frame(latent, step_index, tier, channels, config_fp):
    """Project a latent to RGB pixels and deterministic PNG bytes.

    Different tiers produce different projection dimensions and different
    pixels: the latent spatial size equals the tier resolution and the
    projection matrices are tier-specific.
    """
    matrix = decoder_matrix(tier, channels, config_fp)
    bias = decoder_bias(tier, config_fp)
    projected = latent @ matrix + bias
    scaled = np.float64(128.0) * projected + np.float64(127.5)
    clipped = np.clip(scaled, np.float64(0.0), np.float64(255.0))
    # Banker's rounding (np.round) is deterministic across platforms.
    pixels = np.round(clipped).astype(np.uint8)
    png_bytes = encode_png(pixels)
    return {
        "pixels": pixels,
        "png_bytes": png_bytes,
        "pixel_fp": array_fingerprint(pixels),
        "png_fp": bytes_fingerprint(png_bytes),
        "params": {
            "step_index": int(step_index),
            "tier_id": tier["id"],
            "width": tier["width"],
            "height": tier["height"],
            "channels": int(channels),
            "rounding": "bankers",
            "clip_min": 0,
            "clip_max": 255,
        },
    }
