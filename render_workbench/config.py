"""Loading and validation of render configuration YAML and prompt JSONL.

Validation never raises for ordinary data problems: every problem becomes a
structured failure record carrying one of the documented business codes, so
batch processing can continue with the remaining valid inputs.
"""

import os

import numpy as np
import yaml

from .errors import (
    E_CONFIG_INVALID,
    E_INIT_IMAGE_INVALID,
    E_INPUT_MISSING,
    E_PATCH_OUT_OF_RANGE,
    E_TOKEN_OUT_OF_VOCAB,
)
from .png_codec import decode_png


def _err(code, message, **extra):
    record = {"code": code, "message": message}
    record.update(extra)
    return record


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (isinstance(value, (int, float))
            and not isinstance(value, bool))


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(raw):
    """Validate a raw configuration mapping.

    Returns ``(config, errors)``. ``config`` is the normalized configuration
    (possibly incomplete); ``errors`` is a list of structured failure records.
    """
    errors = []
    cfg = {}
    if not isinstance(raw, dict):
        return cfg, [_err(E_CONFIG_INVALID, "configuration root must be a mapping")]

    vocab = raw.get("vocab")
    if not isinstance(vocab, list) or not vocab:
        errors.append(_err(E_CONFIG_INVALID, "vocab must be a non-empty list"))
        vocab = vocab if isinstance(vocab, list) else []
    elif any(not isinstance(token, str) for token in vocab):
        errors.append(_err(E_CONFIG_INVALID, "vocab entries must be strings"))
    elif len(set(vocab)) != len(vocab):
        errors.append(_err(E_CONFIG_INVALID, "vocab entries must be unique"))
    cfg["vocab"] = [str(token) for token in vocab if isinstance(token, str)]
    vocab_set = set(cfg["vocab"])

    embedding_dim = raw.get("embedding_dim")
    if not _is_int(embedding_dim) or embedding_dim <= 0:
        errors.append(_err(E_CONFIG_INVALID,
                           "embedding_dim must be a positive integer"))
        embedding_dim = 8
    cfg["embedding_dim"] = int(embedding_dim)

    latent_channels = raw.get("latent_channels")
    if not _is_int(latent_channels) or latent_channels <= 0:
        errors.append(_err(E_CONFIG_INVALID,
                           "latent_channels must be a positive integer"))
        latent_channels = 4
    cfg["latent_channels"] = int(latent_channels)

    seed_min = raw.get("seed_min", 0)
    seed_max = raw.get("seed_max", 2 ** 31 - 1)
    if not _is_int(seed_min) or not _is_int(seed_max) or seed_min > seed_max:
        errors.append(_err(E_CONFIG_INVALID,
                           "seed_min/seed_max must be integers with min <= max"))
        seed_min, seed_max = 0, 2 ** 31 - 1
    cfg["seed_min"], cfg["seed_max"] = seed_min, seed_max

    seed = raw.get("seed")
    if not _is_int(seed) or not (seed_min <= seed <= seed_max):
        errors.append(_err(E_CONFIG_INVALID,
                           "seed must be an integer within [seed_min, seed_max]"))
        seed = seed_min
    cfg["seed"] = int(seed)

    denoise_steps = raw.get("denoise_steps")
    if not _is_int(denoise_steps) or denoise_steps < 1:
        errors.append(_err(E_CONFIG_INVALID,
                           "denoise_steps must be a positive integer"))
        denoise_steps = 1
    cfg["denoise_steps"] = int(denoise_steps)

    guidance_scale = raw.get("guidance_scale")
    if not _is_number(guidance_scale) or guidance_scale < 0:
        errors.append(_err(E_CONFIG_INVALID,
                           "guidance_scale must be a non-negative number"))
        guidance_scale = 1.0
    cfg["guidance_scale"] = float(guidance_scale)

    replace_steps = raw.get("img2img_replace_steps")
    if not _is_int(replace_steps) or not (0 <= replace_steps <= denoise_steps):
        errors.append(_err(E_CONFIG_INVALID,
                           "img2img_replace_steps must be in [0, denoise_steps]"))
        replace_steps = 0
    cfg["img2img_replace_steps"] = int(replace_steps)

    patch_tolerance = raw.get("patch_tolerance")
    if not _is_number(patch_tolerance) or patch_tolerance < 0:
        errors.append(_err(E_CONFIG_INVALID,
                           "patch_tolerance must be a non-negative number"))
        patch_tolerance = 0.0
    cfg["patch_tolerance"] = float(patch_tolerance)

    concurrency = raw.get("concurrency")
    if not _is_int(concurrency) or concurrency < 1:
        errors.append(_err(E_CONFIG_INVALID,
                           "concurrency must be a positive integer"))
        concurrency = 1
    cfg["concurrency"] = int(concurrency)

    snapshot_interval = raw.get("snapshot_interval")
    if not _is_int(snapshot_interval) or snapshot_interval < 1:
        errors.append(_err(E_CONFIG_INVALID,
                           "snapshot_interval must be a positive integer"))
        snapshot_interval = 1
    cfg["snapshot_interval"] = int(snapshot_interval)

    tiers, tier_errors = _validate_tiers(raw, denoise_steps)
    errors.extend(tier_errors)
    cfg["tiers"] = tiers
    return cfg, errors


def _validate_tiers(raw, denoise_steps):
    errors = []
    raw_tiers = raw.get("tiers")
    noise_tables = raw.get("noise_tables")
    guidance_tables = raw.get("guidance_tables")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        errors.append(_err(E_CONFIG_INVALID, "tiers must be a non-empty list"))
        return [], errors
    if not isinstance(noise_tables, dict):
        errors.append(_err(E_CONFIG_INVALID, "noise_tables must be a mapping"))
        noise_tables = {}
    if not isinstance(guidance_tables, dict):
        errors.append(_err(E_CONFIG_INVALID, "guidance_tables must be a mapping"))
        guidance_tables = {}

    tiers = []
    prev_w = prev_h = None
    seen_ids = set()
    for index, tier in enumerate(raw_tiers):
        if not isinstance(tier, dict):
            errors.append(_err(E_CONFIG_INVALID,
                               "tier %d must be a mapping" % index))
            continue
        tier_id = tier.get("id")
        width = tier.get("width")
        height = tier.get("height")
        if not isinstance(tier_id, str) or not tier_id:
            errors.append(_err(E_CONFIG_INVALID,
                               "tier %d has invalid id" % index))
            continue
        if tier_id in seen_ids:
            errors.append(_err(E_CONFIG_INVALID,
                               "duplicate tier id %s" % tier_id))
            continue
        if not _is_int(width) or not _is_int(height) or width <= 0 or height <= 0:
            errors.append(_err(E_CONFIG_INVALID,
                               "tier %s width/height must be positive integers"
                               % tier_id))
            continue
        if prev_w is not None and not (width > prev_w and height > prev_h):
            errors.append(_err(E_CONFIG_INVALID,
                               "tier %s must be strictly larger than the "
                               "previous tier (monotonic increase)" % tier_id))
            continue
        seen_ids.add(tier_id)
        noise = noise_tables.get(tier_id)
        guidance = guidance_tables.get(tier_id)
        table_ok = True
        for name, table in (("noise", noise), ("guidance", guidance)):
            if not isinstance(table, list) or len(table) != denoise_steps:
                errors.append(_err(E_CONFIG_INVALID,
                                   "%s table for tier %s must have length "
                                   "denoise_steps=%d"
                                   % (name, tier_id, denoise_steps)))
                table_ok = False
            elif any(not _is_number(value) for value in table):
                errors.append(_err(E_CONFIG_INVALID,
                                   "%s table for tier %s contains non-numeric "
                                   "entries" % (name, tier_id)))
                table_ok = False
        if not table_ok:
            continue
        prev_w, prev_h = width, height
        noise_values = [float(v) for v in noise]
        guidance_values = [float(v) for v in guidance]
        tiers.append({
            "id": tier_id,
            "width": int(width),
            "height": int(height),
            "noise": noise_values,
            "guidance": guidance_values,
            "full_noise": list(noise_values),
            "full_guidance": list(guidance_values),
        })
    return tiers, errors


def load_prompts(path, cfg, base_dir=None):
    """Load and validate prompt records from a JSONL file."""
    errors = []
    prompts = []
    seen_ids = set()
    vocab_set = set(cfg["vocab"])
    base_dir = base_dir or os.path.dirname(os.path.abspath(path))
    if not os.path.exists(path):
        return prompts, [_err(E_INPUT_MISSING, "prompt file not found",
                              path=path)]
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                import json
                record = json.loads(line)
            except ValueError as exc:
                errors.append(_err(E_CONFIG_INVALID,
                                   "invalid JSON on line %d: %s"
                                   % (line_no, exc)))
                continue
            if not isinstance(record, dict):
                errors.append(_err(E_CONFIG_INVALID,
                                   "line %d must be a JSON object" % line_no))
                continue
            prompt_id = record.get("id")
            if not isinstance(prompt_id, str) or not prompt_id:
                errors.append(_err(E_CONFIG_INVALID,
                                   "line %d missing string field id" % line_no))
                continue
            if prompt_id in seen_ids:
                errors.append(_err(E_CONFIG_INVALID,
                                   "duplicate prompt id %s" % prompt_id))
                continue
            tokens = record.get("tokens")
            if not isinstance(tokens, list) or not tokens:
                errors.append(_err(E_CONFIG_INVALID,
                                   "prompt %s tokens must be a non-empty list"
                                   % prompt_id))
                continue
            bad_token = next((t for t in tokens
                              if not isinstance(t, str) or t not in vocab_set),
                             None)
            if bad_token is not None:
                errors.append(_err(E_TOKEN_OUT_OF_VOCAB,
                                   "prompt %s contains token outside vocab: %r"
                                   % (prompt_id, bad_token),
                                   prompt_id=prompt_id))
                continue
            patch_errors = _validate_patches(record.get("patches"), cfg,
                                             prompt_id)
            if patch_errors:
                errors.extend(patch_errors)
                continue
            init_image = record.get("init_image")
            if init_image is not None:
                if not isinstance(init_image, str):
                    errors.append(_err(E_CONFIG_INVALID,
                                       "prompt %s init_image must be a string"
                                       % prompt_id))
                    continue
                image_path = init_image
                if not os.path.isabs(image_path):
                    image_path = os.path.join(base_dir, image_path)
                if not os.path.exists(image_path):
                    errors.append(_err(E_INPUT_MISSING,
                                       "prompt %s init image not found: %s"
                                       % (prompt_id, init_image),
                                       prompt_id=prompt_id))
                    continue
                try:
                    with open(image_path, "rb") as image_handle:
                        pixels = decode_png(image_handle.read())
                except Exception as exc:  # noqa: BLE001 - reported as business error
                    errors.append(_err(E_INIT_IMAGE_INVALID,
                                       "prompt %s init image unreadable: %s"
                                       % (prompt_id, exc),
                                       prompt_id=prompt_id))
                    continue
            else:
                image_path = None
                pixels = None
            seen_ids.add(prompt_id)
            prompts.append({
                "id": prompt_id,
                "tokens": list(tokens),
                "init_image": init_image,
                "init_image_path": image_path,
                "init_pixels": pixels,
                "patches": [_normalize_patch(p) for p in (record.get("patches") or [])],
            })
    return prompts, errors


def _validate_patches(raw_patches, cfg, prompt_id):
    errors = []
    if raw_patches is None:
        return errors
    if not isinstance(raw_patches, list):
        return [_err(E_CONFIG_INVALID,
                     "prompt %s patches must be a list" % prompt_id)]
    channels = cfg["latent_channels"]
    for index, patch in enumerate(raw_patches):
        if not isinstance(patch, dict):
            errors.append(_err(E_CONFIG_INVALID,
                               "prompt %s patch %d must be an object"
                               % (prompt_id, index)))
            continue
        required = ("step", "x", "y", "width", "height", "values")
        missing = [key for key in required if key not in patch]
        if missing:
            errors.append(_err(E_CONFIG_INVALID,
                               "prompt %s patch %d missing fields %s"
                               % (prompt_id, index, missing)))
            continue
        step, x, y, width, height = (patch["step"], patch["x"], patch["y"],
                                     patch["width"], patch["height"])
        values = patch["values"]
        if not all(_is_int(v) for v in (step, x, y, width, height)):
            errors.append(_err(E_CONFIG_INVALID,
                               "prompt %s patch %d geometry must be integers"
                               % (prompt_id, index)))
            continue
        if step < 0 or width <= 0 or height <= 0:
            errors.append(_err(E_CONFIG_INVALID,
                               "prompt %s patch %d has invalid geometry"
                               % (prompt_id, index)))
            continue
        expected_len = width * height * channels
        if not isinstance(values, list) or len(values) != expected_len:
            errors.append(_err(E_CONFIG_INVALID,
                               "prompt %s patch %d values must be a list of "
                               "length %d" % (prompt_id, index, expected_len)))
            continue
        if any(not _is_number(v) for v in values):
            errors.append(_err(E_CONFIG_INVALID,
                               "prompt %s patch %d values must be numeric"
                               % (prompt_id, index)))
            continue
        for tier in cfg["tiers"]:
            if not (0 <= x and 0 <= y
                    and x + width <= tier["width"]
                    and y + height <= tier["height"]):
                errors.append(_err(E_PATCH_OUT_OF_RANGE,
                                   "prompt %s patch %d exceeds tier %s image "
                                   "bounds %dx%d"
                                   % (prompt_id, index, tier["id"],
                                      tier["width"], tier["height"]),
                                   prompt_id=prompt_id, tier=tier["id"]))
                break
    return errors


def _normalize_patch(patch):
    return {
        "step": int(patch["step"]),
        "x": int(patch["x"]),
        "y": int(patch["y"]),
        "width": int(patch["width"]),
        "height": int(patch["height"]),
        "values": [float(v) for v in patch["values"]],
    }


def config_fingerprint_payload(cfg):
    """JSON-compatible payload identifying the full configuration."""
    return {
        "vocab": cfg["vocab"],
        "embedding_dim": cfg["embedding_dim"],
        "latent_channels": cfg["latent_channels"],
        "seed_min": cfg["seed_min"],
        "seed_max": cfg["seed_max"],
        "seed": cfg["seed"],
        "denoise_steps": cfg["denoise_steps"],
        "guidance_scale": cfg["guidance_scale"],
        "img2img_replace_steps": cfg["img2img_replace_steps"],
        "patch_tolerance": cfg["patch_tolerance"],
        "snapshot_interval": cfg["snapshot_interval"],
        "tiers": [
            {
                "id": tier["id"],
                "width": tier["width"],
                "height": tier["height"],
                "noise": tier["noise"],
                "guidance": tier["guidance"],
            }
            for tier in cfg["tiers"]
        ],
    }


def global_semantic_payload(cfg):
    """Tier-independent semantic inputs shared by every pipeline node."""
    return {
        "vocab": cfg["vocab"],
        "embedding_dim": cfg["embedding_dim"],
        "latent_channels": cfg["latent_channels"],
        "seed_min": cfg["seed_min"],
        "seed_max": cfg["seed_max"],
        "seed": cfg["seed"],
        "denoise_steps": cfg["denoise_steps"],
        "guidance_scale": cfg["guidance_scale"],
        "img2img_replace_steps": cfg["img2img_replace_steps"],
        "patch_tolerance": cfg["patch_tolerance"],
    }


def tier_payload(tier):
    """All tier-specific parameters that affect rendered content."""
    return {
        "id": tier["id"],
        "width": tier["width"],
        "height": tier["height"],
        "noise": tier["noise"],
        "guidance": tier["guidance"],
    }
