"""Deterministic fixed-point visual and text feature extraction."""

from __future__ import annotations

import io
import math
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from .config import AlignmentConfig
from .errors import E_CHUNK_MISMATCH, E_FEATURE_NONFINITE, WorkbenchError
from .hashing import canonical_json, hash_json, sha256_text
from .samples import LoadedSample, TextDescription


BASE_DIM = 11
BASE_STATS_SCHEMA = [
    "color_mean_r",
    "color_mean_g",
    "color_mean_b",
    "chroma_r_x3",
    "chroma_g_x3",
    "chroma_b_x3",
    "gradient_strength_x5",
    "edge_density",
    "chunk_center_x",
    "chunk_center_y",
    "activity",
]
COLOR_PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.824, 0.227, 0.204),
    "green": (0.188, 0.616, 0.357),
    "blue": (0.212, 0.404, 0.788),
    "yellow": (0.882, 0.710, 0.184),
    "purple": (0.529, 0.286, 0.682),
}
SHAPE_STATS: dict[str, tuple[float, float, float]] = {
    "circle": (0.13, 0.22, 0.55),
    "rectangle": (0.25, 0.38, 0.72),
    "triangle": (0.21, 0.32, 0.62),
}


def _semantic_projection_fingerprint(config: AlignmentConfig) -> str:
    return hash_json(
        {
            "salt": config.projection_salt,
            "base_dim": BASE_DIM,
            "visual_dim": config.visual_dim,
            "text_dim": config.text_dim,
        }
    )


@lru_cache(maxsize=64)
def _projection_matrix_cached(fingerprint: str, base_dim: int, out_dim: int) -> np.ndarray:
    seed = int(fingerprint[:16], 16)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.13, size=(base_dim, out_dim)).astype(np.float64)
    diagonal = min(base_dim, out_dim)
    for index in range(diagonal):
        matrix[index, index] += 0.88
    return matrix


def projection_matrix(config: AlignmentConfig, out_dim: int) -> tuple[np.ndarray, str]:
    fingerprint = _semantic_projection_fingerprint(config)
    common_dim = max(config.visual_dim, config.text_dim)
    matrix = _projection_matrix_cached(fingerprint, BASE_DIM, common_dim)
    return matrix[:, :out_dim], fingerprint


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _quantize(vector: np.ndarray, bits: int) -> dict[str, Any]:
    scale = 1 << bits
    quantized = np.rint(vector * scale).astype(np.int64)
    reconstructed = quantized.astype(np.float64) / scale
    residual = reconstructed - vector
    return {
        "fixed_decimal_bits": bits,
        "fixed_scale": scale,
        "quantized": [int(value) for value in quantized.tolist()],
        "rounding_residual": [float(value) for value in residual.tolist()],
        "rounding_residual_l2": float(np.linalg.norm(residual)),
        "rounding_residual_max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
    }


def _gradient_and_edges(gray: np.ndarray, edge_threshold: float) -> tuple[float, float]:
    if gray.shape[0] > 2:
        dx = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
    else:
        dx = np.zeros((gray.shape[0], 0), dtype=np.float64)
    if gray.shape[1] > 2:
        dy = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
    else:
        dy = np.zeros((0, gray.shape[1]), dtype=np.float64)
    values: list[np.ndarray] = []
    if dx.size:
        values.append(dx.ravel())
    if dy.size:
        values.append(dy.ravel())
    if not values:
        return 0.0, 0.0
    magnitude = np.concatenate(values)
    gradient_strength = float(np.mean(magnitude))
    edge_density = float(np.mean(magnitude > edge_threshold))
    return gradient_strength, edge_density


def _block_base_stats(block: np.ndarray, config: AlignmentConfig) -> np.ndarray:
    rgb = block.astype(np.float64) / 255.0
    mean = np.mean(rgb, axis=(0, 1))
    chroma = mean - float(np.mean(mean))
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gradient, edge_density = _gradient_and_edges(gray, config.edge_threshold)
    activity = float(np.mean(np.linalg.norm(rgb - np.array([0.965, 0.957, 0.933]), axis=2) > 0.07))
    return np.array(
        [
            mean[0],
            mean[1],
            mean[2],
            chroma[0] * 3.0,
            chroma[1] * 3.0,
            chroma[2] * 3.0,
            gradient * 5.0,
            edge_density,
            0.0,
            0.0,
            activity,
        ],
        dtype=np.float64,
    )


def extract_visual_features(
    sample: LoadedSample,
    config: AlignmentConfig,
    *,
    chunk_size: int | None = None,
    inject_nonfinite: bool = False,
) -> dict[str, Any]:
    context = prepare_visual_context(sample, config, chunk_size=chunk_size)
    blocks: list[dict[str, Any]] = []
    for row in range(context["rows"]):
        for col in range(context["cols"]):
            blocks.append(
                extract_visual_block(
                    context,
                    row,
                    col,
                    config,
                    inject_nonfinite=inject_nonfinite,
                )
            )
    return assemble_visual_artifact(context, blocks, config)


def prepare_visual_context(
    sample: LoadedSample,
    config: AlignmentConfig,
    *,
    chunk_size: int | None = None,
) -> dict[str, Any]:
    chunk_size = chunk_size or config.chunk_size
    width, height = sample.image_size
    if chunk_size <= 0 or width % chunk_size != 0 or height % chunk_size != 0:
        raise WorkbenchError(
            E_CHUNK_MISMATCH,
            "chunk_size must evenly divide both image dimensions",
            {
                "sample": sample.name,
                "image_size": [width, height],
                "chunk_size": chunk_size,
            },
        )
    try:
        with Image.open(io.BytesIO(sample.image_bytes)) as image:
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # Pillow versions expose different decode errors.
        raise WorkbenchError(E_PNG_HEADER, f"image decode failed: {exc}") from exc
    projection, projection_fingerprint = projection_matrix(config, config.visual_dim)
    rows = height // chunk_size
    cols = width // chunk_size
    return {
        "sample": sample,
        "chunk_size": chunk_size,
        "width": width,
        "height": height,
        "rows": rows,
        "cols": cols,
        "rgb": rgb,
        "projection": projection,
        "projection_fingerprint": projection_fingerprint,
    }


def extract_visual_block(
    context: dict[str, Any],
    row: int,
    col: int,
    config: AlignmentConfig,
    *,
    inject_nonfinite: bool = False,
) -> dict[str, Any]:
    sample: LoadedSample = context["sample"]
    chunk_size = int(context["chunk_size"])
    width = int(context["width"])
    height = int(context["height"])
    rgb: np.ndarray = context["rgb"]
    projection: np.ndarray = context["projection"]
    y = row * chunk_size
    x = col * chunk_size
    block = rgb[y : y + chunk_size, x : x + chunk_size]
    base = _block_base_stats(block, config)
    base[8] = (x + chunk_size / 2.0) / width
    base[9] = (y + chunk_size / 2.0) / height
    projected = base @ projection
    if inject_nonfinite and row == 0 and col == 0:
        projected[0] = np.nan
    if not np.all(np.isfinite(projected)):
        raise WorkbenchError(
            E_FEATURE_NONFINITE,
            "visual projection contains a non-finite value",
            {
                "sample": sample.name,
                "row": row,
                "col": col,
                "stage": "visual_projection",
            },
        )
    normalized = _normalize(projected)
    quantized = _quantize(normalized, config.effective_decimal_bits)
    return {
        "block_id": f"{sample.name}:r{row:04d}:c{col:04d}",
        "row": row,
        "col": col,
        "pixel_rect": [x, y, chunk_size, chunk_size],
        "base_stats": [float(value) for value in base.tolist()],
        "color_mean": [float(base[0]), float(base[1]), float(base[2])],
        "chroma_x3": [float(base[3]), float(base[4]), float(base[5])],
        "gradient_strength": float(base[6] / 5.0),
        "edge_density": float(base[7]),
        "chunk_center": [float(base[8]), float(base[9])],
        "activity": float(base[10]),
        **quantized,
    }


def assemble_visual_artifact(
    context: dict[str, Any],
    blocks: list[dict[str, Any]],
    config: AlignmentConfig,
) -> dict[str, Any]:
    sample: LoadedSample = context["sample"]
    blocks = sorted(blocks, key=lambda entry: (entry["row"], entry["col"]))
    return {
        "stage": "visual_features",
        "sample": sample.name,
        "image_fingerprint": sample.image_fingerprint,
        "regions_fingerprint": sample.regions_fingerprint,
        "chunk_size": context["chunk_size"],
        "grid": [context["rows"], context["cols"]],
        "image_size": [context["width"], context["height"]],
        "base_stats_schema": BASE_STATS_SCHEMA,
        "visual_dim": config.visual_dim,
        "projection_fingerprint": context["projection_fingerprint"],
        "fixed_decimal_bits": config.effective_decimal_bits,
        "configured_fixed_decimal_bits": config.fixed_decimal_bits,
        "quantization_level": config.quantization_level,
        "blocks": blocks,
    }


def _shape_stats(entity_type: str) -> tuple[float, float, float]:
    return SHAPE_STATS.get(entity_type, (0.18, 0.28, 0.58))


def _position_coordinates(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, str):
        return None
    parts = value.lower().replace("_", "-").split("-")
    horizontal = next((part for part in parts if part in {"left", "center", "right"}), None)
    vertical = next((part for part in parts if part in {"top", "middle", "bottom"}), None)
    if horizontal is None and vertical is None:
        return None
    x = {"left": 0.15, "center": 0.5, "right": 0.85}.get(horizontal, 0.5)
    y = {"top": 0.15, "middle": 0.5, "bottom": 0.85}.get(vertical, 0.5)
    return x, y


def _stable_base_vector(seed_material: str) -> np.ndarray:
    seed = int(sha256_text(seed_material)[:16], 16)
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=BASE_DIM).astype(np.float64)


def _assertion_semantic_base(
    entity_type: str,
    attribute: str,
    value: Any,
    *,
    semantic_salt: str,
) -> np.ndarray:
    base = np.zeros(BASE_DIM, dtype=np.float64)
    shape_gradient, shape_edge, shape_activity = _shape_stats(entity_type)
    base[6] += shape_gradient * 0.25
    base[7] += shape_edge * 0.25
    base[10] += shape_activity * 0.25

    if attribute == "color" and isinstance(value, str):
        color = COLOR_PALETTE.get(value)
        if color is not None:
            base[0:3] = np.array(color, dtype=np.float64)
            average = sum(color) / 3.0
            base[3:6] = (np.array(color) - average) * 3.0
            base[10] += 0.30
            return base
    elif attribute == "shape" and isinstance(value, str):
        gradient, edge, activity = _shape_stats(value)
        base[6] = gradient
        base[7] = edge
        base[10] = activity
        return base
    elif attribute in {"position", "location"}:
        coordinates = _position_coordinates(value)
        if coordinates is not None:
            base[8], base[9] = coordinates
            return base

    token = canonical_json({"salt": semantic_salt, "type": entity_type, "attribute": attribute, "value": value})
    semantic = _stable_base_vector(token)
    semantic[0:3] = np.abs(semantic[0:3])
    return base + semantic


def _project_text_semantic(base: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return _normalize(base @ projection)


def _default_text_vector(
    *,
    entity_id: str,
    attribute: str,
    projection_fingerprint: str,
    text_dim: int,
) -> np.ndarray:
    seed = int(
        sha256_text(
            canonical_json(
                {
                    "projection_fingerprint": projection_fingerprint,
                    "padding": True,
                    "entity": entity_id,
                    "attribute": attribute,
                }
            )
        )[:16],
        16,
    )
    rng = np.random.default_rng(seed)
    return _normalize(rng.normal(0.0, 1.0, size=text_dim).astype(np.float64))


def extract_text_features(
    sample: LoadedSample,
    config: AlignmentConfig,
    *,
    inject_nonfinite: bool = False,
) -> dict[str, Any]:
    text: TextDescription = sample.text
    projection, projection_fingerprint = projection_matrix(config, config.text_dim)
    entity_types = {entity.id: entity.type for entity in text.entities}
    entries: list[dict[str, Any]] = []
    entity_bases: dict[str, np.ndarray] = {}
    entity_assertion_ids: dict[str, list[str]] = {}
    ordered_assertions = sorted(text.assertions, key=lambda item: (item.id, item.entity, item.attribute))
    for assertion in ordered_assertions:
        base = _assertion_semantic_base(
            entity_types.get(assertion.entity, ""),
            assertion.attribute,
            assertion.value,
            semantic_salt=config.projection_salt,
        )
        entity_bases.setdefault(assertion.entity, np.zeros(BASE_DIM, dtype=np.float64))
        entity_bases[assertion.entity] += base
        entity_assertion_ids.setdefault(assertion.entity, []).append(assertion.id)
        vector = _project_text_semantic(base, projection)
        if inject_nonfinite and not entries:
            vector[0] = np.nan
        if not np.all(np.isfinite(vector)):
            raise WorkbenchError(
                E_FEATURE_NONFINITE,
                "text assertion projection contains a non-finite value",
                {
                    "sample": sample.name,
                    "assertion_id": assertion.id,
                    "stage": "text_projection",
                },
            )
        quantized = _quantize(vector, config.effective_decimal_bits)
        entries.append(
            {
                "assertion_id": assertion.id,
                "entity": assertion.entity,
                "attribute": assertion.attribute,
                "value": assertion.value,
                "padded": False,
                "weight": 1.0,
                **quantized,
            }
        )

    present = {(assertion.entity, assertion.attribute) for assertion in text.assertions}
    padding_positions: list[int] = []
    entity_padding_vectors: dict[str, list[np.ndarray]] = {}
    for entity in sorted(text.entities, key=lambda item: item.id):
        for attribute in config.required_attributes:
            if (entity.id, attribute) in present:
                continue
            vector = _default_text_vector(
                entity_id=entity.id,
                attribute=attribute,
                projection_fingerprint=projection_fingerprint,
                text_dim=config.text_dim,
            )
            if inject_nonfinite and not entries:
                vector[0] = np.nan
            if not np.all(np.isfinite(vector)):
                raise WorkbenchError(
                    E_FEATURE_NONFINITE,
                    "padded text feature contains a non-finite value",
                    {
                        "sample": sample.name,
                        "entity": entity.id,
                        "attribute": attribute,
                        "stage": "text_padding",
                    },
                )
            quantized = _quantize(vector, config.effective_decimal_bits)
            padding_positions.append(len(entries))
            entity_padding_vectors.setdefault(entity.id, []).append(vector)
            entries.append(
                {
                    "assertion_id": f"padding:{entity.id}:{attribute}",
                    "entity": entity.id,
                    "attribute": attribute,
                    "value": None,
                    "padded": True,
                    "weight": 0.20,
                    **quantized,
                }
            )

    entity_groups: list[dict[str, Any]] = []
    for entity in sorted(text.entities, key=lambda item: item.id):
        base = entity_bases.get(entity.id, np.zeros(BASE_DIM, dtype=np.float64))
        group_vector = base @ projection
        for padding_vector in entity_padding_vectors.get(entity.id, []):
            group_vector = group_vector + padding_vector * 0.20
        group_vector = _normalize(group_vector)
        quantized = _quantize(group_vector, config.effective_decimal_bits)
        present_ids = entity_assertion_ids.get(entity.id, [])
        padded_ids = [
            f"padding:{entity.id}:{attribute}"
            for attribute in config.required_attributes
            if (entity.id, attribute) not in present
        ]
        entity_groups.append(
            {
                "entity_id": entity.id,
                "entity_type": entity.type,
                "assertion_ids": present_ids,
                "padded_assertion_ids": padded_ids,
                "padded": not present_ids,
                "weight": 1.0,
                **quantized,
            }
        )
    if not np.all(
        np.isfinite(
            np.asarray(
                [value for entry in entries for value in entry["rounding_residual"]],
                dtype=np.float64,
            )
        )
    ):
        raise WorkbenchError(
            E_FEATURE_NONFINITE,
            "text feature contains a non-finite rounding residual",
            {"sample": sample.name},
        )
    return {
        "stage": "text_features",
        "sample": sample.name,
        "text_fingerprint": sample.text_fingerprint,
        "text_dim": config.text_dim,
        "projection_fingerprint": projection_fingerprint,
        "fixed_decimal_bits": config.effective_decimal_bits,
        "configured_fixed_decimal_bits": config.fixed_decimal_bits,
        "quantization_level": config.quantization_level,
        "assertion_refs": [entry["assertion_id"] for entry in entries if not entry["padded"]],
        "padding_positions": padding_positions,
        "entries": entries,
        "entity_groups": entity_groups,
    }
