from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .canonical import canonical_json_bytes, sha256_bytes
from .raster import rasterize_fill, rasterize_stroke


def _premultiplied_source(color: dict[str, Any], coverage: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coverage16 = coverage.astype(np.uint16)
    alpha8 = (int(color["a"]) * coverage16 + 127) // 255
    blue8 = (int(color["b"]) * alpha8 + 127) // 255
    green8 = (int(color["g"]) * alpha8 + 127) // 255
    red8 = (int(color["r"]) * alpha8 + 127) // 255
    return (
        blue8.astype(np.uint16),
        green8.astype(np.uint16),
        red8.astype(np.uint16),
        alpha8.astype(np.uint16),
    )


def composite_coverage_over(destination: np.ndarray, color: dict[str, Any], coverage: np.ndarray) -> np.ndarray:
    dst = destination.astype(np.uint16, copy=False)
    source_b, source_g, source_r, source_a = _premultiplied_source(color, coverage)
    inverse = 255 - source_a
    out = np.empty_like(destination)
    out[:, :, 0] = source_b + ((dst[:, :, 0] * inverse + 127) // 255)
    out[:, :, 1] = source_g + ((dst[:, :, 1] * inverse + 127) // 255)
    out[:, :, 2] = source_r + ((dst[:, :, 2] * inverse + 127) // 255)
    out[:, :, 3] = source_a + ((dst[:, :, 3] * inverse + 127) // 255)
    return np.minimum(out, 255).astype(np.uint8)


def composite_layer_over(destination: np.ndarray, layer: np.ndarray) -> np.ndarray:
    dst = destination.astype(np.uint16, copy=False)
    src = layer.astype(np.uint16, copy=False)
    inverse = 255 - src[:, :, 3]
    out = np.empty_like(destination)
    out[:, :, 0] = src[:, :, 0] + ((dst[:, :, 0] * inverse + 127) // 255)
    out[:, :, 1] = src[:, :, 1] + ((dst[:, :, 1] * inverse + 127) // 255)
    out[:, :, 2] = src[:, :, 2] + ((dst[:, :, 2] * inverse + 127) // 255)
    out[:, :, 3] = src[:, :, 3] + ((dst[:, :, 3] * inverse + 127) // 255)
    return np.minimum(out, 255).astype(np.uint8)


def layer_content_hash(header: dict[str, Any], pixels: np.ndarray) -> str:
    return hashlib.sha256(canonical_json_bytes(header) + b"\x00" + pixels.tobytes()).hexdigest()


def render_path_layer(
    layer_id: str,
    path: dict[str, Any],
    flat_path: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    fill = rasterize_fill(flat_path, width, height)
    selected_rule = fill["selected_rule"]
    stroke_width = float(path.get("stroke", {}).get("width", 0.0))
    stroke = rasterize_stroke(flat_path, width, height, stroke_width)
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels = composite_coverage_over(pixels, path["fill_color"], fill[selected_rule])
    if stroke_width > 0.0:
        pixels = composite_coverage_over(pixels, path["stroke"]["color"], stroke["coverage"])
    header = {
        "schema": "vgsynth.layer/v1",
        "layer_id": layer_id,
        "path_id": path.get("path_id"),
        "declared_fill_rule": flat_path.get("fill_rule"),
        "fill_rule_used": selected_rule,
        "fill_hash_nonzero": fill["nonzero_info"]["content_hash"],
        "fill_hash_evenodd": fill["evenodd_info"]["content_hash"],
        "fill_hash_selected": fill[selected_rule + "_info"]["content_hash"],
        "stroke_coverage_hash": stroke["info"]["content_hash"],
        "stroke_outline_hash": stroke["outline_hash"],
        "pixel_format": "BGRA8_PREMULTIPLIED",
        "width": width,
        "height": height,
        "fill_below_stroke": True,
    }
    layer_hash = layer_content_hash(header, pixels)
    return {
        "layer_id": layer_id,
        "path_id": path.get("path_id"),
        "pixels": pixels,
        "header": header,
        "layer_hash": layer_hash,
        "pixel_hash": sha256_bytes(pixels.tobytes()),
        "fill": fill,
        "stroke": stroke,
    }


def composite_scene(normalized_scene: dict[str, Any], flat_scene: dict[str, Any]) -> dict[str, Any]:
    width = int(normalized_scene["canvas"]["width"])
    height = int(normalized_scene["canvas"]["height"])
    paths = normalized_scene.get("paths", [])
    flat_paths = flat_scene.get("paths", [])
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    layers: list[dict[str, Any]] = []
    chain = hashlib.sha256(b"vgsynth.composite/v1")
    for index, (path, flat_path) in enumerate(zip(paths, flat_paths)):
        layer = render_path_layer(f"L{index:04d}", path, flat_path, width, height)
        canvas = composite_layer_over(canvas, layer["pixels"])
        chain.update(bytes.fromhex(layer["layer_hash"]))
        layers.append(layer)
    return {
        "width": width,
        "height": height,
        "layers": layers,
        "canvas": canvas,
        "canvas_hash": sha256_bytes(canvas.tobytes()),
        "composite_chain_hash": chain.hexdigest(),
    }
