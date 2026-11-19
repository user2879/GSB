"""Software BGRA rasteriser with shared outline geometry for all styles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import E_CLOSURE_MISMATCH, E_STROKE_SAMPLE, NPRError
from .geometry import FlatElement, GeometryStore, Point
from .model import Element, Layer, Scene
from .snapshot import ElementRaster, RenderSnapshot, STYLE_MODES, build_snapshot
from .strategy import StrategyDecision
from .util import atomic_write_bytes, sha256_bytes, sha256_json


@dataclass(frozen=True, slots=True)
class RenderOutcome:
    snapshot: RenderSnapshot
    actual_elements: frozenset[str]
    actual_layers: frozenset[str]


def _scale_for(scene: Scene, viewport: tuple[int, int]) -> tuple[float, float, float]:
    scale = min(viewport[0] / scene.canvas_width, viewport[1] / scene.canvas_height)
    offset_x = (viewport[0] - scene.canvas_width * scale) * 0.5
    offset_y = (viewport[1] - scene.canvas_height * scale) * 0.5
    return scale, offset_x, offset_y


def _transform(point: Point, scale: float, offset_x: float, offset_y: float) -> Point:
    return Point(point.x * scale + offset_x, point.y * scale + offset_y)


def _hash_unit(element_id: str, salt: str) -> int:
    digest = sha256_bytes(f"{element_id}:{salt}".encode("utf-8"))
    return int(digest[:8], 16)


def _blend_pixel(buffer: bytearray, offset: int, b: int, g: int, r: int, a: int) -> None:
    if a <= 0:
        return
    if a >= 255:
        buffer[offset] = b
        buffer[offset + 1] = g
        buffer[offset + 2] = r
        buffer[offset + 3] = 255
        return
    dst_a = buffer[offset + 3]
    if dst_a == 0:
        buffer[offset] = b
        buffer[offset + 1] = g
        buffer[offset + 2] = r
        buffer[offset + 3] = a
        return
    out_a = a + dst_a * (255 - a) // 255
    if out_a <= 0:
        return
    src_weight = a * 255
    dst_weight = dst_a * (255 - a)
    buffer[offset] = (b * src_weight + buffer[offset] * dst_weight) // (out_a * 255)
    buffer[offset + 1] = (g * src_weight + buffer[offset + 1] * dst_weight) // (out_a * 255)
    buffer[offset + 2] = (r * src_weight + buffer[offset + 2] * dst_weight) // (out_a * 255)
    buffer[offset + 3] = out_a


def _stamp(buffer: bytearray, width: int, height: int, x: int, y: int, radius: int, color: tuple[int, int, int, int]) -> None:
    b, g, r, a = color
    x0 = max(0, x - radius)
    x1 = min(width - 1, x + radius)
    y0 = max(0, y - radius)
    y1 = min(height - 1, y + radius)
    for py in range(y0, y1 + 1):
        row = py * width * 4
        for px in range(x0, x1 + 1):
            _blend_pixel(buffer, row + px * 4, b, g, r, a)


def _draw_line(
    buffer: bytearray,
    width: int,
    height: int,
    start: Point,
    end: Point,
    radius: int,
    color: tuple[int, int, int, int],
    dashed: tuple[int, int, int] | None = None,
) -> None:
    dx = end.x - start.x
    dy = end.y - start.y
    steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
    if dashed is None:
        for step in range(steps + 1):
            t = step / steps
            _stamp(
                buffer,
                width,
                height,
                int(round(start.x + dx * t)),
                int(round(start.y + dy * t)),
                radius,
                color,
            )
        return
    on, off, offset = dashed
    period = max(1, on + off)
    for step in range(steps + 1):
        if (step + offset) % period >= on:
            continue
        t = step / steps
        _stamp(
            buffer,
            width,
            height,
            int(round(start.x + dx * t)),
            int(round(start.y + dy * t)),
            radius,
            color,
        )


def _fill_polygon(buffer: bytearray, width: int, height: int, points: list[Point], color: tuple[int, int, int, int]) -> None:
    if len(points) < 3:
        return
    b, g, r, a = color
    min_y = max(0, int(math.floor(min(point.y for point in points))))
    max_y = min(height - 1, int(math.ceil(max(point.y for point in points))))
    edge_count = len(points)
    for y in range(min_y, max_y + 1):
        scan_y = y + 0.5
        intersections: list[float] = []
        for index in range(edge_count):
            p0 = points[index]
            p1 = points[(index + 1) % edge_count]
            if (p0.y <= scan_y < p1.y) or (p1.y <= scan_y < p0.y):
                t = (scan_y - p0.y) / (p1.y - p0.y)
                intersections.append(p0.x + t * (p1.x - p0.x))
        intersections.sort()
        for index in range(0, len(intersections) - 1, 2):
            x0 = max(0, int(math.ceil(intersections[index])))
            x1 = min(width - 1, int(math.floor(intersections[index + 1])))
            row = y * width * 4
            for x in range(x0, x1 + 1):
                _blend_pixel(buffer, row + x * 4, b, g, r, a)


def _style_colors(element: Element, mode: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None]:
    tint_r, tint_g, tint_b = element.style.tint
    if mode == "outline":
        return (max(0, tint_b // 3), max(0, tint_g // 3), max(0, tint_r // 3), 255), None
    if mode == "sketch":
        tone = max(0.0, min(1.0, element.style.tone))
        base = int(35 + tone * 95)
        alpha = int(150 + element.style.roughness * 95)
        return (base, base, min(255, base + 5), max(80, min(255, alpha))), None
    if mode == "cartoon":
        stroke = (18, 21, 25, 255)
        fill = (
            min(255, int(tint_b * 0.30 + 255 * 0.70)),
            min(255, int(tint_g * 0.30 + 255 * 0.70)),
            min(255, int(tint_r * 0.30 + 255 * 0.70)),
            255,
        )
        return stroke, fill
    raise NPRError(E_STROKE_SAMPLE, "unsupported style mode", mode=mode)


def rasterize_element(
    element: Element,
    flat: FlatElement,
    mode: str,
    viewport: tuple[int, int],
    scene: Scene,
) -> ElementRaster:
    if mode not in STYLE_MODES:
        raise NPRError(E_STROKE_SAMPLE, "unsupported style mode", mode=mode)
    scale, offset_x, offset_y = _scale_for(scene, viewport)
    transformed = [_transform(point, scale, offset_x, offset_y) for point in flat.points]
    stroke_px = max(0.7, element.stroke_width * element.sampling.width_scale * scale)
    padding = int(math.ceil(stroke_px * 0.5)) + 3
    min_x = min(point.x for point in transformed) - padding
    max_x = max(point.x for point in transformed) + padding
    min_y = min(point.y for point in transformed) - padding
    max_y = max(point.y for point in transformed) + padding
    x0 = max(0, int(math.floor(min_x)))
    y0 = max(0, int(math.floor(min_y)))
    x1 = min(viewport[0], int(math.ceil(max_x)))
    y1 = min(viewport[1], int(math.ceil(max_y)))
    local_width = max(0, x1 - x0)
    local_height = max(0, y1 - y0)
    pixels = bytearray(local_width * local_height * 4)
    local_points = [Point(point.x - x0, point.y - y0) for point in transformed]
    stroke_color, fill_color = _style_colors(element, mode)

    if mode == "cartoon" and fill_color is not None and element.closed and len(local_points) >= 3:
        _fill_polygon(pixels, local_width, local_height, local_points, fill_color)

    radius = max(0, int(round(stroke_px * (0.9 if mode == "cartoon" else 0.55))) - 1)
    dashed = None
    if mode == "sketch":
        on = 2 + _hash_unit(element.id, "dash-on") % 5
        off = 2 + _hash_unit(element.id, "dash-off") % 5
        dashed = (on, off, _hash_unit(element.id, "dash-offset") % (on + off))

    indices = flat.indices
    for edge_index in range(0, len(indices), 2):
        start_index = indices[edge_index]
        end_index = indices[edge_index + 1]
        if start_index >= len(local_points) or end_index >= len(local_points):
            continue
        _draw_line(
            pixels,
            local_width,
            local_height,
            local_points[start_index],
            local_points[end_index],
            radius,
            stroke_color,
            dashed,
        )

    payload = {
        "element_id": element.id,
        "element_hash": element.content_hash,
        "geometry_hash": flat.geometry_hash,
        "mode": mode,
        "viewport": list(viewport),
        "x": x0,
        "y": y0,
        "width": local_width,
        "height": local_height,
        "pixels": sha256_bytes(bytes(pixels)),
    }
    render_key = sha256_json(
        {
            "element_hash": element.content_hash,
            "geometry_hash": flat.geometry_hash,
            "mode": mode,
            "viewport": list(viewport),
        }
    )
    render_hash = sha256_json(payload)
    return ElementRaster(
        element_id=element.id,
        element_hash=element.content_hash,
        geometry_hash=flat.geometry_hash,
        render_key=render_key,
        render_hash=render_hash,
        mode=mode,
        x=x0,
        y=y0,
        width=local_width,
        height=local_height,
        pixels=bytes(pixels),
    )



def _blank_buffer(viewport: tuple[int, int]) -> bytearray:
    return bytearray(viewport[0] * viewport[1] * 4)


def _alpha_over_sprite(destination: bytearray, viewport_width: int, viewport_height: int, raster: ElementRaster) -> None:
    if raster.width <= 0 or raster.height <= 0:
        return
    source = raster.pixels
    for y in range(raster.height):
        dy = raster.y + y
        if dy < 0 or dy >= viewport_height:
            continue
        src_row = y * raster.width * 4
        dst_row = dy * viewport_width * 4
        for x in range(raster.width):
            dx = raster.x + x
            if dx < 0 or dx >= viewport_width:
                continue
            source_offset = src_row + x * 4
            alpha = source[source_offset + 3]
            if alpha:
                _blend_pixel(
                    destination,
                    dst_row + dx * 4,
                    source[source_offset],
                    source[source_offset + 1],
                    source[source_offset + 2],
                    alpha,
                )


def _layer_hash(layer: Layer, buffer: bytes) -> str:
    return sha256_json(
        {
            "layer_id": layer.id,
            "order": layer.order,
            "pixel_hash": sha256_bytes(buffer),
        }
    )


def _compose_layer(scene: Scene, layer: Layer, rasters: Mapping[str, ElementRaster], viewport: tuple[int, int]) -> bytes:
    buffer = _blank_buffer(viewport)
    if not layer.visible:
        return bytes(buffer)
    elements = sorted((element for element in scene.elements if element.layer_id == layer.id), key=lambda item: item.id)
    for element in elements:
        _alpha_over_sprite(buffer, viewport[0], viewport[1], rasters[element.id])
    return bytes(buffer)


def _compose_image(scene: Scene, layers: Mapping[str, bytes], viewport: tuple[int, int]) -> bytes:
    width, height = viewport
    image = bytearray([255, 255, 255, 255]) * (width * height)
    for layer in sorted(scene.layers, key=lambda item: (item.order, item.id)):
        if not layer.visible:
            continue
        buffer = layers[layer.id]
        for offset in range(0, len(image), 4):
            alpha = buffer[offset + 3]
            if alpha:
                _blend_pixel(image, offset, buffer[offset], buffer[offset + 1], buffer[offset + 2], alpha)
    return bytes(image)


def _scheduled_elements(scene: Scene, flats: Mapping[str, FlatElement], strategy: StrategyDecision | None) -> list[Element]:
    if strategy is None or strategy.chosen == "per_element":
        return list(scene.elements)
    if strategy.chosen == "layer_merge":
        return sorted(scene.elements, key=lambda element: (element.layer_order, element.layer_id, element.id))
    if strategy.chosen == "tile_batch":
        scale, _offset_x, _offset_y = _scale_for(scene, strategy.stats.viewport)
        tile_size = 64.0
        def tile_key(element: Element) -> tuple[int, int, str]:
            flat = flats[element.id]
            min_x = int((flat.bbox.min_x * scale) // tile_size)
            min_y = int((flat.bbox.min_y * scale) // tile_size)
            return (min_y, min_x, element.id)
        return sorted(scene.elements, key=tile_key)
    return list(scene.elements)


def render_snapshot(
    scene: Scene,
    geometry_store: GeometryStore,
    mode: str,
    viewport: tuple[int, int],
    version: int,
    *,
    prior: RenderSnapshot | None = None,
    dirty_elements: set[str] | None = None,
    dirty_layers: set[str] | None = None,
    strategy: StrategyDecision | None = None,
    gl_backend: object | None = None,
) -> RenderOutcome:
    if mode not in STYLE_MODES:
        raise NPRError(E_STROKE_SAMPLE, "unsupported style mode", mode=mode)
    all_ids = {element.id for element in scene.elements}
    if prior is None:
        dirty = set(all_ids)
        layer_dirty = {layer.id for layer in scene.layers}
    else:
        dirty = set(all_ids if dirty_elements is None else dirty_elements)
        expected_layers = {scene.element_map()[element_id].layer_id for element_id in dirty}
        layer_dirty = expected_layers if dirty_layers is None else set(dirty_layers)
        if layer_dirty != expected_layers:
            raise NPRError(
                E_CLOSURE_MISMATCH,
                "layer closure does not match element closure",
                expected=sorted(expected_layers),
                actual=sorted(layer_dirty),
            )
    rasters: dict[str, ElementRaster] = {}
    flats: dict[str, FlatElement] = {}
    for element in scene.elements:
        record = geometry_store.get(element.id)
        if record is None:
            raise NPRError(E_CLOSURE_MISMATCH, "geometry is missing during rasterisation", element_id=element.id)
        flats[element.id] = record.flat
    for element in _scheduled_elements(scene, flats, strategy):
        flat = flats[element.id]
        if prior is not None and element.id not in dirty:
            cached = prior.element_rasters.get(element.id)
            if cached is None:
                raise NPRError(E_CLOSURE_MISMATCH, "raster cache is missing outside the closure", element_id=element.id)
            if (
                cached.element_hash != element.content_hash
                or cached.geometry_hash != flat.geometry_hash
                or cached.mode != mode
                or prior.viewport != viewport
            ):
                raise NPRError(E_CLOSURE_MISMATCH, "raster cache is stale outside the closure", element_id=element.id)
            rasters[element.id] = cached
        else:
            rasters[element.id] = rasterize_element(element, flat, mode, viewport, scene)

    layer_buffers: dict[str, bytes] = {}
    layer_hashes: dict[str, str] = {}
    for layer in scene.layers:
        if prior is not None and layer.id not in layer_dirty:
            cached = prior.layer_buffers.get(layer.id)
            cached_hash = prior.layer_hashes.get(layer.id)
            if cached is None or cached_hash is None:
                raise NPRError(E_CLOSURE_MISMATCH, "layer cache is missing outside the closure", layer_id=layer.id)
            if len(cached) != viewport[0] * viewport[1] * 4:
                raise NPRError(E_CLOSURE_MISMATCH, "cached layer buffer has the wrong size", layer_id=layer.id)
            if _layer_hash(layer, cached) != cached_hash:
                raise NPRError(E_CLOSURE_MISMATCH, "cached layer hash is inconsistent", layer_id=layer.id)
            layer_buffers[layer.id] = cached
            layer_hashes[layer.id] = cached_hash
        else:
            buffer = _compose_layer(scene, layer, rasters, viewport)
            layer_buffers[layer.id] = buffer
            layer_hashes[layer.id] = _layer_hash(layer, buffer)
    image = _compose_image(scene, layer_buffers, viewport)
    if gl_backend is not None:
        image = gl_backend.roundtrip_bgra(image, viewport[0], viewport[1])  # type: ignore[attr-defined]
    if prior is not None:
        for element_id in all_ids - dirty:
            if prior.skeleton_hashes.get(element_id) != geometry_store.get(element_id).flat.skeleton_hash:  # type: ignore[union-attr]
                raise NPRError(E_CLOSURE_MISMATCH, "skeleton hash changed outside the closure", element_id=element_id)
            if prior.element_hashes.get(element_id) != rasters[element_id].render_hash:
                raise NPRError(E_CLOSURE_MISMATCH, "render hash changed outside the closure", element_id=element_id)
    strategy_payload = strategy.as_dict() if strategy is not None else {}
    snapshot = build_snapshot(
        version=version,
        mode=mode,
        viewport=viewport,
        scene_hash=scene.content_hash,
        image=image,
        layer_buffers=layer_buffers,
        layer_hashes=layer_hashes,
        element_rasters=rasters,
        skeleton_hashes=geometry_store.skeleton_hashes(),
        geometry_hashes=geometry_store.geometry_hashes(),
        flat_geometry=flats,
        strategy=strategy_payload,
    )
    snapshot.verify()
    return RenderOutcome(snapshot=snapshot, actual_elements=frozenset(dirty), actual_layers=frozenset(layer_dirty))


def write_bgra(path: Path, image: bytes) -> None:
    atomic_write_bytes(path, image)


def write_bmp(path: Path, image: bytes, viewport: tuple[int, int]) -> None:
    """Write a deterministic top-down BGRA image as a bottom-up 32-bit BMP."""

    width, height = viewport
    if len(image) != width * height * 4:
        raise NPRError(E_STROKE_SAMPLE, "BGRA image size does not match viewport")
    row_stride = width * 4
    pixel_data = bytearray()
    for y in range(height - 1, -1, -1):
        start = y * row_stride
        pixel_data.extend(image[start : start + row_stride])
    file_header = bytearray(14)
    dib_header = bytearray(40)
    file_size = 14 + 40 + len(pixel_data)
    file_header[0:2] = b"BM"
    file_header[2:6] = file_size.to_bytes(4, "little")
    file_header[10:14] = (54).to_bytes(4, "little")
    dib_header[0:4] = (40).to_bytes(4, "little")
    dib_header[4:8] = width.to_bytes(4, "little", signed=True)
    dib_header[8:12] = height.to_bytes(4, "little", signed=True)
    dib_header[12:14] = (1).to_bytes(2, "little")
    dib_header[14:16] = (32).to_bytes(2, "little")
    dib_header[20:24] = len(pixel_data).to_bytes(4, "little")
    atomic_write_bytes(path, bytes(file_header) + bytes(dib_header) + bytes(pixel_data))
