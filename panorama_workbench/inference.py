from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import Category, WorkbenchConfig
from .fingerprint import tile_content_hash


FEATURE_DIM = 14


@dataclass(frozen=True)
class TileGeometry:
    row: int
    col: int
    x: int
    y: int
    size: int

    @property
    def key(self) -> str:
        return f"{self.row:06d}:{self.col:06d}"


@dataclass
class TileInference:
    geometry: TileGeometry
    content_hash: str
    feature_key: str
    features: np.ndarray
    residual: dict[str, int]

    @property
    def pixel_count(self) -> int:
        return self.geometry.size * self.geometry.size


@dataclass
class TileMask:
    geometry: TileGeometry
    feature_key: str
    mask_key: str
    class_grid: np.ndarray
    class_counts: dict[int, int]
    payload_hash: str


@dataclass(frozen=True)
class Fragment:
    tile_row: int
    tile_col: int
    fragment_index: int
    category_code: int
    pixels: tuple[int, ...]
    area: int
    min_row: int
    min_col: int
    tile_key: str
    fragment_id: str


def tile_geometries(width: int, height: int, size: int, stride: int) -> list[TileGeometry]:
    if width % size != 0 or height % size != 0:
        raise ValueError("window size must divide image dimensions")
    xs = list(range(0, width - size + 1, stride))
    ys = list(range(0, height - size + 1, stride))
    if not xs or xs[-1] != width - size:
        xs.append(width - size)
    if not ys or ys[-1] != height - size:
        ys.append(height - size)
    return [
        TileGeometry(row=row, col=col, x=x, y=y, size=size)
        for row, y in enumerate(ys)
        for col, x in enumerate(xs)
    ]


class DeterministicScorer:
    def __init__(self, config: WorkbenchConfig) -> None:
        self.config = config
        self.q = 1 << config.fixed_bits
        self.categories = tuple(sorted(config.categories, key=lambda item: item.code))
        self.weights = _projection_matrix(config, self.categories, self.q)
        self.quant_step = max(1, int(config.quant_level))

    def features(self, tile_rgb: np.ndarray) -> np.ndarray:
        return compute_tile_features(tile_rgb, self.config.fixed_bits)

    def classify(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        flat = features.reshape(-1, FEATURE_DIM).astype(np.int64, copy=False)
        raw_scores = (flat @ self.weights.T) // self.q
        for index, category in enumerate(self.categories):
            raw_scores[:, index] = (
                raw_scores[:, index] * category.weight_milli
            ) // 1000
        step = self.quant_step
        positive = raw_scores >= 0
        rounded_magnitude = ((np.abs(raw_scores) + step // 2) // step) * step
        quantized = np.where(positive, rounded_magnitude, -rounded_magnitude)
        residual = raw_scores - quantized
        classes = np.asarray(
            [category.code for category in self.categories], dtype=np.int32
        )[np.argmax(quantized, axis=1)]
        summary = {
            "abs_residual_fixed": int(np.abs(residual).sum()),
            "down_count": int((residual < 0).sum()),
            "max_residual_fixed": int(residual.max()) if residual.size else 0,
            "min_residual_fixed": int(residual.min()) if residual.size else 0,
            "none_count": int((residual == 0).sum()),
            "residual_fixed": int(residual.sum()),
            "up_count": int((residual > 0).sum()),
        }
        return classes, summary


def infer_tile(
    rgb: np.ndarray,
    geometry: TileGeometry,
    config: WorkbenchConfig,
    scorer: DeterministicScorer,
) -> TileInference:
    x0, y0 = geometry.x, geometry.y
    tile = np.ascontiguousarray(
        rgb[y0 : y0 + geometry.size, x0 : x0 + geometry.size, :]
    )
    content_hash = tile_content_hash(tile)
    feature_key = feature_cache_key(content_hash, geometry, config)
    features = scorer.features(tile)
    _, residual = scorer.classify(features)
    return TileInference(
        geometry=geometry,
        content_hash=content_hash,
        feature_key=feature_key,
        features=features,
        residual=residual,
    )


def classify_tile(
    inference: TileInference,
    scorer: DeterministicScorer,
    config: WorkbenchConfig,
) -> TileMask:
    classes, residual = scorer.classify(inference.features)
    class_grid = classes.reshape(inference.geometry.size, inference.geometry.size).astype(
        np.int32
    )
    inference.residual = residual
    mask_key = mask_cache_key(
        inference.feature_key,
        config,
        inference.geometry,
    )
    payload_hash = _grid_hash(class_grid)
    counts: dict[int, int] = {}
    for code, count in zip(*np.unique(class_grid, return_counts=True)):
        counts[int(code)] = int(count)
    return TileMask(
        geometry=inference.geometry,
        feature_key=inference.feature_key,
        mask_key=mask_key,
        class_grid=class_grid,
        class_counts=dict(sorted(counts.items())),
        payload_hash=payload_hash,
    )


def feature_cache_key(content_hash: str, geometry: TileGeometry, config: WorkbenchConfig) -> str:
    payload = (
        "tile-feature-v1|"
        f"{config.inference_hash}|{content_hash}|"
        f"{geometry.row}:{geometry.col}:{geometry.x}:{geometry.y}:{geometry.size}"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def mask_cache_key(
    feature_key: str, config: WorkbenchConfig, geometry: TileGeometry
) -> str:
    payload = (
        "tile-mask-v1|"
        f"{feature_key}|{config.category_hash}|{config.quant_level}|"
        f"{geometry.row}:{geometry.col}"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def instance_input_key(
    width: int,
    height: int,
    masks: Iterable[TileMask],
    config: WorkbenchConfig,
) -> str:
    mask_hashes = sorted(
        f"{mask.geometry.row}:{mask.geometry.col}:{mask.payload_hash}" for mask in masks
    )
    payload = (
        "instance-input-v1|"
        f"{width}x{height}|{config.merge_hash}|"
        + "|".join(mask_hashes)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def extract_fragments(
    width: int,
    height: int,
    mask: TileMask,
    background_codes: frozenset[int] = frozenset({0}),
) -> list[Fragment]:
    tile = mask.class_grid
    size = mask.geometry.size
    x0, y0 = mask.geometry.x, mask.geometry.y
    fragments: list[Fragment] = []
    fragment_index = 0
    for category_code in sorted(int(value) for value in np.unique(tile)):
        if category_code in background_codes:
            continue
        category_mask = tile == category_code
        visited = np.zeros_like(category_mask, dtype=bool)
        components: list[list[tuple[int, int]]] = []
        for local_y in range(size):
            for local_x in range(size):
                if not category_mask[local_y, local_x] or visited[local_y, local_x]:
                    continue
                stack = [(local_y, local_x)]
                visited[local_y, local_x] = True
                component: list[tuple[int, int]] = []
                while stack:
                    current_y, current_x = stack.pop()
                    component.append((current_y, current_x))
                    for next_y, next_x in (
                        (current_y - 1, current_x),
                        (current_y + 1, current_x),
                        (current_y, current_x - 1),
                        (current_y, current_x + 1),
                    ):
                        if (
                            0 <= next_y < size
                            and 0 <= next_x < size
                            and category_mask[next_y, next_x]
                            and not visited[next_y, next_x]
                        ):
                            visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
                components.append(component)
        components.sort(key=lambda component: min(component))
        for component in components:
            global_pixels = sorted(
                (y0 + local_y) * width + (x0 + local_x)
                for local_y, local_x in component
            )
            rows = [pixel // width for pixel in global_pixels]
            cols = [pixel % width for pixel in global_pixels]
            fragment_id = (
                f"{mask.geometry.row:06d}:{mask.geometry.col:06d}:"
                f"{fragment_index:06d}:{category_code:06d}"
            )
            fragments.append(
                Fragment(
                    tile_row=mask.geometry.row,
                    tile_col=mask.geometry.col,
                    fragment_index=fragment_index,
                    category_code=category_code,
                    pixels=tuple(global_pixels),
                    area=len(global_pixels),
                    min_row=min(rows),
                    min_col=min(cols),
                    tile_key=mask.geometry.key,
                    fragment_id=fragment_id,
                )
            )
            fragment_index += 1
    return fragments


def compute_tile_features(tile_rgb: np.ndarray, fixed_bits: int) -> np.ndarray:
    q = 1 << fixed_bits
    rgb = np.ascontiguousarray(tile_rgb, dtype=np.int64)
    height, width, _ = rgb.shape
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    gray = (77 * red + 150 * green + 29 * blue) >> 8
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    if width > 1:
        gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    if height > 1:
        gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gradient = np.minimum(255, gx + gy)
    mean_gray = int(gray.sum() // gray.size)
    centered_gray = gray - mean_gray
    variance_numerator = int((centered_gray * centered_gray).sum()) // gray.size
    texture = min(255, math.isqrt(max(0, variance_numerator)))
    edge_density = int((gradient >= 24).sum()) * q // gray.size
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)

    features = np.empty((height, width, FEATURE_DIM), dtype=np.int64)
    features[:, :, 0] = (red - 128) * q // 128
    features[:, :, 1] = (green - 128) * q // 128
    features[:, :, 2] = (blue - 128) * q // 128
    features[:, :, 3] = gradient * q // 255
    features[:, :, 4] = np.abs(centered_gray) * q // 128
    features[:, :, 5] = gray * q // 255
    features[:, :, 6] = (max_channel - min_channel) * q // 255
    features[:, :, 7] = q
    features[:, :, 8] = int(red.sum() // red.size - 128) * q // 128
    features[:, :, 9] = int(green.sum() // green.size - 128) * q // 128
    features[:, :, 10] = int(blue.sum() // blue.size - 128) * q // 128
    features[:, :, 11] = edge_density
    features[:, :, 12] = texture * q // 255
    features[:, :, 13] = mean_gray * q // 255
    return features


def category_color(config: WorkbenchConfig, category: Category) -> tuple[int, int, int]:
    if category.code == 0:
        return (241, 244, 247)
    digest = hashlib.sha256(
        f"color-v1|{config.inference_hash}|{category.code}|{category.name}|"
        f"{category.weight_milli}".encode("utf-8")
    ).digest()
    red = 40 + (digest[0] % 176)
    green = 40 + (digest[1] % 176)
    blue = 40 + (digest[2] % 176)
    return red, green, blue


def _projection_matrix(
    config: WorkbenchConfig, categories: tuple[Category, ...], q: int
) -> np.ndarray:
    maximum = max(1, q // 16)
    span = maximum * 2 + 1
    rows: list[np.ndarray] = []
    for category in categories:
        seed = (
            f"projection-v1|{config.inference_hash}|{category.code}|"
            f"{category.name}|{category.weight_milli}"
        ).encode("utf-8")
        digest = hashlib.sha512(seed).digest()
        values: list[int] = []
        counter = 0
        while len(values) < FEATURE_DIM:
            block = hashlib.sha512(digest + counter.to_bytes(4, "big")).digest()
            counter += 1
            for offset in range(0, len(block), 4):
                if len(values) >= FEATURE_DIM:
                    break
                integer = int.from_bytes(block[offset : offset + 4], "big")
                values.append((integer % span) - maximum)
        target = category_color(config, category)
        centered = [(channel - 128) * q // 128 for channel in target]
        values[0] = centered[0]
        values[1] = centered[1]
        values[2] = centered[2]
        values[8] = centered[0] // 2
        values[9] = centered[1] // 2
        values[10] = centered[2] // 2
        rows.append(np.asarray(values, dtype=np.int64))
    return np.stack(rows, axis=0)


def _grid_hash(grid: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(grid.astype(np.int32))
    return hashlib.sha256(
        f"{contiguous.shape}:".encode("ascii") + contiguous.tobytes()
    ).hexdigest()
