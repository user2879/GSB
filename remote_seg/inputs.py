"""Configuration, index and tile loading with stable business failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import (
    E_BAND_VERSION_UNKNOWN,
    E_CONFIG_INVALID,
    E_FUSION_VERSION_UNKNOWN,
    E_INDEX_INVALID,
    E_TILE_SHAPE_MISMATCH,
    Failure,
    SegmentationError,
)
from .pngio import read_png
from .util import array_content_fingerprint, canonical_hash, read_json


BAND_NAMES = ("red", "green", "nir", "elevation")


@dataclass(frozen=True, slots=True)
class TileSpec:
    tile_id: str
    row: int
    col: int
    path: str
    bounds: tuple[int, int, int, int]
    neighbors: tuple[str, ...]
    band_order_version: str

    @property
    def width(self) -> int:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> int:
        return self.bounds[3] - self.bounds[1]

    @property
    def order_key(self) -> tuple[int, int, str]:
        return (self.row, self.col, self.tile_id)

    @property
    def task_sort_key(self) -> str:
        return f"{self.row:08d}:{self.col:08d}:classify"


@dataclass(frozen=True, slots=True)
class IndexData:
    path: Path
    root: Path
    index_version: str
    band_order_version: str
    bands: tuple[str, ...]
    tile_size: tuple[int, int]
    tiles: tuple[TileSpec, ...]
    raw: dict[str, Any]

    @property
    def by_id(self) -> dict[str, TileSpec]:
        return {tile.tile_id: tile for tile in self.tiles}

    @property
    def by_row_col(self) -> dict[tuple[int, int], TileSpec]:
        return {(tile.row, tile.col): tile for tile in self.tiles}

    @property
    def index_hash(self) -> str:
        return canonical_hash(self.raw, prefix="index-v1|")


@dataclass(frozen=True, slots=True)
class FusionStrategy:
    version: str
    order: int
    weight_power: float
    scale: float
    tie_break: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class FusionRegistry:
    path: Path
    strategies: Mapping[str, FusionStrategy]
    effective_order: tuple[str, ...]
    band_order_versions: frozenset[str]
    raw: dict[str, Any]

    def require(self, version: str) -> FusionStrategy:
        try:
            return self.strategies[version]
        except KeyError as exc:
            raise SegmentationError(
                E_FUSION_VERSION_UNKNOWN,
                f"fusion strategy version {version!r} is not declared",
                stage="config",
                details={"fusion_version": version},
            ) from exc


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    path: Path
    raw: dict[str, Any]
    window_size: int
    stride: int
    context_overlap: int
    feature_thresholds: dict[str, float]
    classification_weights: dict[int, dict[str, float]]
    min_connected_area: int
    fusion_version: str
    recompute_budget: int
    max_workers: int
    class_codes: tuple[int, ...]
    class_names: dict[int, str]

    @property
    def hash(self) -> str:
        return canonical_hash(self.raw, prefix="seg-config-v1|")

    @property
    def classification_hash(self) -> str:
        payload = {
            "window_size": self.window_size,
            "stride": self.stride,
            "context_overlap": self.context_overlap,
            "feature_thresholds": self.feature_thresholds,
            "classification_weights": {
                str(code): self.classification_weights[code] for code in self.class_codes
            },
            "class_codes": self.class_codes,
            "class_names": {str(code): self.class_names[code] for code in self.class_codes},
        }
        return canonical_hash(payload, prefix="classification-v1|")

    @property
    def postprocess_fingerprint(self) -> str:
        return canonical_hash(
            {"min_connected_area": self.min_connected_area},
            prefix="postprocess-v1|",
        )


@dataclass(slots=True)
class LoadedTile:
    spec: TileSpec
    array: np.ndarray
    content_fingerprint: str
    cache_key: str


@dataclass(slots=True)
class TileLoadResult:
    loaded: dict[str, LoadedTile] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)


def _require_mapping(value: Any, name: str, code: str = E_CONFIG_INVALID) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SegmentationError(code, f"{name} must be a JSON object", stage="validation")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SegmentationError(E_CONFIG_INVALID, f"{name} must be an integer", stage="validation")
    if minimum is not None and value < minimum:
        raise SegmentationError(
            E_CONFIG_INVALID,
            f"{name} must be >= {minimum}",
            stage="validation",
            details={"value": value},
        )
    return value


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SegmentationError(E_CONFIG_INVALID, f"{name} must be numeric", stage="validation")
    return float(value)


def load_registry(path: Path) -> FusionRegistry:
    try:
        raw = _require_mapping(read_json(path), "fusion version manifest")
    except (OSError, ValueError) as exc:
        raise SegmentationError(
            E_CONFIG_INVALID, f"cannot read fusion version manifest: {exc}", stage="validation"
        ) from exc
    strategies_raw = raw.get("strategies", raw.get("fusion_strategies"))
    if isinstance(strategies_raw, list):
        entries = strategies_raw
    elif isinstance(strategies_raw, dict):
        entries = []
        for version, item in strategies_raw.items():
            if not isinstance(item, dict):
                raise SegmentationError(E_CONFIG_INVALID, "strategy entry must be an object")
            entry = dict(item)
            entry.setdefault("version", version)
            entries.append(entry)
    else:
        raise SegmentationError(E_CONFIG_INVALID, "strategies must be an object or array")
    strategies: dict[str, FusionStrategy] = {}
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise SegmentationError(E_CONFIG_INVALID, "strategy entry must be an object")
        version = item.get("version")
        if not isinstance(version, str) or not version:
            raise SegmentationError(E_CONFIG_INVALID, "strategy version must be a string")
        if version in strategies:
            raise SegmentationError(E_CONFIG_INVALID, f"duplicate strategy version {version!r}")
        order = item.get("order", index + 1)
        if isinstance(order, bool) or not isinstance(order, int):
            raise SegmentationError(E_CONFIG_INVALID, f"strategy {version} order must be an integer")
        power = _require_float(item.get("weight_power", 1.0), f"strategy {version}.weight_power")
        if power <= 0:
            raise SegmentationError(E_CONFIG_INVALID, f"strategy {version} power must be positive")
        scale = _require_float(item.get("scale", 1.0), f"strategy {version}.scale")
        if scale <= 0:
            raise SegmentationError(E_CONFIG_INVALID, f"strategy {version} scale must be positive")
        tie_break = item.get("tie_break", "row_col_lex")
        if tie_break != "row_col_lex":
            raise SegmentationError(
                E_CONFIG_INVALID,
                f"unsupported tie_break {tie_break!r}; expected row_col_lex",
            )
        strategies[version] = FusionStrategy(
            version=version,
            order=order,
            weight_power=power,
            scale=scale,
            tie_break=tie_break,
            description=str(item.get("description", "")),
        )

    declared_effective = raw.get("effective_order")
    if declared_effective is None:
        declared_effective = [item.version for item in sorted(strategies.values(), key=lambda x: x.order)]
    if not isinstance(declared_effective, list) or not all(
        isinstance(item, str) for item in declared_effective
    ):
        raise SegmentationError(E_CONFIG_INVALID, "effective_order must be an array of strings")
    if set(declared_effective) != set(strategies):
        raise SegmentationError(
            E_CONFIG_INVALID,
            "effective_order must list every strategy exactly once",
            details={"effective_order": declared_effective, "strategies": sorted(strategies)},
        )
    if len(set(declared_effective)) != len(declared_effective):
        raise SegmentationError(E_CONFIG_INVALID, "effective_order contains duplicates")
    if [strategies[version].order for version in declared_effective] != sorted(
        strategies[version].order for version in declared_effective
    ):
        raise SegmentationError(E_CONFIG_INVALID, "effective_order must follow strategy order")

    band_versions_raw = raw.get("band_order_versions", [])
    if isinstance(band_versions_raw, dict):
        band_versions = set(band_versions_raw)
    elif isinstance(band_versions_raw, list):
        band_versions = set(band_versions_raw)
    else:
        raise SegmentationError(E_CONFIG_INVALID, "band_order_versions must be array or object")
    if not all(isinstance(item, str) and item for item in band_versions):
        raise SegmentationError(E_CONFIG_INVALID, "band_order_versions must contain non-empty strings")
    return FusionRegistry(
        path=path,
        strategies=strategies,
        effective_order=tuple(declared_effective),
        band_order_versions=frozenset(band_versions),
        raw=raw,
    )


def load_config(path: Path) -> SegmentationConfig:
    try:
        raw = _require_mapping(read_json(path), "segmentation config")
    except (OSError, ValueError) as exc:
        raise SegmentationError(
            E_CONFIG_INVALID, f"cannot read segmentation config: {exc}", stage="validation"
        ) from exc

    window_size = _require_int(raw.get("window_size"), "window_size", minimum=1)
    stride = _require_int(raw.get("stride"), "stride", minimum=1)
    context_overlap = _require_int(raw.get("context_overlap"), "context_overlap", minimum=0)
    if context_overlap > window_size:
        raise SegmentationError(E_CONFIG_INVALID, "context_overlap cannot exceed window_size")
    thresholds_raw = _require_mapping(raw.get("feature_thresholds"), "feature_thresholds")
    thresholds: dict[str, float] = {}
    for name in ("vegetation", "water", "edge", "slope"):
        thresholds[name] = _require_float(thresholds_raw.get(name), f"feature_thresholds.{name}")

    weights_raw = _require_mapping(raw.get("classification_weights"), "classification_weights")
    classification_weights: dict[int, dict[str, float]] = {}
    for code_key, entry in weights_raw.items():
        try:
            code = int(code_key)
        except (TypeError, ValueError) as exc:
            raise SegmentationError(E_CONFIG_INVALID, "class code must be an integer") from exc
        if code < 0:
            raise SegmentationError(E_CONFIG_INVALID, "class code must be non-negative")
        if not isinstance(entry, dict):
            raise SegmentationError(E_CONFIG_INVALID, f"weights for class {code} must be an object")
        normalized: dict[str, float] = {}
        for name in ("bias", "vegetation", "water", "edge", "slope"):
            normalized[name] = _require_float(entry.get(name, 0.0), f"weights.{code}.{name}")
        classification_weights[code] = normalized
    class_codes = tuple(sorted(classification_weights))
    if not class_codes:
        raise SegmentationError(E_CONFIG_INVALID, "at least one class is required")
    if class_codes != tuple(range(len(class_codes))):
        raise SegmentationError(
            E_CONFIG_INVALID,
            "class codes must be contiguous and start at zero",
            details={"class_codes": class_codes},
        )

    classes_raw = raw.get("classes", [])
    class_names = {code: f"class_{code}" for code in class_codes}
    if classes_raw:
        if not isinstance(classes_raw, list):
            raise SegmentationError(E_CONFIG_INVALID, "classes must be an array")
        seen: set[int] = set()
        for item in classes_raw:
            if not isinstance(item, dict):
                raise SegmentationError(E_CONFIG_INVALID, "class entry must be an object")
            code = item.get("code")
            name = item.get("name")
            if isinstance(code, bool) or not isinstance(code, int) or not isinstance(name, str):
                raise SegmentationError(E_CONFIG_INVALID, "class requires integer code and string name")
            if code in seen:
                raise SegmentationError(E_CONFIG_INVALID, f"duplicate class code {code}")
            seen.add(code)
            class_names[code] = name
        if seen != set(class_codes):
            raise SegmentationError(E_CONFIG_INVALID, "classes must match classification_weights keys")

    postprocess_raw = _require_mapping(raw.get("postprocess"), "postprocess")
    min_area = _require_int(
        postprocess_raw.get("min_connected_area"), "postprocess.min_connected_area", minimum=1
    )
    fusion_raw = _require_mapping(raw.get("overlap_fusion"), "overlap_fusion")
    fusion_version = fusion_raw.get("strategy_version")
    if not isinstance(fusion_version, str) or not fusion_version:
        raise SegmentationError(E_CONFIG_INVALID, "overlap_fusion.strategy_version is required")
    recompute_budget = _require_int(raw.get("recompute_budget"), "recompute_budget", minimum=1)
    max_workers = _require_int(raw.get("max_workers"), "max_workers", minimum=1)
    return SegmentationConfig(
        path=path,
        raw=raw,
        window_size=window_size,
        stride=stride,
        context_overlap=context_overlap,
        feature_thresholds=thresholds,
        classification_weights=classification_weights,
        min_connected_area=min_area,
        fusion_version=fusion_version,
        recompute_budget=recompute_budget,
        max_workers=max_workers,
        class_codes=class_codes,
        class_names=class_names,
    )


def load_index(path: Path) -> IndexData:
    try:
        raw = _require_mapping(read_json(path), "tile index", E_INDEX_INVALID)
    except (OSError, ValueError) as exc:
        raise SegmentationError(
            E_INDEX_INVALID, f"cannot read tile index: {exc}", stage="index"
        ) from exc
    index_version = raw.get("index_version")
    if not isinstance(index_version, str) or not index_version:
        raise SegmentationError(E_INDEX_INVALID, "index_version is required", stage="index")
    band_order_version = raw.get("band_order_version")
    if not isinstance(band_order_version, str) or not band_order_version:
        raise SegmentationError(E_INDEX_INVALID, "band_order_version is required", stage="index")
    bands = raw.get("bands")
    if not isinstance(bands, list) or tuple(bands) != BAND_NAMES:
        raise SegmentationError(
            E_INDEX_INVALID,
            f"bands must be {list(BAND_NAMES)}",
            stage="index",
            details={"bands": bands},
        )
    tile_size_raw = raw.get("tile_size")
    if (
        not isinstance(tile_size_raw, list)
        or len(tile_size_raw) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in tile_size_raw)
    ):
        raise SegmentationError(E_INDEX_INVALID, "tile_size must be [width, height]", stage="index")
    tile_size = (tile_size_raw[0], tile_size_raw[1])
    tiles_raw = raw.get("tiles")
    if not isinstance(tiles_raw, list):
        raise SegmentationError(E_INDEX_INVALID, "tiles must be an array", stage="index")
    specs: list[TileSpec] = []
    seen_ids: set[str] = set()
    seen_cells: set[tuple[int, int]] = set()
    for position, item in enumerate(tiles_raw):
        if not isinstance(item, dict):
            raise SegmentationError(E_INDEX_INVALID, f"tile {position} must be an object", stage="index")
        tile_id = item.get("tile_id")
        row_value = item.get("row")
        col_value = item.get("col")
        tile_path = item.get("path")
        bounds_raw = item.get("bounds")
        neighbors_raw = item.get("neighbors", [])
        tile_band_version = item.get("band_order_version", band_order_version)
        if not isinstance(tile_id, str) or not tile_id:
            raise SegmentationError(E_INDEX_INVALID, f"tile {position} has invalid tile_id", stage="index")
        if tile_id in seen_ids:
            raise SegmentationError(E_INDEX_INVALID, f"duplicate tile_id {tile_id!r}", stage="index")
        if (
            isinstance(row_value, bool)
            or not isinstance(row_value, int)
            or isinstance(col_value, bool)
            or not isinstance(col_value, int)
            or row_value < 0
            or col_value < 0
        ):
            raise SegmentationError(E_INDEX_INVALID, f"tile {tile_id} row/col invalid", stage="index")
        cell = (row_value, col_value)
        if cell in seen_cells:
            raise SegmentationError(
                E_INDEX_INVALID, f"duplicate row/col {cell}", stage="index", tile_id=tile_id
            )
        if not isinstance(tile_path, str) or not tile_path:
            raise SegmentationError(E_INDEX_INVALID, f"tile {tile_id} path invalid", stage="index")
        if (
            not isinstance(bounds_raw, list)
            or len(bounds_raw) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in bounds_raw)
        ):
            raise SegmentationError(E_INDEX_INVALID, f"tile {tile_id} bounds invalid", stage="index")
        x0, y0, x1, y1 = bounds_raw
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise SegmentationError(E_INDEX_INVALID, f"tile {tile_id} bounds empty", stage="index")
        if (x1 - x0, y1 - y0) != tile_size:
            raise SegmentationError(
                E_INDEX_INVALID,
                f"tile {tile_id} bounds do not match tile_size",
                stage="index",
                tile_id=tile_id,
            )
        if not isinstance(neighbors_raw, list) or not all(
            isinstance(v, str) and v for v in neighbors_raw
        ):
            raise SegmentationError(
                E_INDEX_INVALID, f"tile {tile_id} neighbors must be strings", stage="index", tile_id=tile_id
            )
        if len(set(neighbors_raw)) != len(neighbors_raw):
            raise SegmentationError(
                E_INDEX_INVALID, f"tile {tile_id} has duplicate neighbors", stage="index", tile_id=tile_id
            )
        if tile_id in neighbors_raw:
            raise SegmentationError(
                E_INDEX_INVALID, f"tile {tile_id} cannot neighbor itself", stage="index", tile_id=tile_id
            )
        if not isinstance(tile_band_version, str) or not tile_band_version:
            raise SegmentationError(
                E_INDEX_INVALID, f"tile {tile_id} band_order_version invalid", stage="index"
            )
        specs.append(
            TileSpec(
                tile_id=tile_id,
                row=row_value,
                col=col_value,
                path=tile_path,
                bounds=(x0, y0, x1, y1),
                neighbors=tuple(neighbors_raw),
                band_order_version=tile_band_version,
            )
        )
        seen_ids.add(tile_id)
        seen_cells.add(cell)

    by_id = {spec.tile_id: spec for spec in specs}
    for spec in specs:
        for neighbor in spec.neighbors:
            if neighbor not in by_id:
                raise SegmentationError(
                    E_INDEX_INVALID,
                    f"tile {spec.tile_id} references missing neighbor {neighbor}",
                    stage="index",
                    tile_id=spec.tile_id,
                )
            if spec.tile_id not in by_id[neighbor].neighbors:
                raise SegmentationError(
                    E_INDEX_INVALID,
                    f"adjacency is not symmetric: {spec.tile_id} -> {neighbor}",
                    stage="index",
                    tile_id=spec.tile_id,
                )

    return IndexData(
        path=path,
        root=path.parent,
        index_version=index_version,
        band_order_version=band_order_version,
        bands=tuple(bands),
        tile_size=tile_size,
        tiles=tuple(specs),
        raw=raw,
    )


def tile_cache_key(content_fingerprint: str, band_order_version: str, config: SegmentationConfig) -> str:
    return canonical_hash(
        {
            "content_fingerprint": content_fingerprint,
            "band_order_version": band_order_version,
            "classification_fingerprint": config.classification_hash,
        },
        prefix="tile-cache-v1|",
    )


def load_tiles(
    index: IndexData,
    config: SegmentationConfig,
    registry: FusionRegistry,
    *,
    truncate_overrides: Mapping[str, int] | None = None,
    band_count_overrides: Mapping[str, int] | None = None,
) -> TileLoadResult:
    result = TileLoadResult()
    truncate_overrides = truncate_overrides or {}
    band_count_overrides = band_count_overrides or {}
    for spec in sorted(index.tiles, key=lambda item: item.order_key):
        if spec.band_order_version not in registry.band_order_versions:
            result.failures.append(
                Failure(
                    code=E_BAND_VERSION_UNKNOWN,
                    stage="band_version",
                    message=f"band order version {spec.band_order_version!r} is not declared",
                    row=spec.row,
                    col=spec.col,
                    tile_id=spec.tile_id,
                    details={"band_order_version": spec.band_order_version},
                )
            )
            continue
        path = index.root / spec.path
        try:
            decoded = read_png(path, truncate_bytes=truncate_overrides.get(spec.tile_id))
            channels = band_count_overrides.get(spec.tile_id, decoded.channels)
            if channels != decoded.channels:
                raise SegmentationError(
                    E_TILE_SHAPE_MISMATCH,
                    f"decoded {decoded.channels} bands, injected/expected {channels}",
                    stage="shape",
                )
            if decoded.channels != len(BAND_NAMES):
                raise SegmentationError(
                    E_TILE_SHAPE_MISMATCH,
                    f"expected {len(BAND_NAMES)} bands, decoded {decoded.channels}",
                    stage="shape",
                    details={"expected_bands": len(BAND_NAMES), "decoded_bands": decoded.channels},
                )
            if (decoded.width, decoded.height) != (spec.width, spec.height):
                raise SegmentationError(
                    E_TILE_SHAPE_MISMATCH,
                    f"decoded size {decoded.width}x{decoded.height} does not match index "
                    f"{spec.width}x{spec.height}",
                    stage="shape",
                    details={
                        "decoded_size": [decoded.width, decoded.height],
                        "index_size": [spec.width, spec.height],
                    },
                )
        except SegmentationError as exc:
            failure = exc.failure()
            failure.row = spec.row
            failure.col = spec.col
            failure.tile_id = spec.tile_id
            result.failures.append(failure)
            continue
        content_fp = array_content_fingerprint(decoded.array)
        cache_key = tile_cache_key(content_fp, spec.band_order_version, config)
        result.loaded[spec.tile_id] = LoadedTile(
            spec=spec,
            array=decoded.array,
            content_fingerprint=content_fp,
            cache_key=cache_key,
        )
    return result
