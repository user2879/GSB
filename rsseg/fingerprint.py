"""Dependency fingerprints and on-disk result cache.

Tile cache keys derive from band content, index band-order version and the
segmentation configuration fingerprint. Fusion fingerprints derive from the
participating tile cache keys plus the fusion strategy. The cache therefore
invalidates exactly the tiles touched by an edit and only the fusion regions
that consume them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import SegmentationConfig, stable_fingerprint
from .fusion import FusionRegion
from .tiles import TileRecord


def band_content_fingerprint(bands: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(bands.shape).encode("ascii"))
    digest.update(bands.astype(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def tile_cache_key_with_version(tile: TileRecord, bands: np.ndarray,
                                cfg: SegmentationConfig,
                                band_order_version: str) -> str:
    payload = {
        "tile": [tile.row, tile.col],
        "bounds": list(tile.bounds),
        "band_fingerprint": band_content_fingerprint(bands),
        "band_order_version": band_order_version,
        "config_fingerprint": cfg.fingerprint(),
    }
    return stable_fingerprint(payload)


def fusion_fingerprint(region: FusionRegion,
                       tile_keys: dict[tuple[int, int], str],
                       cfg: SegmentationConfig,
                       adjacency_version: str = "") -> str:
    payload = {
        "region": [list(region.a), list(region.b)],
        "global_box": [region.global_x0, region.global_y0, region.global_x1, region.global_y1],
        "tile_keys": [tile_keys[region.a], tile_keys[region.b]],
        "fusion_strategy": cfg.fusion_strategy,
        "adjacency_version": adjacency_version,
    }
    return stable_fingerprint(payload)


def index_adjacency_version(index) -> str:
    payload = {
        "band_order_version": index.band_order_version,
        "edges": sorted(
            [list(edge) for tile in index.tiles for edge in (
                [sorted((tile.key, n)) for n in tile.neighbors if n > tile.key]
            )]
        ),
    }
    return stable_fingerprint(payload)


class ResultStore:
    """File-backed cache of tile results and fusion region results."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tile_dir = self.root / "tiles"
        self.fusion_dir = self.root / "fusions"
        self.tile_dir.mkdir(parents=True, exist_ok=True)
        self.fusion_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tile_name(key: tuple[int, int]) -> str:
        return f"r{key[0]:04d}_c{key[1]:04d}"

    def tile_path(self, key: tuple[int, int], cache_key: str) -> Path:
        return self.tile_dir / f"{self.tile_name(key)}_{cache_key[:16]}.npz"

    def has_tile(self, key: tuple[int, int], cache_key: str) -> bool:
        return self.tile_path(key, cache_key).exists()

    def save_tile(self, key: tuple[int, int], cache_key: str,
                  class_map: np.ndarray, confidence: np.ndarray,
                  score_map: np.ndarray) -> Path:
        path = self.tile_path(key, cache_key)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, class_map=class_map, confidence=confidence, score_map=score_map,
                 cache_key=np.array(cache_key))
        tmp.replace(path)
        return path

    def load_tile(self, key: tuple[int, int], cache_key: str) -> dict[str, np.ndarray]:
        with np.load(self.tile_path(key, cache_key)) as data:
            return {
                "class_map": data["class_map"].copy(),
                "confidence": data["confidence"].copy(),
                "score_map": data["score_map"].copy(),
            }

    def prune_tile(self, key: tuple[int, int], keep_key: str) -> None:
        for path in self.tile_dir.glob(f"{self.tile_name(key)}_*.npz"):
            if keep_key[:16] not in path.name:
                path.unlink()

    def fusion_path(self, region: FusionRegion, fingerprint: str) -> Path:
        return self.fusion_dir / f"{region.name}_{fingerprint[:16]}.npz"

    def has_fusion(self, region: FusionRegion, fingerprint: str) -> bool:
        return self.fusion_path(region, fingerprint).exists()

    def save_fusion(self, region: FusionRegion, fingerprint: str,
                    fused_patch: np.ndarray) -> Path:
        path = self.fusion_path(region, fingerprint)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, fused_patch=fused_patch, fingerprint=np.array(fingerprint))
        tmp.replace(path)
        return path

    def load_fusion(self, region: FusionRegion, fingerprint: str) -> np.ndarray:
        with np.load(self.fusion_path(region, fingerprint)) as data:
            return data["fused_patch"].copy()

    def prune_fusion(self, region: FusionRegion, keep_fingerprint: str) -> None:
        for path in self.fusion_dir.glob(f"{region.name}_*.npz"):
            if keep_fingerprint[:16] not in path.name:
                path.unlink()

    def write_json(self, relative: str, payload: dict) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
        tmp.replace(path)
        return path
