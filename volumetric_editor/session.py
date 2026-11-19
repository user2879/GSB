"""Incremental frame session with exact tile and volume-segment closures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import fail, ErrorCode
from .models import Scene, TileResult
from .rendering import (
    TileSpec,
    assemble_image,
    assemble_trace,
    image_sha256,
    plan_tiles,
    tile_hashes,
    trace_sha256,
)
from .scheduling import SchedulerConfig, execute_tiles, recommended_budgets
from .validation import (
    validate_ior,
    validate_scene,
    validate_scatter,
    validate_tile_size,
)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class FrameSnapshot:
    image_sha256: str
    trace_sha256: str
    tile_hashes: dict[str, str]
    segments_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_image_sha256": self.image_sha256,
            "volume_trace_sha256": self.trace_sha256,
            "tile_hashes": dict(sorted(self.tile_hashes.items())),
            "segments_total": self.segments_total,
        }


class IncrementalSession:
    """Owns a canonical frame cache and performs closure-limited edits."""

    def __init__(self, scene: Scene, tile_size: int = 16) -> None:
        validate_scene(scene)
        self.scene = scene
        self.width = scene.camera.width
        self.height = scene.camera.height
        self.tile_size = validate_tile_size(tile_size, self.width, self.height)
        self.tiles: dict[str, TileResult] = {}
        self.image = np.zeros((self.height, self.width, 4), dtype=np.float32)
        self.visibility = np.zeros((self.height, self.width), dtype=np.uint32)
        self.trace = np.zeros(
            (self.height, self.width, scene.volume.max_steps, 6),
            dtype=np.float32,
        )
        self.inside = np.zeros(
            (self.height, self.width, scene.volume.max_steps),
            dtype=np.uint8,
        )

    @property
    def rendered(self) -> bool:
        return bool(self.tiles)

    def _default_budgets(self, tile_size: int) -> tuple[int, int]:
        return recommended_budgets(self.scene, tile_size)

    def render_full(
        self,
        threads: int = 4,
        queue_memory_budget_bytes: int | None = None,
        compute_budget_units: int | None = None,
        tile_size: int | None = None,
    ) -> dict[str, Any]:
        if tile_size is not None:
            self.tile_size = validate_tile_size(tile_size, self.width, self.height)
        default_memory, default_compute = self._default_budgets(self.tile_size)
        memory_budget = default_memory if queue_memory_budget_bytes is None else queue_memory_budget_bytes
        compute_budget = default_compute if compute_budget_units is None else compute_budget_units
        config = SchedulerConfig(
            thread_limit=threads,
            queue_memory_budget_bytes=int(memory_budget),
            compute_budget_units=int(compute_budget),
        )
        specs = plan_tiles(self.width, self.height, self.tile_size)
        result = execute_tiles(self.scene, specs, config)
        self.tiles = result.tiles
        self._refresh_canonical()
        return {
            "mode": "full_render",
            "tile_size": self.tile_size,
            "scheduler": result.report,
            "frame": self.snapshot().to_dict(),
        }

    def _refresh_canonical(self) -> None:
        self.image = assemble_image(self.tiles, self.width, self.height)
        self.trace, self.inside = assemble_trace(
            self.tiles,
            self.width,
            self.height,
            self.scene.volume.max_steps,
        )
        self.visibility = np.zeros((self.height, self.width), dtype=np.uint32)
        for key in sorted(self.tiles):
            tile = self.tiles[key]
            self.visibility[tile.y0 : tile.y1, tile.x0 : tile.x1] = tile.visibility_mask

    def snapshot(self) -> FrameSnapshot:
        hashes = {key: tile_hashes(tile)["combined_sha256"] for key, tile in sorted(self.tiles.items())}
        return FrameSnapshot(
            image_sha256=image_sha256(self.image),
            trace_sha256=trace_sha256(self.trace),
            tile_hashes=hashes,
            segments_total=int(self.width * self.height * self.scene.volume.max_steps),
        )

    def _scheduler_config(
        self,
        threads: int,
        queue_memory_budget_bytes: int | None,
        compute_budget_units: int | None,
    ) -> SchedulerConfig:
        default_memory, default_compute = self._default_budgets(self.tile_size)
        return SchedulerConfig(
            thread_limit=threads,
            queue_memory_budget_bytes=int(default_memory if queue_memory_budget_bytes is None else queue_memory_budget_bytes),
            compute_budget_units=int(default_compute if compute_budget_units is None else compute_budget_units),
        )

    def _specs_for_keys(self, keys: list[str]) -> list[TileSpec]:
        specs: list[TileSpec] = []
        for key in sorted(keys):
            tile = self.tiles[key]
            specs.append(
                TileSpec(
                    key=tile.key,
                    tx=tile.tx,
                    ty=tile.ty,
                    x0=tile.x0,
                    y0=tile.y0,
                    x1=tile.x1,
                    y1=tile.y1,
                )
            )
        return specs

    def _closure_for_surface(self, surface_id: str) -> list[str]:
        surface = self.scene.surface_by_id(surface_id)
        bit = np.uint32(1 << surface.sort_index)
        affected = []
        for key in sorted(self.tiles):
            tile = self.tiles[key]
            if np.any((tile.visibility_mask & bit) != 0):
                affected.append(key)
        return affected

    def _segment_closure(self) -> list[dict[str, int | str]]:
        steps = self.scene.volume.max_steps
        records: list[dict[str, int | str]] = []
        for key in sorted(self.tiles):
            tile = self.tiles[key]
            active = np.any(tile.volume_inside.astype(bool), axis=-1)
            if not np.any(active):
                continue
            first = np.argmax(tile.volume_inside.astype(bool), axis=-1)
            ys, xs = np.nonzero(active)
            for local_y, local_x in zip(ys.tolist(), xs.tolist(), strict=True):
                start_step = int(first[local_y, local_x])
                records.append(
                    {
                        "tile_key": key,
                        "x": int(tile.x0 + local_x),
                        "y": int(tile.y0 + local_y),
                        "start_step": start_step,
                        "end_step": steps - 1,
                        "count": steps - start_step,
                    }
                )
        return records

    def _check_reused_tiles(self, before: FrameSnapshot, closure_tiles: list[str]) -> bool:
        closure = set(closure_tiles)
        after = self.snapshot()
        for key in sorted(set(before.tile_hashes) - closure):
            if before.tile_hashes[key] != after.tile_hashes[key]:
                return False
        return True

    def edit_ior(
        self,
        surface_id: str,
        new_ior: float,
        threads: int = 4,
        queue_memory_budget_bytes: int | None = None,
        compute_budget_units: int | None = None,
    ) -> dict[str, Any]:
        if not self.rendered:
            raise fail(ErrorCode.BUDGET_DENIED, "render the frame before applying an incremental edit")
        old_surface = self.scene.surface_by_id(surface_id)
        ior = validate_ior(new_ior, surface_id)
        closure_tiles = self._closure_for_surface(surface_id)
        before = self.snapshot()
        old_volume_trace = {key: tile.volume_trace.copy() for key, tile in self.tiles.items()}

        new_scene = self.scene.replace_surface_ior(surface_id, ior)
        config = self._scheduler_config(threads, queue_memory_budget_bytes, compute_budget_units)
        result = execute_tiles(new_scene, self._specs_for_keys(closure_tiles), config)
        actual_recomputed = sorted(result.tiles)
        if actual_recomputed != sorted(closure_tiles):
            raise RuntimeError("IOR recompute set diverged from closure")

        for key in closure_tiles:
            new_tile = result.tiles[key]
            old_tile = self.tiles[key]
            if not np.array_equal(new_tile.volume_trace, old_tile.volume_trace):
                raise RuntimeError("IOR edit unexpectedly changed volume segments")
            if not np.array_equal(new_tile.volume_inside, old_tile.volume_inside):
                raise RuntimeError("IOR edit unexpectedly changed cone geometry")
            self.tiles[key] = new_tile
        self.scene = new_scene
        self._refresh_canonical()
        reused_tiles = sorted(set(self.tiles) - set(closure_tiles))
        reused_unchanged = self._check_reused_tiles(before, closure_tiles)
        if not reused_unchanged:
            raise RuntimeError("IOR edit changed a tile outside its closure")

        after = self.snapshot()
        return {
            "mode": "incremental_edit",
            "edit": {
                "kind": "material_ior",
                "surface_id": surface_id,
                "from": old_surface.ior,
                "to": ior,
            },
            "closure": {
                "tiles": closure_tiles,
                "volume_segment_count": 0,
                "volume_segments": [],
                "digest": _canonical_digest([]),
            },
            "reuse": {
                "tiles": reused_tiles,
                "tile_content_hashes_unchanged": reused_unchanged,
                "volume_segment_count": int(
                    self.width * self.height * self.scene.volume.max_steps
                ),
            },
            "actual_recomputed_tiles": actual_recomputed,
            "actual_recomputed_volume_segment_count": 0,
            "actual_matches_closure": actual_recomputed == sorted(closure_tiles),
            "outside_volume_trace_preserved": all(
                np.array_equal(self.tiles[key].volume_trace, old_volume_trace[key])
                for key in sorted(set(self.tiles) - set(closure_tiles))
            ),
            "scheduler": result.report,
            "frame_before": before.to_dict(),
            "frame_after": after.to_dict(),
        }

    def edit_scatter(
        self,
        new_scatter: float,
        threads: int = 4,
        queue_memory_budget_bytes: int | None = None,
        compute_budget_units: int | None = None,
    ) -> dict[str, Any]:
        if not self.rendered:
            raise fail(ErrorCode.BUDGET_DENIED, "render the frame before applying an incremental edit")
        scatter = validate_scatter(new_scatter)
        old_scatter = self.scene.light_cone.scatter
        segment_records = self._segment_closure()
        closure_tiles = sorted({str(record["tile_key"]) for record in segment_records})
        closure_digest = _canonical_digest(segment_records)
        segment_count = sum(int(record["count"]) for record in segment_records)
        before = self.snapshot()
        old_traces = {key: tile.volume_trace.copy() for key, tile in self.tiles.items()}
        old_inside = {key: tile.volume_inside.copy() for key, tile in self.tiles.items()}

        new_scene = self.scene.replace_cone_scatter(scatter)
        config = self._scheduler_config(threads, queue_memory_budget_bytes, compute_budget_units)
        result = execute_tiles(new_scene, self._specs_for_keys(closure_tiles), config)
        actual_recomputed = sorted(result.tiles)
        if actual_recomputed != closure_tiles:
            raise RuntimeError("scatter recompute set diverged from closure")

        intervals_by_tile: dict[str, list[tuple[int, int, int, int]]] = {}
        for record in segment_records:
            key = str(record["tile_key"])
            intervals_by_tile.setdefault(key, []).append(
                (
                    int(record["y"]),
                    int(record["x"]),
                    int(record["start_step"]),
                    int(record["end_step"]),
                )
            )

        outside_preserved = True
        actual_segment_records: list[dict[str, int | str]] = []
        for key in closure_tiles:
            new_tile = result.tiles[key]
            old_tile = self.tiles[key]
            if not np.array_equal(new_tile.volume_inside, old_inside[key]):
                raise RuntimeError("scatter edit unexpectedly changed cone geometry")
            merged_trace = old_traces[key].copy()
            affected_mask = np.zeros(new_tile.volume_trace.shape, dtype=bool)
            for y, x, start_step, end_step in intervals_by_tile.get(key, []):
                local_y = y - old_tile.y0
                local_x = x - old_tile.x0
                merged_trace[local_y, local_x, start_step : end_step + 1, :] = new_tile.volume_trace[
                    local_y, local_x, start_step : end_step + 1, :
                ]
                affected_mask[local_y, local_x, start_step : end_step + 1, :] = True
                actual_segment_records.append(
                    {
                        "tile_key": key,
                        "x": x,
                        "y": y,
                        "start_step": start_step,
                        "end_step": end_step,
                        "count": end_step - start_step + 1,
                    }
                )
            outside = ~affected_mask
            if np.any(outside):
                if not np.array_equal(merged_trace[outside], old_traces[key][outside]):
                    outside_preserved = False
            new_tile.volume_trace = merged_trace
            self.tiles[key] = new_tile

        self.scene = new_scene
        self._refresh_canonical()
        reused_tiles = sorted(set(self.tiles) - set(closure_tiles))
        reused_unchanged = self._check_reused_tiles(before, closure_tiles)
        if not reused_unchanged or not outside_preserved:
            raise RuntimeError("scatter edit changed content outside its closure")
        actual_segment_count = sum(int(record["count"]) for record in actual_segment_records)
        actual_segment_digest = _canonical_digest(actual_segment_records)
        segments_match = (
            actual_segment_count == segment_count
            and actual_segment_digest == closure_digest
        )
        tiles_match = actual_recomputed == closure_tiles
        if not segments_match:
            raise RuntimeError("scatter segment recompute set diverged from closure")

        after = self.snapshot()
        return {
            "mode": "incremental_edit",
            "edit": {
                "kind": "cone_scatter",
                "from": old_scatter,
                "to": scatter,
            },
            "closure": {
                "tiles": closure_tiles,
                "volume_segment_count": segment_count,
                "volume_segment_rows": segment_records,
                "digest": closure_digest,
            },
            "reuse": {
                "tiles": reused_tiles,
                "tile_content_hashes_unchanged": reused_unchanged,
                "volume_segment_count": int(
                    self.width * self.height * self.scene.volume.max_steps - segment_count
                ),
                "outside_segment_bytes_unchanged": outside_preserved,
            },
            "actual_recomputed_tiles": actual_recomputed,
            "actual_recomputed_volume_segment_count": actual_segment_count,
            "actual_volume_segment_digest": actual_segment_digest,
            "actual_matches_closure": tiles_match and segments_match,
            "outside_volume_trace_preserved": outside_preserved,
            "scheduler": result.report,
            "frame_before": before.to_dict(),
            "frame_after": after.to_dict(),
        }

    def edit_tile_size(self, new_tile_size: int) -> dict[str, Any]:
        if not self.rendered:
            raise fail(ErrorCode.BUDGET_DENIED, "render the frame before retiling")
        before = self.snapshot()
        old_tile_size = self.tile_size
        tile_size = validate_tile_size(new_tile_size, self.width, self.height)
        if tile_size == old_tile_size:
            return {
                "mode": "retile_reuse",
                "edit": {"kind": "tile_size", "from": old_tile_size, "to": tile_size},
                "structural_closure": {"tiles": sorted(self.tiles), "volume_segment_count": 0},
                "recompute_closure": {"tiles": [], "volume_segment_count": 0},
                "reused": {
                    "tiles": sorted(self.tiles),
                    "pixels": int(self.width * self.height),
                    "volume_segments": int(self.width * self.height * self.scene.volume.max_steps),
                },
                "actual_recomputed_tiles": [],
                "frame_before": before.to_dict(),
                "frame_after": before.to_dict(),
            }

        new_specs = plan_tiles(self.width, self.height, tile_size)
        new_tiles: dict[str, TileResult] = {}
        for spec in new_specs:
            image = self.image[spec.y0 : spec.y1, spec.x0 : spec.x1, :].copy()
            trace = self.trace[spec.y0 : spec.y1, spec.x0 : spec.x1, :, :].copy()
            inside = self.inside[spec.y0 : spec.y1, spec.x0 : spec.x1, :].copy()
            visibility = self.visibility[spec.y0 : spec.y1, spec.x0 : spec.x1].copy()
            new_tiles[spec.key] = TileResult(
                key=spec.key,
                tx=spec.tx,
                ty=spec.ty,
                x0=spec.x0,
                y0=spec.y0,
                x1=spec.x1,
                y1=spec.y1,
                image=image,
                visibility_mask=visibility,
                volume_trace=trace,
                volume_inside=inside,
                status="REUSED_RETILE",
            )
        self.tile_size = tile_size
        self.tiles = new_tiles
        self._refresh_canonical()
        after = self.snapshot()
        if before.image_sha256 != after.image_sha256 or before.trace_sha256 != after.trace_sha256:
            raise RuntimeError("retiling changed canonical content")
        return {
            "mode": "retile_reuse",
            "edit": {"kind": "tile_size", "from": old_tile_size, "to": tile_size},
            "structural_closure": {
                "tiles": sorted(new_tiles),
                "volume_segment_count": int(self.width * self.height * self.scene.volume.max_steps),
            },
            "recompute_closure": {
                "tiles": [],
                "volume_segment_count": 0,
                "actual_recomputed_tiles": [],
            },
            "reused": {
                "tiles": sorted(new_tiles),
                "pixels": int(self.width * self.height),
                "volume_segments": int(self.width * self.height * self.scene.volume.max_steps),
                "canonical_image_sha256_unchanged": before.image_sha256 == after.image_sha256,
                "canonical_volume_trace_sha256_unchanged": before.trace_sha256 == after.trace_sha256,
            },
            "actual_recomputed_tiles": [],
            "frame_before": before.to_dict(),
            "frame_after": after.to_dict(),
        }
