"""Incremental multi-resolution engine with precise dependency invalidation."""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import apply_command
from .dependencies import build_parent_map, lower_tile_refs, tiles_intersecting_rect
from .errors import (
    E_TILE_BUDGET_PENDING,
    E_TILE_DEPENDENCY_PENDING,
    E_TILE_FAILED,
    CommandError,
)
from .fixed import round_div, to_fixed
from .imageio import save_fixed_png
from .model import Change, ConfigBundle, EffectiveConfig
from .pyramid import UPSAMPLE_METADATA, bilinear_upsample, block_mean_downsample
from .repair import REPAIR_ALGORITHM, repair_tile
from .style import StyleMapper
from .util import (
    array_fingerprint,
    atomic_write_json,
    bool_fingerprint,
    sha256_json,
    stable_digest_parts,
    tile_sort_key,
)


ENGINE_VERSION = "inpaint-pyramid-engine-v1"


@dataclass
class TileMeta:
    level: int
    row: int
    col: int
    cache_key: str
    input_fingerprint: str
    guide_fingerprint: str
    parameter_fingerprint: str
    quant_levels: int
    lower_refs: tuple[tuple[int, int, int], ...]
    output_fingerprint: str
    mask_pixels: int
    mask_fingerprint: str
    mapped_pixels: int
    retained_mask_repair_pixels: int
    retained_source_pixels: int
    decision_fingerprint: str
    lut_fingerprint: str
    version: int = 1

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.level, self.row, self.col)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "row": self.row,
            "col": self.col,
            "cache_key": self.cache_key,
            "input_fingerprint": self.input_fingerprint,
            "guide_fingerprint": self.guide_fingerprint,
            "parameter_fingerprint": self.parameter_fingerprint,
            "quant_levels": self.quant_levels,
            "lower_refs": [list(ref) for ref in self.lower_refs],
            "output_fingerprint": self.output_fingerprint,
            "mask_pixels": self.mask_pixels,
            "mask_fingerprint": self.mask_fingerprint,
            "mapped_pixels": self.mapped_pixels,
            "retained_mask_repair_pixels": self.retained_mask_repair_pixels,
            "retained_source_pixels": self.retained_source_pixels,
            "decision_fingerprint": self.decision_fingerprint,
            "lut_fingerprint": self.lut_fingerprint,
            "version": self.version,
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "TileMeta":
        return cls(
            level=int(payload["level"]),
            row=int(payload["row"]),
            col=int(payload["col"]),
            cache_key=str(payload["cache_key"]),
            input_fingerprint=str(payload["input_fingerprint"]),
            guide_fingerprint=str(payload["guide_fingerprint"]),
            parameter_fingerprint=str(payload["parameter_fingerprint"]),
            quant_levels=int(payload["quant_levels"]),
            lower_refs=tuple(tuple(int(v) for v in ref) for ref in payload["lower_refs"]),
            output_fingerprint=str(payload["output_fingerprint"]),
            mask_pixels=int(payload["mask_pixels"]),
            mask_fingerprint=str(payload["mask_fingerprint"]),
            mapped_pixels=int(payload["mapped_pixels"]),
            retained_mask_repair_pixels=int(payload["retained_mask_repair_pixels"]),
            retained_source_pixels=int(payload["retained_source_pixels"]),
            decision_fingerprint=str(payload["decision_fingerprint"]),
            lut_fingerprint=str(payload["lut_fingerprint"]),
            version=int(payload.get("version", 1)),
        )


@dataclass
class TileComputation:
    image: np.ndarray
    meta: TileMeta


@dataclass
class CompositeResult:
    image: np.ndarray
    fingerprint: str
    contributions: list[dict[str, Any]]
    contribution_arrays: list[np.ndarray]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "contributions": self.contributions,
            "upsample": UPSAMPLE_METADATA,
        }


@dataclass
class ScheduleOutcome:
    command_id: str
    budget_limit: int
    budget_consumed: int
    budget_exhausted: bool
    status: str
    ledger: list[dict[str, Any]]
    invalidated: list[tuple[int, int, int]]
    recomputed: list[tuple[int, int, int]]
    reused: list[tuple[int, int, int]]
    pending: list[tuple[int, int, int]]
    failures: list[dict[str, Any]]
    elapsed_by_layer: dict[int, float] = field(default_factory=dict)

    def to_report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "budget": {
                "limit": self.budget_limit,
                "consumed": self.budget_consumed,
                "remaining": max(0, self.budget_limit - self.budget_consumed),
                "exhausted": self.budget_exhausted,
                "ledger": self.ledger,
            },
            "invalidated_tiles": [list(key) for key in self.invalidated],
            "recomputed_tiles": [list(key) for key in self.recomputed],
            "reused_tiles": [list(key) for key in self.reused],
            "pending_tiles": [list(key) for key in self.pending],
            "failures": self.failures,
            "elapsed_by_layer": {str(k): v for k, v in sorted(self.elapsed_by_layer.items())},
        }


class IncrementalEngine:
    def __init__(self, bundle: ConfigBundle, mutable: EffectiveConfig | None = None):
        self.bundle = bundle
        self.spec = bundle.spec
        self.mutable = mutable.clone() if mutable is not None else bundle.new_mutable()
        self.masks = self.mutable.all_masks(self.spec)
        self.source_layers = self._build_source_layers()
        self.layer_outputs = [
            np.zeros_like(source, dtype=np.int64) for source in self.source_layers
        ]
        self.style_mapper = StyleMapper(
            bundle.style_rgb, self.spec.fraction_bits, bundle.style_fingerprint
        )
        self.luts = self.style_mapper.prebuild(self.mutable.quant_levels_by_layer)
        self.lower_refs = {
            key: lower_tile_refs(self.spec, *key) for key in self.spec.all_tiles()
        }
        self.parent_map = build_parent_map(self.spec)
        self.valid_tiles: set[tuple[int, int, int]] = set()
        self.dirty_tiles: set[tuple[int, int, int]] = set()
        self.tile_meta: dict[tuple[int, int, int], TileMeta] = {}
        self.tile_versions: dict[tuple[int, int, int], int] = {}
        self.cursor = -1
        self.completed_commands: list[dict[str, Any]] = []
        self.layer_timings = {level: 0.0 for level in range(self.spec.levels)}
        self.failures: list[dict[str, Any]] = []
        self.composite: CompositeResult | None = None
        self.composite_stale = True
        self.concurrency_peak = 0
        self.resumed_from_version: int | None = None
        self.commands_hash: str | None = None

    @classmethod
    def build_full(
        cls,
        bundle: ConfigBundle,
        mutable: EffectiveConfig | None = None,
        cursor: int = -1,
        commands_hash: str | None = None,
    ) -> "IncrementalEngine":
        engine = cls(bundle, mutable)
        engine.cursor = cursor
        engine.commands_hash = commands_hash
        engine.bootstrap()
        return engine

    def _build_source_layers(self) -> list[np.ndarray]:
        layers: list[np.ndarray] = []
        current = to_fixed(self.bundle.source_rgb, self.spec.fraction_bits)
        layers.append(current)
        for level in range(1, self.spec.levels):
            current = block_mean_downsample(current, self.spec.factors[level])
            layers.append(current)
        return layers

    def bootstrap(self) -> ScheduleOutcome:
        self.valid_tiles.clear()
        self.tile_meta.clear()
        self.tile_versions.clear()
        self.dirty_tiles = set(self.spec.all_tiles())
        outcome = self._run_schedule(None, "bootstrap")
        if self.dirty_tiles:
            raise RuntimeError("bootstrap did not resolve all tiles")
        self.refresh_composite(force=True)
        return outcome

    def _parameter_payload(self, level: int) -> dict[str, Any]:
        return {
            "engine_version": ENGINE_VERSION,
            "level": level,
            "iterations": self.mutable.iterations_by_layer[level],
            "guide_strength_q": self.mutable.guide_strength_q[level],
            "style_weights_q": list(self.mutable.style_weights_q[level]),
            "quant_levels": self.mutable.quant_levels_by_layer[level],
            "mask_fill": self.mutable.mask_fill,
            "style_inside": self.mutable.style_inside,
            "style_outside": self.mutable.style_outside,
            "fraction_bits": self.spec.fraction_bits,
            "style_fingerprint": self.bundle.style_fingerprint,
            "repair_algorithm": REPAIR_ALGORITHM,
            "upsample": UPSAMPLE_METADATA,
        }

    def _guide_for_level(self, level: int) -> np.ndarray | None:
        if level == 0:
            return None
        width, height = self.spec.sizes[level]
        return bilinear_upsample(self.layer_outputs[level - 1], height, width)

    def _compute_tile(self, level: int, row: int, col: int, version: int) -> TileComputation:
        x0, y0, x1, y1 = self.spec.tile_bounds(level, row, col)
        source_patch = np.ascontiguousarray(self.source_layers[level][y0:y1, x0:x1])
        mask_patch = np.ascontiguousarray(self.masks[level][y0:y1, x0:x1])
        guide = self._guide_for_level(level)
        guide_patch = None if guide is None else np.ascontiguousarray(guide[y0:y1, x0:x1])
        refs = self.lower_refs[(level, row, col)]

        input_fp = sha256_json(
            {
                "level": level,
                "row": row,
                "col": col,
                "bounds": [x0, y0, x1, y1],
                "source": array_fingerprint(source_patch),
                "mask": bool_fingerprint(mask_patch),
                "mask_fill": self.mutable.mask_fill,
            }
        )
        guide_fp = sha256_json(
            {
                "guide": None if guide_patch is None else array_fingerprint(guide_patch),
                "lower_refs": [list(ref) for ref in refs],
            }
        )
        param_fp = sha256_json(self._parameter_payload(level))
        quant_levels = self.mutable.quant_levels_by_layer[level]
        cache_key = stable_digest_parts(
            "tile-cache-v1",
            level,
            row,
            col,
            input_fp,
            guide_fp,
            param_fp,
            quant_levels,
        )

        repair = repair_tile(
            source_patch,
            guide_patch,
            mask_patch,
            iterations=self.mutable.iterations_by_layer[level],
            guide_strength_q=self.mutable.guide_strength_q[level],
            scale=self.spec.scale,
            mask_fill=self.mutable.mask_fill,
        )
        lut, lut_fp = self.luts[level]
        styled = self.style_mapper.map_patch(
            repair.image,
            mask_patch,
            lut,
            lut_fp,
            self.mutable.style_weights_q[level],
            self.mutable.style_inside,
            self.mutable.style_outside,
            repair.retained_source_mask,
        )
        meta = TileMeta(
            level=level,
            row=row,
            col=col,
            cache_key=cache_key,
            input_fingerprint=input_fp,
            guide_fingerprint=guide_fp,
            parameter_fingerprint=param_fp,
            quant_levels=quant_levels,
            lower_refs=refs,
            output_fingerprint=array_fingerprint(styled.image),
            mask_pixels=int(mask_patch.sum()),
            mask_fingerprint=bool_fingerprint(mask_patch),
            mapped_pixels=styled.mapped_pixels,
            retained_mask_repair_pixels=styled.retained_mask_repair_pixels,
            retained_source_pixels=styled.retained_source_pixels,
            decision_fingerprint=styled.decision_fingerprint,
            lut_fingerprint=styled.lut_fingerprint,
            version=version,
        )
        return TileComputation(image=styled.image, meta=meta)

    def _run_schedule(self, budget: int | None, command_id: str) -> ScheduleOutcome:
        limit = len(self.dirty_tiles) if budget is None else max(0, int(budget))
        planned: list[tuple[int, int, int]] = []
        planned_set: set[tuple[int, int, int]] = set()
        for key in sorted(self.dirty_tiles, key=tile_sort_key):
            if len(planned) >= limit:
                break
            refs = self.lower_refs[key]
            if all(ref in self.valid_tiles or ref in planned_set for ref in refs):
                planned.append(key)
                planned_set.add(key)

        ledger: list[dict[str, Any]] = []
        remaining = limit
        for sequence, key in enumerate(planned):
            level, row, col = key
            ledger.append(
                {
                    "sequence": sequence,
                    "command_id": command_id,
                    "level": level,
                    "row": row,
                    "col": col,
                    "remaining_before": remaining,
                    "remaining_after": remaining - 1,
                }
            )
            remaining -= 1

        self.luts = self.style_mapper.prebuild(self.mutable.quant_levels_by_layer)
        recomputed: list[tuple[int, int, int]] = []
        failures: list[dict[str, Any]] = []
        elapsed_by_layer: dict[int, float] = {}

        for level in range(self.spec.levels):
            level_keys = [key for key in planned if key[0] == level]
            if not level_keys:
                continue
            ready: list[tuple[int, int, int]] = []
            for key in level_keys:
                if all(ref in self.valid_tiles for ref in self.lower_refs[key]):
                    ready.append(key)
                # Unready keys remain dirty; allocation budget was still consumed.
            if not ready:
                continue
            version_by_key = {
                key: self.tile_versions.get(key, 0) + 1 for key in ready
            }
            start = time.perf_counter()
            workers = min(self.bundle.threads, len(ready))
            self.concurrency_peak = max(self.concurrency_peak, workers)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"ipw-L{level}") as executor:
                futures = [
                    executor.submit(
                        self._compute_tile,
                        key[0],
                        key[1],
                        key[2],
                        version_by_key[key],
                    )
                    for key in ready
                ]
                for key, future in zip(ready, futures):
                    try:
                        computation = future.result()
                    except Exception as exc:  # task failures are reported and stay pending
                        failure = {
                            "code": E_TILE_FAILED,
                            "level": key[0],
                            "row": key[1],
                            "col": key[2],
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=3),
                        }
                        failures.append(failure)
                        self.failures.append(failure)
                        continue
                    x0, y0, x1, y1 = self.spec.tile_bounds(*key)
                    self.layer_outputs[key[0]][y0:y1, x0:x1] = computation.image
                    self.tile_meta[key] = computation.meta
                    self.tile_versions[key] = computation.meta.version
                    self.valid_tiles.add(key)
                    self.dirty_tiles.discard(key)
                    recomputed.append(key)
            elapsed = time.perf_counter() - start
            elapsed_by_layer[level] = elapsed_by_layer.get(level, 0.0) + elapsed
            self.layer_timings[level] += elapsed

        recomputed.sort(key=tile_sort_key)
        recomputed_set = set(recomputed)
        reused = sorted(self.valid_tiles - recomputed_set, key=tile_sort_key)
        pending = sorted(self.dirty_tiles, key=tile_sort_key)
        budget_exhausted = bool(pending) and len(planned) >= limit
        if not pending:
            status = "OK"
        elif budget_exhausted:
            status = E_TILE_BUDGET_PENDING
        else:
            status = E_TILE_DEPENDENCY_PENDING
        if not pending:
            self.refresh_composite()
        else:
            self.composite_stale = True
        return ScheduleOutcome(
            command_id=command_id,
            budget_limit=limit,
            budget_consumed=len(planned),
            budget_exhausted=budget_exhausted,
            status=status,
            ledger=ledger,
            invalidated=[],
            recomputed=recomputed,
            reused=reused,
            pending=pending,
            failures=failures,
            elapsed_by_layer=elapsed_by_layer,
        )

    def _propagate_upward(self, seed: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
        seen = set(seed)
        queue = list(seed)
        while queue:
            key = queue.pop()
            for parent in self.parent_map.get(key, ()):
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        return seen

    def _invalidate_keys(self, keys: set[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        newly_invalidated: list[tuple[int, int, int]] = []
        for key in sorted(keys, key=tile_sort_key):
            if key not in self.dirty_tiles:
                newly_invalidated.append(key)
            self.dirty_tiles.add(key)
            self.valid_tiles.discard(key)
        return newly_invalidated

    def _invalidate_change(self, change: Change) -> list[tuple[int, int, int]]:
        if not change.changed or change.kind == "noop" or change.layer is None:
            return []
        if change.kind == "mask_region":
            if change.rects:
                seeds: set[tuple[int, int, int]] = set()
                for rect in change.rects:
                    seeds |= tiles_intersecting_rect(self.spec, change.layer, rect)
            elif change.rect is None:
                seeds = {
                    key for key in self.spec.all_tiles() if key[0] == change.layer
                }
            else:
                seeds = tiles_intersecting_rect(self.spec, change.layer, change.rect)
        elif change.kind == "layer_params":
            seeds = {key for key in self.spec.all_tiles() if key[0] == change.layer}
        else:
            seeds = set()
        return self._invalidate_keys(self._propagate_upward(seeds))

    def process_command(self, raw_command: Any, index: int) -> dict[str, Any]:
        command_id = f"cmd-{index:04d}"
        raw_type = raw_command.get("type") if isinstance(raw_command, dict) else None
        raw_id = raw_command.get("id") if isinstance(raw_command, dict) else None
        entry: dict[str, Any] = {
            "index": index,
            "id": raw_id,
            "type": raw_type,
        }
        invalidated: list[tuple[int, int, int]] = []
        try:
            normalized, change = apply_command(self.mutable, raw_command, self.spec)
            if raw_id is not None:
                command_id = str(raw_id)
            if change.changed and change.layer is not None:
                if change.kind == "mask_region":
                    self.masks[change.layer] = self.mutable.mask_array(change.layer, self.spec)
                invalidated = self._invalidate_change(change)
            entry["status"] = "applied"
            entry["command"] = normalized
            entry["changed"] = bool(change.changed)
        except CommandError as exc:
            entry["status"] = "invalid_skipped"
            entry["error"] = exc.as_dict()
            entry["changed"] = False

        outcome = self._run_schedule(self.bundle.recompute_tile_budget, command_id)
        outcome.invalidated = sorted(invalidated, key=tile_sort_key)
        entry["schedule"] = outcome.to_report()
        entry["effective_config_hash"] = self.bundle.effective_config_hash(self.mutable)
        entry["composite_stale"] = self.composite_stale
        self.cursor = index
        self.completed_commands.append(entry)
        return entry

    def drain_pending(self, max_batches: int = 10000) -> list[ScheduleOutcome]:
        outcomes: list[ScheduleOutcome] = []
        batches = 0
        while self.dirty_tiles and batches < max_batches:
            outcome = self._run_schedule(
                self.bundle.recompute_tile_budget, f"drain-{batches:04d}"
            )
            outcome.invalidated = []
            outcomes.append(outcome)
            batches += 1
            if not outcome.recomputed and outcome.pending:
                break
        return outcomes

    def refresh_composite(self, force: bool = False) -> CompositeResult | None:
        if self.dirty_tiles and not force:
            self.composite_stale = True
            return self.composite
        weights = self.mutable.composition_weights_q
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("composition weights sum to zero")
        contributions: list[np.ndarray] = []
        contribution_meta: list[dict[str, Any]] = []
        weighted_sum = np.zeros(
            (self.spec.base_height, self.spec.base_width, 3), dtype=np.int64
        )
        for level, weight in enumerate(weights):
            upsampled = bilinear_upsample(
                self.layer_outputs[level], self.spec.base_height, self.spec.base_width
            )
            contribution = upsampled * np.int64(weight)
            contributions.append(contribution)
            weighted_sum += contribution
            contribution_meta.append(
                {
                    "level": level,
                    "weight_q": int(weight),
                    "weight": float(weight) / float(self.spec.scale),
                    "contribution_fingerprint": array_fingerprint(contribution),
                    "contribution_sum": [int(v) for v in contribution.sum(axis=(0, 1))],
                    "contribution_mean": [
                        float(v) / float(self.spec.base_width * self.spec.base_height)
                        for v in (contribution.sum(axis=(0, 1)) / self.spec.scale)
                    ],
                }
            )
        image = round_div(weighted_sum, total_weight).astype(np.int64, copy=False)
        self.composite = CompositeResult(
            image=image,
            fingerprint=array_fingerprint(image),
            contributions=contribution_meta,
            contribution_arrays=contributions,
        )
        self.composite_stale = False
        return self.composite

    def layer_fingerprints(self) -> list[str | None]:
        fingerprints: list[str | None] = []
        for level in range(self.spec.levels):
            keys = [
                (level, row, col)
                for row in range(self.spec.tile_rows(level))
                for col in range(self.spec.tile_cols(level))
            ]
            if all(key in self.valid_tiles for key in keys):
                fingerprints.append(
                    sha256_json(
                        {
                            "layer_fingerprint_version": "v1",
                            "level": level,
                            "downsample_factor": self.spec.factors[level],
                            "cumulative_factor": self.spec.cumulative_factors[level],
                            "content": array_fingerprint(self.layer_outputs[level]),
                            "upsample": UPSAMPLE_METADATA,
                        }
                    )
                )
            else:
                fingerprints.append(None)
        return fingerprints

    def _retention_summary(self, level: int) -> dict[str, Any]:
        mask = self.masks[level]
        retained_source = mask & (self.mutable.mask_fill == "preserve_source") & (
            self.mutable.style_inside == "repaired"
        )
        style_mapped_mask = mask & (self.mutable.style_inside == "mapped")
        style_mapped_outside = (~mask) & (self.mutable.style_outside == "mapped")
        retained_repair = ~style_mapped_mask & ~style_mapped_outside
        return {
            "mask_pixels": int(mask.sum()),
            "mask_fingerprint": bool_fingerprint(mask),
            "source_preserved_mask_pixels": int(retained_source.sum()),
            "source_preserved_mask_fingerprint": bool_fingerprint(retained_source),
            "style_mapped_mask_pixels": int(style_mapped_mask.sum()),
            "style_mapped_outside_pixels": int(style_mapped_outside.sum()),
            "repair_retained_pixels": int(retained_repair.sum()),
            "repair_retained_fingerprint": bool_fingerprint(retained_repair),
            "decision_basis": {
                "mask_fill": self.mutable.mask_fill,
                "style_inside": self.mutable.style_inside,
                "style_outside": self.mutable.style_outside,
            },
        }

    def layer_summary(self) -> list[dict[str, Any]]:
        fingerprints = self.layer_fingerprints()
        summaries: list[dict[str, Any]] = []
        for level in range(self.spec.levels):
            tiles: list[dict[str, Any]] = []
            for row in range(self.spec.tile_rows(level)):
                for col in range(self.spec.tile_cols(level)):
                    key = (level, row, col)
                    meta = self.tile_meta.get(key)
                    tiles.append(
                        {
                            "row": row,
                            "col": col,
                            "valid": key in self.valid_tiles,
                            "pending": key in self.dirty_tiles,
                            "version": self.tile_versions.get(key, 0),
                            "cache_key": None
                            if meta is None or key not in self.valid_tiles
                            else meta.cache_key,
                            "output_fingerprint": None
                            if meta is None or key not in self.valid_tiles
                            else meta.output_fingerprint,
                            "lower_refs": [
                                list(ref) for ref in self.lower_refs.get(key, ())
                            ],
                        }
                    )
            summaries.append(
                {
                    "level": level,
                    "size": list(self.spec.sizes[level]),
                    "downsample_factor": self.spec.factors[level],
                    "cumulative_factor": self.spec.cumulative_factors[level],
                    "tile_rows": self.spec.tile_rows(level),
                    "tile_cols": self.spec.tile_cols(level),
                    "fingerprint": fingerprints[level],
                    "parameters": {
                        "iterations": self.mutable.iterations_by_layer[level],
                        "guide_strength_q": self.mutable.guide_strength_q[level],
                        "guide_strength": self.mutable.guide_strength_q[level]
                        / float(self.spec.scale),
                        "style_weights_q": list(self.mutable.style_weights_q[level]),
                        "style_weights": [
                            value / float(self.spec.scale)
                            for value in self.mutable.style_weights_q[level]
                        ],
                        "quant_levels": self.mutable.quant_levels_by_layer[level],
                    },
                    "retention": self._retention_summary(level),
                    "tiles": tiles,
                }
            )
        return summaries

    def state_payload(self) -> dict[str, Any]:
        composite_fp = None if self.composite is None else self.composite.fingerprint
        return {
            "engine_version": ENGINE_VERSION,
            "base_config_hash": self.bundle.base_config_hash,
            "effective_config_hash": self.bundle.effective_config_hash(self.mutable),
            "source_fingerprint": self.bundle.source_fingerprint,
            "style_fingerprint": self.bundle.style_fingerprint,
            "commands_hash": self.commands_hash,
            "cursor": self.cursor,
            "effective_config": self.mutable.to_metadata(),
            "pending_tiles": [list(key) for key in sorted(self.dirty_tiles, key=tile_sort_key)],
            "valid_tiles": [
                self.tile_meta[key].to_metadata()
                for key in sorted(self.valid_tiles, key=tile_sort_key)
                if key in self.tile_meta
            ],
            "tile_versions": {
                f"{key[0]}/{key[1]}/{key[2]}": value
                for key, value in sorted(self.tile_versions.items())
            },
            "layer_fingerprints": self.layer_fingerprints(),
            "composite_fingerprint": composite_fp,
            "composite_stale": self.composite_stale,
        }

    def state_hash(self) -> str:
        return sha256_json(self.state_payload())

    @classmethod
    def restore(
        cls,
        bundle: ConfigBundle,
        state_payload: dict[str, Any],
        tile_loader: Callable[[int, int, int], np.ndarray],
        resumed_version: int,
    ) -> "IncrementalEngine":
        if state_payload.get("source_fingerprint") != bundle.source_fingerprint:
            raise ValueError("snapshot source image fingerprint mismatch")
        if state_payload.get("style_fingerprint") != bundle.style_fingerprint:
            raise ValueError("snapshot style reference fingerprint mismatch")
        mutable = EffectiveConfig.from_metadata(state_payload["effective_config"])
        engine = cls(bundle, mutable)
        engine.cursor = int(state_payload.get("cursor", -1))
        engine.commands_hash = state_payload.get("commands_hash")
        engine.resumed_from_version = resumed_version
        engine.valid_tiles.clear()
        engine.tile_meta.clear()
        engine.tile_versions.clear()
        engine.dirty_tiles.clear()
        for encoded_key, version in state_payload.get("tile_versions", {}).items():
            level_text, row_text, col_text = str(encoded_key).split("/")
            engine.tile_versions[(int(level_text), int(row_text), int(col_text))] = int(version)
        for payload in state_payload.get("valid_tiles", []):
            meta = TileMeta.from_metadata(payload)
            key = meta.key
            image = tile_loader(*key)
            expected_shape = (
                engine.spec.tile_size,
                engine.spec.tile_size,
                3,
            )
            if image.shape != expected_shape:
                raise ValueError(f"tile file has wrong shape for {key}: {image.shape}")
            if array_fingerprint(image) != meta.output_fingerprint:
                raise ValueError(f"tile file fingerprint mismatch for {key}")
            x0, y0, x1, y1 = engine.spec.tile_bounds(*key)
            engine.layer_outputs[key[0]][y0:y1, x0:x1] = image
            engine.tile_meta[key] = meta
            engine.tile_versions[key] = int(meta.version)
            engine.valid_tiles.add(key)
        engine.dirty_tiles = {
            tuple(int(v) for v in key)
            for key in state_payload.get("pending_tiles", [])
        }
        missing = set(engine.spec.all_tiles()) - engine.valid_tiles - engine.dirty_tiles
        engine.dirty_tiles |= missing
        engine.composite_stale = bool(state_payload.get("composite_stale", True))
        if not engine.dirty_tiles:
            engine.refresh_composite(force=True)
        return engine

    def export_outputs(self, output_dir: str | Any) -> dict[str, Any]:
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for level in range(self.spec.levels):
            target = output_path / f"layer_{level:02d}.png"
            save_fixed_png(target, self.layer_outputs[level], self.spec.fraction_bits)
            files.append(target.name)
        if self.composite is not None:
            target = output_path / "composite.png"
            save_fixed_png(target, self.composite.image, self.spec.fraction_bits)
            files.append(target.name)
            np.savez_compressed(
                output_path / "contributions.npz",
                **{
                    f"level_{level:02d}": array
                    for level, array in enumerate(self.composite.contribution_arrays)
                },
            )
            files.append("contributions.npz")
        atomic_write_json(
            output_path / "outputs.json",
            {
                "files": files,
                "layer_fingerprints": self.layer_fingerprints(),
                "composite_fingerprint": None
                if self.composite is None
                else self.composite.fingerprint,
                "composite_stale": self.composite_stale,
            },
        )
        return {
            "directory": str(output_path),
            "files": files,
            "composite_fingerprint": None if self.composite is None else self.composite.fingerprint,
            "composite_stale": self.composite_stale,
        }

    def comparison_metadata(self) -> dict[str, Any]:
        return {
            "layer_fingerprints": self.layer_fingerprints(),
            "tile_cache_keys": {
                f"{key[0]}/{key[1]}/{key[2]}": meta.cache_key
                for key, meta in sorted(self.tile_meta.items())
            },
            "tile_output_fingerprints": {
                f"{key[0]}/{key[1]}/{key[2]}": meta.output_fingerprint
                for key, meta in sorted(self.tile_meta.items())
            },
            "composite_fingerprint": None
            if self.composite is None
            else self.composite.fingerprint,
            "contribution_fingerprints": None
            if self.composite is None
            else [item["contribution_fingerprint"] for item in self.composite.contributions],
        }


def compare_engines(actual: IncrementalEngine, expected: IncrementalEngine) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    if actual.dirty_tiles:
        differences.append(
            {
                "kind": "pending_tiles",
                "tiles": [list(key) for key in sorted(actual.dirty_tiles, key=tile_sort_key)],
            }
        )
    for level in range(actual.spec.levels):
        if not np.array_equal(actual.layer_outputs[level], expected.layer_outputs[level]):
            differences.append({"kind": "layer_pixels", "level": level})
    actual_meta = actual.comparison_metadata()
    expected_meta = expected.comparison_metadata()
    if actual_meta["layer_fingerprints"] != expected_meta["layer_fingerprints"]:
        differences.append(
            {
                "kind": "layer_fingerprints",
                "actual": actual_meta["layer_fingerprints"],
                "expected": expected_meta["layer_fingerprints"],
            }
        )
    if actual_meta["tile_cache_keys"] != expected_meta["tile_cache_keys"]:
        differences.append({"kind": "tile_cache_keys"})
    if actual_meta["tile_output_fingerprints"] != expected_meta["tile_output_fingerprints"]:
        differences.append({"kind": "tile_output_fingerprints"})
    if actual_meta["composite_fingerprint"] != expected_meta["composite_fingerprint"]:
        differences.append(
            {
                "kind": "composite_fingerprint",
                "actual": actual_meta["composite_fingerprint"],
                "expected": expected_meta["composite_fingerprint"],
            }
        )
    if actual_meta["contribution_fingerprints"] != expected_meta["contribution_fingerprints"]:
        differences.append({"kind": "contribution_fingerprints"})
    if actual.composite is not None and expected.composite is not None:
        if not np.array_equal(actual.composite.image, expected.composite.image):
            differences.append({"kind": "composite_pixels"})
        for level, (left, right) in enumerate(
            zip(actual.composite.contribution_arrays, expected.composite.contribution_arrays)
        ):
            if not np.array_equal(left, right):
                differences.append({"kind": "contribution_pixels", "level": level})
    return {
        "equal": not differences,
        "differences": differences,
        "actual": actual_meta,
        "expected": expected_meta,
    }
