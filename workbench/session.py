from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .commands import RuntimeState, apply_command
from .configuration import WorkbenchConfig
from .engine import (
    DONE_STATUSES,
    TileRecord,
    assemble_final,
    build_plan,
    empty_canvases,
    execute_tiles,
)
from .hashing import array_fingerprint
from .masks import Rect
from .snapshot import SnapshotStore
from .visualization import render_tile_status


class BudgetPendingError(Exception):
    """Control marker; CLI serializes E_TILE_BUDGET_PENDING without crashing."""


class WorkbenchSession:
    def __init__(
        self,
        config: WorkbenchConfig,
        pixels: np.ndarray,
        image_fingerprint: str,
        initial_rectangles: dict[str, Rect],
        state: RuntimeState | None = None,
    ) -> None:
        self.config = config
        self.pixels = pixels
        self.image_fingerprint = image_fingerprint
        self.initial_rectangles = dict(initial_rectangles)
        self.state = state or RuntimeState.initial(config, initial_rectangles)
        self.planned = build_plan(config, pixels, image_fingerprint, self.state)
        self.records: dict[str, TileRecord] = {}
        self.repaired_levels = empty_canvases(config)
        self.styled_levels = empty_canvases(config)
        self.pending: list[str] = []
        self.applied_cursor = 0
        self.completed_cursor = 0
        self.commit_seq = 0
        self.latest_report: dict[str, Any] = {}
        self.last_manifest: dict[str, Any] | None = None

    @classmethod
    def fresh(cls, config: WorkbenchConfig, pixels: np.ndarray, image_fingerprint: str,
              rectangles: dict[str, Rect]) -> "WorkbenchSession":
        return cls(config, pixels, image_fingerprint, rectangles)

    @classmethod
    def from_manifest(
        cls,
        config: WorkbenchConfig,
        pixels: np.ndarray,
        image_fingerprint: str,
        rectangles: dict[str, Rect],
        manifest: dict[str, Any],
        products: dict[str, np.ndarray],
    ) -> "WorkbenchSession":
        state = _state_from_manifest(config, manifest)
        session = cls(config, pixels, image_fingerprint, rectangles, state)
        session.applied_cursor = int(manifest.get("applied_command_cursor", manifest.get("command_cursor", 0)))
        session.completed_cursor = int(manifest.get("completed_command_cursor", manifest.get("command_cursor", 0)))
        session.commit_seq = int(manifest.get("commit_seq", 0))
        session.pending = list(manifest.get("pending", []))
        session.records = {
            tile_id: TileRecord(**{key: value for key, value in data.items()
                                   if key in TileRecord.__dataclass_fields__})
            for tile_id, data in manifest.get("tiles", {}).items()
        }
        for level in range(config.levels):
            repair_key = f"repaired_{level}"
            style_key = f"styled_{level}"
            if repair_key in products:
                session.repaired_levels[level] = products[repair_key].copy()
            if style_key in products:
                session.styled_levels[level] = products[style_key].copy()
        return session

    def apply(self, command: dict[str, Any]) -> None:
        self.state = apply_command(self.state, command)
        self.applied_cursor += 1
        self.planned = build_plan(self.config, self.pixels, self.image_fingerprint, self.state)

    def run_batch(
        self,
        budget: int | None = None,
        max_workers: int | None = None,
        injections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if budget is None:
            budget = self.config.tile_budget
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            from .errors import WorkbenchError
            raise WorkbenchError("E_CONFIG_INVALID", "batch budget must be a positive integer", {"budget": budget})

        self.planned = build_plan(self.config, self.pixels, self.image_fingerprint, self.state)
        order = self.planned.order
        reused: list[str] = []
        invalid: list[str] = []
        for tile_id in order:
            old = self.records.get(tile_id)
            desired_key = self.planned.plans[tile_id].cache_key
            if (
                old is not None
                and old.cache_key == desired_key
                and old.status in DONE_STATUSES
                and old.style_fingerprint
            ):
                reused.append(tile_id)
            else:
                invalid.append(tile_id)

        selected = invalid[:budget]
        pending = invalid[budget:]
        workers = self.config.max_workers if max_workers is None else max_workers
        execution = execute_tiles(
            self.config,
            self.planned,
            self.state,
            selected,
            repaired_canvases=self.repaired_levels,
            styled_canvases=self.styled_levels,
            existing_records=self.records,
            max_workers=workers,
            injections=injections,
        )
        self.repaired_levels = execution.repaired_levels
        self.styled_levels = execution.styled_levels

        next_records: dict[str, TileRecord] = {}
        for tile_id in reused:
            record = self.records[tile_id]
            record.status = "reused"
            record.error = None
            next_records[tile_id] = record
        next_records.update(execution.records)
        for tile_id in pending:
            plan = self.planned.plans[tile_id]
            next_records[tile_id] = TileRecord(
                tile_id=tile_id,
                level=plan.level,
                row=plan.row,
                col=plan.col,
                cache_key=plan.cache_key,
                version=plan.cache_key[:16],
                dependencies=list(plan.dependencies),
                input_fingerprint=plan.input_fingerprint,
                guide_fingerprint=plan.guide_fingerprint,
                params_fingerprint=plan.params_fingerprint,
                quant_fingerprint=plan.quant_fingerprint,
                lut_fingerprint=plan.lut_fingerprint,
                status="pending",
            )
        self.records = next_records
        self.pending = pending

        if not pending and not execution.failed:
            self.completed_cursor = self.applied_cursor

        status = "complete"
        code = "OK"
        if pending:
            status = "E_TILE_BUDGET_PENDING"
            code = "E_TILE_BUDGET_PENDING"
        elif execution.failed:
            status = "failed"
            code = "E_INJECTION_FAILURES"

        report = {
            "ok": not pending and not execution.failed,
            "status": status,
            "code": code,
            "budget_limit": budget,
            "budget_charged": selected,
            "budget_used": len(selected),
            "budget_remaining": max(0, budget - len(selected)),
            "invalid_tiles": invalid,
            "reused": reused,
            "recomputed": execution.recomputed,
            "pending": pending,
            "failed": execution.failed,
            "task_order": execution.task_order,
            "injection_hits": execution.injection_hits,
            "max_observed_workers": execution.max_observed_workers,
            "reused_entries": [self._entry(tile_id, "reused") for tile_id in reused],
            "recomputed_entries": [self._entry(tile_id, "recomputed") for tile_id in execution.recomputed],
            "pending_entries": [self._entry(tile_id, "pending") for tile_id in pending],
            "tile_count": len(order),
            "command_cursor": self.completed_cursor,
            "applied_command_cursor": self.applied_cursor,
            "completed_command_cursor": self.completed_cursor,
        }
        self.latest_report = report
        return report

    def _entry(self, tile_id: str, disposition: str) -> dict[str, Any]:
        plan = self.planned.plans[tile_id]
        record = self.records.get(tile_id)
        return {
            "tile_id": tile_id,
            "disposition": disposition,
            "cache_key": plan.cache_key,
            "version": plan.cache_key[:16],
            "recorded_version": record.version if record else None,
            "dependencies": list(plan.dependencies),
        }

    def status_image(self) -> tuple[np.ndarray, str]:
        final = assemble_final(self.config, self.styled_levels)
        if self.pending or self.latest_report.get("failed"):
            return render_tile_status(
                self.config, final, self.pending, self.latest_report.get("failed", [])
            ), "partial_pending.png"
        return final, "final.png"

    def manifest(self, report: dict[str, Any] | None = None) -> dict[str, Any]:
        report = report or self.latest_report
        final = assemble_final(self.config, self.styled_levels)
        image_name = "final.png"
        if self.pending or report.get("failed"):
            image_name = "partial_pending.png"
        repair_done = sum(1 for record in self.records.values() if record.repair_fingerprint)
        style_done = sum(1 for record in self.records.values() if record.style_fingerprint)
        total_tiles = len(self.planned.order)
        has_blocker = bool(self.pending or report.get("failed"))
        return {
            "manifest_version": 1,
            "global_version": self.config.global_version,
            "config_hash": self.config.config_hash,
            "quantization_levels": list(self.state.quant_levels),
            "fractional_bits": list(self.state.fractional_bits),
            "command_cursor": self.completed_cursor,
            "completed_command_cursor": self.completed_cursor,
            "applied_command_cursor": self.applied_cursor,
            "commit_seq": self.commit_seq,
            "status": report.get("status", "complete"),
            "code": report.get("code", "OK"),
            "runtime_state_fingerprint": self.state.fingerprint(),
            "runtime_state": _state_to_manifest(self.state),
            "layer_fingerprints": list(self.planned.pyramid.layer_fingerprints),
            "pyramid_metadata": self.planned.pyramid.metadata,
            "layer_pixel_fingerprints": [array_fingerprint(level) for level in self.styled_levels],
            "final_fingerprint": array_fingerprint(final),
            "stages": {
                "load_validation": {
                    "status": "complete",
                    "config_hash": self.config.config_hash,
                    "image_fingerprint": self.image_fingerprint,
                    "mask_rectangle_count": len(self.state.rectangles),
                },
                "pyramid_build": {
                    "status": "complete",
                    "level_count": self.config.levels,
                    "layer_fingerprints": list(self.planned.pyramid.layer_fingerprints),
                    "tie_break": self.planned.pyramid.metadata["tie_break"],
                },
                "repair": {
                    "status": "blocked" if repair_done < total_tiles and has_blocker else "complete",
                    "complete_tiles": repair_done,
                    "total_tiles": total_tiles,
                },
                "style_transfer": {
                    "status": "blocked" if style_done < total_tiles and has_blocker else "complete",
                    "complete_tiles": style_done,
                    "total_tiles": total_tiles,
                    "quantization_levels": list(self.state.quant_levels),
                },
                "composite": {
                    "status": "pending" if self.pending else ("failed" if report.get("failed") else "complete"),
                    "png": image_name,
                    "fingerprint": array_fingerprint(final),
                },
            },
            "tile_order": list(self.planned.order),
            "tiles": {tile_id: record.to_manifest() for tile_id, record in self.records.items()},
            "pending": list(self.pending),
            "reused": list(report.get("reused", [])),
            "recomputed": list(report.get("recomputed", [])),
            "budget_charged": list(report.get("budget_charged", [])),
            "failed": list(report.get("failed", [])),
            "injection_hits": list(report.get("injection_hits", [])),
            "budget": {
                "limit": report.get("budget_limit", self.config.tile_budget),
                "used": report.get("budget_used", 0),
                "remaining": report.get("budget_remaining", 0),
            },
            "composite_png": image_name,
        }

    def commit(self, store: SnapshotStore) -> dict[str, Any]:
        self.commit_seq += 1
        manifest = self.manifest()
        arrays: dict[str, np.ndarray] = {}
        for level in range(self.config.levels):
            arrays[f"source_{level}"] = self.planned.pyramid.image_levels[level]
            arrays[f"mask_{level}"] = self.planned.pyramid.mask_levels[level].astype(np.uint8)
            arrays[f"repaired_{level}"] = self.repaired_levels[level]
            arrays[f"styled_{level}"] = self.styled_levels[level]
        status_pixels, image_name = self.status_image()
        images = {image_name: status_pixels}
        if image_name != "final.png":
            images["last_canvas.png"] = assemble_final(self.config, self.styled_levels)
        committed = store.commit(manifest, arrays, images)
        self.last_manifest = committed
        return committed


def _state_to_manifest(state: RuntimeState) -> dict[str, Any]:
    return {
        "rectangles": [rect.to_json() for rect in state.rectangles.values()],
        "iterations": list(state.iterations),
        "guidance": list(state.guidance),
        "style_strength": list(state.style_strength),
        "style_weights": [list(weights) for weights in state.style_weights],
        "quant_levels": list(state.quant_levels),
        "fractional_bits": list(state.fractional_bits),
    }


def _state_from_manifest(config: WorkbenchConfig, manifest: dict[str, Any]) -> RuntimeState:
    data = manifest.get("runtime_state")
    if not isinstance(data, dict):
        raise WorkbenchSnapshotStateError("snapshot is missing runtime_state")
    rectangles = {}
    for raw in data.get("rectangles", []):
        rect = Rect(
            id=str(raw["id"]), x=int(raw["x"]), y=int(raw["y"]),
            width=int(raw["width"]), height=int(raw["height"]),
        )
        if rect.x < 0 or rect.y < 0 or rect.x2 > config.width or rect.y2 > config.height:
            from .errors import WorkbenchError
            raise WorkbenchError("E_SNAPSHOT_STALE", "snapshot rectangle is incompatible with current image")
        rectangles[rect.id] = rect
    return RuntimeState(
        rectangles=rectangles,
        iterations=tuple(int(v) for v in data["iterations"]),
        guidance=tuple(float(v) for v in data["guidance"]),
        style_strength=tuple(float(v) for v in data["style_strength"]),
        style_weights=tuple((float(w[0]), float(w[1]), float(w[2])) for w in data["style_weights"]),
        quant_levels=tuple(int(v) for v in data["quant_levels"]),
        fractional_bits=tuple(int(v) for v in data["fractional_bits"]),
    )


class WorkbenchSnapshotStateError(Exception):
    pass
