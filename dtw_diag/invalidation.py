"""Exact cell-level invalidation planning for an edited sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .align import (
    CellProduct,
    band_step_key,
    feasible_coordinates,
    feasible_predecessors,
    local_key_for,
)
from .band import Band
from .config import Config
from .quantization import QuantizedSequence


@dataclass
class InvalidationPlan:
    sequence_id: str
    invalid_cells: list[tuple[int, int]]
    invalid_reasons: dict[tuple[int, int], str]
    removed_cells: list[tuple[int, int]]
    reused_coords: list[tuple[int, int]]
    reused_count: int
    old_feasible_count: int
    new_feasible_count: int

    def reason_rows(self) -> list[dict[str, Any]]:
        return [
            {"row": row, "col": col, "reason": self.invalid_reasons[(row, col)]}
            for row, col in self.invalid_cells
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "invalid_cells": [list(item) for item in self.invalid_cells],
            "invalid_reasons": [
                {"row": row, "col": col, "reason": self.invalid_reasons[(row, col)]}
                for row, col in self.invalid_cells
            ],
            "removed_cells": [list(item) for item in self.removed_cells],
            "reused_cells": [list(item) for item in self.reused_coords],
            "reused_cell_count": self.reused_count,
            "old_feasible_cells": self.old_feasible_count,
            "new_feasible_cells": self.new_feasible_count,
            "invalidated_cells": len(self.invalid_cells),
            "removed_cell_count": len(self.removed_cells),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "InvalidationPlan":
        reasons = {
            (item["row"], item["col"]): item["reason"]
            for item in raw.get("invalid_reasons", [])
        }
        return cls(
            sequence_id=raw["sequence_id"],
            invalid_cells=[tuple(item) for item in raw.get("invalid_cells", [])],
            invalid_reasons=reasons,
            removed_cells=[tuple(item) for item in raw.get("removed_cells", [])],
            reused_coords=[
                tuple(item) for item in raw.get("reused_cells", [])
            ],
            reused_count=raw.get(
                "reused_cell_count",
                len(raw.get("reused_cells", []))
                if isinstance(raw.get("reused_cells"), list)
                else raw.get("reused_cells", 0),
            ),
            old_feasible_count=raw.get("old_feasible_cells", 0),
            new_feasible_count=raw.get("new_feasible_cells", 0),
        )


def build_invalidation_plan(
    sequence_id: str,
    old_cells: dict[tuple[int, int], CellProduct] | None,
    old_row_quantified: QuantizedSequence | None,
    old_col_quantified: QuantizedSequence | None,
    old_band: Band | None,
    old_config: Config | None,
    new_row_quantified: QuantizedSequence,
    new_col_quantified: QuantizedSequence,
    new_band: Band,
    new_config: Config,
) -> InvalidationPlan:
    new_coords = feasible_coordinates(new_band)
    new_coord_set = set(new_coords)
    old_coord_set = set(old_cells or {})
    removed = sorted(old_coord_set - new_coord_set)
    reasons: dict[tuple[int, int], str] = {}

    if (
        old_cells is None
        or old_band is None
        or old_row_quantified is None
        or old_col_quantified is None
        or old_config is None
    ):
        for coord in new_coords:
            reasons[coord] = "NEW_SEQUENCE"
    else:
        step_mode_changed = old_config.step_mode != new_config.step_mode
        for row, col in new_coords:
            if step_mode_changed:
                reasons[(row, col)] = "STEP_MODE_CHANGED"
                continue
            old_cell = old_cells.get((row, col))
            if old_cell is None:
                reasons[(row, col)] = "NEW_FEASIBLE_CELL"
                continue
            if local_key_for(
                sequence_id,
                row,
                col,
                old_row_quantified,
                old_col_quantified,
                old_config,
            ) != local_key_for(
                sequence_id,
                row,
                col,
                new_row_quantified,
                new_col_quantified,
                new_config,
            ):
                reasons[(row, col)] = "LOCAL_POINT_CHANGED"
                continue
            new_preds = tuple(
                (item[0], item[1])
                for item in feasible_predecessors(
                    new_config.step_mode,
                    row,
                    col,
                    new_band.row_count,
                    new_band.col_count,
                    new_band,
                )
            )
            if new_preds != tuple(old_cell.predecessors):
                reasons[(row, col)] = "PREDECESSOR_SET_CHANGED"
                continue
            new_band_step_key = band_step_key(
                new_band,
                row,
                new_config.step_mode,
                new_band.row_count,
                new_band.col_count,
            )
            if old_cell.band_step_key != new_band_step_key:
                reasons[(row, col)] = "BAND_STEP_KEY_CHANGED"

    # Build the new successor graph and propagate invalidation downstream.
    successors: dict[tuple[int, int], list[tuple[int, int]]] = {
        coord: [] for coord in new_coords
    }
    for row, col in new_coords:
        for pred_row, pred_col, _label in feasible_predecessors(
            new_config.step_mode,
            row,
            col,
            new_band.row_count,
            new_band.col_count,
            new_band,
        ):
            successors.setdefault((pred_row, pred_col), []).append((row, col))

    queue = sorted(reasons)
    cursor = 0
    while cursor < len(queue):
        coord = queue[cursor]
        cursor += 1
        for successor in successors.get(coord, []):
            if successor not in reasons:
                reasons[successor] = "DOWNSTREAM_OF_INVALID"
                queue.append(successor)

    invalid_cells = sorted(reasons)
    reused_coords = sorted(new_coord_set - set(invalid_cells))
    reused = len(reused_coords)
    return InvalidationPlan(
        sequence_id=sequence_id,
        invalid_cells=invalid_cells,
        invalid_reasons=reasons,
        removed_cells=removed,
        reused_coords=reused_coords,
        reused_count=reused,
        old_feasible_count=len(old_coord_set),
        new_feasible_count=len(new_coord_set),
    )
