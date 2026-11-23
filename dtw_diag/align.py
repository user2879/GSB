"""Band-constrained fixed-point DTW kernel and result finalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .band import Band
from .config import Config
from .errors import E_NO_PATH, E_PATH_LENGTH_REJECTED, W_PATH_TRUNCATED
from .fingerprints import canonical_json, fingerprint, task_key
from .quantization import QuantizedSequence


SYMMETRIC_ORDER = ("DIAG", "UP", "LEFT")
ASYMMETRIC_ORDER = ("DIAG", "UP")
ASYMMETRIC_HORIZONTAL_ORDER = ("DIAG", "LEFT")


def transition_order(step_mode: str, row_count: int, col_count: int) -> tuple[str, ...]:
    if step_mode == "symmetric":
        return SYMMETRIC_ORDER
    if step_mode == "asymmetric":
        return ASYMMETRIC_ORDER if row_count >= col_count else ASYMMETRIC_HORIZONTAL_ORDER
    raise ValueError(f"unknown step mode: {step_mode}")


def nominal_predecessors(
    step_mode: str,
    row: int,
    col: int,
    row_count: int,
    col_count: int,
) -> tuple[tuple[int, int, str], ...]:
    if row == 0 and col == 0:
        return ()
    order = transition_order(step_mode, row_count, col_count)
    result: list[tuple[int, int, str]] = []
    for label in order:
        if label == "DIAG":
            candidate = (row - 1, col - 1)
        elif label == "UP":
            candidate = (row - 1, col)
        else:
            candidate = (row, col - 1)
        if candidate[0] >= 0 and candidate[1] >= 0:
            result.append((candidate[0], candidate[1], label))
    return tuple(result)


def feasible_predecessors(
    step_mode: str,
    row: int,
    col: int,
    row_count: int,
    col_count: int,
    band: Band,
) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        item
        for item in nominal_predecessors(step_mode, row, col, row_count, col_count)
        if band.contains(item[0], item[1])
    )


def local_cost_for(
    row: int,
    col: int,
    row_quantified: QuantizedSequence,
    col_quantified: QuantizedSequence,
    config: Config,
) -> int:
    left = row_quantified.points[row]
    right = col_quantified.points[col]
    if left.missing or right.missing:
        return config.missing_penalty
    assert left.fixed_units is not None and right.fixed_units is not None
    return config.distance_coefficient * abs(left.fixed_units - right.fixed_units)


def local_key_for(
    sequence_id: str,
    row: int,
    col: int,
    row_quantified: QuantizedSequence,
    col_quantified: QuantizedSequence,
    config: Config,
) -> str:
    left = row_quantified.points[row]
    right = col_quantified.points[col]
    return fingerprint(
        {
            "stage": "local",
            "sequence_id": sequence_id,
            "row": row,
            "col": col,
            "row_point_key": left.point_key,
            "col_point_key": right.point_key,
            "row_missing": left.missing,
            "col_missing": right.missing,
            "distance_coefficient": config.distance_coefficient,
            "missing_penalty": config.missing_penalty,
            "fixed_units": [left.fixed_units, right.fixed_units],
        }
    )


def band_step_key(
    band: Band,
    row: int,
    step_mode: str,
    row_count: int,
    col_count: int,
) -> str:
    row_band = band.row(row)
    return fingerprint(
        {
            "stage": "band_step",
            "row": row,
            "row_band": row_band.as_key_data(),
            "step_mode": step_mode,
            "transition_order": transition_order(step_mode, row_count, col_count),
            "row_count": row_count,
            "col_count": col_count,
        }
    )


@dataclass(frozen=True)
class CellProduct:
    row: int
    col: int
    local_cost: int
    cost: int | None
    backpointer: str | None
    backpointer_coord: tuple[int, int] | None
    predecessors: tuple[tuple[int, int], ...]
    predecessor_entries: tuple[dict[str, Any], ...]
    local_key: str
    band_step_key: str
    cache_key: str
    content_key: str
    task_key: str
    reads: tuple[dict[str, Any], ...]

    @property
    def reachable(self) -> bool:
        return self.cost is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "local_cost": self.local_cost,
            "reachable": self.reachable,
            "cost": self.cost,
            "backpointer": self.backpointer,
            "backpointer_coord": list(self.backpointer_coord)
            if self.backpointer_coord is not None
            else None,
            "predecessors": [list(item) for item in self.predecessors],
            "predecessor_entries": list(self.predecessor_entries),
            "local_key": self.local_key,
            "band_step_key": self.band_step_key,
            "cache_key": self.cache_key,
            "content_key": self.content_key,
            "task_key": self.task_key,
            "reads": list(self.reads),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CellProduct":
        bpc = raw.get("backpointer_coord")
        return cls(
            row=raw["row"],
            col=raw["col"],
            local_cost=raw["local_cost"],
            cost=raw.get("cost"),
            backpointer=raw.get("backpointer"),
            backpointer_coord=tuple(bpc) if bpc is not None else None,
            predecessors=tuple(tuple(item) for item in raw.get("predecessors", [])),
            predecessor_entries=tuple(raw.get("predecessor_entries", [])),
            local_key=raw["local_key"],
            band_step_key=raw["band_step_key"],
            cache_key=raw["cache_key"],
            content_key=raw["content_key"],
            task_key=raw["task_key"],
            reads=tuple(raw.get("reads", [])),
        )


def compute_cell(
    sequence_id: str,
    row: int,
    col: int,
    row_quantified: QuantizedSequence,
    col_quantified: QuantizedSequence,
    band: Band,
    config: Config,
    cells: dict[tuple[int, int], CellProduct],
) -> CellProduct:
    if not band.contains(row, col):
        raise ValueError(f"cell ({row},{col}) is outside the feasible band")

    local_cost = local_cost_for(row, col, row_quantified, col_quantified, config)
    local_key = local_key_for(
        sequence_id, row, col, row_quantified, col_quantified, config
    )
    row_key = band_step_key(
        band, row, config.step_mode, band.row_count, band.col_count
    )
    nominal = nominal_predecessors(
        config.step_mode, row, col, band.row_count, band.col_count
    )
    feasible = feasible_predecessors(
        config.step_mode, row, col, band.row_count, band.col_count, band
    )
    predecessor_entries: list[dict[str, Any]] = []
    predecessor_coords: list[tuple[int, int]] = []
    best_cost: int | None = None
    best_label: str | None = None
    best_coord: tuple[int, int] | None = None

    feasible_map = {(item[0], item[1]): item[2] for item in feasible}
    for pred_row, pred_col, label in nominal:
        in_band = (pred_row, pred_col) in feasible_map
        predecessor = cells.get((pred_row, pred_col)) if in_band else None
        if in_band:
            predecessor_coords.append((pred_row, pred_col))
        if in_band and predecessor is None:
            raise ValueError(
                f"missing predecessor cell ({pred_row},{pred_col}) for ({row},{col})"
            )
        content_key = predecessor.content_key if predecessor is not None else None
        predecessor_entries.append(
            {
                "row": pred_row,
                "col": pred_col,
                "label": label,
                "in_band": in_band,
                "content_key": content_key,
            }
        )
        if predecessor is not None and predecessor.cost is not None:
            candidate = predecessor.cost + local_cost
            if best_cost is None or candidate < best_cost:
                best_cost = candidate
                best_label = label
                best_coord = (pred_row, pred_col)

    if row == 0 and col == 0:
        cost: int | None = local_cost
        backpointer: str | None = "START"
        backpointer_coord: tuple[int, int] | None = None
    elif best_cost is None:
        cost = None
        backpointer = None
        backpointer_coord = None
    else:
        cost = best_cost
        backpointer = best_label
        backpointer_coord = best_coord

    content_key = fingerprint(
        {
            "stage": "cell_content",
            "sequence_id": sequence_id,
            "row": row,
            "col": col,
            "local_cost": local_cost,
            "cost": cost,
            "backpointer": backpointer,
            "backpointer_coord": list(backpointer_coord)
            if backpointer_coord is not None
            else None,
            "local_key": local_key,
            "band_step_key": row_key,
            "predecessor_entries": predecessor_entries,
        }
    )
    cache_key = fingerprint(
        {
            "stage": "cell_cache",
            "sequence_id": sequence_id,
            "row": row,
            "col": col,
            "local_key": local_key,
            "band_step_key": row_key,
            "predecessor_content_keys": [entry["content_key"] for entry in predecessor_entries],
        }
    )
    reads = (
        {"sequence": "row", "index": row},
        {"sequence": "col", "index": col},
    )
    return CellProduct(
        row=row,
        col=col,
        local_cost=local_cost,
        cost=cost,
        backpointer=backpointer,
        backpointer_coord=backpointer_coord,
        predecessors=tuple(predecessor_coords),
        predecessor_entries=tuple(predecessor_entries),
        local_key=local_key,
        band_step_key=row_key,
        cache_key=cache_key,
        content_key=content_key,
        task_key=task_key("cell", sequence_id, row, col),
        reads=reads,
    )


def feasible_coordinates(band: Band) -> list[tuple[int, int]]:
    coordinates: list[tuple[int, int]] = []
    for row_band in band.rows:
        if row_band.empty:
            continue
        for col in range(row_band.col_start, row_band.col_end + 1):
            coordinates.append((row_band.row, col))
    return coordinates


@dataclass
class AlignmentResult:
    sequence_id: str
    status: str
    total_cost: int | None
    raw_total_cost: int | None
    path: tuple[tuple[int, int], ...]
    raw_path_length: int
    path_length: int
    truncated: bool
    cells: tuple[CellProduct, ...]
    segments: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    failure_code: str | None
    failure_message: str | None
    result_key: str
    path_key: str
    segment_key: str
    band: dict[str, Any] = field(default_factory=dict)
    out_of_band_cells: tuple[tuple[int, int], ...] = ()
    feasible_cell_count: int = 0
    out_of_band_cell_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "status": self.status,
            "total_cost": self.total_cost,
            "raw_total_cost": self.raw_total_cost,
            "path": [list(item) for item in self.path],
            "raw_path_length": self.raw_path_length,
            "path_length": self.path_length,
            "truncated": self.truncated,
            "cells": [cell.to_dict() for cell in self.cells],
            "segments": list(self.segments),
            "warnings": list(self.warnings),
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
            "result_key": self.result_key,
            "path_key": self.path_key,
            "segment_key": self.segment_key,
            "band": self.band,
            "out_of_band_cells": [list(item) for item in self.out_of_band_cells],
            "feasible_cell_count": self.feasible_cell_count,
            "out_of_band_cell_count": self.out_of_band_cell_count,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AlignmentResult":
        return cls(
            sequence_id=raw["sequence_id"],
            status=raw["status"],
            total_cost=raw.get("total_cost"),
            raw_total_cost=raw.get("raw_total_cost"),
            path=tuple(tuple(item) for item in raw.get("path", [])),
            raw_path_length=raw.get("raw_path_length", 0),
            path_length=raw.get("path_length", 0),
            truncated=raw.get("truncated", False),
            cells=tuple(CellProduct.from_dict(item) for item in raw.get("cells", [])),
            segments=tuple(raw.get("segments", [])),
            warnings=tuple(raw.get("warnings", [])),
            failure_code=raw.get("failure_code"),
            failure_message=raw.get("failure_message"),
            result_key=raw["result_key"],
            path_key=raw.get("path_key", ""),
            segment_key=raw.get("segment_key", ""),
            band=raw.get("band", {}),
            out_of_band_cells=tuple(
                tuple(item) for item in raw.get("out_of_band_cells", [])
            ),
            feasible_cell_count=raw.get("feasible_cell_count", 0),
            out_of_band_cell_count=raw.get("out_of_band_cell_count", 0),
        )


def _trace_path(
    cells: dict[tuple[int, int], CellProduct], row_count: int, col_count: int
) -> tuple[list[tuple[int, int]] | None, str | None]:
    end = (row_count - 1, col_count - 1)
    terminal = cells.get(end)
    if terminal is None or terminal.cost is None:
        return None, f"no reachable alignment path ends at {list(end)}"
    reverse_path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    current = end
    while True:
        if current in seen:
            return None, "backpointer cycle detected"
        seen.add(current)
        reverse_path.append(current)
        cell = cells.get(current)
        if cell is None:
            return None, f"backpointer references missing cell {list(current)}"
        if cell.backpointer == "START":
            break
        if cell.backpointer_coord is None:
            return None, f"cell {list(current)} has no backpointer"
        current = cell.backpointer_coord
    if current != (0, 0):
        return None, "backpointer chain does not reach (0, 0)"
    reverse_path.reverse()
    return reverse_path, None


def _build_segments(
    path: Iterable[tuple[int, int]],
    cells: dict[tuple[int, int], CellProduct],
    row_quantified: QuantizedSequence,
    col_quantified: QuantizedSequence,
) -> tuple[dict[str, Any], ...]:
    path_list = list(path)
    if not path_list:
        return ()
    segments: list[dict[str, Any]] = []
    start = 0
    current_key = (
        row_quantified.segment_ids[path_list[0][0]],
        col_quantified.segment_ids[path_list[0][1]],
    )
    for index in range(1, len(path_list) + 1):
        next_key = None
        if index < len(path_list):
            row, col = path_list[index]
            next_key = (
                row_quantified.segment_ids[row],
                col_quantified.segment_ids[col],
            )
        if next_key == current_key and index < len(path_list):
            continue
        segment_path = path_list[start:index]
        ordered = sorted(segment_path, key=lambda item: (item[0], item[1]))
        local_cost = sum(cells[coord].local_cost for coord in ordered)
        row_ids = sorted(
            {canonical_json(row_quantified.segment_ids[item[0]]) for item in ordered}
        )
        col_ids = sorted(
            {canonical_json(col_quantified.segment_ids[item[1]]) for item in ordered}
        )
        segments.append(
            {
                "path_start_index": start,
                "path_end_index": index - 1,
                "row_start": segment_path[0][0],
                "row_end": segment_path[-1][0],
                "col_start": segment_path[0][1],
                "col_end": segment_path[-1][1],
                "row_segment_ids": row_ids,
                "col_segment_ids": col_ids,
                "cell_count": len(segment_path),
                "local_cost": local_cost,
                "sum_order": "row_asc_col_asc",
            }
        )
        start = index
        current_key = next_key
    return tuple(segments)


def finalize_alignment(
    sequence_id: str,
    row_quantified: QuantizedSequence,
    col_quantified: QuantizedSequence,
    band: Band,
    cells: dict[tuple[int, int], CellProduct],
    config: Config,
) -> AlignmentResult:
    ordered_cells = tuple(cells[coord] for coord in sorted(cells))
    out_of_band = tuple(
        (row, col)
        for row in range(band.row_count)
        for col in range(band.col_count)
        if not band.contains(row, col)
    )
    band_metadata = band.to_dict()
    raw_path, trace_error = _trace_path(cells, band.row_count, band.col_count)
    base = {
        "sequence_id": sequence_id,
        "cell_manifest_key": fingerprint(
            {
                "stage": "cell_manifest",
                "cells": [
                    {"row": cell.row, "col": cell.col, "content_key": cell.content_key}
                    for cell in ordered_cells
                ],
            }
        ),
        "row_quant_fingerprint": row_quantified.fingerprint,
        "col_quant_fingerprint": col_quantified.fingerprint,
        "band_fingerprint": band.fingerprint,
        "alignment_fingerprint": config.alignment_fingerprint(),
        "max_path_length": config.max_path_length,
        "path_length_policy": config.path_length_policy,
    }
    if raw_path is None:
        path_key = fingerprint(
            {
                **base,
                "status": "rejected",
                "error": E_NO_PATH,
                "path": [],
                "path_length": 0,
            }
        )
        segment_key = fingerprint(
            {"stage": "segment_summary", "path_key": path_key, "segments": []}
        )
        result_key = fingerprint(
            {
                "stage": "alignment_result",
                "path_key": path_key,
                "segment_key": segment_key,
            }
        )
        return AlignmentResult(
            sequence_id=sequence_id,
            status="rejected",
            total_cost=None,
            raw_total_cost=None,
            path=(),
            raw_path_length=0,
            path_length=0,
            truncated=False,
            cells=ordered_cells,
            segments=(),
            warnings=(),
            failure_code=E_NO_PATH,
            failure_message=trace_error,
            result_key=result_key,
            path_key=path_key,
            segment_key=segment_key,
            band=band_metadata,
            out_of_band_cells=out_of_band,
            feasible_cell_count=len(ordered_cells),
            out_of_band_cell_count=len(out_of_band),
        )

    terminal = cells.get((band.row_count - 1, band.col_count - 1))
    raw_total_cost = terminal.cost if terminal is not None else None
    warnings: list[dict[str, Any]] = []
    truncated = False
    output_path = list(raw_path)
    if len(raw_path) > config.max_path_length:
        if config.path_length_policy == "reject":
            path_key = fingerprint(
                {
                    **base,
                    "status": "rejected",
                    "error": E_PATH_LENGTH_REJECTED,
                    "raw_path_length": len(raw_path),
                    "path": [],
                    "path_length": 0,
                }
            )
            segment_key = fingerprint(
                {"stage": "segment_summary", "path_key": path_key, "segments": []}
            )
            result_key = fingerprint(
                {
                    "stage": "alignment_result",
                    "path_key": path_key,
                    "segment_key": segment_key,
                }
            )
            return AlignmentResult(
                sequence_id=sequence_id,
                status="rejected",
                total_cost=None,
                raw_total_cost=raw_total_cost,
                path=(),
                raw_path_length=len(raw_path),
                path_length=0,
                truncated=False,
                cells=ordered_cells,
                segments=(),
                warnings=(),
                failure_code=E_PATH_LENGTH_REJECTED,
                failure_message=(
                    f"path length {len(raw_path)} exceeds max_path_length "
                    f"{config.max_path_length}"
                ),
                result_key=result_key,
                path_key=path_key,
                segment_key=segment_key,
                band=band_metadata,
                out_of_band_cells=out_of_band,
                feasible_cell_count=len(ordered_cells),
                out_of_band_cell_count=len(out_of_band),
            )
        output_path = raw_path[: config.max_path_length]
        truncated = True
        warnings.append(
            {
                "code": W_PATH_TRUNCATED,
                "message": "alignment path truncated by max_path_length",
                "raw_path_length": len(raw_path),
                "max_path_length": config.max_path_length,
            }
        )

    segments = _build_segments(output_path, cells, row_quantified, col_quantified)
    total_cost = sum(cells[coord].local_cost for coord in output_path)
    path_key = fingerprint(
        {
            **base,
            "status": "ok",
            "path": [list(item) for item in output_path],
            "total_cost": total_cost,
            "raw_total_cost": raw_total_cost,
            "truncated": truncated,
            "raw_path_length": len(raw_path),
        }
    )
    segment_key = fingerprint(
        {
            "stage": "segment_summary",
            "path_key": path_key,
            "segments": list(segments),
        }
    )
    result_key = fingerprint(
        {
            "stage": "alignment_result",
            "path_key": path_key,
            "segment_key": segment_key,
        }
    )
    return AlignmentResult(
        sequence_id=sequence_id,
        status="ok",
        total_cost=total_cost,
        raw_total_cost=raw_total_cost,
        path=tuple(output_path),
        raw_path_length=len(raw_path),
        path_length=len(output_path),
        truncated=truncated,
        cells=ordered_cells,
        segments=segments,
        warnings=tuple(warnings),
        failure_code=None,
        failure_message=None,
        result_key=result_key,
        path_key=path_key,
        segment_key=segment_key,
        band=band_metadata,
        out_of_band_cells=out_of_band,
        feasible_cell_count=len(ordered_cells),
        out_of_band_cell_count=len(out_of_band),
    )
