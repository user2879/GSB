"""Cell grid reconstruction from detected line segments."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ParseConfig
from .validation import PageInvalid, region_signature


def _clip_overlap(
    segment: Mapping[str, Any], lower: int, upper: int
) -> tuple[int, int] | None:
    start = max(int(segment["start"]), lower)
    end = min(int(segment["end"]), upper)
    if start > end:
        return None
    return start, end


def _axis_coords(
    segments: Sequence[Mapping[str, Any]],
    region: Sequence[int],
    axis: str,
    minimum_overlap: int,
) -> list[int]:
    left, top, right, bottom = region
    coords: set[int] = set()
    for segment in segments:
        if segment["axis"] != axis:
            continue
        coord = int(segment["coord"])
        if axis == "h":
            if not top <= coord <= bottom:
                continue
            overlap = _clip_overlap(segment, left, right)
        else:
            if not left <= coord <= right:
                continue
            overlap = _clip_overlap(segment, top, bottom)
        if overlap is not None and overlap[1] - overlap[0] + 1 >= minimum_overlap:
            coords.add(coord)
    return sorted(coords)


def _line_ids_at(
    segments: Sequence[Mapping[str, Any]],
    axis: str,
    coord: int,
    start: int,
    end: int,
    tolerance: int,
) -> list[str]:
    matches = []
    for segment in segments:
        if segment["axis"] != axis:
            continue
        if abs(int(segment["coord"]) - coord) > tolerance:
            continue
        if min(int(segment["end"]), end) < max(int(segment["start"]), start):
            continue
        matches.append(segment["line_id"])
    return sorted(matches)


def _edge_coverage(
    segments: Sequence[Mapping[str, Any]],
    axis: str,
    coord: int,
    start: int,
    end: int,
    tolerance: int,
) -> float:
    intervals: list[tuple[int, int]] = []
    for segment in segments:
        if segment["axis"] != axis:
            continue
        if abs(int(segment["coord"]) - coord) > tolerance:
            continue
        clipped_start = max(start, int(segment["start"]))
        clipped_end = min(end, int(segment["end"]))
        if clipped_start <= clipped_end:
            intervals.append((clipped_start, clipped_end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: list[list[int]] = [list(intervals[0])]
    for item_start, item_end in intervals[1:]:
        if item_start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], item_end)
        else:
            merged.append([item_start, item_end])
    covered = sum(item[1] - item[0] + 1 for item in merged)
    return covered / float(max(1, end - start + 1))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def _cell_cache_key(
    gray: np.ndarray,
    table_id: str,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    bbox: Sequence[int],
    row_line_ids: Sequence[str],
    col_line_ids: Sequence[str],
    text: str,
    config: ParseConfig,
) -> str:
    left, top, right, bottom = bbox
    crop = np.ascontiguousarray(gray[top : bottom + 1, left : right + 1]).tobytes()
    payload = {
        "stage": "cell",
        "table_id": table_id,
        "row": [row_start, row_end],
        "col": [col_start, col_end],
        "bbox": list(bbox),
        "row_line_ids": list(row_line_ids),
        "col_line_ids": list(col_line_ids),
        "text": text,
        "crop_sha256": hashlib.sha256(crop).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _truth_cell_for(
    truth_table: Mapping[str, Any],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    bbox: Sequence[int],
) -> Mapping[str, Any] | None:
    for cell in truth_table["cells"]:
        if (
            cell["row"] == row_start
            and cell["row"] + cell["row_span"] - 1 == row_end
            and cell["col"] == col_start
            and cell["col"] + cell["col_span"] - 1 == col_end
            and list(cell["bbox"]) == list(bbox)
        ):
            return cell
    return None


def _previous_cell_lookup(previous_grid: Mapping[str, Any] | None) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    lookup: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    if not previous_grid:
        return lookup
    for table in previous_grid.get("tables", []):
        for cell in table.get("cells", []):
            identity = (
                table.get("table_id"),
                cell.get("row_start"),
                cell.get("row_end"),
                cell.get("col_start"),
                cell.get("col_end"),
            )
            lookup[identity] = cell
    return lookup


def rebuild_grid(
    segments: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    gray: np.ndarray,
    config: ParseConfig,
    line_key: str,
    table_decisions: Mapping[str, Any],
    previous_grid: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    minimum_overlap = max(8, min(config.min_line_length, 16))
    tolerance = config.line_merge_tolerance
    previous_cells = _previous_cell_lookup(previous_grid)
    tables: list[dict[str, Any]] = []
    fragment_drops: list[dict[str, Any]] = []
    reused_cells = 0
    recomputed_cells = 0

    for truth_table in truth["tables"]:
        table_id = truth_table["table_id"]
        region = truth_table["region"]
        row_coords = _axis_coords(segments, region, "h", minimum_overlap)
        col_coords = _axis_coords(segments, region, "v", minimum_overlap)
        if len(row_coords) < 2 or len(col_coords) < 2:
            raise PageInvalid(
                "E_GRID_UNSTABLE",
                f"table {table_id} produced fewer than two detected row/column lines",
            )

        row_line_refs = {
            coord: _line_ids_at(
                segments, "h", coord, region[0], region[2], tolerance
            )
            for coord in row_coords
        }
        col_line_refs = {
            coord: _line_ids_at(
                segments, "v", coord, region[1], region[3], tolerance
            )
            for coord in col_coords
        }

        row_count = len(row_coords) - 1
        col_count = len(col_coords) - 1
        base_cells: list[dict[str, Any]] = []
        for row in range(row_count):
            for col in range(col_count):
                left = col_coords[col]
                right = col_coords[col + 1]
                top = row_coords[row]
                bottom = row_coords[row + 1]
                area = (right - left) * (bottom - top)
                if area < config.cell_min_area:
                    fragment_drops.append(
                        {
                            "table_id": table_id,
                            "row": row,
                            "col": col,
                            "bbox": [left, top, right, bottom],
                            "area": area,
                            "reason": "E_CELL_AREA_TOO_SMALL",
                        }
                    )
                    continue
                base_cells.append(
                    {
                        "row": row,
                        "col": col,
                        "bbox": [left, top, right, bottom],
                        "area": area,
                    }
                )

        index_by_position = {
            (cell["row"], cell["col"]): index for index, cell in enumerate(base_cells)
        }
        union_find = _UnionFind(len(base_cells))
        for index, cell in enumerate(base_cells):
            row = cell["row"]
            col = cell["col"]
            right_neighbor = index_by_position.get((row, col + 1))
            if right_neighbor is not None:
                boundary_x = col_coords[col + 1]
                coverage = _edge_coverage(
                    segments,
                    "v",
                    boundary_x,
                    cell["bbox"][1],
                    cell["bbox"][3],
                    tolerance,
                )
                if coverage < 0.80:
                    union_find.union(index, right_neighbor)
            bottom_neighbor = index_by_position.get((row + 1, col))
            if bottom_neighbor is not None:
                boundary_y = row_coords[row + 1]
                coverage = _edge_coverage(
                    segments,
                    "h",
                    boundary_y,
                    cell["bbox"][0],
                    cell["bbox"][2],
                    tolerance,
                )
                if coverage < 0.80:
                    union_find.union(index, bottom_neighbor)

        groups: dict[int, list[dict[str, Any]]] = {}
        for index, cell in enumerate(base_cells):
            groups.setdefault(union_find.find(index), []).append(cell)

        assembled_cells: list[dict[str, Any]] = []
        for members in groups.values():
            row_start = min(member["row"] for member in members)
            row_end = max(member["row"] for member in members)
            col_start = min(member["col"] for member in members)
            col_end = max(member["col"] for member in members)
            expected_members = (row_end - row_start + 1) * (col_end - col_start + 1)
            if len(members) != expected_members:
                raise PageInvalid(
                    "E_GRID_NON_RECTANGULAR",
                    f"table {table_id} produced a non-rectangular merge at "
                    f"rows[{row_start},{row_end}] cols[{col_start},{col_end}]",
                )
            bbox = [
                col_coords[col_start],
                row_coords[row_start],
                col_coords[col_end + 1],
                row_coords[row_end + 1],
            ]
            truth_cell = _truth_cell_for(
                truth_table, row_start, row_end, col_start, col_end, bbox
            )
            text = truth_cell["text"] if truth_cell is not None else ""
            top_ids = list(row_line_refs.get(bbox[1], []))
            bottom_ids = list(row_line_refs.get(bbox[3], []))
            left_ids = list(col_line_refs.get(bbox[0], []))
            right_ids = list(col_line_refs.get(bbox[2], []))
            row_line_ids = sorted(top_ids + bottom_ids)
            col_line_ids = sorted(left_ids + right_ids)
            cache_key = _cell_cache_key(
                gray,
                table_id,
                row_start,
                row_end,
                col_start,
                col_end,
                bbox,
                row_line_ids,
                col_line_ids,
                text,
                config,
            )
            identity = (table_id, row_start, row_end, col_start, col_end)
            previous = previous_cells.get(identity)
            reused = False
            if previous is not None and previous.get("cache_key") == cache_key:
                reused = True
                reused_cells += 1
            else:
                recomputed_cells += 1
            cell_record = {
                "cell_id": "",
                "table_id": table_id,
                "row_start": row_start,
                "row_end": row_end,
                "col_start": col_start,
                "col_end": col_end,
                "row_span": row_end - row_start + 1,
                "col_span": col_end - col_start + 1,
                "bbox": bbox,
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "row_relation": {"start": row_start, "end": row_end, "span": row_end - row_start + 1},
                "col_relation": {"start": col_start, "end": col_end, "span": col_end - col_start + 1},
                "line_refs": {
                    "top_row_line_ids": top_ids,
                    "bottom_row_line_ids": bottom_ids,
                    "left_col_line_ids": left_ids,
                    "right_col_line_ids": right_ids,
                },
                "row_line_ids": row_line_ids,
                "col_line_ids": col_line_ids,
                "text": text,
                "text_source": "page_truth_annotation" if truth_cell is not None else "unmatched",
                "content_bbox": list(truth_cell.get("content_bbox", [])) if truth_cell is not None else [],
                "truth_cell_id": truth_cell.get("cell_id") if truth_cell is not None else None,
                "cache_key": cache_key,
                "cache_status": "reused" if reused else "recomputed",
            }
            if reused and previous is not None:
                # Keep the exact prior business record; only the cache status changes.
                cell_record = dict(previous)
                cell_record["cache_status"] = "reused"
            assembled_cells.append(cell_record)

        assembled_cells.sort(
            key=lambda item: (
                item["row_start"],
                item["col_start"],
                item["row_end"],
                item["col_end"],
            )
        )
        for ordinal, cell in enumerate(assembled_cells, start=1):
            cell["cell_id"] = f"{table_id}_C{ordinal:04d}"

        table_record = {
            "table_id": table_id,
            "region": list(region),
            "row_count": row_count,
            "col_count": col_count,
            "row_lines": [
                {"coord": coord, "line_ids": row_line_refs[coord]}
                for coord in row_coords
            ],
            "col_lines": [
                {"coord": coord, "line_ids": col_line_refs[coord]}
                for coord in col_coords
            ],
            "cells": assembled_cells,
            "classification": table_decisions.get(table_id),
            "merged_boundaries": [
                {
                    "direction": "vertical",
                    "coord": col_coords[col + 1],
                    "row": row,
                    "rule": "MISSING_INTERNAL_EDGE_COVERAGE_LT_0.80",
                }
                for row in range(row_count)
                for col in range(col_count - 1)
                if _edge_coverage(
                    segments,
                    "v",
                    col_coords[col + 1],
                    row_coords[row],
                    row_coords[row + 1],
                    tolerance,
                )
                < 0.80
            ]
            + [
                {
                    "direction": "horizontal",
                    "coord": row_coords[row + 1],
                    "col": col,
                    "rule": "MISSING_INTERNAL_EDGE_COVERAGE_LT_0.80",
                }
                for row in range(row_count - 1)
                for col in range(col_count)
                if _edge_coverage(
                    segments,
                    "h",
                    row_coords[row + 1],
                    col_coords[col],
                    col_coords[col + 1],
                    tolerance,
                )
                < 0.80
            ],
        }
        tables.append(table_record)

    grid_key = grid_cache_key(
        segments,
        truth,
        config,
        line_key,
        table_decisions,
    )
    return {
        "tables": tables,
        "fragment_drops": fragment_drops,
        "cache_key": grid_key,
        "cell_cache": {
            "reused": reused_cells,
            "recomputed": recomputed_cells,
        },
    }


def _table_role_signature(table_decisions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        table_id: {
            "classified_role": decision.get("classified_role"),
            "table_grid_score": decision.get("table_grid_score"),
            "rationale": decision.get("rationale"),
        }
        for table_id, decision in sorted(table_decisions.items())
    }


def grid_cache_key(
    segments: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    config: ParseConfig,
    line_key: str,
    table_decisions: Mapping[str, Any],
) -> str:
    payload = {
        "stage": "grid",
        "upstream_line_key": line_key,
        "params": config.stage_params("grid"),
        "truth": region_signature(truth, "grid"),
        "table_classification": _table_role_signature(table_decisions),
        "line_ids": [
            [segment["line_id"], segment["axis"], segment["coord"], segment["start"], segment["end"]]
            for segment in segments
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
