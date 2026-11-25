from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ATTRIBUTION_CALIBERS = ("floor", "largest_remainder")


def _largest_remainder(numerators: list[int], denominator: int) -> list[int]:
    if denominator <= 0:
        return [0 for _ in numerators]
    base = [(numerator * 1_000_000) // denominator for numerator in numerators]
    remainders = [(numerator * 1_000_000) % denominator for numerator in numerators]
    remaining = 1_000_000 - sum(base)
    order = sorted(range(len(numerators)), key=lambda index: (-remainders[index], index))
    for index in order[:remaining]:
        base[index] += 1
    return base


def integer_contribution_ppm(
    components: list[tuple[str, int]], tolerance_ppm: int
) -> dict[str, Any]:
    """Return exact integer ppm attribution and the stable caliber decision."""

    denominator = sum(value for _, value in components)
    numerators = [value for _, value in components]
    floor = (
        [0 for _ in numerators]
        if denominator <= 0
        else [(value * 1_000_000) // denominator for value in numerators]
    )
    largest = _largest_remainder(numerators, denominator)
    differences = [abs(a - b) for a, b in zip(floor, largest, strict=True)]
    affected = bool(denominator > 0 and max(differences, default=0) <= tolerance_ppm and any(differences))
    selected = floor
    return {
        "denominator": denominator,
        "calibers": {
            "floor": [{"name": name, "ppm": value} for (name, _), value in zip(components, floor, strict=True)],
            "largest_remainder": [
                {"name": name, "ppm": value}
                for (name, _), value in zip(components, largest, strict=True)
            ],
        },
        "selected_caliber": "floor",
        "selected": [
            {"name": name, "ppm": value}
            for (name, _), value in zip(components, selected, strict=True)
        ],
        "max_caliber_difference_ppm": max(differences, default=0),
        "tolerance_ppm": tolerance_ppm,
        "tolerance_affected": affected,
        "stable_rule": "choose floor; mark affected when max caliber difference <= tolerance and nonzero",
    }


@dataclass
class AttributionTotals:
    uncropped_decoded_bytes: int = 0
    pruned_full_scan_bytes: int = 0
    actual_decoded_bytes: int = 0
    column_pruning_bytes_saved: int = 0
    predicate_pushdown_bytes_saved: int = 0
    batch_partition_bytes_saved: int = 0
    candidate_rows_before: int = 0
    candidate_rows_after: int = 0
    comparisons_before: int = 0
    comparisons_after: int = 0
    physical_vector_comparisons: int = 0
    column_pruning_comparisons_saved: int = 0
    predicate_pushdown_comparisons_saved: int = 0
    batch_partition_comparisons_saved: int = 0
    batch_count: int = 0

    def add(self, item: dict[str, Any]) -> None:
        self.uncropped_decoded_bytes += int(item["uncropped_decoded_bytes"])
        self.pruned_full_scan_bytes += int(item["pruned_full_scan_decoded_bytes"])
        self.actual_decoded_bytes += int(item["actual_decoded_bytes"])
        self.column_pruning_bytes_saved += int(item["bytes"]["column_pruning"])
        self.predicate_pushdown_bytes_saved += int(item["bytes"]["predicate_pushdown"])
        self.batch_partition_bytes_saved += int(item["bytes"]["batch_partition"])
        self.candidate_rows_before += int(item["candidate_rows_before"])
        self.candidate_rows_after += int(item["candidate_rows_after"])
        self.comparisons_before += int(item["comparisons"]["before"])
        self.comparisons_after += int(item["comparisons"]["after"])
        self.physical_vector_comparisons += int(item["comparisons"]["physical_vector"])
        self.column_pruning_comparisons_saved += int(
            item["comparisons"]["column_pruning_saved"]
        )
        self.predicate_pushdown_comparisons_saved += int(
            item["comparisons"]["predicate_pushdown_saved"]
        )
        self.batch_partition_comparisons_saved += int(
            item["comparisons"]["batch_partition_saved"]
        )
        self.batch_count += 1

    def to_dict(self, tolerance_ppm: int) -> dict[str, Any]:
        byte_components = [
            ("column_pruning", self.column_pruning_bytes_saved),
            ("predicate_pushdown", self.predicate_pushdown_bytes_saved),
            ("batch_partition", self.batch_partition_bytes_saved),
        ]
        comparison_components = [
            ("column_pruning", self.column_pruning_comparisons_saved),
            ("predicate_pushdown", self.predicate_pushdown_comparisons_saved),
            ("batch_partition", self.batch_partition_comparisons_saved),
        ]
        return {
            "uncropped_decoded_bytes": self.uncropped_decoded_bytes,
            "pruned_full_scan_decoded_bytes": self.pruned_full_scan_bytes,
            "actual_decoded_bytes": self.actual_decoded_bytes,
            "bytes_saved": {
                "column_pruning": self.column_pruning_bytes_saved,
                "predicate_pushdown": self.predicate_pushdown_bytes_saved,
                "batch_partition": self.batch_partition_bytes_saved,
                "total": self.uncropped_decoded_bytes - self.actual_decoded_bytes,
            },
            "candidate_rows_before": self.candidate_rows_before,
            "candidate_rows_after": self.candidate_rows_after,
            "comparisons": {
                "before": self.comparisons_before,
                "after": self.comparisons_after,
                "physical_vector": self.physical_vector_comparisons,
                "column_pruning_saved": self.column_pruning_comparisons_saved,
                "predicate_pushdown_saved": self.predicate_pushdown_comparisons_saved,
                "batch_partition_saved": self.batch_partition_comparisons_saved,
                "total_saved": self.comparisons_before - self.comparisons_after,
            },
            "bytes_contribution_ppm": integer_contribution_ppm(
                byte_components, tolerance_ppm
            ),
            "comparisons_contribution_ppm": integer_contribution_ppm(
                comparison_components, tolerance_ppm
            ),
            "batch_count": self.batch_count,
            "notes": {
                "decoded_byte_unit": "9 bytes per decoded cell (int64 value + validity byte)",
                "batch_partition": "zero direct byte/comparison savings; it changes peak working set and scheduling only",
                "physical_vector_comparisons": "leaf_count * batch_rows; pushdown savings are measured on effective post-filter candidate rows",
            },
        }


def batch_attribution(
    *,
    rows: int,
    all_column_count: int,
    decode_column_count: int,
    predicate_column_count: int,
    passed_rows: int,
    predicate_leaf_count: int,
    decoded_cell_bytes: int,
    tolerance_ppm: int,
) -> dict[str, Any]:
    uncropped = rows * all_column_count * decoded_cell_bytes
    pruned_full = rows * decode_column_count * decoded_cell_bytes
    output_only_count = max(0, decode_column_count - predicate_column_count)
    actual = (
        rows * predicate_column_count * decoded_cell_bytes
        + passed_rows * output_only_count * decoded_cell_bytes
    )
    column_pruning_saved = uncropped - pruned_full
    pushdown_saved = pruned_full - actual
    bytes_components = [
        ("column_pruning", column_pruning_saved),
        ("predicate_pushdown", pushdown_saved),
        ("batch_partition", 0),
    ]
    comparisons_before = rows * predicate_leaf_count
    comparisons_after = passed_rows * predicate_leaf_count
    comparison_saved = comparisons_before - comparisons_after
    comparison_components = [
        ("column_pruning", 0),
        ("predicate_pushdown", comparison_saved),
        ("batch_partition", 0),
    ]
    return {
        "uncropped_decoded_bytes": uncropped,
        "pruned_full_scan_decoded_bytes": pruned_full,
        "actual_decoded_bytes": actual,
        "bytes": {
            "column_pruning": column_pruning_saved,
            "predicate_pushdown": pushdown_saved,
            "batch_partition": 0,
            "total": uncropped - actual,
        },
        "bytes_contribution_ppm": integer_contribution_ppm(
            bytes_components, tolerance_ppm
        ),
        "candidate_rows_before": rows,
        "candidate_rows_after": passed_rows,
        "comparisons": {
            "before": comparisons_before,
            "after": comparisons_after,
            "physical_vector": comparisons_before,
            "column_pruning_saved": 0,
            "predicate_pushdown_saved": comparison_saved,
            "batch_partition_saved": 0,
            "total_saved": comparison_saved,
        },
        "comparisons_contribution_ppm": integer_contribution_ppm(
            comparison_components, tolerance_ppm
        ),
        "notes": {
            "decoded_byte_unit": "9 bytes per decoded cell (int64 value + validity byte)",
            "candidate_rows": "rows entering the aggregate before and after predicate masking",
            "batch_partition": "zero direct savings; boundaries are exact and non-overlapping",
        },
    }
