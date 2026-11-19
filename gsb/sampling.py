from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .errors import GSBError
from .filtering import null_mask
from .util import (
    FIXED_SCALE,
    canonical_row_order,
    json_number,
    logical_content_digest,
    sha256_json,
    sort_key_value,
    stable_seed,
)


@dataclass(frozen=True)
class SampleLedgerEntry:
    key: int | float
    frequency: int
    rescale_factor: float
    estimated_row_frequency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": json_number(self.key),
            "frequency": self.frequency,
            "rescale_factor": self.rescale_factor,
            "estimated_row_frequency": self.estimated_row_frequency,
        }


@dataclass(frozen=True)
class SampleLedger:
    table: str
    key_column: str
    table_rows: int
    filtered_rows: int
    sample_size: int
    sample_seed: int
    derived_seed: int
    coefficients: Mapping[str, float]
    selected_rows_digest: str
    data_digest: str
    predicate_digest: str
    entries: tuple[SampleLedgerEntry, ...]
    digest: str
    sample_arrays: Mapping[str, np.ndarray]

    def frequency_map(self) -> dict[int | float, int]:
        return {entry.key: entry.frequency for entry in self.entries}

    def to_dict(self, *, include_entries: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "table": self.table,
            "key_column": self.key_column,
            "table_rows": self.table_rows,
            "filtered_rows": self.filtered_rows,
            "sample_size": self.sample_size,
            "sample_seed": self.sample_seed,
            "derived_seed": self.derived_seed,
            "coefficients": dict(self.coefficients),
            "selected_rows_digest": self.selected_rows_digest,
            "data_digest": self.data_digest,
            "predicate_digest": self.predicate_digest,
            "key_count": len(self.entries),
            "digest": self.digest,
        }
        if include_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


def _normalize_key(value: Any) -> int | float:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def draw_sample(
    arrays: Mapping[str, np.ndarray],
    *,
    table: str,
    key_column: str,
    table_rows: int,
    sample_size: int,
    sample_seed: int,
    data_digest: str,
    predicate_digest: str,
    names: list[str] | None = None,
) -> SampleLedger:
    column_names = list(names) if names is not None else list(arrays)
    filtered_rows = int(len(arrays[key_column])) if arrays else 0
    if sample_size <= 0:
        raise GSBError("E_SAMPLE", "抽样规模必须为正整数", {"sample_size": sample_size})
    if sample_size > filtered_rows:
        raise GSBError(
            "E_SAMPLE",
            "抽样规模超过过滤后可用行数",
            {
                "table": table,
                "sample_size": sample_size,
                "filtered_rows": filtered_rows,
                "upper_limit": filtered_rows,
            },
        )
    order = canonical_row_order([arrays[name] for name in column_names])
    derived_seed = stable_seed(sample_seed, table, data_digest, predicate_digest, sample_size, "GSB-SAMPLE-v1")
    rng = np.random.Generator(np.random.PCG64(derived_seed))
    permutation = rng.permutation(filtered_rows)
    selected_positions = np.sort(permutation[:sample_size])
    canonical_arrays = {name: arrays[name][order] for name in column_names}
    sample_arrays = {
        name: np.ascontiguousarray(canonical_arrays[name][selected_positions])
        for name in column_names
    }
    selected_rows_digest = logical_content_digest(sample_arrays, names=column_names)
    keys = sample_arrays[key_column]
    keys = keys[~null_mask(keys)]
    normalized = [_normalize_key(value) for value in keys.tolist()]
    counts: dict[int | float, int] = {}
    for key in normalized:
        counts[key] = counts.get(key, 0) + 1
    coefficient = (filtered_rows / sample_size) if sample_size else 0.0
    entries = tuple(
        SampleLedgerEntry(
            key=key,
            frequency=counts[key],
            rescale_factor=coefficient,
            estimated_row_frequency=counts[key] * coefficient,
        )
        for key in sorted(counts, key=sort_key_value)
    )
    ledger_digest = sha256_json(
        {
            "table": table,
            "key_column": key_column,
            "filtered_rows": filtered_rows,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "derived_seed": derived_seed,
            "selected_rows_digest": selected_rows_digest,
            "data_digest": data_digest,
            "predicate_digest": predicate_digest,
            "entries": [entry.to_dict() for entry in entries],
        }
    )
    return SampleLedger(
        table=table,
        key_column=key_column,
        table_rows=table_rows,
        filtered_rows=filtered_rows,
        sample_size=sample_size,
        sample_seed=sample_seed,
        derived_seed=derived_seed,
        coefficients={
            "rows_per_sample_row": coefficient,
            "filtered_rows": float(filtered_rows),
            "sample_size": float(sample_size),
        },
        selected_rows_digest=selected_rows_digest,
        data_digest=data_digest,
        predicate_digest=predicate_digest,
        entries=entries,
        digest=ledger_digest,
        sample_arrays=sample_arrays,
    )


def sampling_join_estimate(
    left: SampleLedger,
    right: SampleLedger,
    *,
    left_filtered_rows: int,
    right_filtered_rows: int,
) -> dict[str, Any]:
    left_frequency = left.frequency_map()
    right_frequency = right.frequency_map()
    coefficient_left = left_filtered_rows / left.sample_size if left.sample_size else 0.0
    coefficient_right = right_filtered_rows / right.sample_size if right.sample_size else 0.0
    common_keys = sorted(set(left_frequency) & set(right_frequency), key=sort_key_value)
    pair_contributions: list[dict[str, Any]] = []
    sample_hit_pairs = 0
    estimate = 0.0
    fixed_total = 0
    for key in common_keys:
        left_count = left_frequency[key]
        right_count = right_frequency[key]
        hit_pairs = left_count * right_count
        contribution = hit_pairs * coefficient_left * coefficient_right
        contribution_fixed = int(round(contribution * FIXED_SCALE))
        sample_hit_pairs += hit_pairs
        estimate += contribution
        fixed_total += contribution_fixed
        pair_contributions.append(
            {
                "key": json_number(key),
                "left_sample_frequency": left_count,
                "right_sample_frequency": right_count,
                "sample_hit_pairs": hit_pairs,
                "left_coefficient": coefficient_left,
                "right_coefficient": coefficient_right,
                "contribution": contribution,
                "contribution_fixed": contribution_fixed,
            }
        )
    sample_pair_universe = left.sample_size * right.sample_size
    sample_hit_rate = (sample_hit_pairs / sample_pair_universe) if sample_pair_universe else 0.0
    max_possible = float(left_filtered_rows * right_filtered_rows)
    if estimate > max_possible:
        estimate = max_possible
    if fixed_total > int(round(max_possible * FIXED_SCALE)):
        fixed_total = int(round(max_possible * FIXED_SCALE))
    return {
        "estimate": estimate,
        "estimate_from_fixed": fixed_total / FIXED_SCALE,
        "fixed_scale": FIXED_SCALE,
        "estimate_fixed": fixed_total,
        "quantization_error_upper": len(common_keys) * (0.5 / FIXED_SCALE),
        "sample_hit_pairs": sample_hit_pairs,
        "sample_pair_universe": sample_pair_universe,
        "sample_hit_rate": sample_hit_rate,
        "observed_common_keys": len(common_keys),
        "left_coefficient": coefficient_left,
        "right_coefficient": coefficient_right,
        "joint_coefficient": coefficient_left * coefficient_right,
        "max_possible": max_possible,
        "pair_contributions": pair_contributions,
    }
