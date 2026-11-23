from __future__ import annotations

import json
from pathlib import Path

from dtw_diag.config import Config, parse_config
from dtw_diag.errors import Failure
from dtw_diag.models import LibraryData, SequenceRecord


def make_config(**overrides):
    raw = {
        "reference_sequence_id": None,
        "step_mode": "symmetric",
        "bandwidth_permille": 500,
        "distance_coefficient": 1,
        "fixed_point": {"fraction_bits": 8, "quantization_step": 4},
        "missing_marker": None,
        "missing_penalty": 11,
        "max_path_length": 100,
        "path_length_policy": "reject",
        "thread_pool_workers": 1,
        "cell_recompute_budget": 10_000,
        "snapshot_interval_batches": 2,
    }
    raw.update(overrides)
    result = parse_config(raw)
    assert result.config is not None, result.failures
    return result.config


def make_record(
    seq_id: str,
    observations: list,
    *,
    segment_ids: list | None = None,
    timestamp_interval: float = 1.0,
) -> SequenceRecord:
    return SequenceRecord(
        seq_id=seq_id,
        observations=tuple(observations),
        timestamp_interval=timestamp_interval,
        segment_ids=tuple(segment_ids or [0] * len(observations)),
    )


def make_library(records: list[SequenceRecord]) -> LibraryData:
    library = LibraryData()
    for record in records:
        library.records[record.seq_id] = record
    return library


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


SAMPLE_ROOT = Path(__file__).resolve().parents[1]
