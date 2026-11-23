from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import ErrorCode, Failure
from .util import is_int, sha256_json


@dataclass(frozen=True)
class AttentionSample:
    sample_id: str
    query_position: int
    key_positions: tuple[int, ...]
    query_features: np.ndarray
    key_features: np.ndarray
    content_fingerprint: str
    source_line: int

    @property
    def point_count(self) -> int:
        return len(self.key_positions)

    def relative_distances(self) -> np.ndarray:
        positions = np.asarray(self.key_positions, dtype=np.int64)
        return np.abs(positions - np.int64(self.query_position))


@dataclass(frozen=True)
class SampleDataset:
    samples: tuple[AttentionSample, ...]
    failures: tuple[Failure, ...]
    dataset_fingerprint: str
    source_path: Path
    file_sha256: str
    total_lines: int

    @property
    def total_points(self) -> int:
        return sum(sample.point_count for sample in self.samples)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def stats(self) -> dict[str, Any]:
        key_counts = [sample.point_count for sample in self.samples]
        return {
            "sample_count": self.sample_count,
            "valid_sample_count": self.sample_count,
            "invalid_sample_count": len(self.failures),
            "total_points": self.total_points,
            "min_points_per_sample": min(key_counts) if key_counts else 0,
            "max_points_per_sample": max(key_counts) if key_counts else 0,
            "mean_points_per_sample": (
                float(np.mean(np.asarray(key_counts, dtype=np.float64))) if key_counts else 0.0
            ),
            "total_lines": self.total_lines,
        }


def _sample_failure(
    message: str,
    *,
    sample_id: str = "",
    line: int | None = None,
    field: str = "",
    details: dict[str, Any] | None = None,
) -> Failure:
    merged = dict(details or {})
    if field:
        merged["field"] = field
    return Failure(
        code=ErrorCode.SAMPLE_INVALID,
        message=f"{field}: {message}" if field else message,
        sample_id=sample_id,
        stage="sample_validation",
        line=line,
        details=merged,
    )


def _parse_feature_vector(value: Any, dimension: int, field: str) -> tuple[np.ndarray | None, str | None]:
    if not isinstance(value, list):
        return None, f"{field} 必须为数组"
    if len(value) != dimension:
        return None, f"{field} 长度必须为 {dimension}，实际为 {len(value)}"
    parsed: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None, f"{field}[{index}] 必须为数值"
        number = float(item)
        if not math.isfinite(number):
            return None, f"{field}[{index}] 必须为有限数"
        parsed.append(number)
    return np.asarray(parsed, dtype=np.float64), None


def _fingerprint_sample(
    sample_id: str,
    query_position: int,
    key_positions: tuple[int, ...],
    query_features: np.ndarray,
    key_features: np.ndarray,
) -> str:
    return sha256_json(
        {
            "sample_id": sample_id,
            "query_position": query_position,
            "key_positions": list(key_positions),
            "query_features": [float(value).hex() for value in query_features.tolist()],
            "key_features": [
                [float(value).hex() for value in row] for row in key_features.tolist()
            ],
        }
    )


def parse_sample_row(
    row: Any,
    *,
    dimension: int,
    line: int,
) -> tuple[AttentionSample | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(row, dict):
        return None, [_sample_failure("样本行必须为 JSON 对象", line=line)]

    sample_id_raw = row.get("sample_id")
    if not isinstance(sample_id_raw, str) or not sample_id_raw.strip():
        return None, [_sample_failure("sample_id 必须为非空字符串", line=line, field="sample_id")]
    sample_id = sample_id_raw.strip()

    query_position = row.get("query_position")
    if not is_int(query_position) or query_position < 0:
        failures.append(
            _sample_failure(
                "必须为非负整数", sample_id=sample_id, line=line, field="query_position"
            )
        )

    key_positions_raw = row.get("key_positions")
    key_positions: tuple[int, ...] = ()
    if not isinstance(key_positions_raw, list) or not key_positions_raw:
        failures.append(
            _sample_failure(
                "必须为非空整数数组", sample_id=sample_id, line=line, field="key_positions"
            )
        )
    else:
        parsed_positions: list[int] = []
        position_failure = False
        for index, item in enumerate(key_positions_raw):
            if not is_int(item) or item < 0:
                failures.append(
                    _sample_failure(
                        "必须为非负整数",
                        sample_id=sample_id,
                        line=line,
                        field=f"key_positions[{index}]",
                    )
                )
                position_failure = True
            else:
                parsed_positions.append(int(item))
        if len(set(parsed_positions)) != len(parsed_positions):
            failures.append(
                _sample_failure(
                    "键位置列表不得重复",
                    sample_id=sample_id,
                    line=line,
                    field="key_positions",
                )
            )
            position_failure = True
        if not position_failure:
            key_positions = tuple(parsed_positions)

    query_features, query_error = _parse_feature_vector(
        row.get("query_features"), dimension, "query_features"
    )
    if query_error:
        failures.append(_sample_failure(query_error, sample_id=sample_id, line=line))

    key_features_raw = row.get("key_features")
    key_features: np.ndarray | None = None
    if not isinstance(key_features_raw, list):
        failures.append(
            _sample_failure(
                "key_features 必须为二维数组",
                sample_id=sample_id,
                line=line,
                field="key_features",
            )
        )
    elif key_positions_raw is not None and len(key_features_raw) != len(key_positions_raw):
        failures.append(
            _sample_failure(
                "key_features 行数必须等于 key_positions 数量",
                sample_id=sample_id,
                line=line,
                field="key_features",
            )
        )
    else:
        rows: list[np.ndarray] = []
        for index, item in enumerate(key_features_raw):
            vector, error = _parse_feature_vector(item, dimension, f"key_features[{index}]")
            if error:
                failures.append(_sample_failure(error, sample_id=sample_id, line=line))
            elif vector is not None:
                rows.append(vector)
        if len(rows) == len(key_features_raw) and rows:
            key_features = np.vstack(rows).astype(np.float64, copy=False)

    if failures:
        return None, failures
    assert isinstance(query_position, int)
    assert query_features is not None
    assert key_features is not None
    fingerprint = _fingerprint_sample(
        sample_id, query_position, key_positions, query_features, key_features
    )
    return (
        AttentionSample(
            sample_id=sample_id,
            query_position=query_position,
            key_positions=key_positions,
            query_features=query_features,
            key_features=key_features,
            content_fingerprint=fingerprint,
            source_line=line,
        ),
        [],
    )


def parse_samples_lines(
    rows: Iterable[tuple[int, str]], *, dimension: int
) -> tuple[list[AttentionSample], list[Failure], int]:
    samples: list[AttentionSample] = []
    failures: list[Failure] = []
    seen_ids: set[str] = set()
    line_count = 0
    for line_number, text in rows:
        line_count += 1
        if not text.strip():
            failures.append(
                _sample_failure("JSONL 行不得为空", line=line_number, field="jsonl")
            )
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            failures.append(
                _sample_failure(
                    f"JSON 解析失败: {exc.msg}", line=line_number, field="jsonl"
                )
            )
            continue
        raw_id = row.get("sample_id") if isinstance(row, dict) else None
        sample_id = raw_id.strip() if isinstance(raw_id, str) else ""
        if sample_id and sample_id in seen_ids:
            failures.append(
                _sample_failure(
                    "样本 ID 重复",
                    sample_id=sample_id,
                    line=line_number,
                    field="sample_id",
                )
            )
            continue
        if sample_id:
            seen_ids.add(sample_id)
        sample, row_failures = parse_sample_row(row, dimension=dimension, line=line_number)
        failures.extend(row_failures)
        if sample is not None:
            samples.append(sample)
    samples.sort(key=lambda item: item.sample_id)
    return samples, failures, line_count


def build_dataset_fingerprint(samples: Iterable[AttentionSample], dimension: int) -> str:
    payload = [
        {
            "sample_id": sample.sample_id,
            "content_fingerprint": sample.content_fingerprint,
        }
        for sample in sorted(samples, key=lambda item: item.sample_id)
    ]
    return sha256_json({"dimension": dimension, "samples": payload, "schema": "rope-diag-dataset-v1"})


def load_samples(path: Path, *, dimension: int) -> SampleDataset:
    try:
        from .util import sha256_file

        file_hash = sha256_file(path)
        with path.open("r", encoding="utf-8") as handle:
            rows = [(index + 1, line.rstrip("\r\n")) for index, line in enumerate(handle)]
    except OSError as exc:
        failure = Failure(
            code=ErrorCode.IO_ERROR,
            message=f"无法读取样本 JSONL: {exc}",
            stage="sample_validation",
            details={"path": path.as_posix()},
        )
        return SampleDataset((), (failure,), "", path, "", 0)

    samples, failures, line_count = parse_samples_lines(rows, dimension=dimension)
    if not samples and not failures:
        failures.append(_sample_failure("样本文件为空", field="jsonl"))
    failures.sort(key=lambda failure: failure.sort_key)
    dataset_fingerprint = build_dataset_fingerprint(samples, dimension)
    return SampleDataset(
        samples=tuple(samples),
        failures=tuple(failures),
        dataset_fingerprint=dataset_fingerprint,
        source_path=path,
        file_sha256=file_hash,
        total_lines=line_count,
    )
