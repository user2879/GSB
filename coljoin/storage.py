"""列式块组样本生成与本地磁盘存储。

样本完全由块组声明中的固定整数种子驱动：
- 行值经 :class:`SplitMix64` 确定性生成，列种子为 ``seed + 列序号``；
- 空值位图按声明 null_rate 由同一 PRNG 决定，逐字节可复现；
- 直方图在生成后用 NumPy 等宽分箱统计，并保证首尾相接、计数和等于行数；
- 列支持 plain/dictionary/rle 三种编码，解码结果与原始列逐元素一致。
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from .canonical import canonical_bytes, digest
from .errors import SpecError
from .model import BlockGroupSpec, ColumnSpec, HistogramBucket
from .prng import SplitMix64

NULL_RATE_DEFAULT = 0.05


def _column_seed(spec: BlockGroupSpec, index: int) -> int:
    return spec.seed + 101 * (index + 1)


def _generate_column(
    spec: BlockGroupSpec, column: ColumnSpec, index: int
) -> np.ndarray:
    rng = SplitMix64(_column_seed(spec, index))
    rng_cfg = spec.value_ranges.get(column.name, {})
    if column.dtype == "integer":
        low = int(rng_cfg.get("low", 0))
        high = int(rng_cfg.get("high", max(1, spec.rows * 3)))
        if high < low:
            raise SpecError(
                "整数列值域 high 小于 low",
                object_id=spec.bg_id,
                field=f"value_ranges.{column.name}",
            )
        values = np.array(
            [rng.next_int(low, high) for _ in range(spec.rows)], dtype=np.int64
        )
    else:
        low = float(rng_cfg.get("low", 0.0))
        high = float(rng_cfg.get("high", 100.0))
        values = np.array(
            [rng.next_float(low, high) for _ in range(spec.rows)], dtype=np.float64
        )
    return values


def _generate_null_bitmap(spec: BlockGroupSpec, null_rate: float) -> List[int]:
    rng = SplitMix64(spec.seed + 7)
    if spec.rows == 0:
        return []
    threshold = int(round(null_rate * (1 << 64)))
    bitmap = [1 if (rng.next_u64() < threshold) else 0 for _ in range(spec.rows)]
    return bitmap


def _build_histogram(
    values: np.ndarray, bitmap: List[int], dtype: str, bucket_count: int = 8
) -> Tuple[List[HistogramBucket], int]:
    valid = values[np.array(bitmap, dtype=np.int8) == 0] if len(values) else values
    null_count = int(sum(bitmap))
    if valid.size == 0:
        edge = 0.0
        buckets = [
            HistogramBucket(edge, edge, 0) for _ in range(max(1, bucket_count))
        ]
        return buckets, null_count
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if hi == lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bucket_count + 1)
    counts, _ = np.histogram(valid, bins=edges)
    counts = counts.astype(np.int64).tolist()
    if sum(counts) != valid.size:
        counts[-1] += valid.size - sum(counts)
    buckets = [
        HistogramBucket(float(edges[i]), float(edges[i + 1]), int(counts[i]))
        for i in range(bucket_count)
    ]
    assert sum(b.count for b in buckets) == valid.size
    return buckets, null_count


def encode_plain(values: np.ndarray) -> Dict[str, Any]:
    return {"encoding": "plain", "data": values.tolist()}


def encode_dictionary(values: np.ndarray) -> Dict[str, Any]:
    uniques, indices = np.unique(values, return_inverse=True)
    return {
        "encoding": "dictionary",
        "dictionary": uniques.tolist(),
        "indices": indices.astype(np.int64).tolist(),
    }


def _run_lengths(values: np.ndarray) -> List[List[Any]]:
    runs: List[List[Any]] = []
    if values.size == 0:
        return runs
    prev = values[0]
    length = 1
    for current in values[1:]:
        if current == prev:
            length += 1
        else:
            runs.append([prev.item(), length])
            prev = current
            length = 1
    runs.append([prev.item(), length])
    return runs


def encode_rle(values: np.ndarray) -> Dict[str, Any]:
    return {"encoding": "rle", "runs": _run_lengths(values)}


def encode_column(values: np.ndarray, encoding: str) -> Dict[str, Any]:
    if encoding == "plain":
        return encode_plain(values)
    if encoding == "dictionary":
        return encode_dictionary(values)
    if encoding == "rle":
        return encode_rle(values)
    raise SpecError("未知列编码", field="encoding")


def decode_column(payload: Dict[str, Any]) -> np.ndarray:
    encoding = payload["encoding"]
    if encoding == "plain":
        return np.array(payload["data"])
    if encoding == "dictionary":
        dictionary = np.array(payload["dictionary"])
        indices = np.array(payload["indices"], dtype=np.int64)
        return dictionary[indices]
    if encoding == "rle":
        pieces = []
        for value, length in payload["runs"]:
            pieces.append(np.repeat(value, length))
        return np.concatenate(pieces) if pieces else np.array([])
    raise SpecError("未知列编码", field="encoding")


def materialize_block_group(spec: BlockGroupSpec, null_rate: float = NULL_RATE_DEFAULT):
    """生成 (列数据 dict, 空值位图, 直方图 dict, ndv dict)。"""
    columns: Dict[str, np.ndarray] = {}
    histograms: Dict[str, List[HistogramBucket]] = {}
    null_counts: Dict[str, int] = {}
    ndvs: Dict[str, int] = {}
    bitmap = _generate_null_bitmap(spec, null_rate)
    for index, column in enumerate(spec.columns):
        values = _generate_column(spec, column, index)
        columns[column.name] = values
        buckets, null_count = _build_histogram(values, bitmap, column.dtype)
        histograms[column.name] = buckets
        null_counts[column.name] = null_count
        mask = np.array(bitmap, dtype=np.int8) == 0
        ndvs[column.name] = int(np.unique(values[mask]).size) if mask.any() else 0
    return columns, bitmap, histograms, null_counts, ndvs


def _build_generated_spec(
    spec: BlockGroupSpec,
    bitmap: List[int],
    histograms: Dict[str, List[HistogramBucket]],
    null_counts: Dict[str, int],
    ndvs: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "bg_id": spec.bg_id,
        "rows": spec.rows,
        "seed": spec.seed,
        "null_bitmap": bitmap,
        "columns": [
            {
                "name": col.name,
                "dtype": col.dtype,
                "encoding": col.encoding,
                "null_count": null_counts[col.name],
                "ndv": ndvs[col.name],
                "histogram": [
                    {"low": b.low, "high": b.high, "count": b.count}
                    for b in histograms[col.name]
                ],
            }
            for col in spec.columns
        ],
        "value_ranges": spec.value_ranges,
    }


def write_workspace(
    groups: List[BlockGroupSpec], data_dir: str, null_rate: float = NULL_RATE_DEFAULT
) -> Dict[str, Any]:
    """把块组样本与生成后的严格声明写入 ``data_dir``，返回清单。"""
    os.makedirs(data_dir, exist_ok=True)
    manifest = {"format": "coljoin-data-v1", "block_groups": []}
    for spec in groups:
        bg_dir = os.path.join(data_dir, spec.bg_id)
        os.makedirs(bg_dir, exist_ok=True)
        columns, bitmap, histograms, null_counts, ndvs = materialize_block_group(
            spec, null_rate
        )
        generated = _build_generated_spec(
            spec, bitmap, histograms, null_counts, ndvs
        )
        spec_path = os.path.join(bg_dir, "spec.json")
        with open(spec_path, "wb") as handle:
            handle.write(canonical_bytes(generated))
        for column in spec.columns:
            payload = encode_column(columns[column.name], column.encoding)
            payload["null_bitmap"] = bitmap
            with open(os.path.join(bg_dir, f"{column.name}.col.json"), "wb") as handle:
                handle.write(canonical_bytes(payload))
        manifest["block_groups"].append(
            {
                "bg_id": spec.bg_id,
                "dir": bg_dir,
                "spec": spec_path,
                "spec_digest": digest(generated),
            }
        )
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "wb") as handle:
        handle.write(canonical_bytes(manifest))
    return manifest


def load_generated_specs(data_dir: str) -> List[BlockGroupSpec]:
    """从已物化工作区重新加载严格块组声明（供执行器读取）。"""
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    specs: List[BlockGroupSpec] = []
    for entry in sorted(manifest["block_groups"], key=lambda item: item["bg_id"]):
        with open(entry["spec"], "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        specs.extend(load_block_groups_from_payload({"block_groups": [raw]}))
    return specs


def load_block_groups_from_payload(payload: Dict[str, Any]) -> List[BlockGroupSpec]:
    from .loader import _validate_block_group

    seen: set = set()
    return [_validate_block_group(raw, seen) for raw in payload["block_groups"]]


def read_column(data_dir: str, bg_id: str, column: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(data_dir, bg_id, f"{column}.col.json")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = decode_column(payload)
    bitmap = np.array(payload["null_bitmap"], dtype=np.int8)
    return values, bitmap
