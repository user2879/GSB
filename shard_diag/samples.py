from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_bytes, content_digest, saturate_int64, stable_uint32
from .errors import SpecError


LOSS_ALGORITHM = "int-fixed-forward-v1"
CHUNK_ROWS = 256


def derive_sample_seed(node_seed: int, shard_id: str, quota_version: int) -> int:
    material = f"sample|seed={node_seed}|shard={shard_id}|quota={quota_version}"
    return stable_uint32(material)


def param_digest(node: dict[str, Any]) -> str:
    return content_digest(
        {
            "algorithm": LOSS_ALGORITHM,
            "node_id": node["node_id"],
            "fixed_scale": node["fixed_scale"],
            "feature_count": len(node["weights_q"]),
            "weights_q": list(node["weights_q"]),
            "bias_q": node["bias_q"],
        },
        prefix="pd1",
    )


def sample_rows_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_bytes(row))
        digest.update(b"\n")
    return "sd1:" + digest.hexdigest()


def _validate_row(row: Any, feature_count: int, entry: str) -> dict[str, int | list[int]]:
    if not isinstance(row, dict):
        raise SpecError("样本行必须是对象", entry=entry, field="row")
    x = row.get("x")
    y = row.get("y")
    if not isinstance(x, list) or len(x) != feature_count:
        raise SpecError(f"x 必须是长度 {feature_count} 的数组", entry=entry, field="x")
    clean_x: list[int] = []
    for index, value in enumerate(x):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SpecError("x 元素必须是整数", entry=entry, field=f"x[{index}]")
        clean_x.append(value)
    if isinstance(y, bool) or not isinstance(y, int):
        raise SpecError("y 必须是整数", entry=entry, field="y")
    return {"x": clean_x, "y": int(y)}


def generate_rows(
    node: dict[str, Any],
    shard: dict[str, Any],
    quota_version: int,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    feature_count = len(node["weights_q"])
    seed = derive_sample_seed(int(node["seed"]), shard["shard_id"], int(quota_version))
    rng = np.random.RandomState(seed)
    row_count = int(shard["sample_rows"])
    if row_count == 0:
        rows: list[dict[str, Any]] = []
    else:
        values = rng.randint(-4, 5, size=(row_count, feature_count)).astype(np.int64)
        teacher = np.array([((index % 3) - 1) for index in range(feature_count)], dtype=np.int64)
        noise = rng.randint(-1, 2, size=row_count).astype(np.int64)
        labels = values @ teacher + int(seed % 7) - 3 + noise
        rows = [
            {"x": [int(v) for v in values[index]], "y": int(labels[index])}
            for index in range(row_count)
        ]

    shard_overrides = (overrides or {}).get(shard["shard_id"], {})
    if not isinstance(shard_overrides, dict):
        raise SpecError("样本替换必须是 row_index -> sample 对象", entry=f"overrides.{shard['shard_id']}", field="overrides")
    for raw_index, raw_row in shard_overrides.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise SpecError("样本行号必须是十进制整数", entry=f"overrides.{shard['shard_id']}", field=raw_index) from exc
        if index < 0 or index >= row_count:
            raise SpecError(
                f"替换行号超出 0..{row_count - 1}",
                entry=f"overrides.{shard['shard_id']}",
                field=raw_index,
            )
        rows[index] = _validate_row(raw_row, feature_count, f"overrides.{shard['shard_id']}.{raw_index}")  # type: ignore[assignment]
    return rows


def load_overrides(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "samples" / "overrides.json"
    if not path.exists():
        return {"format_version": 1, "overrides": {}}
    with path.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or doc.get("format_version") != 1:
        raise SpecError("overrides.json format_version 必须为 1", entry=str(path), field="format_version")
    overrides = doc.get("overrides", {})
    if not isinstance(overrides, dict):
        raise SpecError("overrides 必须是对象", entry=str(path), field="overrides")
    return overrides


def save_overrides(root: str | Path, overrides: dict[str, Any]) -> Path:
    path = Path(root) / "samples" / "overrides.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format_version": 1, "overrides": overrides}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


class LossAccumulator:
    def __init__(self, node: dict[str, Any], shard_id: str, quota_version: int) -> None:
        self.node = node
        self.shard_id = shard_id
        self.quota_version = int(quota_version)
        self.feature_count = len(node["weights_q"])
        self.scale = int(node["fixed_scale"])
        self.weights = np.asarray(node["weights_q"], dtype=np.int64)
        self.bias = np.int64(node["bias_q"])
        self.exact_sse = 0
        self.row_count = 0
        self.row_digest = hashlib.sha256()

    def add_rows(self, rows: list[dict[str, Any]]) -> int:
        added = 0
        for offset in range(0, len(rows), CHUNK_ROWS):
            raw_chunk = rows[offset : offset + CHUNK_ROWS]
            chunk = [
                _validate_row(row, self.feature_count, f"batch_row[{offset + index}]")
                for index, row in enumerate(raw_chunk)
            ]
            x = np.asarray([row["x"] for row in chunk], dtype=np.int64)
            y = np.asarray([row["y"] for row in chunk], dtype=np.int64)
            prediction_q = x @ self.weights + self.bias
            target_q = y * np.int64(self.scale)
            residual_q = prediction_q - target_q
            residual_object = residual_q.astype(object)
            squares = residual_object * residual_object
            self.exact_sse += int(np.sum(squares, dtype=object))
            for row in chunk:
                self.row_digest.update(canonical_bytes(row))
                self.row_digest.update(b"\n")
            self.row_count += len(chunk)
            added += len(chunk)
        return added

    def summary(self) -> dict[str, Any]:
        loss_sum_q_exact = self.exact_sse // self.scale
        mean_loss_q_exact = loss_sum_q_exact // self.row_count if self.row_count else 0
        sse_q = saturate_int64(self.exact_sse)
        loss_sum_q = saturate_int64(loss_sum_q_exact)
        mean_loss_q = saturate_int64(mean_loss_q_exact)
        saturated = (
            sse_q != self.exact_sse
            or loss_sum_q != loss_sum_q_exact
            or mean_loss_q != mean_loss_q_exact
        )
        rows_digest = "sd1:" + self.row_digest.hexdigest()
        params_digest = param_digest(self.node)
        basis = {
            "algorithm": LOSS_ALGORITHM,
            "truncation": "nonnegative_floor_divide; signed_divide_toward_zero",
            "overflow": "saturate_to_signed_int64_after_exact_accumulation",
            "node_id": self.node["node_id"],
            "shard_id": self.shard_id,
            "participating_rows": self.row_count,
            "quota_version": self.quota_version,
            "param_digest": params_digest,
            "sample_digest": rows_digest,
            "fixed_scale": self.scale,
            "sum_squared_residuals": sse_q,
            "sum_loss_q": loss_sum_q,
            "mean_loss_q": mean_loss_q,
            "overflow_saturated": saturated,
        }
        digest = "ld1:" + hashlib.sha256(canonical_bytes(basis)).hexdigest()
        return {**basis, "loss_digest": digest}


def compute_loss_summary(
    node: dict[str, Any],
    shard_id: str,
    quota_version: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    accumulator = LossAccumulator(node, shard_id, quota_version)
    accumulator.add_rows(rows)
    return accumulator.summary()
