"""样本描述 JSONL 的加载校验与固定种子确定性生成 (业务要求 1)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .config import Config, TensorSpec
from .errors import CqError, E_SAMPLE_INVALID

SAMPLE_GENERATOR_SEED = 20261123


@dataclass
class SampleTensor:
    name: str
    spec: TensorSpec
    values: np.ndarray  # float64, 形状与 spec.shape 一致
    missing: np.ndarray  # bool, 同形状

    def column_view(self, column: int) -> tuple[np.ndarray, np.ndarray]:
        """取第 column 列 (沿 column_axis) 的值与缺失标记, 展平为一维."""
        v = np.moveaxis(self.values, self.spec.column_axis, -1)[..., column]
        m = np.moveaxis(self.missing, self.spec.column_axis, -1)[..., column]
        return np.ascontiguousarray(v).reshape(-1), np.ascontiguousarray(m).reshape(-1)

    def set_column(self, column: int, values: np.ndarray, missing: np.ndarray) -> None:
        """替换第 column 列的逐元素值与缺失标记 (moveaxis 返回视图, 原位写入)."""
        shape_rest = tuple(d for i, d in enumerate(self.spec.shape) if i != self.spec.column_axis)
        np.moveaxis(self.values, self.spec.column_axis, -1)[..., column] = values.reshape(shape_rest)
        np.moveaxis(self.missing, self.spec.column_axis, -1)[..., column] = missing.reshape(shape_rest)


@dataclass
class SampleSet:
    tensors: dict[str, SampleTensor]

    def tensor(self, name: str) -> SampleTensor:
        return self.tensors[name]


def _validate_element(tensor: str, idx: int, raw: dict, shape: tuple[int, ...]) -> tuple[tuple[int, ...], float, bool]:
    index = raw.get("index")
    if (
        not isinstance(index, list)
        or len(index) != len(shape)
        or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= shape[k] for k, i in enumerate(index))
    ):
        raise CqError(E_SAMPLE_INVALID, "元素索引越界或非法", tensor=tensor, element=idx, index=index)
    missing = bool(raw.get("missing", False))
    value = raw.get("value")
    if missing:
        v = 0.0 if value is None else float(value)
    else:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise CqError(E_SAMPLE_INVALID, "非缺失元素必须为有限浮点值", tensor=tensor, element=idx, value=value)
        v = float(value)
    return tuple(index), v, missing


def load_samples(path: str, config: Config) -> SampleSet:
    """逐行读取 JSONL 并校验: 形状与配置一致, 每个元素恰好出现一次."""
    values: dict[str, np.ndarray] = {}
    missing: dict[str, np.ndarray] = {}
    seen: dict[str, set[tuple[int, ...]]] = {}
    for spec in config.tensors:
        values[spec.name] = np.zeros(spec.shape, dtype=np.float64)
        missing[spec.name] = np.zeros(spec.shape, dtype=bool)
        seen[spec.name] = set()

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CqError(E_SAMPLE_INVALID, "JSONL 行解析失败", line=line_no, error=str(exc))
            name = rec.get("tensor")
            if name not in values:
                raise CqError(E_SAMPLE_INVALID, "样本引用未知张量", line=line_no, tensor=name)
            spec = config.tensor(name)
            if tuple(rec.get("shape", [])) != spec.shape:
                raise CqError(
                    E_SAMPLE_INVALID,
                    "样本形状与配置不一致",
                    line=line_no,
                    tensor=name,
                    sample_shape=rec.get("shape"),
                    config_shape=list(spec.shape),
                )
            elements = rec.get("elements")
            if not isinstance(elements, list):
                raise CqError(E_SAMPLE_INVALID, "elements 必须为列表", line=line_no, tensor=name)
            for eidx, eraw in enumerate(elements):
                index, v, m = _validate_element(name, eidx, eraw, spec.shape)
                if index in seen[name]:
                    raise CqError(E_SAMPLE_INVALID, "元素重复定义", line=line_no, tensor=name, index=list(index))
                seen[name].add(index)
                values[name][index] = v
                missing[name][index] = m

    tensors: dict[str, SampleTensor] = {}
    for spec in config.tensors:
        expected = int(np.prod(spec.shape))
        if len(seen[spec.name]) != expected:
            raise CqError(
                E_SAMPLE_INVALID,
                "样本元素数量与形状不符",
                tensor=spec.name,
                expected=expected,
                got=len(seen[spec.name]),
            )
        tensors[spec.name] = SampleTensor(
            name=spec.name, spec=spec, values=values[spec.name], missing=missing[spec.name]
        )
    return SampleSet(tensors=tensors)


def generate_samples(config: Config, seed: int = SAMPLE_GENERATOR_SEED) -> list[dict]:
    """按固定种子确定性生成样本描述记录 (每行一个张量)."""
    records: list[dict] = []
    for t_idx, spec in enumerate(config.tensors):
        rng = np.random.default_rng(seed + t_idx * 100003)
        size = spec.shape
        base = rng.standard_normal(size)
        # 逐列施加不同尺度与偏置, 使逐列标定产生真实差异
        col_scale = 0.25 + np.exp(rng.standard_normal(spec.num_columns) * 0.8)
        col_shift = rng.standard_normal(spec.num_columns) * 2.0
        bshape = [1] * len(size)
        bshape[spec.column_axis] = spec.num_columns
        vals = base * col_scale.reshape(bshape) + col_shift.reshape(bshape)
        # 注入少量离群值触发饱和截断
        outlier_mask = rng.random(size) < 0.02
        vals = vals + outlier_mask * rng.choice([-1.0, 1.0], size=size) * (8.0 + 6.0 * rng.random(size))
        # 注入少量缺失
        miss = rng.random(size) < 0.01
        elements = []
        for index in np.ndindex(size):
            rec = {"index": list(index), "value": round(float(vals[index]), 10)}
            if miss[index]:
                rec["missing"] = True
            elements.append(rec)
        records.append({"tensor": spec.name, "shape": list(size), "elements": elements})
    return records


def write_samples_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def samples_to_records(samples: SampleSet, config: Config) -> list[dict]:
    """把内存样本集导出为 JSONL 记录 (用于全量重建验证的输入物化)."""
    records = []
    for spec in config.tensors:
        st = samples.tensor(spec.name)
        elements = []
        for index in np.ndindex(spec.shape):
            rec = {"index": list(index), "value": float(st.values[index])}
            if st.missing[index]:
                rec["missing"] = True
            elements.append(rec)
        records.append({"tensor": spec.name, "shape": list(spec.shape), "elements": elements})
    return records
