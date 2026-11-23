"""确定性多传感器时序样本的现场生成与 JSONL 读写。

观测 = 截距 + 线性趋势 + 正弦周期 + 固定种子高斯噪声；
每个序列使用 :class:`numpy.random.SeedSequence` 派生独立 Generator，
不触碰全局随机状态，任意顺序、任意次数运行结果逐元素一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    sensor_id: str
    length: int
    seed: int
    slope: float
    period: int
    amplitude: float
    noise_sigma: float
    missing_every: int = 0          # >0 时每 N 个点制造一个缺失
    interval_ms: int = 1000
    start_ms: int = 0


def _series_rng(seed: int, series_id: str) -> np.random.Generator:
    ss = np.random.SeedSequence([seed, int.from_bytes(series_id.encode("utf-8"), "little") & 0xFFFFFFFF])
    return np.random.default_rng(ss)


def generate_series(spec: SeriesSpec) -> list[dict]:
    rng = _series_rng(spec.seed, spec.series_id)
    idx = np.arange(spec.length, dtype=np.float64)
    trend = spec.slope * idx
    seasonal = spec.amplitude * np.sin(2.0 * np.pi * idx / max(spec.period, 1))
    noise = rng.normal(0.0, spec.noise_sigma, size=spec.length)
    values = 10.0 + trend + seasonal + noise
    rows: list[dict] = []
    for i in range(spec.length):
        missing = spec.missing_every > 0 and i > 0 and (i % spec.missing_every == 0)
        rows.append({
            "series_id": spec.series_id,
            "sensor_id": spec.sensor_id,
            "timestamp_ms": spec.start_ms + i * spec.interval_ms,
            "value": None if missing else round(float(values[i]), 6),
            "missing": bool(missing),
        })
    return rows


def default_specs(seed: int = 20261124, series_count: int = 4, length: int = 96) -> list[SeriesSpec]:
    specs: list[SeriesSpec] = []
    for k in range(series_count):
        specs.append(SeriesSpec(
            series_id=f"S{k:03d}",
            sensor_id=f"sensor-{k % 3}",
            length=length,
            seed=seed,
            slope=0.05 * (k + 1),
            period=12 + 2 * k,
            amplitude=2.0 + 0.25 * k,
            noise_sigma=0.15,
            missing_every=11 if k % 2 == 0 else 0,
        ))
    return specs


def write_jsonl(path: str | Path, specs: list[SeriesSpec]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for spec in specs:
            for row in generate_series(spec):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            grouped.setdefault(row["series_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r["timestamp_ms"])
    return grouped
