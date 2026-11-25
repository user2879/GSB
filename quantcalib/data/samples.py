"""Deterministic local sample generation under a fixed integer seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quantcalib.data.calibration import write_calibration
from quantcalib.data.config import QuantConfig, write_config
from quantcalib.data.weightpack import write_weight_pack
from quantcalib.fingerprint import fingerprint
from quantcalib.models import CalibrationSet, WeightLayer
from quantcalib.parsers.receipts import write_q1, write_q2, write_q3
from quantcalib.models import UnifiedReceipt

DEFAULT_SEED = 20261125

# 3-layer MLP: 8 -> 6 -> 4 -> 3. Group granularities divide each size.
_LAYER_SPECS = [
    ("layer.fc1", (8, 6), 16),    # 48 elements
    ("layer.fc2", (6, 4), 12),    # 24 elements
    ("layer.fc3", (4, 3), 6),     # 12 elements
]


def generate_weights(seed: int = DEFAULT_SEED) -> list[WeightLayer]:
    rng = np.random.default_rng(seed)
    layers: list[WeightLayer] = []
    for index, (name, shape, granularity) in enumerate(_LAYER_SPECS):
        # Slight scale differences exercise asymmetric clipping paths.
        scale = 0.8 + 0.4 * index
        values = rng.normal(loc=0.0, scale=scale, size=shape)
        layers.append(WeightLayer(
            name=name,
            shape=shape,
            weights=values.astype(np.float64),
            content_version=index + 1,
            group_granularity=granularity,
        ))
    return layers


def generate_calibration(layers: list[WeightLayer],
                         seed: int = DEFAULT_SEED,
                         sample_count: int = 10) -> CalibrationSet:
    rng = np.random.default_rng(seed + 7)
    inputs = rng.normal(size=(sample_count, layers[0].shape[0]))
    hidden = inputs
    for index, layer in enumerate(layers):
        hidden = hidden @ layer.weights
        if index < len(layers) - 1:
            hidden = np.maximum(hidden, 0.0)
    reference = hidden + rng.normal(scale=1e-3, size=hidden.shape)
    return CalibrationSet(inputs=inputs, reference_outputs=reference, seed=seed)


def generate_receipts(layers: list[WeightLayer],
                      out_dir: str | Path) -> list[UnifiedReceipt]:
    out_dir = Path(out_dir)
    q1: list[UnifiedReceipt] = []
    q2: list[UnifiedReceipt] = []
    q3: list[UnifiedReceipt] = []
    for index, layer in enumerate(layers):
        coverage = 900 + 25 * index
        digest = fingerprint(layer.weights.reshape(-1)[:layer.group_granularity])[:16]
        q1.append(UnifiedReceipt(
            layer=layer.name, coverage_permille=coverage,
            source_version="q1",
            missing_fields=["param_version", "time_ms", "group_scale_digest",
                            "duplicate_refs", "weight_snapshot"],
            precision_labels={"layer": "utf8", "coverage_permille": "int32"},
        ))
        q2.append(UnifiedReceipt(
            layer=layer.name, coverage_permille=coverage,
            source_version="q2",
            missing_fields=["group_scale_digest", "duplicate_refs",
                            "weight_snapshot"],
            precision_labels={"layer": "utf8", "coverage_permille": "int32",
                              "param_version": "int32", "time_ms": "int64"},
            param_version=layer.content_version,
            time_ms=120 + index * 35,
        ))
        q3.append(UnifiedReceipt(
            layer=layer.name, coverage_permille=coverage + 5,
            source_version="q3", missing_fields=[],
            precision_labels={"layer": "utf8", "coverage_permille": "int32",
                              "param_version": "int64", "time_ms": "int64",
                              "group_scale_digest": "utf8",
                              "duplicate_refs": "utf8",
                              "weight_snapshot": "utf8"},
            param_version=layer.content_version,
            time_ms=140 + index * 40,
            group_scale_digest=digest,
            duplicate_refs=[layers[0].name] if index else [],
            weight_snapshot=fingerprint(layer.weights)[:16],
        ))

    write_q1(out_dir / "history.q1", q1)
    write_q2(out_dir / "history.q2.json", q2)
    write_q3(out_dir / "history.q3.bin", q3)
    return q1 + q2 + q3


def generate_workspace(workdir: str | Path,
                       cfg: QuantConfig | None = None) -> dict[str, str]:
    workdir = Path(workdir)
    for sub in ("weights", "calib", "config", "receipts", "out"):
        (workdir / sub).mkdir(parents=True, exist_ok=True)

    cfg = cfg or QuantConfig(seed=DEFAULT_SEED)
    layers = generate_weights(cfg.seed)
    calibration = generate_calibration(layers, cfg.seed)
    generate_receipts(layers, workdir / "receipts")
    write_weight_pack(workdir / "weights" / "weights.jsonl", layers)
    write_calibration(workdir / "calib" / "calibration.json", calibration)
    write_config(workdir / "config" / "quant.yaml", cfg)
    return {
        "weights": str(workdir / "weights" / "weights.jsonl"),
        "calibration": str(workdir / "calibration" / "calibration.json"),
        "config": str(workdir / "config" / "quant.yaml"),
        "receipts": str(workdir / "receipts"),
    }
