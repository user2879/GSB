from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .binning import EdgeTable
from .cache import ArtifactCache, CacheTask
from .canonical import fingerprint
from .config import FeatureConfig
from .fixed import fraction_scale, quantize_real
from .samples import effective_dense_value, effective_sparse_value
from .state import Sample


FEATURE_SCHEMA = "feature-quant-v1"


@dataclass
class FeatureSnapshot:
    entries: dict[tuple[str, str], dict[str, dict[str, Any]]]
    fixed_vectors: dict[tuple[str, str], dict[str, int]]
    raw_vectors: dict[tuple[str, str], dict[str, float]]
    edge_tables: dict[str, EdgeTable]
    ledger: list[dict[str, Any]]
    reuse_count: int
    recompute_count: int
    root_key: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(sorted({name for row in self.entries.values() for name in row}))

    def vector_fixed(self, sample_key: tuple[str, str], feature_names: tuple[str, ...]) -> list[int]:
        row = self.fixed_vectors[sample_key]
        return [row[name] for name in feature_names]

    def vector_raw(self, sample_key: tuple[str, str], feature_names: tuple[str, ...]) -> list[float]:
        row = self.raw_vectors[sample_key]
        return [row[name] for name in feature_names]

    def entries_sorted(self, sample_key: tuple[str, str]) -> list[dict[str, Any]]:
        row = self.entries[sample_key]
        return [row[name] for name in sorted(row)]


def _quant_record(
    *,
    feature: str,
    kind: str,
    raw_value: float,
    tier: str,
    fractional_bits: int,
    bin_id: int | None = None,
    dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    scale = fraction_scale(fractional_bits)
    rounding = quantize_real(raw_value, fractional_bits, tier)
    dequantized = rounding.value / scale
    return {
        "feature": feature,
        "kind": kind,
        "raw_value": raw_value,
        "bin": bin_id,
        "quantization_tier": tier,
        "fractional_bits": fractional_bits,
        "step_num": 1,
        "step_den": scale,
        "step": 1.0 / scale,
        "integer": rounding.value,
        "dequantized": dequantized,
        "residual_units": rounding.residual,
        "residual": raw_value - dequantized,
        "rounding_direction": rounding.direction,
        "clipped": rounding.clipped,
        "dependencies": list(dependencies),
    }


def compute_feature_snapshot(
    config: FeatureConfig,
    samples: list[Sample],
    edge_tables: dict[str, EdgeTable],
    quantization_tier: str,
    fractional_bits: int,
    cache: ArtifactCache,
    workers: int,
) -> FeatureSnapshot:
    ordered_samples = sorted(samples, key=lambda item: item.key)
    dense_by_name = {feature.name: feature for feature in config.dense_features}
    sparse_names = {feature.name for feature in config.sparse_features}
    ledger: list[dict[str, Any]] = []
    reuse = 0
    recompute = 0
    entries: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
        sample.key: {} for sample in ordered_samples
    }

    source_tasks: list[CacheTask] = []
    source_meta: list[tuple[tuple[str, str], str]] = []
    for sample in ordered_samples:
        for feature_name in sorted(list(dense_by_name) + sorted(sparse_names)):
            if feature_name in dense_by_name:
                feature = dense_by_name[feature_name]
                edge_table = edge_tables[feature_name]
                raw = effective_dense_value(sample, feature_name)
                key = fingerprint(
                    {
                        "schema": FEATURE_SCHEMA,
                        "kind": "dense",
                        "sample": sample.key,
                        "feature": feature_name,
                        "raw": raw,
                        "edge": edge_table.fingerprint(),
                        "tier": quantization_tier,
                        "fractional_bits": fractional_bits,
                    }
                )

                def compute_dense(
                    sample=sample,
                    feature=feature,
                    edge_table=edge_table,
                    raw=raw,
                ) -> dict[str, Any]:
                    bin_id = edge_table.assign(raw)
                    assert feature.bins is not None
                    center = (bin_id + 0.5) / feature.bins
                    return _quant_record(
                        feature=feature.name,
                        kind="dense",
                        raw_value=center,
                        tier=quantization_tier,
                        fractional_bits=fractional_bits,
                        bin_id=bin_id,
                    )

                source_tasks.append(
                    CacheTask(
                        stage="feature_quantization",
                        key=key,
                        samples=(sample.sample_id,),
                        features=(feature_name,),
                        compute=compute_dense,
                        upstream_key=edge_table.fingerprint(),
                    )
                )
                source_meta.append((sample.key, feature_name))
            else:
                raw = effective_sparse_value(sample, feature_name)
                key = fingerprint(
                    {
                        "schema": FEATURE_SCHEMA,
                        "kind": "sparse",
                        "sample": sample.key,
                        "feature": feature_name,
                        "raw": raw,
                        "tier": quantization_tier,
                        "fractional_bits": fractional_bits,
                    }
                )

                def compute_sparse(sample=sample, feature_name=feature_name, raw=raw) -> dict[str, Any]:
                    return _quant_record(
                        feature=feature_name,
                        kind="sparse",
                        raw_value=raw,
                        tier=quantization_tier,
                        fractional_bits=fractional_bits,
                    )

                source_tasks.append(
                    CacheTask(
                        stage="feature_quantization",
                        key=key,
                        samples=(sample.sample_id,),
                        features=(feature_name,),
                        compute=compute_sparse,
                    )
                )
                source_meta.append((sample.key, feature_name))

    source_result = cache.get_many(source_tasks, workers=workers)
    ledger.extend(source_result.ledger)
    reuse += source_result.reused_count
    recompute += source_result.recomputed_count
    source_keys: dict[tuple[tuple[str, str], str], str] = {}
    for meta, task, payload in zip(source_meta, source_tasks, source_result.payloads):
        entries[meta[0]][meta[1]] = payload
        source_keys[meta] = task.key

    cross_tasks: list[CacheTask] = []
    cross_meta: list[tuple[tuple[str, str], str]] = []
    for sample in ordered_samples:
        for cross in sorted(config.crosses, key=lambda item: item.name):
            left_payload = entries[sample.key][cross.left]
            right_payload = entries[sample.key][cross.right]
            left_raw = float(left_payload["raw_value"])
            right_raw = float(right_payload["raw_value"])
            raw = left_raw * right_raw
            left_key = source_keys[(sample.key, cross.left)]
            right_key = source_keys[(sample.key, cross.right)]
            key = fingerprint(
                {
                    "schema": FEATURE_SCHEMA,
                    "kind": "cross",
                    "sample": sample.key,
                    "feature": cross.name,
                    "left": cross.left,
                    "right": cross.right,
                    "left_key": left_key,
                    "right_key": right_key,
                    "raw": raw,
                    "tier": quantization_tier,
                    "fractional_bits": fractional_bits,
                }
            )

            def compute_cross(cross=cross, raw=raw) -> dict[str, Any]:
                return _quant_record(
                    feature=cross.name,
                    kind="cross",
                    raw_value=raw,
                    tier=quantization_tier,
                    fractional_bits=fractional_bits,
                    dependencies=(cross.left, cross.right),
                )

            cross_tasks.append(
                CacheTask(
                    stage="feature_quantization",
                    key=key,
                    samples=(sample.sample_id,),
                    features=tuple(sorted((cross.name, cross.left, cross.right))),
                    compute=compute_cross,
                    upstream_key=fingerprint({"left": left_key, "right": right_key}),
                )
            )
            cross_meta.append((sample.key, cross.name))

    cross_result = cache.get_many(cross_tasks, workers=workers)
    ledger.extend(cross_result.ledger)
    reuse += cross_result.reused_count
    recompute += cross_result.recomputed_count
    for meta, payload in zip(cross_meta, cross_result.payloads):
        entries[meta[0]][meta[1]] = payload

    fixed_vectors: dict[tuple[str, str], dict[str, int]] = {}
    raw_vectors: dict[tuple[str, str], dict[str, float]] = {}
    for sample in ordered_samples:
        fixed_vectors[sample.key] = {
            name: int(payload["integer"]) for name, payload in entries[sample.key].items()
        }
        raw_vectors[sample.key] = {
            name: float(payload["raw_value"]) for name, payload in entries[sample.key].items()
        }

    root_payload = {
        "schema": FEATURE_SCHEMA,
        "tier": quantization_tier,
        "fractional_bits": fractional_bits,
        "edge_fingerprints": {name: table.fingerprint() for name, table in sorted(edge_tables.items())},
        "samples": [
            {
                "sample": sample.sample_id,
                "features": [
                    {
                        "name": name,
                        "integer": entries[sample.key][name]["integer"],
                        "raw": entries[sample.key][name]["raw_value"],
                    }
                    for name in sorted(entries[sample.key])
                ],
            }
            for sample in ordered_samples
        ],
    }
    return FeatureSnapshot(
        entries=entries,
        fixed_vectors=fixed_vectors,
        raw_vectors=raw_vectors,
        edge_tables=edge_tables,
        ledger=ledger,
        reuse_count=reuse,
        recompute_count=recompute,
        root_key=fingerprint(root_payload),
    )
