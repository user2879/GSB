"""Deterministic layered feature pipeline in fixed point integer domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .fingerprint import canonical_json, short_digest
from .fixedpoint import FixedConfig, divide_round, quantize, scaled, sqrt_fixed
from .models import Artifact, Pipeline, Sample, StageDef

RAW_PREFIX = "raw"
STAGE_PREFIX = {
    "standardize": "std",
    "bucketize": "bkt",
    "interact": "int",
    "train_view": "view",
}


def raw_artifact_id(sample_id: str) -> str:
    return f"{RAW_PREFIX}:{sample_id}"


def artifact_id(stage_name: str, scope: str) -> str:
    kind_prefix = stage_name  # stage names already carry stable meaning.
    return f"{kind_prefix}:{scope}"


def param_fingerprint(stage: StageDef, fixed: FixedConfig, param_versions: Dict[str, int]) -> str:
    relevant_prefixes = (f"{stage.name}.",)
    relevant_versions = {
        key: value for key, value in param_versions.items() if key.startswith(relevant_prefixes)
    }
    body = {
        "stage": stage.name,
        "kind": stage.kind,
        "upstream": list(stage.upstream),
        "params": stage.params,
        "fixed": {"frac_bits": fixed.frac_bits, "gear": fixed.gear},
        "param_versions": {key: relevant_versions[key] for key in sorted(relevant_versions)},
    }
    return short_digest(body)


def content_digest(
    stage_name: str,
    scope: str,
    payload: Any,
    upstream_digests: List[str],
    parameter_fingerprint: str,
) -> str:
    body = {
        "stage": stage_name,
        "scope": scope,
        "payload": payload,
        "upstream_digests": sorted(upstream_digests),
        "parameter_fingerprint": parameter_fingerprint,
    }
    return short_digest(body)


def cache_key(artifact_id_value: str, digest: str) -> str:
    return short_digest({"artifact_id": artifact_id_value, "digest": digest}, length=24)


@dataclass
class StageResult:
    artifacts: Tuple[Artifact, ...]
    quant_log: Tuple[dict, ...]


class FixedPipeline:
    """Computes every stage deterministically from samples and definitions."""

    def __init__(self, definition: Pipeline, fixed: FixedConfig):
        definition  # retained through the methods below via explicit arguments
        self.definition = definition
        fixed.validate()
        self.fixed = fixed

    def raw_artifacts(self, samples: Tuple[Sample, ...]) -> Tuple[Artifact, ...]:
        artifacts: List[Artifact] = []
        for sample in samples:
            values = [scaled(value, self.fixed.frac_bits) for value in sample.features]
            quantized_values: List[int] = []
            residuals: List[int] = []
            for value in values:
                q, residual = quantize(value, self.fixed.gear)
                quantized_values.append(q)
                residuals.append(residual)
            scope = f"sample:{sample.sample_id}"
            payload = {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "raw_ints": quantized_values,
            }
            pf = short_digest({"stage": RAW_PREFIX, "fixed": self.fixed.__dict__})
            digest = content_digest(RAW_PREFIX, scope, payload, [], pf)
            artifacts.append(
                Artifact(
                    artifact_id=raw_artifact_id(sample.sample_id),
                    stage=RAW_PREFIX,
                    scope=scope,
                    payload=payload,
                    digest=digest,
                    upstream=(),
                    params={"frac_bits": self.fixed.frac_bits, "gear": self.fixed.gear},
                    param_versions={},
                    quant_step=self.fixed.gear,
                    residual=sum(abs(item) for item in residuals),
                )
            )
        return tuple(artifacts)

    def compute_stage(
        self,
        stage: StageDef,
        samples: Tuple[Sample, ...],
        upstream: Dict[str, Artifact],
        param_versions: Dict[str, int],
    ) -> StageResult:
        dispatch = {
            "standardize": self._standardize,
            "bucketize": self._bucketize,
            "interact": self._interact,
            "train_view": self._train_view,
        }
        return dispatch[stage.kind](stage, samples, upstream, param_versions)

    def _make_artifact(
        self,
        stage: StageDef,
        scope: str,
        payload: Any,
        upstream_artifacts: List[Artifact],
        param_versions: Dict[str, int],
        residual: int,
    ) -> Artifact:
        pf = param_fingerprint(stage, self.fixed, param_versions)
        digest = content_digest(
            stage.name,
            scope,
            payload,
            [item.digest for item in upstream_artifacts],
            pf,
        )
        return Artifact(
            artifact_id=artifact_id(stage.name, scope),
            stage=stage.name,
            scope=scope,
            payload=payload,
            digest=digest,
            upstream=tuple(item.artifact_id for item in upstream_artifacts),
            params={
                "stage_params": stage.params,
                "frac_bits": self.fixed.frac_bits,
                "gear": self.fixed.gear,
            },
            param_versions=dict(param_versions),
            quant_step=self.fixed.gear,
            residual=residual,
        )

    def _standardize(
        self,
        stage: StageDef,
        samples: Tuple[Sample, ...],
        upstream: Dict[str, Artifact],
        param_versions: Dict[str, int],
    ) -> StageResult:
        count = len(samples)
        width = self.definition.feature_count
        scale = self.fixed.scale
        sums = [0] * width
        for sample in samples:
            raw = upstream[raw_artifact_id(sample.sample_id)]
            for idx, value in enumerate(raw.payload["raw_ints"]):
                sums[idx] += value
        means = [divide_round(total, count) for total in sums]
        sum_sq = [0] * width
        for sample in samples:
            raw = upstream[raw_artifact_id(sample.sample_id)]
            for idx, value in enumerate(raw.payload["raw_ints"]):
                delta = value - means[idx]
                sum_sq[idx] += delta * delta
        # Variance in fixed units; divide count before sqrt to keep magnitude.
        variances = [divide_round(total, count) for total in sum_sq]
        stds = [sqrt_fixed(max(variance, scale)) for variance in variances]
        stats_scope = "global"
        stats_payload = {
            "sum": sums,
            "mean": means,
            "variance": variances,
            "std": stds,
            "accumulation_order": [sample.sample_id for sample in samples],
        }
        stats = self._make_artifact(
            stage,
            stats_scope,
            stats_payload,
            [upstream[raw_artifact_id(sample.sample_id)] for sample in samples],
            param_versions,
            0,
        )
        artifacts: List[Artifact] = [stats]
        quant_log: List[dict] = []
        for sample in samples:
            raw = upstream[raw_artifact_id(sample.sample_id)]
            z_values: List[int] = []
            residual_total = 0
            for idx, value in enumerate(raw.payload["raw_ints"]):
                numerator = (value - means[idx]) * scale
                z = divide_round(numerator, stds[idx])
                qz, residual = quantize(z, self.fixed.gear)
                z_values.append(qz)
                residual_total += abs(residual)
                quant_log.append(
                    {
                        "sample_id": sample.sample_id,
                        "feature": idx,
                        "step": self.fixed.gear,
                        "before": z,
                        "after": qz,
                        "residual": residual,
                    }
                )
            scope = f"sample:{sample.sample_id}"
            artifacts.append(
                self._make_artifact(
                    stage,
                    scope,
                    {"sample_id": sample.sample_id, "z_ints": z_values},
                    [raw, stats],
                    param_versions,
                    residual_total,
                )
            )
        return StageResult(tuple(artifacts), tuple(quant_log))

    def _bucketize(
        self,
        stage: StageDef,
        samples: Tuple[Sample, ...],
        upstream: Dict[str, Artifact],
        param_versions: Dict[str, int],
    ) -> StageResult:
        bucket_count = int(stage.params["bucket_count"])
        std_stage = self._require_upstream(stage, "standardize")
        stats = upstream[artifact_id(std_stage, "global")]
        means = stats.payload["mean"]
        stds = stats.payload["std"]
        # Equal width edges in fixed point: mean +/- 3 std, bucket_count cells.
        edges: List[List[int]] = []
        quant_log: List[dict] = []
        for mean, std in zip(means, stds):
            low = mean - 3 * std
            high = mean + 3 * std
            span = high - low
            width_step = divide_round(span, bucket_count)
            feature_edges = []
            for idx in range(bucket_count + 1):
                edge, edge_residual = quantize(low + width_step * idx, self.fixed.gear)
                feature_edges.append(edge)
            edges.append(feature_edges)
            quant_log.append(
                {
                    "feature": len(quant_log),
                    "step": self.fixed.gear,
                    "width_step": width_step,
                    "edges": feature_edges,
                }
            )
        artifacts: List[Artifact] = []
        for sample in samples:
            std_art = upstream[artifact_id(std_stage, f"sample:{sample.sample_id}")]
            buckets: List[int] = []
            residual_total = 0
            for idx, z in enumerate(std_art.payload["z_ints"]):
                bucket = bucket_count - 1
                for edge_idx in range(1, bucket_count):
                    if z < edges[idx][edge_idx]:
                        bucket = edge_idx - 1
                        break
                qbucket, residual = bucket, 0
                residual_total += abs(residual)
                buckets.append(qbucket)
            scope = f"sample:{sample.sample_id}"
            artifacts.append(
                self._make_artifact(
                    stage,
                    scope,
                    {"sample_id": sample.sample_id, "bucket_ids": buckets, "edges": edges},
                    [std_art, stats],
                    param_versions,
                    residual_total,
                )
            )
        return StageResult(tuple(artifacts), tuple(quant_log))

    def _interact(
        self,
        stage: StageDef,
        samples: Tuple[Sample, ...],
        upstream: Dict[str, Artifact],
        param_versions: Dict[str, int],
    ) -> StageResult:
        degree = int(stage.params["degree"])
        std_stage = self._require_upstream(stage, "standardize")
        scale = self.fixed.scale
        artifacts: List[Artifact] = []
        quant_log: List[dict] = []
        for sample in samples:
            std_art = upstream[artifact_id(std_stage, f"sample:{sample.sample_id}")]
            z_values = std_art.payload["z_ints"]
            if degree > len(z_values):
                raise ValueError("interaction degree exceeds feature count")
            products: List[int] = []
            residual_total = 0
            for start in range(len(z_values) - degree + 1):
                product = scale
                for offset in range(degree):
                    product = divide_round(product * z_values[start + offset], scale)
                qproduct, residual = quantize(product, self.fixed.gear)
                products.append(qproduct)
                residual_total += abs(residual)
                quant_log.append(
                    {
                        "sample_id": sample.sample_id,
                        "start": start,
                        "step": self.fixed.gear,
                        "before": product,
                        "after": qproduct,
                        "residual": residual,
                    }
                )
            scope = f"sample:{sample.sample_id}"
            artifacts.append(
                self._make_artifact(
                    stage,
                    scope,
                    {"sample_id": sample.sample_id, "degree": degree, "interaction_ints": products},
                    [std_art],
                    param_versions,
                    residual_total,
                )
            )
        return StageResult(tuple(artifacts), tuple(quant_log))

    def _train_view(
        self,
        stage: StageDef,
        samples: Tuple[Sample, ...],
        upstream: Dict[str, Artifact],
        param_versions: Dict[str, int],
    ) -> StageResult:
        interact_stage = self._require_upstream(stage, "interact")
        weights = [scaled(value, self.fixed.frac_bits) for value in stage.params["weights"]]
        threshold = scaled(stage.params["threshold"], self.fixed.frac_bits)
        rows: List[dict] = []
        score_sum = 0
        included = 0
        residual_total = 0
        quant_log: List[dict] = []
        for sample in samples:
            interaction = upstream[
                artifact_id(interact_stage, f"sample:{sample.sample_id}")
            ]
            values = interaction.payload["interaction_ints"]
            score = 0
            for idx, value in enumerate(values):
                weight = weights[min(idx, len(weights) - 1)]
                score += divide_round(value * weight, self.fixed.scale)
            qscore, residual = quantize(score, self.fixed.gear)
            residual_total += abs(residual)
            keep = qscore >= threshold
            score_sum += qscore
            included += 1 if keep else 0
            quant_log.append(
                {
                    "sample_id": sample.sample_id,
                    "step": self.fixed.gear,
                    "before": score,
                    "after": qscore,
                    "residual": residual,
                }
            )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "score_int": qscore,
                    "included": keep,
                    "label": sample.label,
                }
            )
        payload = {
            "rows": rows,
            "score_sum": score_sum,
            "included_count": included,
            "weights_int": weights,
            "threshold_int": threshold,
            "order": [sample.sample_id for sample in samples],
        }
        upstream_artifacts = [
            upstream[artifact_id(interact_stage, f"sample:{sample.sample_id}")]
            for sample in samples
        ]
        artifact = self._make_artifact(
            stage, "global", payload, upstream_artifacts, param_versions, residual_total
        )
        return StageResult((artifact,), tuple(quant_log))

    def _require_upstream(self, stage: StageDef, kind: str) -> str:
        for ref in stage.upstream:
            ref_stage = self.definition.stage(ref)
            if ref_stage.kind == kind:
                return ref
        raise ValueError(f"stage {stage.name} must reference a {kind} stage")
