from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .binning import build_edge_tables
from .cache import ArtifactCache, CacheTask
from .canonical import canonical_json, fingerprint, sha256_text
from .config import (
    FeatureConfig,
    TrainingConfig,
)
from .diagnostics import DIAGNOSTIC_SCHEMA, build_diagnostic_payload
from .errors import Failure
from .features import FeatureSnapshot, compute_feature_snapshot
from .fixed import ALLOWED_FRACTIONAL_BITS, ALLOWED_TIERS, TIER_BITS, fraction_scale
from .fusion import FUSION_SCHEMA, compute_fusion
from .model import MODEL_SCHEMA, build_dims, initialize_parameters, train_models
from .state import PipelineState
from .visibility import CVRVisibility, evaluate_cvr_visibility


PIPELINE_SCHEMA = "ctr-cvr-diagnostic-pipeline-v1"


@dataclass
class BaseComputation:
    feature_snapshot: FeatureSnapshot
    visibilities: dict[tuple[str, str], CVRVisibility]
    labels_key: str
    labels_payload: dict[str, Any]
    training_payload: dict[str, Any]
    tower_payload: dict[str, Any]
    fusion_payload: dict[str, Any]
    ledger: list[dict[str, Any]]
    reused_count: int
    recomputed_count: int
    stage_keys: dict[str, Any]


def _labels_for_state(
    state: PipelineState,
    training: TrainingConfig,
    fractional_bits: int,
) -> tuple[dict[tuple[str, str], CVRVisibility], dict[str, Any]]:
    scale = fraction_scale(fractional_bits)
    visibilities: dict[tuple[str, str], CVRVisibility] = {}
    rows = []
    for sample in sorted(state.samples.values(), key=lambda item: item.key):
        visibility = evaluate_cvr_visibility(
            sample,
            state.current_event_time_ms,
            state.conversion_delay_cutoff_ms,
            training,
            scale,
        )
        visibilities[sample.key] = visibility
        rows.append(
            {
                "sample": sample.sample_id,
                "click": sample.click,
                "conversion": sample.conversion,
                "delay_ms": sample.conversion_delay_ms,
                "visibility": visibility.to_dict(),
            }
        )
    payload = {
        "training_event_time_ms": state.current_event_time_ms,
        "cutoff_ms": state.conversion_delay_cutoff_ms,
        "samples": rows,
    }
    return visibilities, payload


def _tower_payload(training_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "dual-tower-output-v1",
        "tier": training_payload["tier"],
        "fractional_bits": training_payload["fractional_bits"],
        "ctr_fixed": training_payload["ctr_fixed"],
        "ctr_reference": training_payload["ctr_reference"],
        "cvr_fixed": training_payload["cvr_fixed"],
        "cvr_reference": training_payload["cvr_reference"],
        "loss_trace": training_payload["loss_trace"],
        "reference_loss_trace": training_payload["reference_loss_trace"],
    }


def compute_base(
    state: PipelineState,
    config: FeatureConfig,
    training: TrainingConfig,
    cache: ArtifactCache,
    *,
    workers: int,
) -> BaseComputation:
    samples = list(state.samples.values())
    edge_tables = build_edge_tables(config, samples, state.explicit_edges)
    feature_snapshot = compute_feature_snapshot(
        config,
        samples,
        edge_tables,
        state.quantization_tier,
        state.fixed_fractional_bits,
        cache,
        workers,
    )
    ledger = list(feature_snapshot.ledger)
    reused = feature_snapshot.reuse_count
    recomputed = feature_snapshot.recompute_count

    visibilities, labels_payload = _labels_for_state(
        state, training, state.fixed_fractional_bits
    )
    labels_key = fingerprint(labels_payload)
    training_signal_key = fingerprint(
        {
            "samples": [
                {
                    "sample": sample.sample_id,
                    "click": sample.click,
                    "effective_cvr_target": visibilities[sample.key].effective_target,
                    "visible": visibilities[sample.key].visible,
                    "reason": visibilities[sample.key].reason,
                    "weight_q": visibilities[sample.key].weight,
                }
                for sample in sorted(samples, key=lambda item: item.key)
            ]
        }
    )
    dims = build_dims(config)
    initial_parameter_fingerprint = fingerprint(
        initialize_parameters(
            dims,
            config.seed ^ training.seed,
            state.fixed_fractional_bits,
            state.quantization_tier,
        )
    )
    shared_key = fingerprint(
        {
            "schema": MODEL_SCHEMA,
            "feature_root": feature_snapshot.root_key,
            "training_signal_key": training_signal_key,
            "tier": state.quantization_tier,
            "fractional_bits": state.fixed_fractional_bits,
            "model_dims": dims.payload(),
            "parameter_fingerprint": initial_parameter_fingerprint,
            "feature_seed": config.seed,
            "training_seed": training.seed,
            "training": training.raw,
        }
    )
    sample_ids = tuple(sample.sample_id for sample in sorted(samples, key=lambda item: item.key))
    feature_names = feature_snapshot.feature_names
    shared_task = CacheTask(
        stage="shared_hidden",
        key=shared_key,
        samples=sample_ids,
        features=feature_names,
        upstream_key=feature_snapshot.root_key,
        compute=lambda: train_models(
            config,
            training,
            samples,
            feature_snapshot,
            visibilities,
            state.quantization_tier,
            state.fixed_fractional_bits,
        ),
    )
    shared_result = cache.get_many([shared_task], workers=workers)
    ledger.extend(shared_result.ledger)
    reused += shared_result.reused_count
    recomputed += shared_result.recomputed_count
    training_payload = shared_result.payloads[0]

    tower_key = fingerprint(
        {
            "schema": "dual-tower-output-v1",
            "shared_key": shared_key,
            "dims": dims.payload(),
        }
    )
    tower_task = CacheTask(
        stage="tower_outputs",
        key=tower_key,
        samples=sample_ids,
        features=feature_names,
        upstream_key=shared_key,
        compute=lambda: _tower_payload(training_payload),
    )
    tower_result = cache.get_many([tower_task], workers=workers)
    ledger.extend(tower_result.ledger)
    reused += tower_result.reused_count
    recomputed += tower_result.recomputed_count
    tower_payload = tower_result.payloads[0]

    fusion_key = fingerprint(
        {
            "schema": FUSION_SCHEMA,
            "tower_key": tower_key,
            "fusion": config.fusion.to_dict(),
            "tier": state.quantization_tier,
            "fractional_bits": state.fixed_fractional_bits,
        }
    )
    fusion_task = CacheTask(
        stage="fusion_ranking",
        key=fusion_key,
        samples=sample_ids,
        features=feature_names,
        upstream_key=tower_key,
        compute=lambda: compute_fusion(
            training_payload,
            config,
            state.quantization_tier,
            state.fixed_fractional_bits,
        ),
    )
    fusion_result = cache.get_many([fusion_task], workers=workers)
    ledger.extend(fusion_result.ledger)
    reused += fusion_result.reused_count
    recomputed += fusion_result.recomputed_count
    fusion_payload = fusion_result.payloads[0]
    return BaseComputation(
        feature_snapshot=feature_snapshot,
        visibilities=visibilities,
        labels_key=labels_key,
        labels_payload=labels_payload,
        training_payload=training_payload,
        tower_payload=tower_payload,
        fusion_payload=fusion_payload,
        ledger=ledger,
        reused_count=reused,
        recomputed_count=recomputed,
        stage_keys={
            "feature_quantization": feature_snapshot.root_key,
            "shared_hidden": shared_key,
            "tower_outputs": tower_key,
            "fusion_ranking": fusion_key,
        },
    )


def _neighbor_definitions(
    tier: str,
    fractional_bits: int,
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    tier_list = sorted(ALLOWED_TIERS, key=lambda name: TIER_BITS[name])
    current_tier_index = tier_list.index(tier)
    for offset, direction in ((-1, "previous"), (1, "next")):
        index = current_tier_index + offset
        if 0 <= index < len(tier_list):
            candidate = tier_list[index]
            if fractional_bits <= TIER_BITS[candidate] - 2:
                definitions.append(
                    {
                        "variant_kind": "quantization_tier",
                        "direction": direction,
                        "from": tier,
                        "to": candidate,
                        "tier": candidate,
                        "fractional_bits": fractional_bits,
                    }
                )
    bit_list = sorted(ALLOWED_FRACTIONAL_BITS)
    current_bit_index = bit_list.index(fractional_bits)
    for offset, direction in ((-1, "previous"), (1, "next")):
        index = current_bit_index + offset
        if 0 <= index < len(bit_list):
            candidate = bit_list[index]
            if candidate <= TIER_BITS[tier] - 2:
                definitions.append(
                    {
                        "variant_kind": "fractional_bits",
                        "direction": direction,
                        "from": fractional_bits,
                        "to": candidate,
                        "tier": tier,
                        "fractional_bits": candidate,
                    }
                )
    return definitions


def _state_material_key(state: PipelineState) -> str:
    return fingerprint(
        {
            "samples": [
                sample.feature_payload() | {
                    "click": sample.click,
                    "conversion": sample.conversion,
                    "delay": sample.conversion_delay_ms,
                }
                for sample in sorted(state.samples.values(), key=lambda item: item.key)
            ],
            "explicit_edges": {name: list(edges) for name, edges in sorted(state.explicit_edges.items())},
            "tier": state.quantization_tier,
            "fractional_bits": state.fixed_fractional_bits,
            "cutoff_ms": state.conversion_delay_cutoff_ms,
            "event_time_ms": state.current_event_time_ms,
        }
    )


def _budget_failures(
    state: PipelineState,
    training_payload: dict[str, Any],
) -> list[Failure]:
    failures: list[Failure] = []
    pending = training_payload.get("pending_samples", [])
    if not pending:
        return failures
    schedule = training_payload["schedule"]
    for sample_id in pending:
        request_id, item_id = sample_id.split(":", 1)
        failures.append(
            Failure(
                code="E_TRAIN_BUDGET_PENDING",
                message="sample point exceeds global_train_sample_budget and remains pending",
                request_id=request_id,
                item_id=item_id,
                sample_id=sample_id,
                field="global_train_sample_budget",
                details={
                    "selected_count": len(schedule["selected"]),
                    "pending_count": len(schedule["pending"]),
                },
            )
        )
    return failures


def _ledger_signature(ledger: list[dict[str, Any]]) -> str:
    normalized = []
    for entry in ledger:
        normalized.append(
            {
                "order": entry["order"],
                "stage": entry["stage"],
                "key": entry["key"],
                "samples": entry["samples"],
                "features": entry["features"],
                "upstream_key": entry.get("upstream_key"),
                "extra_refs": entry.get("extra_refs"),
            }
        )
    return fingerprint(normalized)


def analyze_state(
    state: PipelineState,
    config: FeatureConfig,
    training: TrainingConfig,
    cache: ArtifactCache,
    *,
    mode: str,
    command_index: int | None = None,
    command_type: str | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    workers = training.thread_pool_workers if workers is None else workers
    base = compute_base(state, config, training, cache, workers=workers)
    neighbors = _neighbor_definitions(
        state.quantization_tier, state.fixed_fractional_bits
    )
    variant_material = fingerprint(
        {
            "base_feature_root": base.feature_snapshot.root_key,
            "labels_key": base.labels_key,
            "state_material": _state_material_key(state),
            "training": training.raw,
            "neighbors": neighbors,
            "tolerances": training.tolerances.to_dict(),
        }
    )
    diagnostic_key = fingerprint(
        {
            "schema": DIAGNOSTIC_SCHEMA,
            "shared_hidden": base.stage_keys["shared_hidden"],
            "tower_outputs": base.stage_keys["tower_outputs"],
            "fusion_ranking": base.stage_keys["fusion_ranking"],
            "variant_material": variant_material,
        }
    )
    sample_ids = tuple(
        sample.sample_id for sample in sorted(state.samples.values(), key=lambda item: item.key)
    )

    def build_diagnostics() -> dict[str, Any]:
        neighbor_results = []
        for definition in neighbors:
            variant_state = state.clone()
            variant_state.quantization_tier = definition["tier"]
            variant_state.fixed_fractional_bits = definition["fractional_bits"]
            variant_base = compute_base(
                variant_state,
                config,
                training,
                ArtifactCache(None, collect_ledger=False),
                workers=workers,
            )
            neighbor_results.append(
                {
                    **definition,
                    "base_stage_keys": variant_base.stage_keys,
                    "fusion": variant_base.fusion_payload,
                }
            )
        return build_diagnostic_payload(
            config,
            training,
            base.feature_snapshot,
            base.training_payload,
            base.fusion_payload,
            neighbor_results,
            base.labels_payload,
        )

    diagnostic_task = CacheTask(
        stage="diagnostic_summary",
        key=diagnostic_key,
        samples=sample_ids,
        features=base.feature_snapshot.feature_names,
        upstream_key=base.stage_keys["fusion_ranking"],
        compute=build_diagnostics,
    )
    diagnostic_result = cache.get_many([diagnostic_task], workers=workers)
    diagnostics = diagnostic_result.payloads[0]
    ledger = base.ledger + diagnostic_result.ledger
    reused = base.reused_count + diagnostic_result.reused_count
    recomputed = base.recomputed_count + diagnostic_result.recomputed_count
    stage_keys = {**base.stage_keys, "diagnostic_summary": diagnostic_key}

    failures = list(state.failures)
    failures.extend(_budget_failures(state, base.training_payload))
    failures = sorted(failures, key=lambda item: item.sort_key())
    loss_trace = base.training_payload["loss_trace"]
    fixed_parameters = base.training_payload["fixed_parameters"]
    reference_parameters = base.training_payload["reference_parameters"]
    checkpoint_fingerprints = [
        checkpoint["parameter_fingerprint"]
        for checkpoint in base.training_payload.get("checkpoints", [])
    ]
    deterministic_core = {
        "schema": PIPELINE_SCHEMA,
        "command_index": command_index,
        "command_type": command_type,
        "state_material": _state_material_key(state),
        "active_config": {
            "tier": state.quantization_tier,
            "fractional_bits": state.fixed_fractional_bits,
            "cutoff_ms": state.conversion_delay_cutoff_ms,
            "event_time_ms": state.current_event_time_ms,
            "fusion": config.fusion.to_dict(),
            "tolerances": training.tolerances.to_dict(),
        },
        "binning": {
            name: edge_table.to_dict()
            for name, edge_table in sorted(base.feature_snapshot.edge_tables.items())
        },
        "stage_keys": stage_keys,
        "pointwise": diagnostics["points"],
        "stage_summary": diagnostics["stage_summary"],
        "quantization_flips": diagnostics["quantization_flips"],
        "reference_flips": diagnostics["reference_flips"],
        "metrics": diagnostics["metrics"],
        "selected_samples": base.training_payload["selected_samples"],
        "pending_samples": base.training_payload["pending_samples"],
        "loss_trace": loss_trace,
        "reference_loss_trace": base.training_payload["reference_loss_trace"],
        "fixed_parameters": fixed_parameters,
        "reference_parameters": reference_parameters,
        "parameter_fingerprint": fingerprint(fixed_parameters),
        "arithmetic_trace_fingerprint": fingerprint(base.training_payload["arithmetic_trace"]),
        "checkpoint_fingerprints": checkpoint_fingerprints,
        "failures": [failure.to_dict() for failure in failures],
        "ledger_signature": _ledger_signature(ledger),
    }
    core_digest = fingerprint(deterministic_core)
    report = {
        "schema_version": "1.0",
        "pipeline_schema": PIPELINE_SCHEMA,
        "run": {
            "mode": mode,
            "command_index": command_index,
            "command_type": command_type,
            "state_id": sha256_text(canonical_json(deterministic_core["state_material"]))[:16],
            "core_digest": core_digest,
            "tier": state.quantization_tier,
            "fractional_bits": state.fixed_fractional_bits,
            "quantization_scale": fraction_scale(state.fixed_fractional_bits),
            "training_event_time_ms": state.current_event_time_ms,
            "conversion_delay_cutoff_ms": state.conversion_delay_cutoff_ms,
        },
        "config": {
            "feature_config": config.raw,
            "training_config": training.raw,
            "model_dims": build_dims(config).payload(),
        },
        "binning_metadata": {
            name: edge_table.to_dict()
            for name, edge_table in sorted(base.feature_snapshot.edge_tables.items())
        },
        "training": {
            "selected_samples": base.training_payload["selected_samples"],
            "pending_samples": base.training_payload["pending_samples"],
            "loss_trace": loss_trace,
            "reference_loss_trace": base.training_payload["reference_loss_trace"],
            "final_fixed_parameters": fixed_parameters,
            "final_reference_parameters": reference_parameters,
            "parameter_fingerprint": fingerprint(fixed_parameters),
            "arithmetic_trace": base.training_payload["arithmetic_trace"],
            "arithmetic_trace_count": len(base.training_payload["arithmetic_trace"]),
            "checkpoints": base.training_payload.get("checkpoints", []),
        },
        "pointwise": diagnostics["points"],
        "stage_summary": diagnostics["stage_summary"],
        "quantization_flips": diagnostics["quantization_flips"],
        "reference_flips": diagnostics["reference_flips"],
        "metrics": diagnostics["metrics"],
        "neighbor_definitions": diagnostics["neighbor_definitions"],
        "cache": {
            "stage_keys": stage_keys,
            "ledger": ledger,
            "ledger_signature": _ledger_signature(ledger),
            "reused_count": reused,
            "recomputed_count": recomputed,
            "entry_count": len(ledger),
            "reuse_ratio": (reused / len(ledger)) if ledger else 0.0,
        },
        "failures": [failure.to_dict() for failure in failures],
        "deterministic_core_digest": core_digest,
    }
    return report


def report_core(report: dict[str, Any]) -> dict[str, Any]:
    """Stable view used to compare incremental, full and serial/concurrent runs."""
    return {
        "run": {
            "command_index": report["run"]["command_index"],
            "command_type": report["run"]["command_type"],
            "tier": report["run"]["tier"],
            "fractional_bits": report["run"]["fractional_bits"],
            "training_event_time_ms": report["run"]["training_event_time_ms"],
            "conversion_delay_cutoff_ms": report["run"]["conversion_delay_cutoff_ms"],
            "state_id": report["run"]["state_id"],
            "core_digest": report["run"]["core_digest"],
        },
        "pointwise": report["pointwise"],
        "stage_summary": report["stage_summary"],
        "quantization_flips": report["quantization_flips"],
        "reference_flips": report["reference_flips"],
        "metrics": report["metrics"],
        "failures": report["failures"],
        "ledger_signature": report["cache"]["ledger_signature"],
        "training": {
            "selected_samples": report["training"]["selected_samples"],
            "pending_samples": report["training"]["pending_samples"],
            "loss_trace": report["training"]["loss_trace"],
            "final_fixed_parameters": report["training"]["final_fixed_parameters"],
            "parameter_fingerprint": report["training"]["parameter_fingerprint"],
        },
    }
