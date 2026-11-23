from __future__ import annotations

from typing import Any

from .fixedpoint import quantize
from .spec import Binding, Environment, Source, Specs


def _binding_for(environment: Environment, key: str, source_type_order: tuple[str, ...]) -> Binding:
    return environment.bindings.get(key, Binding(required_source_types=source_type_order, min_observed_seq=None))


def build_chain(specs: Specs, environment_id: str, key: str) -> dict[str, Any]:
    environment = specs.environments[environment_id]
    type_rank = {source_type: index for index, source_type in enumerate(specs.source_type_order)}
    binding = _binding_for(environment, key, specs.source_type_order)

    candidates: list[tuple[tuple[int, int, str], Source]] = []
    for source in environment.sources.values():
        if key in source.observations:
            candidates.append(((-source.priority_bias, type_rank[source.source_type], source.source_id), source))
    candidates.sort(key=lambda item: item[0])

    steps: list[dict[str, Any]] = []
    effective: dict[str, Any] | None = None
    for rank, (_, source) in enumerate(candidates, start=1):
        observation = source.observations[key]
        quantized = quantize(observation.value, specs.session.scale_power)
        binding_failures: list[str] = []
        if source.source_type not in binding.required_source_types:
            binding_failures.append("source_type_not_required")
        if binding.min_observed_seq is not None and observation.observed_seq < binding.min_observed_seq:
            binding_failures.append("observed_seq_below_binding_minimum")
        eligible = not binding_failures
        step = {
            "rank": rank,
            "source_id": source.source_id,
            "source_type": source.source_type,
            "priority_bias": source.priority_bias,
            "source_type_rank": type_rank[source.source_type],
            "sort_basis": [
                "priority_bias_desc",
                "source_type_order_asc",
                "source_id_asc",
            ],
            "sort_key": [-source.priority_bias, type_rank[source.source_type], source.source_id],
            "value_text": observation.value_text,
            "value_fixed": quantized.fixed,
            "quantization_error_fixed_units": quantized.as_dict()["quantization_error_fixed_units"],
            "quantization_error_upper_fixed_units": quantized.error_upper_fixed_units,
            "saturated": quantized.saturated,
            "observed_seq": observation.observed_seq,
            "captured_at_ms": observation.captured_at_ms,
            "binding_eligible": eligible,
            "binding_failures": binding_failures,
            "decision": "effective" if eligible and effective is None else ("shadowed" if eligible else "skipped_by_binding"),
        }
        steps.append(step)
        if eligible and effective is None:
            effective = {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "priority_bias": source.priority_bias,
                "environment_id": environment_id,
                "value_text": observation.value_text,
                "value_fixed": quantized.fixed,
                "observed_seq": observation.observed_seq,
                "captured_at_ms": observation.captured_at_ms,
                "quantization_error_fixed_units": quantized.as_dict()["quantization_error_fixed_units"],
                "quantization_error_upper_fixed_units": quantized.error_upper_fixed_units,
                "saturated": quantized.saturated,
            }

    signature_parts = [f"{step['source_type']}:{step['source_id']}@{step['priority_bias']}" for step in steps]
    return {
        "environment_id": environment_id,
        "depth": environment.depth,
        "key": key,
        "binding": {
            "required_source_types": list(binding.required_source_types),
            "min_observed_seq": binding.min_observed_seq,
        },
        "steps": steps,
        "chain_length": len(steps),
        "chain_types": [step["source_type"] for step in steps],
        "chain_signature": " -> ".join(signature_parts) if signature_parts else "<empty>",
        "effective": effective,
    }


def build_chains_for_view(specs: Specs, view_id: str, key: str) -> list[dict[str, Any]]:
    view = specs.views[view_id]
    return [build_chain(specs, environment_id, key) for environment_id in view.environment_priority]
