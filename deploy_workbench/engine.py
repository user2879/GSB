from __future__ import annotations

from decimal import Decimal
from typing import Any

from .canonical import decimal_text, digest_value
from .fixed import FixedQuantizer, compare_keys, evaluate_task, heat_level
from .model import EnvironmentSpec, SourceDef, WorkbenchSpec


def _source_sort_key(
    source: SourceDef,
    priority_rank: dict[str, int],
    type_bias: dict[str, int],
) -> tuple[int, int, str]:
    return (type_bias[source.type], priority_rank[source.id], source.id)


def resolve_environment(
    spec: WorkbenchSpec,
    env: EnvironmentSpec,
    quantizer: FixedQuantizer,
) -> dict[str, Any]:
    priority_rank = {source_id: index for index, source_id in enumerate(env.source_priority)}
    all_keys = env.all_entry_keys()
    key_results: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        candidates = [
            spec.sources[source_id]
            for source_id in env.available_sources
            if key in env.entries[source_id]
        ]
        candidates.sort(key=lambda source: _source_sort_key(source, priority_rank, spec.source_type_bias))
        chain: list[dict[str, Any]] = []
        binding = env.bindings.get(key)
        selected_row: int | None = None
        for row_number, source in enumerate(candidates, start=1):
            raw_value = env.entries[source.id][key]
            quantized = quantizer.quantize(raw_value)
            binding_satisfied = True if binding is None else binding.allows(source)
            selected = binding_satisfied and selected_row is None
            if selected:
                selected_row = row_number
            chain.append(
                {
                    "rank": row_number,
                    "source_id": source.id,
                    "source_type": source.type,
                    "raw_value": decimal_text(raw_value),
                    "value_fixed": quantized.fixed,
                    "value_residual": decimal_text(quantized.residual),
                    "type_bias": spec.source_type_bias[source.type],
                    "explicit_priority": priority_rank[source.id],
                    "sort_tuple": [
                        spec.source_type_bias[source.type],
                        priority_rank[source.id],
                        source.id,
                    ],
                    "sort_basis": (
                        f"type_bias={spec.source_type_bias[source.type]};"
                        f"priority={priority_rank[source.id]};source_id={source.id}"
                    ),
                    "binding_satisfied": binding_satisfied,
                    "selected": selected,
                }
            )

        if selected_row is None:
            status = "no_source" if not candidates else "no_binding_match"
            key_results[key] = {
                "key": key,
                "resolved": False,
                "status": status,
                "chain": chain,
                "chain_digest": digest_value({"key": key, "chain": chain}),
                "available_source_count": len(candidates),
            }
            continue

        selected = chain[selected_row - 1]
        value = Decimal(selected["raw_value"])
        quantized = quantizer.quantize(value)
        unit = quantizer.unit
        key_results[key] = {
            "key": key,
            "resolved": True,
            "status": "resolved",
            "chain": chain,
            "chain_digest": digest_value({"key": key, "chain": chain}),
            "available_source_count": len(candidates),
            "effective_source_id": selected["source_id"],
            "effective_source_type": selected["source_type"],
            "effective_raw": selected["raw_value"],
            "effective_fixed": quantized.fixed,
            "effective_value": decimal_text(Decimal(quantized.fixed) / Decimal(quantizer.scale)),
            "quantization_error": decimal_text(quantized.residual),
            "quantization_error_abs": decimal_text(quantized.error),
            "quantization_error_bound": decimal_text(unit),
            "comparison_error_bound": decimal_text(unit),
            "fixed_scale": quantizer.scale,
            "saturated": quantized.saturated,
            "heat_level": heat_level(quantized.error, unit),
        }

    tasks = {
        task.id: evaluate_task(task, key_results, quantizer, spec.error_dominance_threshold)
        for task in spec.task_order()
    }
    comparisons = compare_keys(key_results, quantizer)
    chain_digest = digest_value(
        {key: key_results[key]["chain_digest"] for key in sorted(key_results)}
    )
    environment_digest = digest_value(
        {
            "environment_id": env.id,
            "depth": env.depth,
            "keys": key_results,
            "tasks": tasks,
            "comparisons": comparisons,
        }
    )
    return {
        "environment_id": env.id,
        "depth": env.depth,
        "available_sources": list(env.available_sources),
        "source_priority": list(env.source_priority),
        "quantizer": quantizer.policy(),
        "keys": key_results,
        "tasks": tasks,
        "comparisons": comparisons,
        "source_chain_digests": {
            key: key_results[key]["chain_digest"] for key in sorted(key_results)
        },
        "source_chain_digest": chain_digest,
        "conclusion_digest": environment_digest,
        "resolved_key_count": sum(1 for value in key_results.values() if value["resolved"]),
        "unresolved_key_count": sum(1 for value in key_results.values() if not value["resolved"]),
    }


def resolve_all(spec: WorkbenchSpec, quantizer: FixedQuantizer) -> dict[str, dict[str, Any]]:
    return {
        env.id: resolve_environment(spec, env, quantizer)
        for env in spec.environment_order()
    }

