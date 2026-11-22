from __future__ import annotations

from decimal import Decimal
from typing import Any

from .constants import E_CONFIG_INVALID
from .plan import ConstructionState
from .util import (
    as_decimal,
    canonical_json,
    fingerprint_object,
    stable_int,
    stable_key_hash,
)


def candidate_id_for(task_type: str, sample_seq: int) -> str:
    return f"cand-{task_type}-{sample_seq:06d}"


def stable_sort_key(candidate_id: str, stage: str) -> str:
    return f"{candidate_id}|{stage}"


def weighted_pick(weights: dict[str, float], *digest_parts: Any) -> str:
    active = sorted((key, as_decimal(value)) for key, value in weights.items() if as_decimal(value) > 0)
    if not active:
        raise ValueError("weighted_pick requires at least one positive weight")
    total = sum((weight for _, weight in active), as_decimal(0))
    raw = stable_int(*digest_parts, bits=128)
    target = (Decimal(raw) * total) / (Decimal(2) ** 128)
    cumulative = as_decimal(0)
    for key, weight in active:
        cumulative += weight
        if target <= cumulative:
            return key
    return active[-1][0]


def _template_fingerprint(template: dict[str, Any]) -> str:
    return fingerprint_object(
        {
            "variant_id": template["template_variant_id"],
            "instruction_template": template["instruction_template"],
            "slots": template["slots"],
            "reference_answer": template["reference_answer"],
        }
    )


def _source_fingerprint(name: str, values: list[str]) -> str:
    return fingerprint_object({"name": name, "values": values})


def prepare_sampling(
    *,
    state: ConstructionState,
    task_type: str,
    sample_seq: int,
    seed_file_fingerprint: str,
) -> dict[str, Any]:
    task = state.tasks[task_type]
    template_weights = task["template_weights"]
    source_weights = task["slot_source_weights"]

    variant_id = weighted_pick(
        template_weights,
        seed_file_fingerprint,
        task_type,
        sample_seq,
        fingerprint_object(template_weights),
        "template",
    )
    source_name = weighted_pick(
        source_weights,
        seed_file_fingerprint,
        task_type,
        sample_seq,
        fingerprint_object(source_weights),
        "source",
    )
    template = state.templates[task_type][variant_id]
    source_values = state.sources[source_name]
    template_fp = _template_fingerprint(template)
    source_fp = _source_fingerprint(source_name, source_values)

    slot_values: dict[str, str] = {}
    for slot in template["slots"]:
        value_index = stable_int(
            seed_file_fingerprint,
            task_type,
            source_name,
            slot,
            sample_seq,
            "slot-value",
            bits=128,
        ) % len(source_values)
        slot_values[slot] = source_values[value_index]
    candidate_id = candidate_id_for(task_type, sample_seq)
    template_reference = {
        "variant_id": variant_id,
        "cache_key": stable_key_hash("template-v1", task_type, variant_id, template_fp),
        "content_fingerprint": template_fp,
    }
    source_references = [
        {
            "slot": slot,
            "source": source_name,
            "cache_key": stable_key_hash("source-v1", task_type, slot, source_name, source_fp),
            "content_fingerprint": source_fp,
        }
        for slot in template["slots"]
    ]
    cache_key = stable_key_hash(
        "sampling-v1",
        seed_file_fingerprint,
        task_type,
        sample_seq,
        variant_id,
        template_fp,
        source_name,
        source_fp,
        canonical_json(slot_values),
    )
    return {
        "stage": "sampling",
        "candidate_id": candidate_id,
        "task_type": task_type,
        "sample_seq": sample_seq,
        "template_variant_id": variant_id,
        "source_name": source_name,
        "slot_values": slot_values,
        "template_reference": template_reference,
        "slot_source_references": source_references,
        "cache_key": cache_key,
        "stable_sort_key": stable_sort_key(candidate_id, "sampling"),
    }


def materialize_sampling(selection: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    try:
        instruction = template["instruction_template"].format_map(selection["slot_values"])
        reference = template["reference_answer"].format_map(selection["slot_values"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"template rendering failed for {selection['task_type']}/{selection['template_variant_id']}: {exc}"
        ) from exc
    content_fingerprint = fingerprint_object(
        {
            "task_type": selection["task_type"],
            "variant_id": selection["template_variant_id"],
            "source_name": selection["source_name"],
            "slot_values": selection["slot_values"],
            "instruction": instruction,
            "reference_answer": reference,
        }
    )
    artifact = dict(selection)
    artifact["instruction"] = instruction
    artifact["reference_answer"] = reference
    artifact["content_fingerprint"] = content_fingerprint
    return artifact


def sample_candidate(
    *,
    state: ConstructionState,
    task_type: str,
    sample_seq: int,
    seed_file_fingerprint: str,
) -> dict[str, Any]:
    selection = prepare_sampling(
        state=state,
        task_type=task_type,
        sample_seq=sample_seq,
        seed_file_fingerprint=seed_file_fingerprint,
    )
    template = state.templates[task_type][selection["template_variant_id"]]
    return materialize_sampling(selection, template)


def task_sampling_fingerprint(state: ConstructionState, task_type: str, seed_file_fingerprint: str) -> str:
    task = state.tasks[task_type]
    return fingerprint_object(
        {
            "seed_file": seed_file_fingerprint,
            "task_type": task_type,
            "template_weights": task["template_weights"],
            "slot_source_weights": task["slot_source_weights"],
        }
    )


def ensure_sampling_available(state: ConstructionState, task_type: str) -> None:
    if task_type not in state.tasks:
        raise ValueError(f"unknown task type: {task_type}")
    if not any(as_decimal(value) > 0 for value in state.tasks[task_type]["template_weights"].values()):
        raise ValueError(f"task {task_type} has no positive template weight")
    if not any(as_decimal(value) > 0 for value in state.tasks[task_type]["slot_source_weights"].values()):
        raise ValueError(f"task {task_type} has no positive source weight")
