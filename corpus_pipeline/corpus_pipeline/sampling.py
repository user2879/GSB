from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import Code, failure_record, fail
from .fixed import QuantizedWeight, fraction_slot_key, quantize_weight
from .models import Document, PipelineConfig, content_hash
from .synthetic import derive_seed


ORDINAL_MODEL = "interleaved-stable-v1"


def domain_document_fingerprint(documents: Sequence[Document]) -> str:
    return content_hash(
        [
            [doc.doc_id, doc.content_hash, doc.batch_index, doc.generator_seed]
            for doc in sorted(documents, key=lambda item: item.doc_id)
        ]
    )


def deterministic_domain_order(domain_id: str, domain_seed: int, global_seed: int, documents: Sequence[Document]) -> list[Document]:
    """Return a seeded random order with a lexicographic collision tie-break.

    The domain content fingerprint participates in the seed.  Consequently a
    replacement or append is a real reorder of the sampling stream, not a
    cosmetic relabeling, and all affected shard boundaries move with it.
    """

    if not documents:
        return []
    seed = derive_seed(
        global_seed,
        domain_seed,
        domain_id,
        domain_document_fingerprint(documents),
        "deterministic-domain-order-v1",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    keys = rng.integers(0, np.iinfo(np.uint64).max, size=len(documents), dtype=np.uint64)
    decorated = [(int(key), doc.doc_id, doc) for key, doc in zip(keys, documents, strict=True)]
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in decorated]


def quantized_domain_weights(config: PipelineConfig) -> dict[str, QuantizedWeight]:
    return {
        domain.id: quantize_weight(
            domain.weight,
            config.precision.decimal_places,
            config.precision.quantization_tier,
            field_name=f"domains[{domain.id}].weight",
        )
        for domain in config.domains
    }


def allocate_quotas(
    config: PipelineConfig,
    *,
    capacities: Mapping[str, int],
    quantized_weights: Mapping[str, QuantizedWeight] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Allocate deterministic sample points under budget and domain quotas.

    Each domain exposes ``capacity`` weighted-fair slots.  Slot ``s`` has
    virtual finish time ``s / fixed_weight``.  Selecting the globally smallest
    ``budget`` slots is work-conserving, respects per-domain caps, and does not
    depend on worker timing.
    """

    weights = quantized_weights or quantized_domain_weights(config)
    domain_order = sorted(config.domain_ids)
    slots: list[tuple[Fraction, str, int]] = []
    for domain_id in domain_order:
        capacity = max(0, int(capacities.get(domain_id, 0)))
        weight = weights[domain_id].fixed
        if weight <= 0:
            continue
        for slot in range(1, capacity + 1):
            slots.append((fraction_slot_key(slot, weight), domain_id, slot))
    slots.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = slots[: config.sampling.budget]

    allocations = {domain_id: 0 for domain_id in domain_order}
    selected_slots: dict[str, list[int]] = {domain_id: [] for domain_id in domain_order}
    for _, domain_id, slot in selected:
        allocations[domain_id] += 1
        selected_slots[domain_id].append(slot)

    non_zero_weights = [weights[domain_id].fixed for domain_id in domain_order]
    if all(weight == 0 for weight in non_zero_weights):
        fail(
            Code.INVALID_WEIGHT,
            "all domain weights quantize to zero at the configured precision",
            precision=config.precision.to_dict(),
        )
    allocation_meta = {
        "algorithm": "weighted-fair-queue-v1",
        "budget": config.sampling.budget,
        "per_domain_quota": config.sampling.per_domain_quota,
        "capacities": dict(sorted(capacities.items())),
        "allocations": allocations,
        "selected_slots": {domain: sorted(value) for domain, value in selected_slots.items()},
        "available_slots": len(slots),
        "unused_budget": max(0, config.sampling.budget - len(selected)),
        "weights": {domain: weights[domain].to_dict() for domain in domain_order},
    }
    return allocations, allocation_meta


def _unsampled_reason(
    domain_id: str,
    *,
    allocation: int,
    capacity: int,
    document_count: int,
    quota: int,
    fixed_weight: int,
) -> tuple[Code, str]:
    if fixed_weight <= 0:
        return Code.ZERO_WEIGHT_DOMAIN, "domain has a zero quantized sampling weight"
    if document_count > quota and allocation >= quota:
        return Code.DOMAIN_QUOTA_EXHAUSTED, "per-domain quota exhausted"
    if allocation < capacity:
        return Code.BUDGET_EXHAUSTED, "global sample-point budget exhausted before this document"
    return Code.DOMAIN_QUOTA_EXHAUSTED, "per-domain quota exhausted"


def build_sample_plan(config: PipelineConfig, documents: Mapping[str, Sequence[Document]]) -> dict[str, Any]:
    domain_order = sorted(config.domain_ids)
    domain_index = {domain_id: index for index, domain_id in enumerate(domain_order)}
    weights = quantized_domain_weights(config)
    capacities = {
        domain_id: min(config.sampling.per_domain_quota, len(documents.get(domain_id, ())))
        for domain_id in domain_order
    }
    allocations, allocation_meta = allocate_quotas(
        config,
        capacities=capacities,
        quantized_weights=weights,
    )

    failures: list[dict[str, Any]] = []
    ordered_by_domain: dict[str, list[Document]] = {}
    selected_by_domain: dict[str, list[Document]] = {}
    for domain_id in domain_order:
        domain = config.domain(domain_id)
        domain_docs = list(documents.get(domain_id, ()))
        ordered = deterministic_domain_order(domain_id, domain.seed, config.seed, domain_docs)
        ordered_by_domain[domain_id] = ordered
        allocation = allocations[domain_id]
        selected = ordered[:allocation]
        selected_by_domain[domain_id] = selected
        capacity = capacities[domain_id]
        quota = config.sampling.per_domain_quota
        selected_ids = {doc.doc_id for doc in selected}
        reason_code, reason_message = _unsampled_reason(
            domain_id,
            allocation=allocation,
            capacity=capacity,
            document_count=len(domain_docs),
            quota=quota,
            fixed_weight=weights[domain_id].fixed,
        )
        for doc in sorted(domain_docs, key=lambda item: item.doc_id):
            if doc.doc_id not in selected_ids:
                failures.append(
                    failure_record(
                        reason_code,
                        reason_message,
                        domain=domain_id,
                        doc_id=doc.doc_id,
                        allocation=allocation,
                        capacity=capacity,
                        quota=quota,
                    )
                )

    shards: list[dict[str, Any]] = []
    all_document_entries: list[dict[str, Any]] = []
    shard_size = config.sampling.shard_size
    domain_count = len(domain_order)
    for domain_id in domain_order:
        selected = selected_by_domain[domain_id]
        domain = config.domain(domain_id)
        domain_source_fp = domain_document_fingerprint(documents.get(domain_id, ()))
        domain_component = {
            "domain": domain.to_dict(),
            "source_documents_fingerprint": domain_source_fp,
            "global_seed": config.seed,
            "allocation": allocations[domain_id],
            "allocation_meta": {
                "budget": config.sampling.budget,
                "per_domain_quota": config.sampling.per_domain_quota,
                "selected_slots": allocation_meta["selected_slots"][domain_id],
                "all_capacity": capacities[domain_id],
            },
            "quantized_weight": weights[domain_id].to_dict(),
            "shard_size": shard_size,
            "precision": config.precision.to_dict(),
            "ordinal_model": ORDINAL_MODEL,
        }
        domain_context_fingerprint = content_hash(domain_component)
        for shard_index, start in enumerate(range(0, len(selected), shard_size)):
            shard_docs = selected[start : start + shard_size]
            shard_id = f"{domain_id}/shard-{shard_index:03d}"
            local_ordinals: list[int] = []
            doc_entries: list[dict[str, Any]] = []
            for offset, doc in enumerate(shard_docs):
                local_ordinal = start + offset + 1
                global_ordinal = (local_ordinal - 1) * domain_count + domain_index[domain_id] + 1
                local_ordinals.append(local_ordinal)
                entry = {
                    "doc_id": doc.doc_id,
                    "local_ordinal": local_ordinal,
                    "global_ordinal": global_ordinal,
                    "content_hash": doc.content_hash,
                }
                doc_entries.append(entry)
                all_document_entries.append(
                    {
                        "domain": domain_id,
                        "shard_id": shard_id,
                        **entry,
                    }
                )
            sampling_fingerprint = content_hash(
                {
                    "domain_context_fingerprint": domain_context_fingerprint,
                    "shard_index": shard_index,
                    "shard_id": shard_id,
                    "documents": doc_entries,
                    "local_ordinals": local_ordinals,
                }
            )
            sampling_payload = {
                "shard_id": shard_id,
                "domain": domain_id,
                "shard_index": shard_index,
                "documents": doc_entries,
                "domain_context_fingerprint": domain_context_fingerprint,
                "sampling_fingerprint": sampling_fingerprint,
                "local_ordinals": local_ordinals,
            }
            shards.append(sampling_payload)

    all_document_entries.sort(key=lambda item: (item["global_ordinal"], item["domain"], item["doc_id"]))
    display_positions = {
        (entry["shard_id"], entry["doc_id"]): position
        for position, entry in enumerate(all_document_entries, start=1)
    }
    for shard in shards:
        for entry in shard["documents"]:
            entry["display_position"] = display_positions[(shard["shard_id"], entry["doc_id"])]

    failures.sort(
        key=lambda item: (
            item["code"],
            item.get("domain", ""),
            item.get("shard_id", ""),
            item.get("doc_id", ""),
            item.get("message", ""),
        )
    )
    plan_fingerprint = content_hash(
        {
            "config": config.config_fingerprint_payload(),
            "allocation": allocation_meta,
            "shards": [
                [shard["shard_id"], shard["sampling_fingerprint"]]
                for shard in sorted(shards, key=lambda item: item["shard_id"])
            ],
            "failures": failures,
        }
    )
    return {
        "ordinal_model": ORDINAL_MODEL,
        "seed": config.seed,
        "precision": config.precision.to_dict(),
        "budget": config.sampling.budget,
        "budget_consumed": sum(allocations.values()),
        "budget_remaining": max(0, config.sampling.budget - sum(allocations.values())),
        "allocations": allocations,
        "allocation_meta": allocation_meta,
        "shards": shards,
        "failures": failures,
        "plan_fingerprint": plan_fingerprint,
        "documents_by_domain": {
            domain_id: [doc.doc_id for doc in selected_by_domain[domain_id]]
            for domain_id in domain_order
        },
        "ordered_documents_by_domain": {
            domain_id: [doc.doc_id for doc in ordered_by_domain[domain_id]]
            for domain_id in domain_order
        },
    }


def dedup_parameter_fingerprint(config: PipelineConfig) -> str:
    return content_hash(
        {
            "threshold_permille": config.dedup.similarity_threshold_permille,
            "boundary_rule": config.dedup.boundary_rule,
            "precision": config.precision.to_dict(),
            "max_dedup_comparisons": config.sampling.max_dedup_comparisons,
        }
    )


def shard_documents_from_plan(
    shard: Mapping[str, Any],
    documents_by_domain: Mapping[str, Sequence[Document]],
) -> list[Document]:
    domain = shard["domain"]
    by_id = {doc.doc_id: doc for doc in documents_by_domain.get(domain, ())}
    result: list[Document] = []
    for entry in shard["documents"]:
        doc = by_id.get(entry["doc_id"])
        if doc is None:
            fail(
                Code.STATE_CORRUPT,
                "sample plan references a document missing from the input stream",
                domain=domain,
                shard_id=shard["shard_id"],
                doc_id=entry["doc_id"],
            )
        result.append(doc)
    return result
