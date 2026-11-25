"""Deterministic per-bucket reservoir sampling.

For an accepted record with 1-based bucket ordinal ``i`` and capacity ``k``:

* ordinals ``1..k`` fill slots ``1..k`` in arrival order;
* ordinal ``i > k`` derives a 256-bit random source from the bucket
  fingerprint and record content, then admits when ``random < k / i``;
* admitted records replace slot ``((i - 1) % k) + 1`` and retain the replaced
  record identifier in the admission history.

The criterion is evaluated with integer arithmetic; float output is only for
human-readable diagnostics.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .canonical import canonical_json, fingerprint, stable_number, sha256_text
from .models import AcceptedRecord

RANDOM_SPACE = 1 << 256


def record_fingerprint(record: AcceptedRecord) -> str:
    return fingerprint(record.content_dict())


def quantization_step(quantization: dict[str, Any]) -> int:
    decimals = int(quantization.get("decimals", 3))
    tier = str(quantization.get("tier", "fine"))
    step = 10 ** max(0, 3 - decimals)
    if tier == "coarse":
        step *= 10
    return step


def quantized_weight(metric: float | int, quantization: dict[str, Any]) -> int:
    step = quantization_step(quantization)
    raw = float(metric) * 1000.0
    return int(math.floor(raw / step + 0.5)) * step


def bucket_params(
    config: dict[str, Any],
    bucket_id: str,
    capacity: int,
    priority: int,
    profile: str,
    global_budget: int,
    quantization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "lds-bucket-v1",
        "bucket_id": bucket_id,
        "capacity": capacity,
        "priority": priority,
        "profile": profile,
        "global_budget": global_budget,
        "quantization": {
            "decimals": int(quantization.get("decimals", 3)),
            "tier": str(quantization.get("tier", "fine")),
        },
        "seed": int(config["seed"]),
        "reservoir": "algorithm-r-int256-slot-cycle-v1",
    }


def bucket_fingerprint(params: dict[str, Any]) -> str:
    return fingerprint(params)


def bucket_random_source_digest(params: dict[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "schema": "lds-random-source-v1",
                "bucket_fingerprint": bucket_fingerprint(params),
                "seed": params["seed"],
            }
        )
    )


def derive_random_source(
    bucket_fp: str,
    arrival_ordinal: int,
    record_fp: str,
    seed: int,
) -> tuple[str, int, float]:
    material = canonical_json(
        {
            "schema": "lds-random-tick-v1",
            "bucket_fingerprint": bucket_fp,
            "arrival_ordinal": arrival_ordinal,
            "record_fingerprint": record_fp,
            "seed": seed,
        }
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    numerator = int(digest, 16)
    unit = numerator / RANDOM_SPACE
    return digest, numerator, unit


def build_bucket_history(
    *,
    bucket_id: str,
    records: list[AcceptedRecord],
    capacity: int,
    priority: int,
    config: dict[str, Any],
    profile: str,
    global_budget: int,
    quantization: dict[str, Any],
) -> dict[str, Any]:
    if capacity <= 0:
        raise ValueError("bucket capacity must be positive")
    params = bucket_params(
        config,
        bucket_id,
        capacity,
        priority,
        profile,
        global_budget,
        quantization,
    )
    bucket_fp = bucket_fingerprint(params)
    random_source_digest = bucket_random_source_digest(params)
    record_fps = {record.log_id: record_fingerprint(record) for record in records}
    ordered = sorted(records, key=lambda record: record.order_key)
    slots: dict[int, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []

    for ordinal, record in enumerate(ordered, start=1):
        digest, numerator, unit = derive_random_source(
            bucket_fp, ordinal, record_fps[record.log_id], int(config["seed"])
        )
        target_slot = ((ordinal - 1) % capacity) + 1
        threshold_num = capacity
        threshold_den = ordinal
        if ordinal <= capacity:
            admitted = True
            basis = f"initial_fill:ordinal={ordinal}<=capacity={capacity}"
        else:
            admitted = numerator * ordinal < capacity * RANDOM_SPACE
            symbol = "<" if admitted else ">="
            basis = (
                f"reservoir:random={digest[:16]}... {symbol} "
                f"capacity/ordinal={capacity}/{ordinal}"
            )

        replaced = slots.get(target_slot) if admitted else None
        if admitted:
            event_slot = target_slot
            slots[target_slot] = {
                "log_id": record.log_id,
                "arrival_ordinal": ordinal,
                "arrival_seq": record.arrival_seq,
                "record_index": record.record_index,
                "record_fingerprint": record_fps[record.log_id],
                "slot_no": target_slot,
                "priority": priority,
            }
        else:
            event_slot = 0

        history.append(
            {
                "log_id": record.log_id,
                "arrival_seq": record.arrival_seq,
                "record_index": record.record_index,
                "arrival_ordinal": ordinal,
                "record_fingerprint": record_fps[record.log_id],
                "random_digest": digest,
                "random_numerator_hex": f"{numerator:064x}",
                "random_unit": unit,
                "threshold_numerator": threshold_num,
                "threshold_denominator": threshold_den,
                "admitted": admitted,
                "target_slot_no": target_slot,
                "slot_no": event_slot,
                "replaced_log_id": (
                    replaced["log_id"] if replaced is not None else None
                ),
                "decision": (
                    "admitted" if admitted else "rejected_by_bucket"
                ),
                "basis": basis,
                "metric": record.metric,
                "quantized_weight": quantized_weight(record.metric, quantization),
                "timestamp": record.timestamp,
                "device_id": record.device_id,
                "severity": record.severity,
                "template_id": record.template_id,
                "injection_ids": list(record.injection_ids),
            }
        )

    current_slots = [
        {
            **slots[slot_no],
            "metric": next(
                event["metric"]
                for event in history
                if event["log_id"] == slots[slot_no]["log_id"]
            ),
            "quantized_weight": next(
                event["quantized_weight"]
                for event in history
                if event["log_id"] == slots[slot_no]["log_id"]
            ),
            "timestamp": next(
                event["timestamp"]
                for event in history
                if event["log_id"] == slots[slot_no]["log_id"]
            ),
            "device_id": next(
                event["device_id"]
                for event in history
                if event["log_id"] == slots[slot_no]["log_id"]
            ),
            "severity": next(
                event["severity"]
                for event in history
                if event["log_id"] == slots[slot_no]["log_id"]
            ),
            "template_id": next(
                event["template_id"]
                for event in history
                if event["log_id"] == slots[slot_no]["log_id"]
            ),
            "priority": priority,
            "bucket_id": bucket_id,
        }
        for slot_no in sorted(slots)
    ]
    return {
        "bucket_id": bucket_id,
        "bucket_fingerprint": bucket_fp,
        "random_source_digest": random_source_digest,
        "params": params,
        "capacity": capacity,
        "priority": priority,
        "record_count": len(history),
        "record_fingerprints": record_fps,
        "history": history,
        "slots": current_slots,
        "final_slot_count": len(current_slots),
    }


def bucket_task_sort_key(bucket: dict[str, Any] | tuple[str, int, str]) -> tuple[str, int, str]:
    """Stable scheduling key derived from bucket id and earliest arrival ordinal."""
    if isinstance(bucket, tuple):
        return bucket
    history = bucket.get("history", [])
    first_ordinal = history[0]["arrival_ordinal"] if history else 0
    return (
        str(bucket["bucket_id"]),
        int(first_ordinal),
        str(bucket.get("bucket_fingerprint", "")),
    )

