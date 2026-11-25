"""Generate deterministic local fixtures used by tests and the acceptance demo."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from log_sampler.cache import bucket_content_fingerprint
from log_sampler.models import AcceptedRecord
from log_sampler.reservoir import (
    bucket_fingerprint,
    bucket_params,
    derive_random_source,
    record_fingerprint,
)
from log_sampler.state import apply_command
from log_sampler.validation import (
    bucket_capacity,
    resolve_sampling,
    validate_input,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
SEED = 20261126

DEVICE_SEVERITIES = [
    ("edge-a", "critical"),
    ("edge-a", "info"),
    ("edge-a", "warning"),
    ("edge-b", "error"),
    ("edge-b", "info"),
    ("edge-b", "debug"),
    ("core-x", "critical"),
    ("core-x", "info"),
    ("core-x", "warning"),
]


def make_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": SEED,
        "active_profile": "balanced",
        "evidence_format": "p3",
        "max_workers": 4,
        "default_bucket_priority": 10,
        "devices": [
            {
                "device_id": "edge-a",
                "severities": ["info", "warning", "critical"],
            },
            {
                "device_id": "edge-b",
                "severities": ["debug", "info", "error"],
            },
            {
                "device_id": "core-x",
                "severities": ["info", "warning", "critical"],
            },
        ],
        "profiles": {
            "balanced": {
                "global_budget": 8,
                "bucket_capacity": 3,
                "quantization": {"decimals": 3, "tier": "fine"},
            },
            "tight": {
                "global_budget": 5,
                "bucket_capacity": 3,
                "quantization": {"decimals": 3, "tier": "fine"},
            },
        },
        "allowed_quantization": {
            "decimals": [2, 3, 4],
            "tiers": ["coarse", "fine"],
        },
        "bucket_overrides": {
            "edge-a|critical": {"capacity": 2, "priority": 30},
            "core-x|warning": {"capacity": 3, "priority": 20},
            "edge-b|error": {"capacity": 3, "priority": 15},
        },
    }


def make_record(index: int, metric: float | None = None) -> dict[str, Any]:
    device_id, severity = DEVICE_SEVERITIES[(index - 1) % len(DEVICE_SEVERITIES)]
    template = f"tmpl-{((index - 1) % 11) + 1:02d}"
    return {
        "log_id": f"log-{index:04d}",
        "timestamp": 1_792_800_000 + index * 17,
        "device_id": device_id,
        "severity": severity,
        "template_id": template,
        "metric": metric if metric is not None else 0.5,
    }


def make_batches() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    batches: list[dict[str, Any]] = []
    index = 1
    for arrival_seq in range(1, 7):
        records = []
        for record_index in range(6):
            metric = round(rng.uniform(0.03, 0.997), 7)
            records.append(make_record(index, metric))
            index += 1
        batches.append({"arrival_seq": arrival_seq, "records": records})
    # Keep one global-cut adjacency one per-mille apart.  At the final
    # coarse/2-decimal setting this deterministically exercises the
    # TRUNCATION_TIE diagnostic without disturbing rank order.
    for batch in batches:
        for record in batch["records"]:
            if record["log_id"] == "log-0009":
                record["metric"] = 0.249
    return batches


def make_injections() -> list[dict[str, Any]]:
    return [
        {
            "id": "INJ-001",
            "type": "non_finite_metric",
            "batch_seq": 1,
            "log_id": "log-0002",
        },
        {
            "id": "INJ-002",
            "type": "unknown_bucket",
            "batch_seq": 2,
            "log_id": "log-0008",
        },
        {
            "id": "INJ-003",
            "type": "batch_order_regression",
            "batch_seq": 5,
        },
        {
            "id": "INJ-004",
            "type": "missing_random_digest",
            "batch_seq": 4,
            "log_id": "log-0020",
        },
    ]


def make_commands(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batch_three = copy.deepcopy(batches[2])
    for record in batch_three["records"]:
        if record["log_id"] == "log-0015":
            record["metric"] = 0.8123456
            record["template_id"] = "tmpl-replaced-15"
    appended = []
    for index in range(37, 43):
        appended.append(make_record(index, round(0.2 + (index - 37) * 0.077, 7)))
    return [
        {
            "id": "cmd-01",
            "type": "replace_batch",
            "batch": batch_three,
        },
        {"id": "cmd-02", "type": "delete_log", "log_id": "log-0018"},
        {
            "id": "cmd-03",
            "type": "set_bucket_capacity",
            "device_id": "edge-a",
            "severity": "warning",
            "capacity": 2,
        },
        {
            "id": "cmd-04",
            "type": "set_bucket_priority",
            "device_id": "edge-b",
            "severity": "error",
            "priority": 40,
        },
        {"id": "cmd-05", "type": "set_profile", "profile": "tight"},
        {
            "id": "cmd-06",
            "type": "set_quantization",
            "decimals": 2,
            "tier": "coarse",
        },
        {"id": "cmd-07", "type": "set_evidence_format", "format": "p1"},
        {
            "id": "cmd-08",
            "type": "append_batch",
            "batch": {"arrival_seq": 7, "records": appended},
        },
    ]


def effective(
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    effective_config = copy.deepcopy(config)
    effective_batches = copy.deepcopy(batches)
    for command in commands:
        apply_command(effective_config, effective_batches, command)
    return effective_config, effective_batches


def accepted_from(
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    injections: list[dict[str, Any]],
) -> list[AcceptedRecord]:
    validation = validate_input(config, batches, injections)
    return validation.accepted_records


def boundary_count(
    config: dict[str, Any],
    records: list[AcceptedRecord],
) -> int:
    sampling = resolve_sampling(config)
    quantization = sampling["quantization"]
    coarse_digits = max(1, int(quantization["decimals"]) - 1) if quantization["tier"] == "coarse" else int(quantization["decimals"])
    half_step = 0.5 * (10 ** -coarse_digits)
    count = 0
    grouped: dict[str, list[AcceptedRecord]] = {}
    for record in records:
        grouped.setdefault(record.bucket_id, []).append(record)
    for bucket_id, bucket_records in grouped.items():
        capacity = bucket_capacity(config, bucket_id)
        params = bucket_params(
            config,
            bucket_id,
            capacity,
            int(
                config.get("bucket_overrides", {})
                .get(bucket_id, {})
                .get("priority", config.get("default_bucket_priority", 10))
            ),
            sampling["profile"],
            int(sampling["global_budget"]),
            quantization,
        )
        fp = bucket_fingerprint(params)
        for ordinal, record in enumerate(
            sorted(bucket_records, key=lambda item: item.order_key), start=1
        ):
            if ordinal <= capacity:
                continue
            _, _, unit = derive_random_source(
                fp, ordinal, record_fingerprint(record), int(config["seed"])
            )
            if abs(unit - capacity / ordinal) <= half_step:
                count += 1
    return count


def tune_boundary(
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    injections: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> None:
    eff_config, eff_batches = effective(config, batches, commands)
    records = accepted_from(eff_config, eff_batches, injections)
    if boundary_count(eff_config, records) > 0:
        return
    target = "log-0033"
    for attempt in range(1, 5001):
        candidate_metric = round((attempt % 997_000) / 997_000.0, 7)
        trial_batches = copy.deepcopy(batches)
        for batch in trial_batches:
            for record in batch["records"]:
                if record["log_id"] == target:
                    record["metric"] = candidate_metric
        trial_config, trial_effective = effective(
            config, trial_batches, commands
        )
        trial_records = accepted_from(
            trial_config, trial_effective, injections
        )
        if boundary_count(trial_config, trial_records) > 0:
            for batch in batches:
                for record in batch["records"]:
                    if record["log_id"] == target:
                        record["metric"] = candidate_metric
            return
    raise RuntimeError("unable to tune a deterministic boundary sample")


def dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    config = make_config()
    batches = make_batches()
    injections = make_injections()
    commands = make_commands(batches)
    tune_boundary(config, batches, injections, commands)
    dump(SAMPLES / "config.json", config)
    dump(SAMPLES / "batches.json", {"schema_version": 1, "batches": batches})
    dump(SAMPLES / "injections.json", {"schema_version": 1, "injections": injections})
    dump(SAMPLES / "commands.json", {"schema_version": 1, "commands": commands})
    eff_config, eff_batches = effective(config, batches, commands)
    final_records = accepted_from(eff_config, eff_batches, injections)
    print(
        json.dumps(
            {
                "samples": str(SAMPLES),
                "base_batches": len(batches),
                "base_records": sum(len(batch["records"]) for batch in batches),
                "accepted_after_commands": len(final_records),
                "boundary_events": boundary_count(eff_config, final_records),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
