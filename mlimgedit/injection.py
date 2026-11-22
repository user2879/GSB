"""Independent fault-injection configuration.

Injection entries reference a sample, a stage and one of the supported fault
types. Injection is applied in-memory on top of the on-disk inputs so the
original generated dataset is never mutated, and every hit is reported with its
entry id and the business code it maps to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    E_BBOX_OUT_OF_RANGE,
    E_LABEL_UNKNOWN,
    E_SAMPLE_FORMAT,
    ConfigError,
)

FAULT_TRUNCATE_PNG = "truncate_png"
FAULT_UNKNOWN_LABEL = "unknown_label"
FAULT_BBOX_OUT_OF_RANGE = "bbox_out_of_range"
FAULT_DROP_LABEL = "drop_label"

SUPPORTED_FAULTS = (
    FAULT_TRUNCATE_PNG,
    FAULT_UNKNOWN_LABEL,
    FAULT_BBOX_OUT_OF_RANGE,
    FAULT_DROP_LABEL,
)

FAULT_CODE = {
    FAULT_TRUNCATE_PNG: E_SAMPLE_FORMAT,
    FAULT_UNKNOWN_LABEL: E_LABEL_UNKNOWN,
    FAULT_BBOX_OUT_OF_RANGE: E_BBOX_OUT_OF_RANGE,
    FAULT_DROP_LABEL: E_SAMPLE_FORMAT,
}

VALIDATION_STAGE = "validation"


@dataclass(frozen=True)
class Injection:
    injection_id: str
    sample: str
    stage: str
    fault: str
    enabled: bool = True

    @property
    def business_code(self) -> str:
        return FAULT_CODE[self.fault]


def load_injections(path: str | Path | None) -> dict[str, list[Injection]]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("injections", [])
    grouped: dict[str, list[Injection]] = {}
    for idx, raw in enumerate(entries):
        injection_id = str(raw.get("id", f"INJ-{idx:03d}"))
        sample = raw.get("sample")
        stage = raw.get("stage", VALIDATION_STAGE)
        fault = raw.get("fault")
        enabled = bool(raw.get("enabled", True))
        if not sample or not isinstance(sample, str):
            raise ConfigError(f"injection {injection_id} missing sample")
        if fault not in SUPPORTED_FAULTS:
            raise ConfigError(f"injection {injection_id} uses unsupported fault {fault!r}")
        injection = Injection(injection_id, sample, stage, fault, enabled)
        grouped.setdefault(sample, []).append(injection)
    for sample_entries in grouped.values():
        sample_entries.sort(key=lambda item: item.injection_id)
    return grouped


def first_active(injections: dict[str, list[Injection]], sample: str, stage: str):
    for injection in injections.get(sample, []):
        if injection.enabled and injection.stage == stage:
            return injection
    return None
