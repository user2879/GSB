"""Independent, auditable fault-injection configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import (
    E_CALIBRATION_INVALID,
    E_CONFIG_INVALID,
    E_GROUP_INVALID,
    E_INJECTION_MISS,
    E_QUANT_BOUNDS,
    E_RECEIPT_ORPHAN,
    E_WEIGHT_INVALID,
    FailureList,
)
from .model import Dataset, Receipt


KIND_TO_CODE = {
    "nonfinite_weight": E_WEIGHT_INVALID,
    "group_indivisible": E_GROUP_INVALID,
    "bit_width_out_of_range": E_QUANT_BOUNDS,
    "calibration_dimension_mismatch": E_CALIBRATION_INVALID,
    "invalid_quantile_bounds": E_CONFIG_INVALID,
    "orphan_receipt": E_RECEIPT_ORPHAN,
}


@dataclass
class FaultInjection:
    injection_id: str
    kind: str
    layer: str | None = None
    sample_index: int | None = None
    value: Any = None


def load_faults(path: Path, failures: FailureList) -> list[FaultInjection]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        failures.add(E_CONFIG_INVALID, f"cannot load fault config: {exc}", "fault_injection", path=str(path))
        return []
    if not isinstance(data, dict) or not isinstance(data.get("injections"), list):
        failures.add(
            E_CONFIG_INVALID,
            "fault config must contain an injections array",
            "fault_injection",
            path=str(path),
        )
        return []
    result: list[FaultInjection] = []
    seen: set[str] = set()
    for index, item in enumerate(data["injections"]):
        if not isinstance(item, dict):
            failures.add(
                E_CONFIG_INVALID,
                "fault injection must be an object",
                "fault_injection",
                details={"index": index},
            )
            continue
        injection_id = item.get("id", f"INJ-{index + 1:02d}")
        kind = item.get("kind")
        if not isinstance(injection_id, str) or not injection_id or injection_id in seen:
            failures.add(
                E_CONFIG_INVALID,
                "fault injection id must be unique and non-empty",
                "fault_injection",
                details={"index": index},
            )
            continue
        seen.add(injection_id)
        if kind not in KIND_TO_CODE:
            failures.add(
                E_CONFIG_INVALID,
                f"unsupported fault kind {kind!r}",
                "fault_injection",
                injection_id=injection_id,
            )
            continue
        result.append(
            FaultInjection(
                injection_id=injection_id,
                kind=str(kind),
                layer=item.get("layer"),
                sample_index=item.get("sample_index"),
                value=item.get("value"),
            )
        )
    return result


def apply_faults(
    dataset: Dataset, injections: list[FaultInjection]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    failures = dataset.failures
    for injection in injections:
        expected = KIND_TO_CODE[injection.kind]
        hit = False
        message = ""
        layer = injection.layer
        if injection.kind == "nonfinite_weight":
            target = next((item for item in dataset.layers if item.name == layer), None)
            if target is not None and target.weights:
                weights = list(target.weights)
                weights[0] = float("nan")
                target.weights = tuple(weights)
                hit = True
                message = f"injected non-finite weight into layer {layer!r}"
        elif injection.kind == "group_indivisible":
            target = next((item for item in dataset.layers if item.name == layer), None)
            if target is not None:
                value = injection.value if isinstance(injection.value, int) else target.numel + 1
                target.group_granularity = int(value)
                if dataset.config is not None:
                    dataset.config.layer_overrides.setdefault(target.name, {})[
                        "group_granularity"
                    ] = int(value)
                hit = True
                message = f"injected indivisible group granularity {value} into layer {layer!r}"
        elif injection.kind == "bit_width_out_of_range":
            target = next((item for item in dataset.layers if item.name == layer), None)
            if target is not None and dataset.config is not None:
                value = injection.value if isinstance(injection.value, int) else 32
                dataset.config.layer_overrides.setdefault(target.name, {})["bit_width"] = int(value)
                hit = True
                message = f"injected bit width {value} into layer {layer!r}"
        elif injection.kind == "calibration_dimension_mismatch":
            index = injection.sample_index if isinstance(injection.sample_index, int) else 0
            if dataset.calibration is not None and 0 <= index < len(dataset.calibration.samples):
                samples = list(dataset.calibration.samples)
                sample = samples[index]
                samples[index] = type(sample)(
                    sample_id=sample.sample_id,
                    input_vector=sample.input_vector[:-1],
                    reference_output=sample.reference_output,
                    source_index=sample.source_index,
                )
                dataset.calibration.samples = tuple(samples)
                hit = True
                message = f"injected calibration input dimension mismatch at sample index {index}"
        elif injection.kind == "invalid_quantile_bounds":
            if dataset.config is not None:
                dataset.config.defaults["percentile_lower_permille"] = 990
                dataset.config.defaults["percentile_upper_permille"] = 10
                hit = True
                message = "injected invalid quantile bounds 990/10"
        elif injection.kind == "orphan_receipt":
            fake_layer = layer or "__injected_orphan__"
            dataset.receipts.receipts.append(
                Receipt(
                    receipt_id=f"injected-{injection.injection_id}",
                    source_file="fault-injection",
                    source_line=0,
                    source_version="q1",
                    layer=fake_layer,
                    compression_permille=500,
                    parameter_version=None,
                    elapsed_ms=None,
                    group_scale_digest=None,
                    supersedes=(),
                    missing_fields=("parameter_version", "elapsed_ms", "group_scale_digest", "supersedes"),
                    raw={},
                )
            )
            hit = True
            message = f"injected receipt for orphan layer {fake_layer!r}"
        if hit:
            failures.add(
                expected,
                message,
                "fault_injection",
                layer=layer,
                injection_id=injection.injection_id,
                details={"kind": injection.kind},
            )
            results.append(
                {
                    "id": injection.injection_id,
                    "kind": injection.kind,
                    "target_layer": layer,
                    "sample_index": injection.sample_index,
                    "hit": True,
                    "code": expected,
                    "message": message,
                }
            )
        else:
            failures.add(
                E_INJECTION_MISS,
                f"fault injection {injection.injection_id} target was not found",
                "fault_injection",
                layer=layer,
                injection_id=injection.injection_id,
                details={"kind": injection.kind},
            )
            results.append(
                {
                    "id": injection.injection_id,
                    "kind": injection.kind,
                    "target_layer": layer,
                    "sample_index": injection.sample_index,
                    "hit": False,
                    "code": E_INJECTION_MISS,
                    "message": "target not found",
                }
            )
    return results
