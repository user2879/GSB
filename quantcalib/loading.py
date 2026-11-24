"""Validated loading for weights, calibration data, YAML config and receipts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import (
    E_CALIBRATION_INVALID,
    E_CONFIG_INVALID,
    E_GROUP_INVALID,
    E_IO,
    E_QUANT_BOUNDS,
    E_RECEIPT_INVALID,
    E_RECEIPT_ORPHAN,
    E_VERSION_ORDER,
    E_WEIGHT_DUPLICATE,
    E_WEIGHT_INVALID,
    E_WEIGHT_SHAPE,
    FailureList,
)
from .model import (
    CalibrationSample,
    CalibrationSet,
    Config,
    Dataset,
    DistillationConfig,
    Layer,
    Receipt,
    ReceiptCorpus,
    SchedulingConfig,
)
from .utils import first_present, flatten_numeric


WEIGHT_NAME_KEYS = ("layer_name", "name", "layer")
WEIGHT_SHAPE_KEYS = ("shape", "weight_shape")
WEIGHT_VALUES_KEYS = ("weights", "values", "data")
WEIGHT_VERSION_KEYS = ("content_version", "layer_content_version", "version")
WEIGHT_GROUP_KEYS = ("group_granularity", "group_size", "grouping")

CONFIG_DEFAULTS: dict[str, Any] = {
    "scheme": "asymmetric",
    "bit_width": 8,
    "group_granularity": None,
    "percentile_lower_permille": 10,
    "percentile_upper_permille": 990,
    "fractional_bits": 8,
    "quantization_levels": 1,
    "scale_effective_digits": 15,
}

MIN_BIT_WIDTH = 2
MAX_BIT_WIDTH = 16
MAX_FRACTIONAL_BITS = 24


def _read_json_file(path: Path, failures: FailureList, stage: str, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        failures.add(code, f"cannot load JSON: {exc}", stage, path=str(path))
        return None


def load_weights(path: Path, failures: FailureList) -> list[Layer]:
    layers: list[Layer] = []
    names: set[str] = set()
    previous_version: int | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        failures.add(E_IO, f"cannot read weight package: {exc}", "loading", path=str(path))
        return layers

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.add(
                E_WEIGHT_INVALID,
                f"invalid JSONL record: {exc.msg}",
                "loading",
                path=str(path),
                details={"line": line_number},
            )
            continue
        if not isinstance(record, dict):
            failures.add(
                E_WEIGHT_INVALID,
                "weight record must be a JSON object",
                "loading",
                path=str(path),
                details={"line": line_number},
            )
            continue

        name = first_present(record, WEIGHT_NAME_KEYS)
        if not isinstance(name, str) or not name.strip():
            failures.add(
                E_WEIGHT_INVALID,
                "layer name must be a non-empty string",
                "loading",
                path=str(path),
                details={"line": line_number},
            )
            continue
        if name in names:
            failures.add(
                E_WEIGHT_DUPLICATE,
                f"duplicate layer name {name!r}",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue

        shape_value = first_present(record, WEIGHT_SHAPE_KEYS)
        if not isinstance(shape_value, list) or not shape_value:
            failures.add(
                E_WEIGHT_SHAPE,
                "shape must be a non-empty array of positive integers",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape_value):
            failures.add(
                E_WEIGHT_SHAPE,
                "shape dimensions must be positive integers",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number, "shape": shape_value},
            )
            continue
        shape = tuple(int(dim) for dim in shape_value)

        values_value = first_present(record, WEIGHT_VALUES_KEYS)
        try:
            weights = flatten_numeric(values_value)
        except (TypeError, ValueError) as exc:
            failures.add(
                E_WEIGHT_INVALID,
                f"invalid weight elements: {exc}",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue
        if any(not math.isfinite(value) for value in weights):
            failures.add(
                E_WEIGHT_INVALID,
                "all weight elements must be finite",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue
        expected = math.prod(shape)
        if expected != len(weights):
            failures.add(
                E_WEIGHT_SHAPE,
                f"shape product {expected} does not match {len(weights)} weight elements",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue

        version = first_present(record, WEIGHT_VERSION_KEYS)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            failures.add(
                E_WEIGHT_INVALID,
                "content version must be a non-negative integer",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue
        if previous_version is not None and version < previous_version:
            failures.add(
                E_VERSION_ORDER,
                f"content version {version} is lower than preceding version {previous_version}",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue

        group = first_present(record, WEIGHT_GROUP_KEYS)
        if isinstance(group, bool) or not isinstance(group, int) or group <= 0:
            failures.add(
                E_GROUP_INVALID,
                "group granularity must be a positive integer",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue
        if len(weights) % group != 0:
            failures.add(
                E_GROUP_INVALID,
                f"group granularity {group} does not divide {len(weights)} elements",
                "loading",
                layer=name,
                path=str(path),
                details={"line": line_number},
            )
            continue

        previous_version = version
        names.add(name)
        layers.append(
            Layer(
                name=name,
                shape=shape,
                weights=tuple(weights),
                content_version=version,
                group_granularity=group,
                source_line=line_number,
            )
        )
    return layers


def load_calibration(path: Path, failures: FailureList) -> CalibrationSet | None:
    data = _read_json_file(path, failures, "loading", E_CALIBRATION_INVALID)
    if data is None:
        return None
    if not isinstance(data, dict):
        failures.add(E_CALIBRATION_INVALID, "calibration root must be an object", "loading", path=str(path))
        return None
    samples_value = data.get("samples")
    if not isinstance(samples_value, list) or not samples_value:
        failures.add(E_CALIBRATION_INVALID, "samples must be a non-empty array", "loading", path=str(path))
        return None

    input_dim = data.get("input_dim")
    output_dim = data.get("output_dim")
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
        failures.add(E_CALIBRATION_INVALID, "input_dim must be a positive integer", "loading", path=str(path))
        return None
    if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
        failures.add(E_CALIBRATION_INVALID, "output_dim must be a positive integer", "loading", path=str(path))
        return None

    samples: list[CalibrationSample] = []
    ids: set[str] = set()
    for index, item in enumerate(samples_value):
        if not isinstance(item, dict):
            failures.add(
                E_CALIBRATION_INVALID,
                "sample must be an object",
                "loading",
                path=str(path),
                details={"sample_index": index},
            )
            continue
        sample_id = item.get("id", f"sample-{index:04d}")
        if not isinstance(sample_id, str) or not sample_id:
            failures.add(
                E_CALIBRATION_INVALID,
                "sample id must be a non-empty string",
                "loading",
                path=str(path),
                details={"sample_index": index},
            )
            continue
        if sample_id in ids:
            failures.add(
                E_CALIBRATION_INVALID,
                f"duplicate sample id {sample_id!r}",
                "loading",
                path=str(path),
                details={"sample_index": index},
            )
            continue
        input_value = first_present(item, ("input", "input_vector", "x"))
        reference_value = first_present(item, ("reference_output", "reference", "output", "y"))
        try:
            input_vector = flatten_numeric(input_value)
            reference_output = flatten_numeric(reference_value)
        except (TypeError, ValueError) as exc:
            failures.add(
                E_CALIBRATION_INVALID,
                f"sample vectors must contain finite numbers: {exc}",
                "loading",
                path=str(path),
                details={"sample_index": index, "sample_id": sample_id},
            )
            continue
        if len(input_vector) != input_dim:
            failures.add(
                E_CALIBRATION_INVALID,
                f"sample input dimension {len(input_vector)} != input_dim {input_dim}",
                "loading",
                path=str(path),
                details={"sample_index": index, "sample_id": sample_id},
            )
            continue
        if len(reference_output) != output_dim:
            failures.add(
                E_CALIBRATION_INVALID,
                f"sample output dimension {len(reference_output)} != output_dim {output_dim}",
                "loading",
                path=str(path),
                details={"sample_index": index, "sample_id": sample_id},
            )
            continue
        if any(not math.isfinite(value) for value in input_vector + reference_output):
            failures.add(
                E_CALIBRATION_INVALID,
                "sample vectors must be finite",
                "loading",
                path=str(path),
                details={"sample_index": index, "sample_id": sample_id},
            )
            continue
        ids.add(sample_id)
        samples.append(
            CalibrationSample(
                sample_id=sample_id,
                input_vector=tuple(input_vector),
                reference_output=tuple(reference_output),
                source_index=index,
            )
        )
    if not samples:
        return None
    return CalibrationSet(input_dim=input_dim, output_dim=output_dim, samples=tuple(samples))


def _int_field(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
    failures: FailureList,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    code: str = E_CONFIG_INVALID,
) -> int:
    value = first_present(mapping, keys, default)
    if isinstance(value, bool) or not isinstance(value, int):
        failures.add(code, f"{label} must be an integer", "loading")
        return int(default)
    if minimum is not None and value < minimum:
        failures.add(code, f"{label} must be >= {minimum}", "loading")
        return int(default)
    if maximum is not None and value > maximum:
        failures.add(code, f"{label} must be <= {maximum}", "loading")
        return int(default)
    return value


def _float_field(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
    failures: FailureList,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = first_present(mapping, keys, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        failures.add(E_CONFIG_INVALID, f"{label} must be numeric", "loading")
        return float(default)
    number = float(value)
    if not math.isfinite(number):
        failures.add(E_CONFIG_INVALID, f"{label} must be finite", "loading")
        return float(default)
    if minimum is not None and number < minimum:
        failures.add(E_CONFIG_INVALID, f"{label} must be >= {minimum}", "loading")
    if maximum is not None and number > maximum:
        failures.add(E_CONFIG_INVALID, f"{label} must be <= {maximum}", "loading")
    return number


_LAYER_CONFIG_KEYS = {
    "scheme": ("scheme", "quant_scheme", "mode"),
    "bit_width": ("bit_width", "bits", "target_bit_width"),
    "group_granularity": ("group_granularity", "group_size", "grouping"),
    "percentile_lower_permille": (
        "percentile_lower_permille",
        "quantile_lower_permille",
        "lower_permille",
        "percentile_lower",
    ),
    "percentile_upper_permille": (
        "percentile_upper_permille",
        "quantile_upper_permille",
        "upper_permille",
        "percentile_upper",
    ),
    "fractional_bits": ("fractional_bits", "fixed_fractional_bits", "fraction_bits"),
    "quantization_levels": ("quantization_levels", "quant_levels", "levels"),
    "scale_effective_digits": ("scale_effective_digits", "effective_digits", "scale_significant_digits"),
}


def _normalize_layer_block(block: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for canonical, keys in _LAYER_CONFIG_KEYS.items():
        value = first_present(block, keys, None)
        if value is not None:
            normalized[canonical] = value
    return normalized


def load_config(path: Path, failures: FailureList) -> Config | None:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        failures.add(E_CONFIG_INVALID, f"cannot load YAML config: {exc}", "loading", path=str(path))
        return None
    if not isinstance(raw, dict):
        failures.add(E_CONFIG_INVALID, "config root must be a mapping", "loading", path=str(path))
        return None

    quant_block = raw.get("quantization", raw)
    if not isinstance(quant_block, dict):
        failures.add(E_CONFIG_INVALID, "quantization must be a mapping", "loading", path=str(path))
        return None

    defaults = dict(CONFIG_DEFAULTS)
    defaults.update(_normalize_layer_block(quant_block))
    scheme = str(defaults.get("scheme", "asymmetric")).lower()
    if scheme not in {"symmetric", "asymmetric"}:
        failures.add(E_CONFIG_INVALID, "scheme must be symmetric or asymmetric", "loading", path=str(path))
        scheme = "asymmetric"
    defaults["scheme"] = scheme

    if defaults.get("group_granularity") is not None:
        defaults["group_granularity"] = _int_field(
            defaults,
            ("group_granularity",),
            None,
            failures,
            "group_granularity",
            minimum=1,
        )

    layers_block = raw.get("layers", {})
    if layers_block is None:
        layers_block = {}
    if not isinstance(layers_block, dict):
        failures.add(E_CONFIG_INVALID, "layers must be a mapping", "loading", path=str(path))
        layers_block = {}
    overrides: dict[str, dict[str, Any]] = {}
    for layer_name, block in layers_block.items():
        if not isinstance(layer_name, str) or not isinstance(block, dict):
            failures.add(E_CONFIG_INVALID, "each layer override must be a named mapping", "loading")
            continue
        candidate = dict(defaults)
        candidate.update(_normalize_layer_block(block))
        overrides[layer_name] = _validate_layer_quant_mapping(candidate, failures, layer_name)

    defaults = _validate_layer_quant_mapping(dict(defaults), failures, None)

    distill_block = raw.get("distillation", raw.get("distill", {}))
    if distill_block is None:
        distill_block = {}
    if not isinstance(distill_block, dict):
        failures.add(E_CONFIG_INVALID, "distillation must be a mapping", "loading", path=str(path))
        distill_block = {}
    temperature = _float_field(
        distill_block,
        ("temperature", "distill_temperature"),
        1.0,
        failures,
        "distillation temperature",
        minimum=1e-12,
    )
    steps = _int_field(
        distill_block,
        ("steps", "distill_steps"),
        2,
        failures,
        "distillation steps",
        minimum=0,
    )
    soft_weight = _float_field(
        distill_block,
        ("soft_label_weight", "soft_weight", "weight"),
        0.5,
        failures,
        "soft label weight",
        minimum=0.0,
        maximum=1.0,
    )
    tolerance = _float_field(
        distill_block,
        ("tolerance", "relative_tolerance"),
        0.01,
        failures,
        "tolerance",
        minimum=0.0,
    )

    schedule_block = raw.get("scheduling", raw.get("scheduler", {}))
    if schedule_block is None:
        schedule_block = {}
    if not isinstance(schedule_block, dict):
        failures.add(E_CONFIG_INVALID, "scheduling must be a mapping", "loading", path=str(path))
        schedule_block = {}
    workers = _int_field(
        schedule_block,
        ("thread_pool_workers", "workers", "concurrency"),
        1,
        failures,
        "thread_pool_workers",
        minimum=1,
    )
    global_budget = _int_field(
        schedule_block,
        ("global_calibration_point_budget", "global_point_budget", "global_budget"),
        1_000_000,
        failures,
        "global calibration point budget",
        minimum=0,
    )
    quota = _int_field(
        schedule_block,
        ("per_layer_calibration_quota", "per_layer_quota", "layer_quota"),
        1_000_000,
        failures,
        "per-layer calibration quota",
        minimum=0,
    )
    batch_budget = _int_field(
        schedule_block,
        ("batch_quantization_point_budget", "batch_point_budget", "batch_budget"),
        1_000_000,
        failures,
        "batch quantization point budget",
        minimum=1,
    )
    snapshot_interval = _int_field(
        schedule_block,
        ("snapshot_commit_interval_batches", "snapshot_interval_batches", "snapshot_interval"),
        1,
        failures,
        "snapshot commit interval",
        minimum=1,
    )
    return Config(
        raw=raw,
        defaults=defaults,
        layer_overrides=overrides,
        distillation=DistillationConfig(temperature, steps, soft_weight, tolerance),
        scheduling=SchedulingConfig(workers, global_budget, quota, batch_budget, snapshot_interval),
    )


def _validate_layer_quant_mapping(
    candidate: dict[str, Any], failures: FailureList, layer_name: str | None
) -> dict[str, Any]:
    subject = f"layer {layer_name!r}" if layer_name else "global quantization"
    scheme = str(candidate.get("scheme", "asymmetric")).lower()
    if scheme not in {"symmetric", "asymmetric"}:
        failures.add(E_CONFIG_INVALID, f"{subject}: scheme must be symmetric or asymmetric", "loading", layer=layer_name)
        scheme = "asymmetric"
    candidate["scheme"] = scheme
    bit_width = _int_field(
        candidate,
        ("bit_width",),
        CONFIG_DEFAULTS["bit_width"],
        failures,
        f"{subject} bit_width",
        minimum=MIN_BIT_WIDTH,
        maximum=MAX_BIT_WIDTH,
        code=E_QUANT_BOUNDS,
    )
    candidate["bit_width"] = bit_width
    lower = _int_field(
        candidate,
        ("percentile_lower_permille",),
        CONFIG_DEFAULTS["percentile_lower_permille"],
        failures,
        f"{subject} percentile lower",
    )
    upper = _int_field(
        candidate,
        ("percentile_upper_permille",),
        CONFIG_DEFAULTS["percentile_upper_permille"],
        failures,
        f"{subject} percentile upper",
    )
    if not (0 <= lower < upper <= 1000):
        failures.add(
            E_CONFIG_INVALID,
            f"{subject}: percentile bounds must satisfy 0 <= lower < upper <= 1000",
            "loading",
            layer=layer_name,
            details={"lower_permille": lower, "upper_permille": upper},
        )
    candidate["percentile_lower_permille"] = lower
    candidate["percentile_upper_permille"] = upper
    candidate["fractional_bits"] = _int_field(
        candidate,
        ("fractional_bits",),
        CONFIG_DEFAULTS["fractional_bits"],
        failures,
        f"{subject} fractional_bits",
        minimum=0,
        maximum=MAX_FRACTIONAL_BITS,
    )
    candidate["quantization_levels"] = _int_field(
        candidate,
        ("quantization_levels",),
        CONFIG_DEFAULTS["quantization_levels"],
        failures,
        f"{subject} quantization_levels",
        minimum=1,
    )
    candidate["scale_effective_digits"] = _int_field(
        candidate,
        ("scale_effective_digits",),
        CONFIG_DEFAULTS["scale_effective_digits"],
        failures,
        f"{subject} scale_effective_digits",
        minimum=1,
        maximum=17,
    )
    if candidate.get("group_granularity") is not None:
        candidate["group_granularity"] = _int_field(
            candidate,
            ("group_granularity",),
            None,
            failures,
            f"{subject} group_granularity",
            minimum=1,
        )
    return candidate


def validate_effective_config(dataset: Dataset) -> None:
    if dataset.config is None:
        return
    for layer in dataset.layers:
        effective = dataset.config.for_layer(layer)
        if layer.numel % effective.group_granularity != 0:
            dataset.failures.add(
                E_GROUP_INVALID,
                f"effective group granularity {effective.group_granularity} does not divide {layer.numel} elements",
                "loading",
                layer=layer.name,
            )


def _receipt_version(record: dict[str, Any]) -> str:
    declared = first_present(record, ("receipt_version", "schema_version", "qversion"))
    if declared is not None:
        text = str(declared).lower()
        if text in {"1", "q1"}:
            return "q1"
        if text in {"2", "q2"}:
            return "q2"
        if text in {"3", "q3"}:
            return "q3"
    if first_present(record, ("group_scale_digest", "scale_digest", "group_scales_sha256")) is not None:
        return "q3"
    if first_present(record, ("supersedes", "superseded_receipts", "overrides")) is not None:
        return "q3"
    if first_present(record, ("parameter_version", "params_version", "parameter_version_seq")) is not None:
        return "q2"
    if first_present(record, ("elapsed_ms", "duration_ms", "runtime_ms")) is not None:
        return "q2"
    return "q1"


def load_receipts(
    paths: Iterable[Path], valid_layers: set[str], failures: FailureList
) -> ReceiptCorpus:
    corpus = ReceiptCorpus()
    seen_ids: set[str] = set()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        corpus.supplied_paths.append(str(path))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            failures.add(E_IO, f"cannot read receipt file: {exc}", "loading", path=str(path))
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.add(
                    E_RECEIPT_INVALID,
                    f"invalid receipt JSON: {exc.msg}",
                    "loading",
                    path=str(path),
                    details={"line": line_number},
                )
                continue
            if not isinstance(record, dict):
                failures.add(
                    E_RECEIPT_INVALID,
                    "receipt must be a JSON object",
                    "loading",
                    path=str(path),
                    details={"line": line_number},
                )
                continue
            version = _receipt_version(record)
            layer = first_present(record, ("layer_name", "layer", "name"))
            if not isinstance(layer, str) or not layer:
                failures.add(
                    E_RECEIPT_INVALID,
                    "receipt layer must be a non-empty string",
                    "loading",
                    path=str(path),
                    details={"line": line_number, "source_version": version},
                )
                continue
            compression = first_present(
                record,
                ("compression_permille", "compression_ratio_permille", "ratio_permille", "compression"),
            )
            if isinstance(compression, bool) or not isinstance(compression, int):
                failures.add(
                    E_RECEIPT_INVALID,
                    "compression permille must be an integer",
                    "loading",
                    layer=layer,
                    path=str(path),
                    details={"line": line_number, "source_version": version},
                )
                continue
            receipt_id = record.get("receipt_id")
            if not isinstance(receipt_id, str) or not receipt_id:
                receipt_id = f"{path.name}#L{line_number}"
            if receipt_id in seen_ids:
                receipt_id = f"{receipt_id}@{line_number}"
            seen_ids.add(receipt_id)

            missing: list[str] = []
            parameter_version = first_present(
                record, ("parameter_version", "params_version", "parameter_version_seq")
            )
            if parameter_version is None:
                missing.append("parameter_version")
            elif isinstance(parameter_version, bool) or not isinstance(parameter_version, int):
                failures.add(
                    E_RECEIPT_INVALID,
                    "parameter_version must be an integer",
                    "loading",
                    layer=layer,
                    path=str(path),
                    details={"line": line_number},
                )
                parameter_version = None

            elapsed_ms = first_present(record, ("elapsed_ms", "duration_ms", "runtime_ms"))
            if elapsed_ms is None:
                missing.append("elapsed_ms")
            elif isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int) or elapsed_ms < 0:
                failures.add(
                    E_RECEIPT_INVALID,
                    "elapsed_ms must be a non-negative integer",
                    "loading",
                    layer=layer,
                    path=str(path),
                    details={"line": line_number},
                )
                elapsed_ms = None

            digest = first_present(record, ("group_scale_digest", "scale_digest", "group_scales_sha256"))
            if digest is None:
                missing.append("group_scale_digest")
            elif not isinstance(digest, str):
                failures.add(
                    E_RECEIPT_INVALID,
                    "group_scale_digest must be a string",
                    "loading",
                    layer=layer,
                    path=str(path),
                    details={"line": line_number},
                )
                digest = None

            supersedes_value = first_present(record, ("supersedes", "superseded_receipts", "overrides"))
            if supersedes_value is None:
                missing.append("supersedes")
                supersedes: tuple[str, ...] = ()
            elif isinstance(supersedes_value, str):
                supersedes = (supersedes_value,)
            elif isinstance(supersedes_value, list) and all(isinstance(item, str) for item in supersedes_value):
                supersedes = tuple(supersedes_value)
            else:
                failures.add(
                    E_RECEIPT_INVALID,
                    "supersedes must be a string or array of strings",
                    "loading",
                    layer=layer,
                    path=str(path),
                    details={"line": line_number},
                )
                missing.append("supersedes")
                supersedes = ()

            if layer not in valid_layers:
                failures.add(
                    E_RECEIPT_ORPHAN,
                    f"receipt references unknown layer {layer!r}",
                    "loading",
                    layer=layer,
                    path=str(path),
                    details={"line": line_number, "receipt_id": receipt_id},
                )
            corpus.receipts.append(
                Receipt(
                    receipt_id=receipt_id,
                    source_file=path.name,
                    source_line=line_number,
                    source_version=version,
                    layer=layer,
                    compression_permille=compression,
                    parameter_version=parameter_version,
                    elapsed_ms=elapsed_ms,
                    group_scale_digest=digest,
                    supersedes=supersedes,
                    missing_fields=tuple(missing),
                    raw=record,
                )
            )
    return corpus


def load_dataset(
    *,
    weights_path: Path,
    calibration_path: Path,
    config_path: Path,
    receipt_paths: Iterable[Path] = (),
) -> Dataset:
    failures = FailureList()
    layers = load_weights(weights_path, failures)
    calibration = load_calibration(calibration_path, failures)
    config = load_config(config_path, failures)
    corpus = load_receipts(receipt_paths, {layer.name for layer in layers}, failures)
    dataset = Dataset(
        layers=layers,
        calibration=calibration,
        config=config,
        receipts=corpus,
        failures=failures,
        weights_path=str(weights_path),
        calibration_path=str(calibration_path),
        config_path=str(config_path),
    )
    validate_effective_config(dataset)
    return dataset
