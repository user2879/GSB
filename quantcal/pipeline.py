"""End-to-end deterministic calibration, exchange, cache and snapshot pipeline."""

from __future__ import annotations

import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .cache import StageCache, content_fingerprint, make_cache_key
from .calibration import (
    QUANTILE_METHOD,
    TIE_RULE,
    calibrate_quantiles,
    derive_scale_zero,
    exchange_scaled_result,
    export_quantile_range,
    load_samples,
    load_tensor_specs,
    tensor_values_and_refs,
)
from .config import load_config
from .errors import QuantCalError
from .format_v1 import parse_v1, render_v1
from .format_v2 import (
    downgrade_v2_to_v1,
    parse_v2,
    to_v2_document,
    upgrade_v1_to_v2,
)
from .models import (
    ExchangeSet,
    TensorParam,
    scheme_saturation_max,
)
from .quantize import precision_summary, quantize_tensor
from .snapshot import SnapshotManager, snapshot_public_view
from .util import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_text,
    fingerprint,
    load_json,
    sha256_file,
)


@dataclass(frozen=True)
class InputPaths:
    tensor_manifest: Path
    samples: Path
    exchange_v1: Path
    exchange_v2: Path
    config: Path
    injections: Path

    @classmethod
    def discover(cls, input_dir: str | Path) -> "InputPaths":
        root = Path(input_dir)
        return cls(
            tensor_manifest=root / "tensors.json",
            samples=root / "samples.jsonl",
            exchange_v1=root / "calibration_v1.txt",
            exchange_v2=root / "calibration_v2.json",
            config=root / "config.json",
            injections=root / "injections.json",
        )


@dataclass
class TensorCalibration:
    name: str
    spec: dict[str, Any]
    resolved: dict[str, Any]
    values: list[float]
    layout: list[dict[str, Any]]
    sample_ids: list[str]
    values_fingerprint: str
    point_count: int
    quantile: dict[str, Any]
    scale: dict[str, Any]
    statuses: dict[str, dict[str, Any]]


@dataclass
class TensorFinal:
    name: str
    calibration: TensorCalibration
    quantize: dict[str, Any]
    precision: dict[str, Any]
    statuses: dict[str, dict[str, Any]]

    def result_payload(self) -> dict[str, Any]:
        quantile = self.calibration.quantile
        scale = self.calibration.scale
        return {
            "name": self.name,
            "shape": self.calibration.spec["shape"],
            "distribution": self.calibration.spec["distribution"],
            "sample_ids": self.calibration.sample_ids,
            "point_count": self.calibration.point_count,
            "quantile_lower": quantile["quantile_lower"],
            "quantile_upper": quantile["quantile_upper"],
            "quantile_metadata": quantile["quantile_metadata"],
            "scale": scale["scale"],
            "calibrated_scale": scale["calibrated_scale"],
            "zero_point": scale["zero_point"],
            "calibrated_zero_point": scale["calibrated_zero_point"],
            "quantization_bits": scale["quantization_bits"],
            "quantization_scheme": scale["quantization_scheme"],
            "calibrated_scheme": scale["calibrated_scheme"],
            "saturation_max": scale["saturation_max"],
            "scale_guard": scale.get("scale_guard"),
            "exchange_loss": scale.get("exchange_loss"),
            "config": self.calibration.resolved,
            "quantized_values": self.quantize["quantized_values"],
            "dequantized_values": self.quantize["dequantized_values"],
            "quantized_summary": self.quantize["quantized_summary"],
            "effective_scale": self.quantize["effective_scale"],
            "effective_scale_coefficient": self.quantize["effective_scale_coefficient"],
            "effective_scale_guard": self.quantize["effective_scale_guard"],
            "clamp_bounds": self.quantize["clamp_bounds"],
            "clipped_positions": self.quantize["clipped_positions"],
            "clipped_count": self.quantize["clipped_count"],
            "clipped_element_ratio": self.quantize["clipped_element_ratio"],
            "rounding_mode": self.quantize["rounding_mode"],
            "saturation_strategy": self.quantize["saturation_strategy"],
            "max_abs_error": self.precision["max_abs_error"],
            "mean_abs_error": self.precision["mean_abs_error"],
            "mse": self.precision["mse"],
            "error_attribution": self.precision["error_attribution"],
            "max_error_flat_index": self.precision["max_error_flat_index"],
        }


@dataclass
class PipelineResult:
    result: dict[str, Any]
    diagnostics: dict[str, Any]
    output_dir: Path
    exit_code: int
    snapshot: dict[str, Any] | None
    exchange_text: str | None = None
    exchange_document: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)


def _tensor_spec_payload(spec: Any) -> dict[str, Any]:
    return {
        "name": spec.name,
        "shape": list(spec.shape),
        "distribution": spec.distribution,
        "special_positions": list(spec.special_positions),
    }


def _safe_tensor_name(name: str) -> None:
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise QuantCalError(
            "E_FORMAT_INVALID", "tensor name contains a path separator", tensor=name, stage="parse"
        )


def _fingerprint_jsonable(value: Any) -> Any:
    """Convert non-finite floats to stable tokens for hashing only."""

    if isinstance(value, float):
        if math.isnan(value):
            return "float:nan"
        if math.isinf(value):
            return "float:+inf" if value > 0 else "float:-inf"
        return value
    if isinstance(value, dict):
        return {str(key): _fingerprint_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_jsonable(item) for item in value]
    return value


def _load_and_validate_exchanges(
    paths: InputPaths, names: list[str]
) -> tuple[ExchangeSet, ExchangeSet, dict[str, Any]]:
    try:
        v1_text = paths.exchange_v1.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise QuantCalError("E_FORMAT_INVALID", f"cannot read v1 exchange: {exc}", stage="parse") from exc
    v1_input = parse_v1(v1_text, names)
    v2_input = parse_v2(load_json(paths.exchange_v2), names)

    upgraded, upgrade_losses = upgrade_v1_to_v2(v1_input)
    downgraded, downgrade_text, probe_sidecar = downgrade_v2_to_v1(upgraded)
    reparsed = parse_v1(downgrade_text, names)
    roundtrip_exact = (
        [item.name for item in v1_input.tensors] == [item.name for item in reparsed.tensors]
        and v1_input.quantization_bits == reparsed.quantization_bits
        and all(
            abs(left.scale - right.scale) <= 1e-15
            for left, right in zip(v1_input.tensors, reparsed.tensors)
        )
        and all(left.zero_point == right.zero_point for left, right in zip(v1_input.tensors, reparsed.tensors))
    )
    probe_losses = probe_sidecar["tensors"]
    v2_down, _, v2_sidecar = downgrade_v2_to_v1(v2_input)
    v2_reparsed = parse_v1(render_v1(v2_down), names)
    interop = {
        "input_v1": {
            "tensor_order": [item.name for item in v1_input.tensors],
            "quantization_bits": v1_input.quantization_bits,
            "upgraded_precision_tags": sorted({item.precision_tag for item in upgraded.tensors}),
            "upgrade_loss_by_tensor": upgrade_losses["losses"],
            "roundtrip_exact": roundtrip_exact,
            "roundtrip_probe_sidecar": probe_sidecar,
        },
        "input_v2": {
            "tensor_order": [item.name for item in v2_input.tensors],
            "quantization_bits": v2_input.quantization_bits,
            "downgrade_loss_by_tensor": v2_sidecar["tensors"],
            "v1_tensor_order": [item.name for item in v2_reparsed.tensors],
            "loss_codes": v2_sidecar["losses"],
        },
        "v1_to_v2_to_v1_exact": roundtrip_exact,
        "probe_loss_count": len(probe_losses),
    }
    return v1_input, v2_input, interop


def _exchange_from_calibrations(
    calibrations: dict[str, TensorCalibration],
    version: str,
    use_calibrated_values: bool = False,
    default_bits: int = 8,
) -> ExchangeSet:
    tensors: list[TensorParam] = []
    for name in sorted(calibrations):
        calibration = calibrations[name]
        scale = calibration.scale
        resolved = calibration.resolved
        scheme = str(scale["calibrated_scheme"] if use_calibrated_values else scale["quantization_scheme"])
        tensor_scale = float(scale["calibrated_scale"] if use_calibrated_values else scale["scale"])
        tensor_zero = int(
            scale["calibrated_zero_point"] if use_calibrated_values else scale["zero_point"]
        )
        bits = int(scale["quantization_bits"])
        saturation = scheme_saturation_max(scheme, bits)
        range_lower, range_upper, range_expanded = export_quantile_range(
            float(calibration.quantile["quantile_lower"]),
            float(calibration.quantile["quantile_upper"]),
        )
        tensors.append(
            TensorParam(
                name=name,
                scale=tensor_scale,
                zero_point=tensor_zero,
                scheme=scheme,
                rounding=str(resolved["rounding_mode"]),
                precision_tag=(
                    "v1_milli_exact_1e-3"
                    if version == "v1"
                    else f"calibrated_float64_q{bits}"
                ),
                quantization_bits=bits,
                saturation_max=saturation,
                quantile_lower=range_lower,
                quantile_upper=range_upper,
                source={
                    "pipeline": "quantcal.pipeline.v1",
                    "quantile_method": QUANTILE_METHOD,
                    "tie_rule": TIE_RULE,
                    "scale_guard": scale.get("scale_guard"),
                    "measured_quantile_range": [
                        float(calibration.quantile["quantile_lower"]),
                        float(calibration.quantile["quantile_upper"]),
                    ],
                    "quantile_range_expanded_for_zero_span": range_expanded,
                },
            )
        )
    bundle_bits = tensors[0].quantization_bits if tensors else default_bits
    return ExchangeSet(version=version, quantization_bits=bundle_bits, tensors=tensors)


def _stage_files(calibrations: Iterable[TensorCalibration]) -> dict[str, Any]:
    ordered = sorted(calibrations, key=lambda item: item.name)
    files: dict[str, Any] = {
        "index.json": {
            "format": "quantcal.calibration-stage.v1",
            "tensor_order": [item.name for item in ordered],
        }
    }
    for index, item in enumerate(ordered):
        files[f"tensors/{index:04d}.json"] = {
            "tensor": item.name,
            "quantile": item.quantile,
            "scale": item.scale,
            "sample_ids": item.sample_ids,
            "quantile_method": QUANTILE_METHOD,
            "tie_rule": TIE_RULE,
        }
    return files


def _quantize_stage_files(finals: Iterable[TensorFinal]) -> dict[str, Any]:
    ordered = sorted(finals, key=lambda item: item.name)
    files: dict[str, Any] = {
        "index.json": {
            "format": "quantcal.quantize-stage.v1",
            "tensor_order": [item.name for item in ordered],
        }
    }
    for index, item in enumerate(ordered):
        files[f"tensors/{index:04d}.json"] = item.quantize
    return files


def _precision_stage_files(finals: Iterable[TensorFinal]) -> dict[str, Any]:
    ordered = sorted(finals, key=lambda item: item.name)
    files: dict[str, Any] = {
        "index.json": {
            "format": "quantcal.precision-stage.v1",
            "tensor_order": [item.name for item in ordered],
        }
    }
    for index, item in enumerate(ordered):
        files[f"tensors/{index:04d}.json"] = item.precision
    return files


def _error_record(error: QuantCalError, tensor: str | None = None, stage: str | None = None) -> dict[str, Any]:
    record = error.to_dict()
    if tensor is not None and "tensor" not in record:
        record["tensor"] = tensor
    if stage is not None and "stage" not in record:
        record["stage"] = stage
    return record


def _wrap_stage(stage: str, tensor: str, function: Any) -> Any:
    try:
        return function()
    except QuantCalError as exc:
        if exc.tensor is None:
            exc.tensor = tensor
        if exc.stage is None:
            exc.stage = stage
        raise
    except Exception as exc:  # isolate one tensor from unexpected numerical failures
        raise QuantCalError(
            "E_INTERNAL",
            f"{type(exc).__name__}: {exc}",
            tensor=tensor,
            stage=stage,
        ) from exc


def run_pipeline(
    input_dir: str | Path,
    work_dir: str | Path,
    exchange_version: str = "v2",
    report_path: str | Path | None = None,
    resume: bool = False,
    force_rebuild: bool = False,
    max_concurrency: int | None = None,
    batch_point_budget: int | None = None,
    injection_hits: list[dict[str, Any]] | None = None,
) -> PipelineResult:
    if exchange_version not in {"v1", "v2"}:
        raise QuantCalError("E_FORMAT_INVALID", "exchange_version must be v1 or v2", stage="parse")
    paths = InputPaths.discover(input_dir)
    work = Path(work_dir)
    if force_rebuild:
        for name in ("state", "stages", "tmp", "output"):
            shutil.rmtree(work / name, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    specs = load_tensor_specs(paths.tensor_manifest)
    for spec in specs:
        _safe_tensor_name(spec.name)
    spec_map = {spec.name: spec for spec in specs}
    config = load_config(paths.config)
    samples = load_samples(str(paths.samples), specs)
    sample_content_fingerprints = {
        sample.sample_id: fingerprint(
            _fingerprint_jsonable(
                {"sample_id": sample.sample_id, "features": sample.features}
            )
        )
        for sample in samples
    }
    names = sorted(spec_map)
    if max_concurrency is not None:
        if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool):
            raise QuantCalError(
                "E_FORMAT_INVALID", "max_concurrency override must be an integer", stage="config"
            )
        config.max_concurrency = max_concurrency
    if batch_point_budget is not None:
        if not isinstance(batch_point_budget, int) or isinstance(batch_point_budget, bool):
            raise QuantCalError(
                "E_FORMAT_INVALID", "batch_point_budget override must be an integer", stage="config"
            )
        config.batch_point_budget = batch_point_budget
    if config.max_concurrency < 1 or config.batch_point_budget < 0:
        raise QuantCalError("E_FORMAT_INVALID", "invalid runtime scheduler override", stage="config")

    _, _, interop = _load_and_validate_exchanges(paths, names)
    snapshot_manager = SnapshotManager(work, config.fingerprint, exchange_version, resume=resume)
    snapshot_manager.startup()
    cache = StageCache(work, enabled=not force_rebuild)

    parse_stage_payload = {
        "format": "quantcal.parse-stage.v1",
        "tensor_manifest": [_tensor_spec_payload(spec_map[name]) for name in names],
        "quantile_method": QUANTILE_METHOD,
        "tie_rule": TIE_RULE,
        "config_fingerprint": config.fingerprint,
        "exchange_version": exchange_version,
        "input_exchange_fingerprints": {
            "v1": sha256_file(paths.exchange_v1),
            "v2": sha256_file(paths.exchange_v2),
        },
        "input_interop": interop,
    }
    parse_hash = fingerprint(parse_stage_payload)
    snapshot_manager.publish_stage("parse", {"parse.json": parse_stage_payload}, parse_hash)

    errors: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    prepared: dict[str, dict[str, Any]] = {}
    remaining_budget = config.batch_point_budget
    budget_closed = False
    deduction_sequence: list[dict[str, Any]] = []
    for name in names:
        try:
            resolved = config.resolve(name)
            values, refs, layout = tensor_values_and_refs(name, samples)
            values_fp = fingerprint(
                {
                    "tensor": name,
                    "sample_ids": refs,
                    "referenced_sample_fingerprints": [
                        sample_content_fingerprints[sample_id] for sample_id in refs
                    ],
                    "values": values,
                    "layout": layout,
                    "shape": list(spec_map[name].shape),
                }
            )
            quantile_params = {
                "lower_permille": resolved.quantile_lower_permille,
                "upper_permille": resolved.quantile_upper_permille,
                "method": QUANTILE_METHOD,
                "tie_rule": TIE_RULE,
            }
            quantile_key = make_cache_key(
                "quantile", values_fp, fingerprint(quantile_params), exchange_version
            )
            lookup = cache.lookup("quantile", quantile_key)
            prepared[name] = {
                "resolved": resolved,
                "values": values,
                "refs": refs,
                "layout": layout,
                "values_fingerprint": values_fp,
                "quantile_params": quantile_params,
                "quantile_key": quantile_key,
                "quantile_lookup": lookup,
                "spec": _tensor_spec_payload(spec_map[name]),
            }
            if lookup.payload is not None:
                continue
            cost = len(values)
            if budget_closed or cost > remaining_budget:
                budget_closed = True
                pending.append(
                    {
                        "tensor": name,
                        "stage": "calibrate",
                        "code": "E_CALIB_BUDGET_PENDING",
                        "points_required": cost,
                        "points_remaining": remaining_budget,
                    }
                )
                prepared[name]["pending"] = True
            else:
                remaining_budget -= cost
                deduction_sequence.append(
                    {
                        "tensor": name,
                        "points": cost,
                        "remaining_budget": remaining_budget,
                    }
                )
        except QuantCalError as exc:
            errors.append(_error_record(exc, name))

    runnable_names = [
        name
        for name in names
        if name in prepared and not prepared[name].get("pending")
    ]

    def calibrate_one(name: str) -> TensorCalibration:
        item = prepared[name]
        resolved = item["resolved"]
        lookup = item["quantile_lookup"]
        statuses: dict[str, dict[str, Any]] = {}
        if lookup.payload is not None:
            quantile = lookup.payload
            qstatus = "reused"
        else:
            quantile = _wrap_stage(
                "calibrate",
                name,
                lambda: calibrate_quantiles(
                    name, item["values"], item["refs"], item["layout"], resolved
                ),
            )
            cache.store("quantile", item["quantile_key"], quantile)
            qstatus = "computed"
        qhash = content_fingerprint("quantile", quantile)
        statuses["quantile"] = {
            "key": item["quantile_key"],
            "content_fingerprint": qhash,
            "status": qstatus,
            "task_sort_key": f"{name}:quantile",
        }
        scale_params = {
            "quantization_bits": resolved.quantization_bits,
            "quantization_scheme": resolved.quantization_scheme,
            "zero_range_guard_scale": 0.001,
            "zero_point_rounding": "nearest_even",
            "exchange_version": exchange_version,
        }
        scale_key = make_cache_key("scale", qhash, fingerprint(scale_params), exchange_version)
        scale_lookup = cache.lookup("scale", scale_key)
        if scale_lookup.payload is not None:
            scale = scale_lookup.payload
            if "saturation_max" not in scale:
                scale["saturation_max"] = scheme_saturation_max(
                    str(scale["quantization_scheme"]), int(scale["quantization_bits"])
                )
            sstatus = "reused"
        else:
            raw_scale = _wrap_stage(
                "calibrate", name, lambda: derive_scale_zero(name, quantile, resolved)
            )
            scale = _wrap_stage(
                "calibrate",
                name,
                lambda: exchange_scaled_result(raw_scale, exchange_version),
            )
            scale["quantile_lower"] = quantile["quantile_lower"]
            scale["quantile_upper"] = quantile["quantile_upper"]
            cache.store("scale", scale_key, scale)
            sstatus = "computed"
        shash = content_fingerprint("scale", scale)
        statuses["scale"] = {
            "key": scale_key,
            "content_fingerprint": shash,
            "status": sstatus,
            "task_sort_key": f"{name}:scale",
        }
        return TensorCalibration(
            name=name,
            spec=item["spec"],
            resolved=resolved.to_dict(),
            values=item["values"],
            layout=item["layout"],
            sample_ids=item["refs"],
            values_fingerprint=item["values_fingerprint"],
            point_count=len(item["values"]),
            quantile=quantile,
            scale=scale,
            statuses=statuses,
        )

    calibrations: dict[str, TensorCalibration] = {}
    completed_for_checkpoint: list[str] = []
    with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
        futures = {name: executor.submit(calibrate_one, name) for name in runnable_names}
        for name in runnable_names:
            try:
                calibrations[name] = futures[name].result()
            except QuantCalError as exc:
                errors.append(_error_record(exc, name, exc.stage or "calibrate"))
            except Exception as exc:
                errors.append(
                    _error_record(
                        QuantCalError(
                            "E_INTERNAL",
                            f"{type(exc).__name__}: {exc}",
                            tensor=name,
                            stage="calibrate",
                        ),
                        name,
                        "calibrate",
                    )
                )
            completed_for_checkpoint.append(name)
            if len(completed_for_checkpoint) % config.snapshot_commit_interval == 0:
                snapshot_manager.checkpoint("calibrate", completed_for_checkpoint)
    snapshot_manager.checkpoint("calibrate", completed_for_checkpoint)

    calibrate_files = _stage_files(calibrations.values())
    calibrate_hash = fingerprint(calibrate_files)
    snapshot_manager.publish_stage("calibrate", calibrate_files, calibrate_hash)

    final_exchange = _exchange_from_calibrations(calibrations, exchange_version)
    sidecar: dict[str, Any]
    if exchange_version == "v1":
        source_v2 = _exchange_from_calibrations(
            calibrations, "v2", use_calibrated_values=True, default_bits=config.quantization_bits
        )
        _, exchange_text, sidecar = downgrade_v2_to_v1(source_v2)
        if calibrations:
            reparsed = parse_v1(exchange_text, sorted(calibrations))
            if [item.name for item in reparsed.tensors] != [item.name for item in source_v2.tensors]:
                raise QuantCalError("E_INTERNAL", "v1 exchange reparse changed tensor order", stage="exchange")
        exchange_document = None
        exchange_fp = fingerprint({"version": "v1", "text": exchange_text})
    else:
        exchange_document = to_v2_document(final_exchange)
        exchange_text = canonical_json_text(exchange_document)
        sidecar = {
            "format": "quantcal.exchange-sidecar.v1",
            "selected_version": "v2",
            "v1_input_upgrade_losses": interop["input_v1"]["upgrade_loss_by_tensor"],
            "v1_input_upgrade_roundtrip_exact": interop["v1_to_v2_to_v1_exact"],
            "note": "v2 preserves named scheme, rounding, precision tag and quantile range",
        }
        exchange_fp = fingerprint({"version": "v2", "document": exchange_document})

    def quantize_one(calibration: TensorCalibration) -> TensorFinal:
        name = calibration.name
        resolved_payload = calibration.resolved
        from .config import ResolvedTensorConfig

        resolved = ResolvedTensorConfig(**resolved_payload)
        statuses = dict(calibration.statuses)
        quantize_params = {
            "fixed_fractional_bits": resolved.fixed_fractional_bits,
            "rounding_mode": resolved.rounding_mode,
            "saturation_strategy": resolved.saturation_strategy,
            "quantization_bits": calibration.scale["quantization_bits"],
            "quantization_scheme": calibration.scale["quantization_scheme"],
            "zero_point": calibration.scale["zero_point"],
            "scale": calibration.scale["scale"],
            "saturation_max": calibration.scale["saturation_max"],
            "exchange_version": exchange_version,
        }
        scale_hash = content_fingerprint("scale", calibration.scale)
        quantize_key = make_cache_key(
            "quantize",
            fingerprint(
                {
                    "scale_content": scale_hash,
                    "values": calibration.values_fingerprint,
                }
            ),
            fingerprint(quantize_params),
            exchange_version,
        )
        lookup = cache.lookup("quantize", quantize_key)
        if lookup.payload is not None:
            quantized = lookup.payload
            qzstatus = "reused"
        else:
            quantized = _wrap_stage(
                "quantize",
                name,
                lambda: quantize_tensor(
                    name,
                    calibration.values,
                    calibration.layout,
                    resolved,
                    calibration.scale,
                    exchange_version,
                ),
            )
            quantized["shape"] = calibration.spec["shape"]
            cache.store("quantize", quantize_key, quantized)
            qzstatus = "computed"
        qzhash = content_fingerprint("quantize", quantized)
        statuses["quantize"] = {
            "key": quantize_key,
            "content_fingerprint": qzhash,
            "status": qzstatus,
            "task_sort_key": f"{name}:quantize",
        }
        precision_key = make_cache_key(
            "precision",
            qzhash,
            fingerprint({"metric_schema": "max_abs/mean_abs/mse/v1"}),
            exchange_version,
        )
        precision_lookup = cache.lookup("precision", precision_key)
        if precision_lookup.payload is not None:
            metrics = precision_lookup.payload
            pstatus = "reused"
        else:
            metrics = _wrap_stage(
                "precision",
                name,
                lambda: precision_summary(quantized, calibration.values),
            )
            cache.store("precision", precision_key, metrics)
            pstatus = "computed"
        statuses["precision"] = {
            "key": precision_key,
            "content_fingerprint": content_fingerprint("precision", metrics),
            "status": pstatus,
            "task_sort_key": f"{name}:precision",
        }
        return TensorFinal(
            name=name,
            calibration=calibration,
            quantize=quantized,
            precision=metrics,
            statuses=statuses,
        )

    finals: dict[str, TensorFinal] = {}
    quantize_checkpoint: list[str] = []
    calibration_order = [name for name in names if name in calibrations]
    with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
        futures = {
            name: executor.submit(quantize_one, calibrations[name])
            for name in calibration_order
        }
        for name in calibration_order:
            try:
                finals[name] = futures[name].result()
            except QuantCalError as exc:
                errors.append(_error_record(exc, name, exc.stage or "quantize"))
            except Exception as exc:
                errors.append(
                    _error_record(
                        QuantCalError(
                            "E_INTERNAL",
                            f"{type(exc).__name__}: {exc}",
                            tensor=name,
                            stage="quantize",
                        ),
                        name,
                        "quantize",
                    )
                )
            quantize_checkpoint.append(name)
            if len(quantize_checkpoint) % config.snapshot_commit_interval == 0:
                snapshot_manager.checkpoint("quantize", quantize_checkpoint)
    snapshot_manager.checkpoint("quantize", quantize_checkpoint)
    quantize_files = _quantize_stage_files(finals.values())
    quantize_hash = fingerprint(quantize_files)
    snapshot_manager.publish_stage("quantize", quantize_files, quantize_hash)
    precision_files = _precision_stage_files(finals.values())
    precision_hash = fingerprint(precision_files)
    snapshot_manager.publish_stage("precision", precision_files, precision_hash)

    output_dir = work / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if exchange_version == "v1":
        exchange_path = output_dir / "calibration_v1.txt"
        atomic_write_text(exchange_path, exchange_text or "")
    else:
        exchange_path = output_dir / "calibration_v2.json"
        atomic_write_json(exchange_path, exchange_document)
    sidecar_path = output_dir / "exchange_sidecar.json"
    atomic_write_json(sidecar_path, sidecar)

    errors.sort(key=lambda item: (item.get("tensor") or "", item.get("stage") or "", item.get("code") or ""))
    pending.sort(key=lambda item: item["tensor"])
    ordered_finals = [finals[name] for name in sorted(finals)]
    successful_names = [item.name for item in ordered_finals]
    result_tensors = [item.result_payload() for item in ordered_finals]
    clipped_total = sum(item.quantize["clipped_count"] for item in ordered_finals)
    error_values = [item.precision["max_abs_error"] for item in ordered_finals]
    result = {
        "format": "quantcal.result.v1",
        "exchange_version": exchange_version,
        "config_fingerprint": config.fingerprint,
        "exchange_version_fingerprint": exchange_fp,
        "tensor_order": successful_names,
        "tensors": result_tensors,
        "errors": errors,
        "pending": pending,
        "summary": {
            "manifest_tensor_count": len(names),
            "completed_tensor_count": len(successful_names),
            "failed_tensor_count": len({item.get("tensor") for item in errors if item.get("tensor")}),
            "pending_tensor_count": len(pending),
            "clipped_element_count": clipped_total,
            "max_abs_error": max(error_values) if error_values else None,
            "mean_of_tensor_max_abs_error": (
                sum(error_values) / len(error_values) if error_values else None
            ),
        },
    }
    result_path = output_dir / "result.json"
    atomic_write_json(result_path, result)

    stage_status: dict[str, dict[str, list[str]]] = {
        stage: {"reused": [], "computed": []}
        for stage in ("quantile", "scale", "quantize", "precision")
    }
    ledger_tensors: dict[str, Any] = {}
    for final in ordered_finals:
        stage_records: dict[str, Any] = {}
        for stage, record in final.statuses.items():
            stage_records[stage] = record
            stage_status[stage][record["status"]].append(final.name)
        ledger_tensors[final.name] = stage_records
    for stage in stage_status:
        stage_status[stage]["reused"].sort()
        stage_status[stage]["computed"].sort()
    reused_tensors = sorted(
        name
        for name, records in ledger_tensors.items()
        if all(record["status"] == "reused" for record in records.values())
    )
    recomputed_tensors = sorted(
        name
        for name, records in ledger_tensors.items()
        if any(record["status"] == "computed" for record in records.values())
    )
    ledger = {
        "format": "quantcal.ledger.v1",
        "tensor_order": successful_names,
        "tensors": ledger_tensors,
        "stages": stage_status,
        "reused_tensors": reused_tensors,
        "recomputed_tensors": recomputed_tensors,
        "budget": {
            "limit": config.batch_point_budget,
            "deduction_sequence": deduction_sequence,
            "pending": pending,
        },
        "failed_tensors": sorted({item.get("tensor") for item in errors if item.get("tensor")}),
    }
    ledger_path = output_dir / "ledger.json"
    atomic_write_json(ledger_path, ledger)

    diagnostics = {
        "format": "quantcal.diagnostics.v1",
        "global_version": snapshot_manager.version,
        "config_fingerprint": config.fingerprint,
        "exchange_version": exchange_version,
        "exchange_version_fingerprint": exchange_fp,
        "quantile_method": QUANTILE_METHOD,
        "tie_rule": TIE_RULE,
        "input_exchange_validation": interop,
        "tensor_order": successful_names,
        "schedule": {
            "max_concurrency": config.max_concurrency,
            "batch_point_budget": config.batch_point_budget,
            "snapshot_commit_interval": config.snapshot_commit_interval,
            "budget_deduction_sequence": deduction_sequence,
            "pending": pending,
            "tensor_order": successful_names,
            "task_sort_keys": sorted(
                record["task_sort_key"]
                for records in ledger_tensors.values()
                for record in records.values()
            ),
        },
        "cache_ledger": ledger,
        "injection_hits": sorted(injection_hits or [], key=lambda item: item.get("injection_id", "")),
        "errors": errors,
        "tensor_metrics": [
            {
                "tensor": item.name,
                "clipped_element_ratio": item.precision["clipped_element_ratio"],
                "max_abs_error": item.precision["max_abs_error"],
                "mean_abs_error": item.precision["mean_abs_error"],
                "mse": item.precision["mse"],
                "error_attribution": item.precision["error_attribution"],
            }
            for item in ordered_finals
        ],
        "summary": result["summary"],
        "outputs": {
            "result": "result.json",
            "ledger": "ledger.json",
            "exchange": exchange_path.name,
            "sidecar": "exchange_sidecar.json",
        },
    }
    diagnostics_path = output_dir / "diagnostics.json"
    atomic_write_json(diagnostics_path, diagnostics)
    if report_path is not None:
        atomic_write_json(report_path, diagnostics)

    bypass_hash = sha256_file(sidecar_path)
    snapshot = snapshot_manager.finalize(
        {
            "completed_tensors": successful_names,
            "failed_tensors": sorted({item.get("tensor") for item in errors if item.get("tensor")}),
            "pending_tensors": [item["tensor"] for item in pending],
            "exchange_version_fingerprint": exchange_fp,
            "bypass_metadata_hash": bypass_hash,
            "result_fingerprint": fingerprint(result),
            "sidecar_fingerprint": fingerprint(sidecar),
            "summary": result["summary"],
        }
    )
    diagnostics["snapshot"] = snapshot_public_view(snapshot)
    atomic_write_json(diagnostics_path, diagnostics)
    if report_path is not None:
        atomic_write_json(report_path, diagnostics)

    if errors:
        exit_code = 17
        for item in errors:
            code = item.get("code")
            if code in {
                "E_FORMAT_INVALID",
                "E_ZERO_POINT_INVALID",
                "E_QUANT_RANGE_INVALID",
                "E_NON_FINITE_SAMPLE",
            }:
                exit_code = 2
                break
    elif pending:
        exit_code = 15
    else:
        exit_code = 0
    return PipelineResult(
        result=result,
        diagnostics=diagnostics,
        output_dir=output_dir,
        exit_code=exit_code,
        snapshot=snapshot,
        exchange_text=exchange_text,
        exchange_document=exchange_document,
        errors=errors,
        pending=pending,
    )
