"""End-to-end deterministic quantization, distillation, and incremental engine."""

from __future__ import annotations

import copy
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .calibration import (
    ParsedCalibration,
    compare_v2_records,
    downgrade_v2_to_v1,
    parse_calibration_file,
    reverse_downgrade,
    upgrade_v1_to_v2,
    write_downgrade_artifacts,
)
from .codes import (
    E_CALIB_PARSE,
    E_QUANT_BUDGET_PENDING,
    E_SIDECAR_MISSING,
    E_WEIGHT_NONFINITE,
    E_WEIGHT_SCHEMA,
)
from .distillation import (
    distill_cache_key,
    distill_layer,
    distill_param_fingerprint,
    distillation_failure,
)
from .inputs import (
    layer_config,
    load_config,
    load_samples,
    load_weights,
)
from .io_utils import (
    atomic_write_json,
    combined_fingerprint,
    load_json,
    result_digest,
    sha256_file,
    sha256_json,
)
from .models import FailureRecord, WeightPack
from .quantile import (
    calibrate_layer,
    layer_quant_key,
    layer_result_from_dict,
    quantile_failure,
    quantile_parameter_fingerprint,
)
from .quantize import (
    quantization_cache_key,
    quantization_failure,
    quantization_result_from_dict,
    quantize_layer,
    rescale_config_fingerprint,
)
from .snapshot import Snapshot, SnapshotStore


@dataclass(frozen=True)
class RunPaths:
    directory: Path
    weights: Path
    calibration_v1: Path
    calibration_v2: Path
    samples: Path
    exchange_spec: Path
    config: Path

    @classmethod
    def discover(cls, directory: str | Path, config: str | Path | None = None) -> "RunPaths":
        root = Path(directory)
        return cls(
            directory=root,
            weights=root / "weights.jsonl",
            calibration_v1=root / "calibration_v1.txt",
            calibration_v2=root / "calibration_v2.json",
            samples=root / "calibration_samples.json",
            exchange_spec=root / "exchange_spec.json",
            config=Path(config) if config is not None else root / "config.yaml",
        )


@dataclass
class PreparedInputs:
    paths: RunPaths
    config: dict[str, Any]
    config_hash: str
    weights: dict[str, WeightPack]
    weight_failures: list[FailureRecord]
    weight_layers: set[str]
    samples: list[dict[str, Any]]
    sample_fingerprint: str
    sample_failures: list[FailureRecord]
    parsed_v1: ParsedCalibration
    parsed_v2: ParsedCalibration
    exchange_fingerprint: str
    exchange_spec: dict[str, Any]
    injections: list[dict[str, Any]]
    injection_fingerprint: str


def prepare_inputs(
    paths: RunPaths,
    *,
    config_override: dict[str, Any] | None = None,
    weights_override: dict[str, WeightPack] | None = None,
    injections: list[dict[str, Any]] | None = None,
) -> PreparedInputs:
    injections = list(injections or [])
    config, config_hash = load_config(paths.config, config_override)
    if weights_override is None:
        weights, weight_failures, weight_layers = load_weights(paths.weights, injections)
    else:
        weights = copy.deepcopy(weights_override)
        weight_failures = []
        weight_layers = set(weights)
    samples, sample_fingerprint, sample_failures = load_samples(paths.samples)
    if paths.exchange_spec.exists():
        try:
            exchange_spec = load_json(paths.exchange_spec)
        except (OSError, ValueError):
            exchange_spec = {"schema_version": "qcal-exchange-spec/missing", "error": "unreadable"}
        spec_hash = sha256_file(paths.exchange_spec)
    else:
        exchange_spec = {"schema_version": "qcal-exchange-spec/missing"}
        spec_hash = "<missing>"
    exchange_version = str(config["exchange_version"])
    exchange_fingerprint = combined_fingerprint(
        "exchange_version",
        exchange_version,
        spec_hash,
        exchange_spec.get("schema_version", "unspecified"),
    )
    parsed_v1 = parse_calibration_file(paths.calibration_v1, "v1", weight_layers, injections)
    parsed_v2 = parse_calibration_file(paths.calibration_v2, "v2", weight_layers, injections)
    return PreparedInputs(
        paths=paths,
        config=config,
        config_hash=config_hash,
        weights=weights,
        weight_failures=weight_failures,
        weight_layers=weight_layers,
        samples=samples,
        sample_fingerprint=sample_fingerprint,
        sample_failures=sample_failures,
        parsed_v1=parsed_v1,
        parsed_v2=parsed_v2,
        exchange_fingerprint=exchange_fingerprint,
        exchange_spec=exchange_spec,
        injections=injections,
        injection_fingerprint=sha256_json(injections),
    )


class ForcedInterrupt(RuntimeError):
    def __init__(self, stage: str, exit_code: int = 75) -> None:
        super().__init__(f"forced interrupt after stage {stage}")
        self.stage = stage
        self.exit_code = exit_code


def _entry_failure(entry: dict[str, Any]) -> FailureRecord | None:
    payload = entry.get("failure")
    if not isinstance(payload, dict):
        return None
    return FailureRecord(
        layer=str(payload["layer"]),
        stage=str(payload["stage"]),
        code=str(payload["code"]),
        message=str(payload.get("message", "")),
        injection_id=payload.get("injection_id"),
        detail=dict(payload.get("detail", {})),
    )


def _deduplicate_failures(failures: list[FailureRecord]) -> list[FailureRecord]:
    unique: dict[tuple[str, str, str, str, str], FailureRecord] = {}
    for failure in failures:
        key = (
            failure.layer,
            failure.stage,
            failure.code,
            failure.message,
            failure.injection_id or "",
        )
        unique[key] = failure
    return [unique[key] for key in sorted(unique)]


class QcalEngine:
    def __init__(
        self,
        paths: RunPaths,
        *,
        output_dir: str | Path,
        state_dir: str | Path,
        prepared: PreparedInputs | None = None,
        injection_path: str | Path | None = None,
        fresh: bool = False,
        force_interrupt_after_stage: str | None = None,
        budget_override: int | None = None,
        workers_override: int | None = None,
    ) -> None:
        injections: list[dict[str, Any]] = []
        if injection_path is not None and Path(injection_path).exists():
            raw_injections = load_json(injection_path)
            if isinstance(raw_injections, dict):
                injections = list(raw_injections.get("injections", []))
            elif isinstance(raw_injections, list):
                injections = raw_injections
        self.injections = injections
        self.paths = paths
        self.prepared = prepared or prepare_inputs(paths, injections=injections)
        self.output_dir = Path(output_dir)
        self.state_dir = Path(state_dir)
        if fresh and self.state_dir.exists():
            shutil.rmtree(self.state_dir)
        self.store = SnapshotStore(self.state_dir)
        self.snapshot = self.store.load_current()
        self.previous_manifest = copy.deepcopy(self.snapshot.manifest)
        self.force_interrupt_after_stage = force_interrupt_after_stage
        self.budget_override = budget_override
        self.workers_override = workers_override
        self.stage_timings: dict[str, float] = {}
        self._stage_index = 0
        self.recomputed: dict[str, list[str]] = {
            "parse_validation": [],
            "conversion": [],
            "quantile_calibration": [],
            "fixed_point_rescale": [],
            "distillation_correction": [],
        }
        self.reused: dict[str, list[str]] = {key: [] for key in self.recomputed}
        self.pending_by_stage: dict[str, list[str]] = {}
        self.failures: list[FailureRecord] = []
        self.sidecar_hash = "<none>"
        self.sidecar_missing_injected = any(
            item.get("type") == "sidecar_missing" for item in self.injections
        )

    @property
    def config(self) -> dict[str, Any]:
        return self.prepared.config

    @property
    def weights(self) -> dict[str, WeightPack]:
        return self.prepared.weights

    @property
    def exchange_fingerprint(self) -> str:
        return self.prepared.exchange_fingerprint

    @property
    def budget(self) -> int:
        if self.budget_override is not None:
            return max(0, int(self.budget_override))
        return max(0, int(self.config["scheduler"].get("quant_points_budget", 0)))

    @property
    def workers(self) -> int:
        if self.workers_override is not None:
            return max(1, int(self.workers_override))
        return max(1, int(self.config["scheduler"].get("workers", 1)))

    def _commit(self, stage: str, changed_stages: list[str]) -> None:
        self._stage_index += 1
        interval = max(1, int(self.config.get("snapshot", {}).get("commit_every_stages", 1)))
        force_for_interrupt = self.force_interrupt_after_stage == stage
        is_final_stage = stage == "distillation_correction"
        if interval > 1 and self._stage_index % interval != 0 and not force_for_interrupt and not is_final_stage:
            return
        self.snapshot = self.store.commit_stage(
            self.snapshot,
            stage,
            config_hash=self.prepared.config_hash,
            exchange_fingerprint=self.exchange_fingerprint,
            sidecar_hash=self.sidecar_hash,
            changed_stages=changed_stages,
        )

    def _maybe_interrupt(self, stage: str) -> None:
        if self.force_interrupt_after_stage == stage:
            raise ForcedInterrupt(stage)

    def _stage_begin(self) -> float:
        return time.perf_counter()

    def _stage_end(self, stage: str, started: float) -> None:
        self.stage_timings[stage] = float(time.perf_counter() - started)

    def _parse_stage(self) -> tuple[list[str], set[str]]:
        started = self._stage_begin()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        conversion_dir = self.output_dir / "conversion"
        conversion_dir.mkdir(parents=True, exist_ok=True)
        self.failures.extend(self.prepared.weight_failures)
        self.failures.extend(self.prepared.sample_failures)
        self.failures.extend(self.prepared.parsed_v1.failures)
        self.failures.extend(self.prepared.parsed_v2.failures)

        # Build both conversion directions regardless of which result is
        # selected. The sidecar is written with an atomic file replace.
        v1_to_v2 = upgrade_v1_to_v2(self.prepared.parsed_v1.records)
        v2_to_v1, sidecar, thousandths = downgrade_v2_to_v1(self.prepared.parsed_v2.records)
        text_path, sidecar_path = write_downgrade_artifacts(
            conversion_dir,
            v2_to_v1,
            sidecar,
            thousandths,
        )
        atomic_write_json(
            conversion_dir / "calibration_v1_to_v2.json",
            {
                "exchange_version": "v2",
                "layers": [v1_to_v2[layer].to_dict() for layer in sorted(v1_to_v2)],
            },
        )
        self.sidecar_hash = sha256_file(sidecar_path)
        effective_sidecar_path = sidecar_path
        if self.sidecar_missing_injected:
            effective_sidecar_path = conversion_dir / "injected-missing.sidecar.json"
        restored_v2, reverse_failures, reverse_status = reverse_downgrade(
            v2_to_v1, effective_sidecar_path
        )
        if self.sidecar_missing_injected:
            injection_id = next(
                (
                    str(item.get("id", "INJ-UNKNOWN"))
                    for item in self.injections
                    if item.get("type") == "sidecar_missing"
                ),
                "INJ-UNKNOWN",
            )
            for failure in reverse_failures:
                failure.injection_id = injection_id
        self.failures.extend(reverse_failures)
        round_trip = compare_v2_records(self.prepared.parsed_v2.records, restored_v2)
        atomic_write_json(
            conversion_dir / "reverse_status.json",
            {
                "status": reverse_status,
                "round_trip": round_trip,
                "residuals": {
                    layer: sidecar["layers"][layer]["truncation_residual"]
                    for layer in sorted(sidecar["layers"])
                },
            },
        )

        parse_entries = self.snapshot.entries.setdefault("parse_validation", {})
        conversion_entries = self.snapshot.entries.setdefault("conversion", {})
        layers = sorted(
            set(self.prepared.weight_layers)
            | set(self.prepared.weights)
            | set(self.prepared.parsed_v1.raw_by_layer)
            | set(self.prepared.parsed_v2.raw_by_layer)
            | set(self.prepared.parsed_v1.records)
            | set(self.prepared.parsed_v2.records)
        )
        selected_version = str(self.config["exchange_version"])
        selected_parsed = (
            self.prepared.parsed_v1 if selected_version == "v1" else self.prepared.parsed_v2
        )
        valid_parse_layers: set[str] = set()
        for layer in layers:
            pack = self.weights.get(layer)
            selected = selected_parsed.records.get(layer)
            selected_raw = selected_parsed.raw_by_layer.get(layer)
            key = combined_fingerprint(
                "parse_validation",
                layer,
                pack.content_fingerprint if pack else "<missing-weight>",
                selected.to_dict() if selected else "<missing-calibration>",
                sha256_json(selected_raw) if selected_raw is not None else "<missing-raw>",
                self.exchange_fingerprint,
                self.prepared.injection_fingerprint,
            )
            failure: FailureRecord | None = None
            if pack is None:
                code = E_WEIGHT_NONFINITE if layer in self.prepared.weight_layers else E_WEIGHT_SCHEMA
                failure = FailureRecord(
                    layer=layer,
                    stage="parse_validation",
                    code=code,
                    message="weight layer is unavailable after validation",
                )
            elif selected is None:
                failure = FailureRecord(
                    layer=layer,
                    stage="parse_validation",
                    code=E_CALIB_PARSE,
                    message=f"selected {selected_version} calibration record is unavailable",
                )
            old = parse_entries.get(layer)
            if old and old.get("key") == key:
                self.reused["parse_validation"].append(layer)
                entry = old
            else:
                self.recomputed["parse_validation"].append(layer)
                if failure is not None:
                    entry = {
                        "key": key,
                        "status": "skipped",
                        "sort_key": f"{layer}:parse_validation",
                        "skip_reason": failure.to_dict(),
                    }
                else:
                    entry = {
                        "key": key,
                        "status": "ok",
                        "sort_key": f"{layer}:parse_validation",
                        "value": {
                            "layer": layer,
                            "selected_version": selected_version,
                            "selected_calibration": selected.to_dict() if selected else None,
                            "v1_calibration": (
                                self.prepared.parsed_v1.records[layer].to_dict()
                                if layer in self.prepared.parsed_v1.records
                                else None
                            ),
                            "v2_calibration": (
                                self.prepared.parsed_v2.records[layer].to_dict()
                                if layer in self.prepared.parsed_v2.records
                                else None
                            ),
                            "content_fingerprint": pack.content_fingerprint if pack else None,
                        },
                    }
            parse_entries[layer] = entry
            if entry.get("status") == "ok":
                valid_parse_layers.add(layer)

            v1_record = self.prepared.parsed_v1.records.get(layer)
            v2_record = self.prepared.parsed_v2.records.get(layer)
            conversion_key = combined_fingerprint(
                "conversion",
                layer,
                pack.content_fingerprint if pack else "<missing-weight>",
                v1_record.to_dict() if v1_record else None,
                v2_record.to_dict() if v2_record else None,
                self.sidecar_hash,
                self.exchange_fingerprint,
                self.prepared.injection_fingerprint,
            )
            old_conversion = conversion_entries.get(layer)
            if old_conversion and old_conversion.get("key") == conversion_key:
                self.reused["conversion"].append(layer)
            else:
                self.recomputed["conversion"].append(layer)
                sidecar_layer = sidecar.get("layers", {}).get(layer)
                conversion_value = {
                    "layer": layer,
                    "v1_to_v2": v1_to_v2[layer].to_dict() if layer in v1_to_v2 else None,
                    "v2_to_v1": v2_to_v1[layer].to_dict() if layer in v2_to_v1 else None,
                    "sidecar": sidecar_layer,
                    "reverse_v2": restored_v2[layer].to_dict() if layer in restored_v2 else None,
                    "reverse_exact": (
                        restored_v2[layer].to_dict() == v2_record.to_dict()
                        if layer in restored_v2 and v2_record is not None
                        else None
                    ),
                }
                conversion_entries[layer] = {
                    "key": conversion_key,
                    "status": "ok",
                    "sort_key": f"{layer}:conversion",
                    "value": conversion_value,
                }

        self._commit("parse_validation", ["parse_validation", "conversion"])
        self._stage_end("parse_validation", started)
        self._maybe_interrupt("parse_validation")
        return self.recomputed["parse_validation"], valid_parse_layers

    def _parallel_compute(
        self,
        jobs: list[tuple[str, Callable[[], Any]]],
        stage: str,
    ) -> list[tuple[str, bool, Any, Exception | None, str]]:
        if not jobs:
            return []
        workers = min(self.workers, len(jobs))
        results: list[tuple[str, bool, Any, Exception | None, str]] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"qcal-{stage}") as executor:
            future_map = {}
            for layer, function in jobs:
                sort_key = f"{layer}:{stage}"
                future = executor.submit(function)
                future_map[future] = (layer, sort_key)
            for future in as_completed(future_map):
                layer, sort_key = future_map[future]
                try:
                    value = future.result()
                    results.append((layer, True, value, None, sort_key))
                except Exception as exc:  # isolated per-layer failure
                    results.append((layer, False, None, exc, sort_key))
        results.sort(key=lambda item: (item[0], item[4]))
        return results

    def _select_jobs(
        self,
        stage: str,
        layers: list[str],
        desired_keys: dict[str, str],
        point_counts: dict[str, int],
    ) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]], list[str]]:
        entries = self.snapshot.entries.setdefault(stage, {})
        jobs: list[tuple[str, str]] = []
        reuse_entries: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        remaining = self.budget
        budget_exhausted = False
        for layer in layers:
            key = desired_keys[layer]
            old = entries.get(layer)
            if old and old.get("key") == key:
                reuse_entries[layer] = old
                self.reused[stage].append(layer)
                continue
            points = int(point_counts.get(layer, 0))
            if not budget_exhausted and self.budget > 0:
                if points > remaining:
                    budget_exhausted = True
                    pending.append(layer)
                    continue
                remaining -= points
            if budget_exhausted:
                pending.append(layer)
                continue
            jobs.append((layer, key))
        self.pending_by_stage[stage] = pending
        return jobs, reuse_entries, pending

    def _store_stage_result(
        self,
        stage: str,
        layer: str,
        key: str,
        ok: bool,
        value: Any,
        error: Exception | None,
        sort_key: str,
        failure_factory: Callable[[str, Exception], FailureRecord],
    ) -> None:
        entries = self.snapshot.entries.setdefault(stage, {})
        if ok:
            entries[layer] = {
                "key": key,
                "status": "ok",
                "sort_key": sort_key,
                "value": value,
            }
        else:
            assert error is not None
            failure = failure_factory(layer, error)
            entries[layer] = {
                "key": key,
                "status": "failed",
                "sort_key": sort_key,
                "failure": failure.to_dict(),
            }
        self.recomputed[stage].append(layer)

    def _quantile_stage(self, valid_parse_layers: set[str]) -> None:
        started = self._stage_begin()
        layers = [layer for layer in sorted(self.weights) if layer in valid_parse_layers]
        desired_keys: dict[str, str] = {}
        point_counts: dict[str, int] = {}
        for layer in layers:
            pack = self.weights[layer]
            layer_cfg = layer_config(self.config, layer)
            parameter_fp = quantile_parameter_fingerprint(layer_cfg, self.config["calibration"])
            desired_keys[layer] = layer_quant_key(pack, parameter_fp, self.exchange_fingerprint)
            point_counts[layer] = pack.numel
        jobs, reuse_entries, pending = self._select_jobs(
            "quantile_calibration", layers, desired_keys, point_counts
        )
        compute_jobs: list[tuple[str, Callable[[], Any]]] = []
        for layer, key in jobs:
            pack = self.weights[layer]
            layer_cfg = copy.deepcopy(layer_config(self.config, layer))
            calibration_cfg = copy.deepcopy(self.config["calibration"])
            exchange_version = str(self.config["exchange_version"])
            compute_jobs.append(
                (
                    layer,
                    lambda pack=pack, layer_cfg=layer_cfg, calibration_cfg=calibration_cfg, exchange_version=exchange_version: calibrate_layer(
                        pack, layer_cfg, calibration_cfg, exchange_version
                    ).to_dict(),
                )
            )
        job_keys = dict(jobs)
        results = self._parallel_compute(compute_jobs, "quantile_calibration")
        for layer, ok, value, error, sort_key in results:
            self._store_stage_result(
                "quantile_calibration",
                layer,
                job_keys[layer],
                ok,
                value,
                error,
                sort_key,
                quantile_failure,
            )
        self._commit("quantile_calibration", ["quantile_calibration"])
        self._stage_end("quantile_calibration", started)
        self._maybe_interrupt("quantile_calibration")

    def _rescale_stage(self) -> None:
        started = self._stage_begin()
        quant_entries = self.snapshot.entries.get("quantile_calibration", {})
        layers = sorted(
            layer for layer, entry in quant_entries.items() if entry.get("status") == "ok"
        )
        desired_keys: dict[str, str] = {}
        point_counts: dict[str, int] = {}
        full_precision: dict[str, bool] = {}
        for layer in layers:
            quant_entry = quant_entries[layer]
            layer_cfg = layer_config(self.config, layer)
            is_full = bool(layer_cfg.get("full_precision", False)) or layer in set(
                self.config.get("full_precision_layers", [])
            )
            full_precision[layer] = is_full
            fp = rescale_config_fingerprint(layer_cfg, self.config["fixed_point"], is_full)
            desired_keys[layer] = quantization_cache_key(
                str(quant_entry["key"]), fp, self.exchange_fingerprint
            )
            point_counts[layer] = int(self.weights[layer].numel)
        jobs, reuse_entries, pending = self._select_jobs(
            "fixed_point_rescale", layers, desired_keys, point_counts
        )
        compute_jobs: list[tuple[str, Callable[[], Any]]] = []
        for layer, key in jobs:
            quant_payload = quant_entries[layer]["value"]
            pack = self.weights[layer]
            fixed_cfg = copy.deepcopy(self.config["fixed_point"])
            is_full = full_precision[layer]
            compute_jobs.append(
                (
                    layer,
                    lambda quant_payload=quant_payload, pack=pack, fixed_cfg=fixed_cfg, is_full=is_full: quantize_layer(
                        pack,
                        layer_result_from_dict(quant_payload),
                        fixed_cfg,
                        full_precision=is_full,
                    ).to_dict(),
                )
            )
        job_keys = dict(jobs)
        results = self._parallel_compute(compute_jobs, "fixed_point_rescale")
        for layer, ok, value, error, sort_key in results:
            self._store_stage_result(
                "fixed_point_rescale",
                layer,
                job_keys[layer],
                ok,
                value,
                error,
                sort_key,
                quantization_failure,
            )
        self._commit("fixed_point_rescale", ["fixed_point_rescale"])
        self._stage_end("fixed_point_rescale", started)
        self._maybe_interrupt("fixed_point_rescale")

    def _distill_stage(self) -> None:
        started = self._stage_begin()
        rescale_entries = self.snapshot.entries.get("fixed_point_rescale", {})
        layers = sorted(
            layer for layer, entry in rescale_entries.items() if entry.get("status") == "ok"
        )
        distill_fp = distill_param_fingerprint(self.config["distillation"])
        desired_keys: dict[str, str] = {}
        for layer in layers:
            desired_keys[layer] = distill_cache_key(
                str(rescale_entries[layer]["key"]),
                self.prepared.sample_fingerprint,
                distill_fp,
                self.exchange_fingerprint,
            )
        jobs, reuse_entries, pending = self._select_jobs(
            "distillation_correction",
            layers,
            desired_keys,
            {layer: 0 for layer in layers},
        )
        compute_jobs: list[tuple[str, Callable[[], Any]]] = []
        for layer, key in jobs:
            quant_payload = rescale_entries[layer]["value"]
            pack = self.weights[layer]
            samples = copy.deepcopy(self.prepared.samples)
            distill_cfg = copy.deepcopy(self.config["distillation"])
            compute_jobs.append(
                (
                    layer,
                    lambda pack=pack, quant_payload=quant_payload, samples=samples, distill_cfg=distill_cfg: distill_layer(
                        pack,
                        quantization_result_from_dict(quant_payload),
                        samples,
                        distill_cfg,
                    ),
                )
            )
        job_keys = dict(jobs)
        results = self._parallel_compute(compute_jobs, "distillation_correction")
        for layer, ok, value, error, sort_key in results:
            self._store_stage_result(
                "distillation_correction",
                layer,
                job_keys[layer],
                ok,
                value,
                error,
                sort_key,
                distillation_failure,
            )
            entry = self.snapshot.entries["distillation_correction"].get(layer, {})
            if ok and entry.get("status") == "ok":
                entry["calibration_samples"] = [str(sample["id"]) for sample in self.prepared.samples]
                entry["sample_fingerprint"] = self.prepared.sample_fingerprint
        for layer, entry in self.snapshot.entries.get("distillation_correction", {}).items():
            if entry.get("status") == "ok" and "calibration_samples" not in entry:
                entry["calibration_samples"] = [str(sample["id"]) for sample in self.prepared.samples]
                entry["sample_fingerprint"] = self.prepared.sample_fingerprint
        self._commit("distillation_correction", ["distillation_correction"])
        self._stage_end("distillation_correction", started)
        self._maybe_interrupt("distillation_correction")

    def _collect_failures(self, relevant_layers: set[str]) -> list[FailureRecord]:
        failures = list(self.failures)
        for stage in (
            "parse_validation",
            "quantile_calibration",
            "fixed_point_rescale",
            "distillation_correction",
        ):
            for layer, entry in self.snapshot.entries.get(stage, {}).items():
                if layer not in relevant_layers and stage != "parse_validation":
                    continue
                failure = _entry_failure(entry)
                if failure is not None:
                    failures.append(failure)
        return _deduplicate_failures(failures)

    def _report_digest_payload(self, relevant_layers: set[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"exchange_version": self.config["exchange_version"]}
        for stage in ("parse_validation", "conversion", "quantile_calibration", "fixed_point_rescale", "distillation_correction"):
            stage_payload: dict[str, Any] = {}
            for layer in sorted(relevant_layers):
                entry = self.snapshot.entries.get(stage, {}).get(layer)
                if not entry:
                    continue
                if entry.get("status") == "ok":
                    stage_payload[layer] = entry.get("value")
                else:
                    stage_payload[layer] = {"failure": entry.get("failure")}
            payload[stage] = stage_payload
        return payload

    def run(self) -> dict[str, Any]:
        _, valid_parse_layers = self._parse_stage()
        self._quantile_stage(valid_parse_layers)
        self._rescale_stage()
        self._distill_stage()

        relevant_layers = (
            set(self.prepared.weight_layers)
            | set(self.prepared.weights)
            | set(self.prepared.parsed_v1.raw_by_layer)
            | set(self.prepared.parsed_v2.raw_by_layer)
        )
        failures = self._collect_failures(relevant_layers)
        pending: list[dict[str, str]] = []
        for layer in sorted(
            set(self.pending_by_stage.get("quantile_calibration", []))
            | set(self.pending_by_stage.get("fixed_point_rescale", []))
            | set(self.pending_by_stage.get("distillation_correction", []))
        ):
            pending.append({"layer": layer, "code": E_QUANT_BUDGET_PENDING})

        quant_entries = self.snapshot.entries.get("fixed_point_rescale", {})
        quantization_errors: dict[str, Any] = {}
        distillation_records: list[dict[str, Any]] = []
        for layer in sorted(quant_entries):
            entry = quant_entries[layer]
            if entry.get("status") != "ok":
                continue
            value = entry["value"]
            quantization_errors[layer] = {
                "bit_width": int(value["bit_width"]),
                "fixed_point_fraction_bits": int(value["fixed_point_fraction_bits"]),
                "rounding_mode": str(value["rounding_mode"]),
                "full_precision": bool(value["full_precision"]),
                "max_abs_error": float(value["max_abs_error"]),
                "mse": float(value["mse"]),
                "clip_rate": float(value["clip_rate"]),
                "clipped_count": int(value["clipped_count"]),
            }
        for layer, entry in self.snapshot.entries.get("distillation_correction", {}).items():
            if entry.get("status") == "ok":
                distillation_records.extend(entry.get("value", []))
        distillation_records.sort(key=lambda item: (item["sample_id"], item["layer"]))

        quantization_parameters = {
            layer: entry["value"]
            for layer, entry in sorted(
                self.snapshot.entries.get("quantile_calibration", {}).items()
            )
            if entry.get("status") == "ok"
        }
        digest_payload = self._report_digest_payload(relevant_layers)
        digest = result_digest(digest_payload)
        sidecar_path = self.output_dir / "conversion" / "calibration_v2_to_v1.sidecar.json"
        sidecar = load_json(sidecar_path) if sidecar_path.exists() else {"layers": {}}
        conversion_residuals = {
            layer: float(item["truncation_residual"])
            for layer, item in sorted(sidecar.get("layers", {}).items())
        }
        report: dict[str, Any] = {
            "schema_version": "qcal-run-report/1.0",
            "result_digest": digest,
            "exchange_version": self.config["exchange_version"],
            "config_hash": self.prepared.config_hash,
            "exchange_version_fingerprint": self.exchange_fingerprint,
            "sidecar_hash": self.sidecar_hash,
            "failures": [failure.to_dict() for failure in failures],
            "failure_count": len(failures),
            "pending": pending,
            "pending_count": len(pending),
            "quantization_errors": quantization_errors,
            "quantization_parameters": quantization_parameters,
            "distillation_record_count": len(distillation_records),
            "distillation_records": distillation_records,
            "conversion": {
                "residuals": conversion_residuals,
                "round_trip_restored": not any(
                    failure.code in (E_SIDECAR_MISSING,)
                    for failure in failures
                ),
                "sidecar_hash": self.sidecar_hash,
            },
            "reuse": {
                "recomputed": {stage: sorted(set(values)) for stage, values in self.recomputed.items()},
                "reused": {stage: sorted(set(values)) for stage, values in self.reused.items()},
            },
            "cache_ledger": {
                stage: {
                    layer: {
                        "key": entry.get("key"),
                        "status": entry.get("status"),
                        "sort_key": entry.get("sort_key"),
                        "calibration_samples": entry.get("calibration_samples", []),
                        "sample_fingerprint": entry.get("sample_fingerprint"),
                    }
                    for layer, entry in sorted(
                        self.snapshot.entries.get(stage, {}).items()
                    )
                }
                for stage in (
                    "parse_validation",
                    "conversion",
                    "quantile_calibration",
                    "fixed_point_rescale",
                    "distillation_correction",
                )
            },
            "version_changes": {
                "previous_global_version": self.previous_manifest.get("global_version", 0),
                "current_global_version": self.snapshot.global_version,
                "config_hash_changed": self.previous_manifest.get("config_hash")
                not in (None, self.prepared.config_hash),
                "exchange_fingerprint_changed": self.previous_manifest.get(
                    "exchange_version_fingerprint"
                )
                not in (None, self.exchange_fingerprint),
                "sidecar_hash_changed": self.previous_manifest.get("sidecar_hash")
                not in (None, self.sidecar_hash),
            },
            "pending_by_stage": self.pending_by_stage,
            "stage_timings_seconds": dict(sorted(self.stage_timings.items())),
            "snapshot": {
                "global_version": self.snapshot.global_version,
                "manifest": self.snapshot.manifest,
                "completed_layers": self.snapshot.manifest.get("completed_layers", {}),
                "previous_manifest": self.previous_manifest,
                "commit_every_stages": max(
                    1, int(self.config.get("snapshot", {}).get("commit_every_stages", 1))
                ),
            },
            "sorting": {
                "layer_order": "lexicographic",
                "task_sort_key": "<layer>:<stage>",
                "distillation_order": "sample_id then layer",
            },
        }
        atomic_write_json(self.output_dir / "run_report.json", report)
        atomic_write_json(
            self.output_dir / "canonical_result.json",
            {
                "digest": digest,
                "payload": digest_payload,
            },
        )
        return report


def recovered_stage_flags(snapshot: Snapshot) -> dict[str, bool]:
    stages = (
        "parse_validation",
        "quantile_calibration",
        "fixed_point_rescale",
        "distillation_correction",
    )
    return {
        stage: bool(snapshot.manifest.get("completed_layers", {}).get(stage))
        for stage in stages
    }


def snapshot_summary(state_dir: str | Path) -> dict[str, Any]:
    snapshot = SnapshotStore(state_dir).load_current()
    return {
        "global_version": snapshot.global_version,
        "completed_layers": snapshot.manifest.get("completed_layers", {}),
        "recovered_stages": recovered_stage_flags(snapshot),
        "manifest_digest": snapshot.manifest.get("manifest_digest"),
    }
