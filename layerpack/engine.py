"""End-to-end incremental compression engine with deterministic artifacts."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cache import ArtifactCache
from .calibration import compute_calibration_key, evaluate_layer
from .config import CompressionConfig
from .deployment import (
    build_manifest,
    compute_manifest_key,
    export_package,
)
from .errors import (
    E_POINT_BUDGET_PENDING,
    Failure,
    sort_failures,
)
from .fingerprint import canonical_json, sha256_json, to_builtin
from .loader import load_inputs
from .model import CalibrationResult, Layer, LoadedState, QuantResult
from .quantize import (
    compute_dequant_key,
    compute_quant_key,
    quantize_layer,
)
from .scheduler import PointBudgetScheduler, ScheduledTask
from .snapshot import SnapshotStore


class ShutdownController:
    """Cooperative SIGINT/SIGTERM handling for stage-boundary shutdown."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._previous: dict[int, Any] = {}

    def request(self, signum: int | None = None, frame: Any = None) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self.request)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


@dataclass
class RunSummary:
    status: str
    core: dict[str, Any]
    diagnostics: dict[str, Any]
    business_codes: list[str]
    snapshot_version: int
    restore_point: dict[str, Any] | None
    package_path: str | None
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "business_codes": self.business_codes,
            "exit_code": self.exit_code,
            "snapshot_version": self.snapshot_version,
            "restore_point": self.restore_point,
            "package_path": self.package_path,
            "result": self.core,
            "diagnostics": self.diagnostics,
        }

    def comparable(self) -> dict[str, Any]:
        """The fields required to be byte-for-byte equal by the acceptance."""

        return self.core


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(to_builtin(payload)))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class LayerPackEngine:
    """Load inputs, run four atomic phases and produce deployable artifacts."""

    def __init__(
        self,
        weights_path: str | Path,
        calibration_path: str | Path,
        config_path: str | Path,
        state_dir: str | Path,
        *,
        use_cache: bool = True,
        shutdown: ShutdownController | None = None,
        write_reports: bool = True,
    ) -> None:
        self.weights_path = str(weights_path)
        self.calibration_path = str(calibration_path)
        self.config_path = str(config_path)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cache = ArtifactCache(self.state_dir)
        self.snapshots = SnapshotStore(self.state_dir)
        self.use_cache = use_cache
        self.shutdown = shutdown or ShutdownController()
        self.write_reports = write_reports

    def run(self, *, full_rebuild: bool = False) -> RunSummary:
        allow_cache = self.use_cache and not full_rebuild
        initial_restore = self.snapshots.restore_point()
        loaded, config = load_inputs(
            self.weights_path, self.calibration_path, self.config_path
        )
        failures: list[Failure] = list(loaded.failures)
        valid_layer_names = [
            name
            for name in sorted(loaded.layers)
            if name not in config.invalid_layers
        ]
        layers = {name: loaded.layers[name] for name in valid_layer_names}

        invalidated: dict[str, Any] = {
            "quant_parameters": [],
            "dequantized_weights": [],
            "calibration_deviations": [],
            "deployment_manifest": False,
        }
        reused: dict[str, Any] = {
            "quant_parameters": [],
            "dequantized_weights": [],
            "calibration_deviations": [],
            "deployment_manifest": False,
        }
        keys: dict[str, Any] = {
            "quant_parameters": {},
            "dequantized_weights": {},
            "calibration_deviations": {},
            "deployment_manifest": "",
        }
        budget_events: list[dict[str, Any]] = []
        pending_queue: list[dict[str, Any]] = []
        max_concurrency = 0
        batch_count = 0
        quant_results: dict[str, QuantResult] = {}
        calibration_results: dict[str, CalibrationResult] = {}
        deployment_manifest: dict[str, Any] | None = None
        deployment_key: str | None = None
        package_path: str | None = None
        current_manifest = self.snapshots.latest()

        loading_payload = self._loading_payload(loaded, config, failures)
        current_manifest = self._commit_phase(
            "loading",
            loading_payload,
            config,
            loaded,
            sorted(layers),
            "in_progress",
            [],
            current_manifest,
        )

        if self.shutdown.requested:
            return self._finish(
                status="interrupted",
                config=config,
                loaded=loaded,
                layers=layers,
                quant_results=quant_results,
                calibration_results=calibration_results,
                deployment_manifest=None,
                failures=failures,
                invalidated=invalidated,
                reused=reused,
                keys=keys,
                budget_events=budget_events,
                pending_queue=pending_queue,
                max_concurrency=max_concurrency,
                batch_count=batch_count,
                initial_restore=initial_restore,
                current_manifest=self._commit_status(
                    current_manifest,
                    "loading",
                    "interrupted",
                    loading_payload,
                    config,
                    loaded,
                    sorted(layers),
                    [],
                ),
                package_path=None,
            )

        # Phase 2: quantization and dequantization.
        quant_tasks: list[ScheduledTask[QuantResult]] = []
        last_checkpoint_count = 0

        def quant_payload(status: str) -> dict[str, Any]:
            return {
                "config_hash": config.config_hash,
                "calibration_hash": loaded.calibration_hash,
                "completed_layers": sorted(quant_results),
                "layers": [
                    {
                        "layer": name,
                        "quant_key": quant_results[name].key,
                        "dequant_key": quant_results[name].dequant_key,
                        "parameters_sha256": quant_results[name].parameters.get(
                            "parameters_sha256", ""
                        ),
                        "quantized_sha256": quant_results[name].quantized_sha256,
                        "dequantized_sha256": quant_results[name].dequantized_sha256,
                        "distortion": quant_results[name].distortion,
                    }
                    for name in sorted(quant_results)
                ],
                "pending_queue": pending_queue,
                "status": status,
            }

        def quant_checkpoint(batch_results: list[tuple[str, QuantResult]], batch: int) -> None:
            nonlocal current_manifest, last_checkpoint_count
            for _, result in batch_results:
                quant_results[result.layer_name] = result
            if (
                config.snapshot_commit_interval > 0
                and
                len(quant_results) - last_checkpoint_count
                >= config.snapshot_commit_interval
            ):
                current_manifest = self._commit_phase(
                    "quantization",
                    quant_payload("in_progress"),
                    config,
                    loaded,
                    sorted(quant_results),
                    "in_progress",
                    [],
                    current_manifest,
                )
                last_checkpoint_count = len(quant_results)

        for name in valid_layer_names:
            layer = layers[name]
            settings = config.effective(name)
            quant_key = compute_quant_key(layer, settings)
            expected_dequant_key = compute_dequant_key(layer, settings, quant_key)
            keys["quant_parameters"][name] = quant_key
            keys["dequantized_weights"][name] = expected_dequant_key
            cached = self.cache.load_quant(quant_key) if allow_cache else None
            if (
                cached is not None
                and cached.layer_name == name
                and cached.dequantized.shape == layer.shape
            ):
                if not cached.dequant_key:
                    cached.dequant_key = expected_dequant_key
                cached.reused = True
                quant_results[name] = cached
                reused["quant_parameters"].append(name)
                reused["dequantized_weights"].append(name)
                continue

            invalidated["quant_parameters"].append(name)
            invalidated["dequantized_weights"].append(name)

            def action(
                current_layer: Layer = layer,
                current_settings=settings,
                current_key: str = quant_key,
            ) -> QuantResult:
                result = quantize_layer(
                    current_layer, current_settings, key=current_key
                )
                self.cache.save_quant(result)
                return result

            quant_tasks.append(
                ScheduledTask(
                    layer=name,
                    stage="quantization",
                    points=layer.element_count,
                    action=action,
                )
            )

        if quant_tasks:
            scheduler = PointBudgetScheduler(
                max_workers=config.concurrency,
                point_budget_per_batch=config.point_budget_per_batch,
                stage_delay_ms=config.stage_delay_ms,
                batch_callback=quant_checkpoint,
            )
            scheduled = scheduler.run(quant_tasks)
            budget_events.extend(event.to_dict() for event in scheduled.events)
            pending_queue.extend(
                {
                    **item.to_dict(),
                    "business_code": E_POINT_BUDGET_PENDING,
                }
                for item in scheduled.pending
            )
            max_concurrency = max(max_concurrency, scheduled.max_observed_concurrency)
            batch_count += scheduled.batch_count
            for _, result in scheduled.results:
                quant_results[result.layer_name] = result

        current_manifest = self._commit_phase(
            "quantization",
            quant_payload("partial" if pending_queue else "complete"),
            config,
            loaded,
            sorted(quant_results),
            "in_progress",
            pending_queue,
            current_manifest,
        )
        if self.shutdown.requested:
            return self._finish(
                status="interrupted",
                config=config,
                loaded=loaded,
                layers=layers,
                quant_results=quant_results,
                calibration_results=calibration_results,
                deployment_manifest=None,
                failures=failures,
                invalidated=invalidated,
                reused=reused,
                keys=keys,
                budget_events=budget_events,
                pending_queue=pending_queue,
                max_concurrency=max_concurrency,
                batch_count=batch_count,
                initial_restore=initial_restore,
                current_manifest=self._commit_status(
                    current_manifest,
                    "quantization",
                    "interrupted",
                    quant_payload("interrupted"),
                    config,
                    loaded,
                    sorted(quant_results),
                    pending_queue,
                ),
                package_path=None,
            )
        if pending_queue:
            current_manifest = self._commit_status(
                current_manifest,
                "quantization",
                "pending",
                quant_payload("partial"),
                config,
                loaded,
                sorted(quant_results),
                pending_queue,
            )
            return self._finish(
                status="pending",
                config=config,
                loaded=loaded,
                layers=layers,
                quant_results=quant_results,
                calibration_results=calibration_results,
                deployment_manifest=None,
                failures=failures,
                invalidated=invalidated,
                reused=reused,
                keys=keys,
                budget_events=budget_events,
                pending_queue=pending_queue,
                max_concurrency=max_concurrency,
                batch_count=batch_count,
                initial_restore=initial_restore,
                current_manifest=current_manifest,
                package_path=None,
            )

        # Phase 3: calibration evaluation and soft-label correction.
        calibration_tasks: list[ScheduledTask[CalibrationResult]] = []
        last_calibration_checkpoint = 0

        def calibration_payload(status: str) -> dict[str, Any]:
            return {
                "config_hash": config.config_hash,
                "calibration_hash": loaded.calibration_hash,
                "completed_layers": sorted(calibration_results),
                "layers": [
                    {
                        "layer": name,
                        "calibration_key": calibration_results[name].key,
                        "summary": calibration_results[name].summary,
                    }
                    for name in sorted(calibration_results)
                ],
                "status": status,
            }

        def calibration_checkpoint(
            batch_results: list[tuple[str, CalibrationResult]], batch: int
        ) -> None:
            nonlocal current_manifest, last_calibration_checkpoint
            for _, result in batch_results:
                calibration_results[result.layer_name] = result
            if (
                config.snapshot_commit_interval > 0
                and
                len(calibration_results) - last_calibration_checkpoint
                >= config.snapshot_commit_interval
            ):
                current_manifest = self._commit_phase(
                    "calibration",
                    calibration_payload("in_progress"),
                    config,
                    loaded,
                    sorted(quant_results),
                    "in_progress",
                    [],
                    current_manifest,
                )
                last_calibration_checkpoint = len(calibration_results)

        layer_order = sorted(quant_results)
        layer_sample_fingerprints: dict[str, list[dict[str, str]]] = {}
        for name in layer_order:
            layer = layers[name]
            applicable_samples = sorted(
                [
                    sample
                    for sample in loaded.calibration.samples
                    if sample.applies_to(name)
                    and len(sample.inputs) == layer.shape[-1]
                ],
                key=lambda sample: sample.sample_id,
            )
            layer_sample_fingerprints[name] = [
                {
                    "sample_id": sample.sample_id,
                    "sha256": sha256_json(
                        {
                            "kind": "calibration-sample-v1",
                            "sample": sample.to_input_dict(),
                        }
                    ),
                }
                for sample in applicable_samples
            ]
        for index, name in enumerate(layer_order):
            layer = layers[name]
            quant = quant_results[name]
            samples = [
                sample
                for sample in loaded.calibration.samples
                if sample.applies_to(name)
                and len(sample.inputs) == layer.shape[-1]
            ]
            upstream = [quant_results[item].dequant_key for item in layer_order[: index + 1]]
            upstream_samples: list[dict[str, str]] = []
            for dependency in layer_order[: index + 1]:
                upstream_samples.extend(layer_sample_fingerprints[dependency])
            calibration_key = compute_calibration_key(
                layer=layer,
                quant=quant,
                sample_fingerprints=layer_sample_fingerprints[name],
                upstream_sample_fingerprints=upstream_samples,
                upstream_dequant_keys=upstream,
                config=config,
            )
            keys["calibration_deviations"][name] = calibration_key
            cached = (
                self.cache.load_calibration(calibration_key)
                if allow_cache
                else None
            )
            if cached is not None and cached.layer_name == name:
                cached.reused = True
                calibration_results[name] = cached
                reused["calibration_deviations"].append(name)
                continue

            invalidated["calibration_deviations"].append(name)

            def action(
                current_layer: Layer = layer,
                current_quant: QuantResult = quant,
                current_samples=samples,
                current_key: str = calibration_key,
            ) -> CalibrationResult:
                result = evaluate_layer(
                    layer=current_layer,
                    quant=current_quant,
                    samples=current_samples,
                    config=config,
                    key=current_key,
                )
                self.cache.save_calibration(result)
                return result

            calibration_tasks.append(
                ScheduledTask(
                    layer=name,
                    stage="calibration",
                    points=layer.element_count * len(samples),
                    action=action,
                )
            )

        if calibration_tasks:
            scheduler = PointBudgetScheduler(
                max_workers=config.concurrency,
                point_budget_per_batch=config.point_budget_per_batch,
                stage_delay_ms=config.stage_delay_ms,
                batch_callback=calibration_checkpoint,
            )
            scheduled = scheduler.run(calibration_tasks)
            budget_events.extend(event.to_dict() for event in scheduled.events)
            pending_queue.extend(
                {
                    **item.to_dict(),
                    "business_code": E_POINT_BUDGET_PENDING,
                }
                for item in scheduled.pending
            )
            max_concurrency = max(max_concurrency, scheduled.max_observed_concurrency)
            batch_count += scheduled.batch_count
            for _, result in scheduled.results:
                calibration_results[result.layer_name] = result

        current_manifest = self._commit_phase(
            "calibration",
            calibration_payload("partial" if pending_queue else "complete"),
            config,
            loaded,
            sorted(quant_results),
            "in_progress",
            pending_queue,
            current_manifest,
        )
        if self.shutdown.requested:
            return self._finish(
                status="interrupted",
                config=config,
                loaded=loaded,
                layers=layers,
                quant_results=quant_results,
                calibration_results=calibration_results,
                deployment_manifest=None,
                failures=failures,
                invalidated=invalidated,
                reused=reused,
                keys=keys,
                budget_events=budget_events,
                pending_queue=pending_queue,
                max_concurrency=max_concurrency,
                batch_count=batch_count,
                initial_restore=initial_restore,
                current_manifest=self._commit_status(
                    current_manifest,
                    "calibration",
                    "interrupted",
                    calibration_payload("interrupted"),
                    config,
                    loaded,
                    sorted(quant_results),
                    pending_queue,
                ),
                package_path=None,
            )
        if pending_queue:
            current_manifest = self._commit_status(
                current_manifest,
                "calibration",
                "pending",
                calibration_payload("partial"),
                config,
                loaded,
                sorted(quant_results),
                pending_queue,
            )
            return self._finish(
                status="pending",
                config=config,
                loaded=loaded,
                layers=layers,
                quant_results=quant_results,
                calibration_results=calibration_results,
                deployment_manifest=None,
                failures=failures,
                invalidated=invalidated,
                reused=reused,
                keys=keys,
                budget_events=budget_events,
                pending_queue=pending_queue,
                max_concurrency=max_concurrency,
                batch_count=batch_count,
                initial_restore=initial_restore,
                current_manifest=current_manifest,
                package_path=None,
            )

        # Phase 4: deployment manifest and package.
        deployment_key = compute_manifest_key(
            config=config,
            calibration_hash=loaded.calibration_hash,
            quant_results=quant_results.values(),
            calibration_results=calibration_results.values(),
        )
        keys["deployment_manifest"] = deployment_key
        cached_manifest = (
            self.cache.load_manifest(deployment_key) if allow_cache else None
        )
        if cached_manifest is not None:
            deployment_manifest = cached_manifest
            reused["deployment_manifest"] = True
        else:
            invalidated["deployment_manifest"] = True
            deployment_manifest = build_manifest(
                config=config,
                calibration_hash=loaded.calibration_hash,
                layers=layers,
                quant_results=quant_results,
                calibration_results=calibration_results,
            )
            self.cache.save_manifest(deployment_key, deployment_manifest)

        final_status = "failed" if failures else "complete"
        if self.shutdown.requested:
            final_status = "interrupted"
        if final_status != "interrupted":
            package_path = export_package(
                state_root=self.state_dir,
                manifest=deployment_manifest,
                quant_results=quant_results,
            )

        deployment_payload = {
            "config_hash": config.config_hash,
            "calibration_hash": loaded.calibration_hash,
            "manifest_key": deployment_key,
            "manifest": deployment_manifest,
            "package_path": package_path,
            "status": final_status,
        }
        current_manifest = self._commit_phase(
            "deployment",
            deployment_payload,
            config,
            loaded,
            sorted(quant_results),
            final_status,
            [],
            current_manifest,
        )
        if self.shutdown.requested and final_status != "interrupted":
            final_status = "interrupted"
            current_manifest = self._commit_status(
                current_manifest,
                "deployment",
                "interrupted",
                deployment_payload,
                config,
                loaded,
                sorted(quant_results),
                [],
            )

        return self._finish(
            status=final_status,
            config=config,
            loaded=loaded,
            layers=layers,
            quant_results=quant_results,
            calibration_results=calibration_results,
            deployment_manifest=deployment_manifest,
            failures=failures,
            invalidated=invalidated,
            reused=reused,
            keys=keys,
            budget_events=budget_events,
            pending_queue=pending_queue,
            max_concurrency=max_concurrency,
            batch_count=batch_count,
            initial_restore=initial_restore,
            current_manifest=current_manifest,
            package_path=package_path,
        )

    def _loading_payload(
        self,
        loaded: LoadedState,
        config: CompressionConfig,
        failures: list[Failure],
    ) -> dict[str, Any]:
        return {
            "weights_hash": loaded.weights_hash,
            "calibration_hash": loaded.calibration_hash,
            "config_hash": config.config_hash,
            "layers": [
                {
                    "name": name,
                    "shape": list(loaded.layers[name].shape),
                    "content_version": loaded.layers[name].content_version,
                    "content_fingerprint": loaded.layers[name].content_fingerprint,
                }
                for name in sorted(loaded.layers)
            ],
            "calibration_sample_ids": sorted(
                sample.sample_id for sample in loaded.calibration.samples
            ),
            "invalid_layers": sorted(config.invalid_layers),
            "failures": [item.to_dict() for item in sort_failures(failures)],
        }

    def _commit_phase(
        self,
        phase: str,
        payload: dict[str, Any],
        config: CompressionConfig,
        loaded: LoadedState,
        completed_layers: list[str],
        status: str,
        pending_queue: list[dict[str, Any]],
        base: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if config.stage_delay_ms:
            time.sleep(config.stage_delay_ms / 1000.0)
        return self.snapshots.commit_phase(
            phase=phase,
            payload=payload,
            config_hash=config.config_hash,
            calibration_hash=loaded.calibration_hash,
            completed_layers=completed_layers,
            status=status,
            pending_queue=pending_queue,
            base=base,
        )

    def _commit_status(
        self,
        manifest: dict[str, Any],
        phase: str,
        status: str,
        payload: dict[str, Any],
        config: CompressionConfig,
        loaded: LoadedState,
        completed_layers: list[str],
        pending_queue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._commit_phase(
            phase,
            payload,
            config,
            loaded,
            completed_layers,
            status,
            pending_queue,
            manifest,
        )

    def _finish(
        self,
        *,
        status: str,
        config: CompressionConfig,
        loaded: LoadedState,
        layers: dict[str, Layer],
        quant_results: dict[str, QuantResult],
        calibration_results: dict[str, CalibrationResult],
        deployment_manifest: dict[str, Any] | None,
        failures: list[Failure],
        invalidated: dict[str, Any],
        reused: dict[str, Any],
        keys: dict[str, Any],
        budget_events: list[dict[str, Any]],
        pending_queue: list[dict[str, Any]],
        max_concurrency: int,
        batch_count: int,
        initial_restore: dict[str, Any] | None,
        current_manifest: dict[str, Any] | None,
        package_path: str | None,
    ) -> RunSummary:
        ordered_failures = sort_failures(failures)
        core: dict[str, Any] = {
            "layers": [],
            "calibration": [],
            "deployment_manifest": deployment_manifest,
            "failure_list": [item.to_dict() for item in ordered_failures],
        }
        for name in sorted(quant_results):
            quant = quant_results[name]
            layer = layers[name]
            core["layers"].append(
                {
                    "layer": name,
                    "shape": list(layer.shape),
                    "content_version": layer.content_version,
                    "content_fingerprint": layer.content_fingerprint,
                    "settings": config.effective(name).fingerprint_payload(),
                    "quantization_parameters": quant.parameters,
                    "quantized_sha256": quant.quantized_sha256,
                    "dequantized_sha256": quant.dequantized_sha256,
                    "distortion": quant.distortion,
                }
            )
        for name in sorted(calibration_results):
            core["calibration"].append(calibration_results[name].to_public_dict())

        error_layers: list[dict[str, Any]] = []
        quant_sse = 0.0
        quant_elements = 0
        before_sse = 0.0
        after_sse = 0.0
        calibration_elements = 0
        for name in sorted(quant_results):
            distortion = quant_results[name].distortion
            calibration = calibration_results.get(name)
            quant_sse += float(distortion.get("sum_squared_error", 0.0))
            quant_elements += int(distortion.get("element_count", 0))
            before = calibration.summary["before"] if calibration else None
            after = calibration.summary["after"] if calibration else None
            correction = calibration.summary["correction"] if calibration else None
            if calibration:
                before_sse += float(before.get("sum_squared_error", 0.0))
                after_sse += float(after.get("sum_squared_error", 0.0))
                calibration_elements += int(before.get("element_count", 0))
            error_layers.append(
                {
                    "layer": name,
                    "quantization": distortion,
                    "calibration_before": before,
                    "calibration_after": after,
                    "correction": correction,
                }
            )
        core["error_report"] = {
            "layers": error_layers,
            "totals": {
                "quantization": {
                    "sum_squared_error": quant_sse,
                    "mse": (quant_sse / quant_elements) if quant_elements else 0.0,
                    "element_count": quant_elements,
                },
                "calibration_before": {
                    "sum_squared_error": before_sse,
                    "mse": (
                        before_sse / calibration_elements
                        if calibration_elements
                        else 0.0
                    ),
                    "element_count": calibration_elements,
                },
                "calibration_after": {
                    "sum_squared_error": after_sse,
                    "mse": (
                        after_sse / calibration_elements
                        if calibration_elements
                        else 0.0
                    ),
                    "element_count": calibration_elements,
                },
            },
        }

        business_codes = sorted({item.code for item in ordered_failures})
        if pending_queue:
            business_codes.append(E_POINT_BUDGET_PENDING)
            business_codes = sorted(set(business_codes))

        layer_error_table: list[dict[str, Any]] = []
        degeneration_markers: list[dict[str, Any]] = []
        soft_label_disabled: list[dict[str, Any]] = []
        for name in sorted(quant_results):
            calibration = calibration_results.get(name)
            layer_error_table.append(
                {
                    "layer": name,
                    "quantization": quant_results[name].distortion,
                    "calibration": calibration.summary if calibration else None,
                }
            )
            parameters = quant_results[name].parameters
            if int(parameters.get("degenerated_group_count", 0)) > 0:
                degeneration_markers.append(
                    {
                        "layer": name,
                        "count": int(parameters.get("degenerated_group_count", 0)),
                        "sources": parameters.get("degeneration_sources", []),
                    }
                )
            if calibration and calibration.summary.get("soft_label_disabled"):
                soft_label_disabled.append(
                    {
                        "layer": name,
                        "reason": calibration.summary.get(
                            "soft_label_disabled_reason"
                        ),
                    }
                )

        affected = sorted(
            set(invalidated["quant_parameters"])
            | set(invalidated["calibration_deviations"])
            | set(config.invalid_layers)
        )
        snapshot_version = (
            int(current_manifest.get("global_version", 0))
            if current_manifest is not None
            else 0
        )
        final_restore = self.snapshots.restore_point()
        if final_restore is None and current_manifest is not None:
            final_restore = {
                "snapshot_version": snapshot_version,
                "status": current_manifest.get("status", status),
                "completed_layers": list(current_manifest.get("completed_layers", [])),
                "config_hash": current_manifest.get("config_hash", ""),
                "calibration_hash": current_manifest.get("calibration_hash", ""),
                "phases": {
                    name: {
                        "version": int(value.get("version", 0)),
                        "payload_sha256": value.get("payload_sha256", ""),
                    }
                    for name, value in current_manifest.get("phases", {}).items()
                },
            }
        diagnostics: dict[str, Any] = {
            "status": status,
            "business_codes": business_codes,
            "completed_layers": sorted(quant_results),
            "completed_layer_count": len(quant_results),
            "calibrated_layer_count": len(calibration_results),
            "affected_layers": affected,
            "layer_error_table": layer_error_table,
            "degeneration_markers": degeneration_markers,
            "soft_label_disabled": soft_label_disabled,
            "pending_queue": pending_queue,
            "budget_deductions": budget_events,
            "invalidated": {
                "quant_parameters": sorted(invalidated["quant_parameters"]),
                "dequantized_weights": sorted(invalidated["dequantized_weights"]),
                "calibration_deviations": sorted(
                    invalidated["calibration_deviations"]
                ),
                "deployment_manifest": bool(invalidated["deployment_manifest"]),
            },
            "reused": {
                "quant_parameters": sorted(reused["quant_parameters"]),
                "dequantized_weights": sorted(reused["dequantized_weights"]),
                "calibration_deviations": sorted(
                    reused["calibration_deviations"]
                ),
                "deployment_manifest": bool(reused["deployment_manifest"]),
            },
            "counts": {
                "invalidated": sum(
                    len(invalidated[key])
                    for key in (
                        "quant_parameters",
                        "dequantized_weights",
                        "calibration_deviations",
                    )
                )
                + int(bool(invalidated["deployment_manifest"])),
                "reused": sum(
                    len(reused[key])
                    for key in (
                        "quant_parameters",
                        "dequantized_weights",
                        "calibration_deviations",
                    )
                )
                + int(bool(reused["deployment_manifest"])),
            },
            "keys": keys,
            "snapshot_version": snapshot_version,
            "restore_point": final_restore,
            "initial_restore_point": initial_restore,
            "phase_versions": (
                current_manifest.get("phases", {})
                if current_manifest is not None
                else {}
            ),
            "snapshot_commit_interval": config.snapshot_commit_interval,
            "point_budget_per_batch": config.point_budget_per_batch,
            "concurrency": config.concurrency,
            "max_observed_concurrency": max_concurrency,
            "batch_count": batch_count,
            "failures": [item.to_dict() for item in ordered_failures],
            "failure_codes": sorted({item.code for item in ordered_failures}),
            "config_hash": config.config_hash,
            "calibration_hash": loaded.calibration_hash,
            "weights_hash": loaded.weights_hash,
            "package_path": package_path,
        }
        exit_code = 2 if status == "failed" else (3 if status == "pending" else 0)
        summary = RunSummary(
            status=status,
            core=core,
            diagnostics=diagnostics,
            business_codes=business_codes,
            snapshot_version=snapshot_version,
            restore_point=final_restore,
            package_path=package_path,
            exit_code=exit_code,
        )
        if self.write_reports:
            _atomic_json(self.state_dir / "reports" / "diagnostics.json", diagnostics)
            _atomic_json(self.state_dir / "reports" / "latest.json", summary.to_dict())
        return summary
