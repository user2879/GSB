"""Latency-bounded command submission with distinguishable interim frames."""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .engine import LayerPackEngine, RunSummary, ShutdownController
from .errors import E_CONFIG_INVALID, E_INTERACT_BUDGET_PENDING
from .session import CommandError, SessionState, SessionStore


@dataclass
class InteractiveFrame:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


@dataclass
class _Operation:
    command_id: str
    command_type: str
    affected_layers: list[str]
    candidate: SessionState
    start_time: float
    budget_ms: int = 0
    event: threading.Event = field(default_factory=threading.Event)
    result: RunSummary | None = None
    error: Exception | None = None
    thread: threading.Thread | None = None
    finalized: bool = False


class CompressionSession:
    """Interactive facade around the incremental engine.

    Each accepted command runs against a private copy of session state. The
    copy is persisted only after that run reaches a final, non-interaction
    pending state. Therefore an interim frame can never make its command
    visible to a later session process.
    """

    def __init__(
        self,
        weights_path: str | Path,
        calibration_path: str | Path,
        config_path: str | Path,
        state_dir: str | Path,
        *,
        reset_session: bool = False,
        shutdown: ShutdownController | None = None,
        write_reports: bool = True,
    ) -> None:
        self.weights_path = str(weights_path)
        self.calibration_path = str(calibration_path)
        self.config_path = str(config_path)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = SessionStore(self.state_dir)
        self.state = self.store.load_or_create(
            self.weights_path,
            self.calibration_path,
            self.config_path,
            reset=reset_session,
        )
        self.shutdown = shutdown or ShutdownController()
        self.write_reports = write_reports
        self.current_result: RunSummary | None = None
        self._active: _Operation | None = None
        self._sequence = 0

    def _latency_budget_ms(self) -> int:
        runtime = self.state.config.get("runtime", {})
        raw = {}
        if isinstance(runtime, dict):
            raw = runtime
        value = raw.get(
            "interaction_latency_budget_ms",
            raw.get(
                "latency_budget_ms",
                self.state.config.get(
                    "interaction_latency_budget_ms",
                    self.state.config.get("latency_budget_ms", 100),
                ),
            ),
        )
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return 100
        return int(value)

    def _work_dir(self, label: str) -> Path:
        root = self.state_dir / "work"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"{label}-", dir=root))

    def _execute_state(self, state: SessionState, label: str) -> RunSummary:
        work_dir = self._work_dir(label)
        try:
            paths = state.export_to(work_dir)
            engine = LayerPackEngine(
                paths["weights"],
                paths["calibration"],
                paths["config"],
                self.state_dir,
                use_cache=True,
                shutdown=self.shutdown,
                write_reports=self.write_reports,
            )
            return engine.run()
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def initialize(self, *, wait: bool = True) -> RunSummary | None:
        """Perform the initial full run so later commands can reuse artifacts."""

        if self.current_result is not None:
            return self.current_result
        operation = _Operation(
            command_id="initialize",
            command_type="initialize",
            affected_layers=[],
            candidate=self.state,
            start_time=time.monotonic(),
        )

        def initialize_worker() -> None:
            try:
                operation.result = self._execute_state(self.state, "initialize")
                if operation.result.status not in {"pending", "interrupted"}:
                    self.store.save(self.state)
                    self.current_result = operation.result
            except Exception as exc:  # pragma: no cover - defensive surface
                operation.error = exc
            finally:
                operation.event.set()

        thread = threading.Thread(
            target=initialize_worker, name="layerpack-initialize", daemon=True
        )
        operation.thread = thread
        thread.start()
        if wait:
            operation.event.wait()
        if operation.error is not None:
            raise operation.error
        return operation.result

    def _latest_summary(self) -> dict[str, Any] | None:
        if self.current_result is None:
            return None
        result = self.current_result
        quant_mse = 0.0
        max_abs = 0.0
        quant_elements = 0
        for layer in result.core.get("layers", []):
            distortion = layer.get("distortion", {})
            quant_mse += float(distortion.get("sum_squared_error", 0.0))
            quant_elements += int(distortion.get("element_count", 0))
            max_abs = max(max_abs, float(distortion.get("max_abs_error", 0.0)))
        before_sse = 0.0
        after_sse = 0.0
        element_count = 0
        for calibration in result.core.get("calibration", []):
            summary = calibration.get("summary", {})
            before = summary.get("before", {})
            after = summary.get("after", {})
            before_sse += float(before.get("sum_squared_error", 0.0))
            after_sse += float(after.get("sum_squared_error", 0.0))
            element_count += int(before.get("element_count", 0))
        return {
            "status": result.status,
            "snapshot_version": result.snapshot_version,
            "completed_layer_count": result.diagnostics.get(
                "completed_layer_count", 0
            ),
            "calibrated_layer_count": result.diagnostics.get(
                "calibrated_layer_count", 0
            ),
            "error_statistics": {
                "quantization_mse": (
                    quant_mse / quant_elements if quant_elements else 0.0
                ),
                "quantization_max_abs_error": max_abs,
                "calibration_before_mse": (
                    before_sse / element_count if element_count else 0.0
                ),
                "calibration_after_mse": (
                    after_sse / element_count if element_count else 0.0
                ),
            },
        }

    def _interim_frame(
        self, operation: _Operation, elapsed_ms: float, budget_ms: int
    ) -> InteractiveFrame:
        latest = self._latest_summary()
        completed_layers = (
            list(self.current_result.diagnostics.get("completed_layers", []))
            if self.current_result is not None
            else []
        )
        total_layers = len(self.state.layers)
        progress_layers = (
            int(latest.get("completed_layer_count", 0)) if latest else 0
        )
        payload = {
            "frame_type": "interim",
            "status": "pending",
            "business_code": E_INTERACT_BUDGET_PENDING,
            "command_id": operation.command_id,
            "command_type": operation.command_type,
            "latency_budget_ms": budget_ms,
            "elapsed_ms": round(elapsed_ms, 6),
            "affected_layers": operation.affected_layers,
            "completed_layers": completed_layers,
            "completed_layer_count": progress_layers,
            "total_layer_count": total_layers,
            "progress": {
                "completed": progress_layers,
                "total": total_layers,
                "fraction": (progress_layers / total_layers) if total_layers else 1.0,
            },
            "latest_completed_summary": latest,
            "error_statistics": latest.get("error_statistics", {}) if latest else {},
            "persisted": False,
        }
        return InteractiveFrame(payload)

    def _final_frame(
        self, operation: _Operation, elapsed_ms: float, budget_ms: int
    ) -> InteractiveFrame:
        if operation.error is not None:
            return InteractiveFrame(
                {
                    "frame_type": "final",
                    "status": "failed",
                    "business_code": E_CONFIG_INVALID,
                    "command_id": operation.command_id,
                    "command_type": operation.command_type,
                    "affected_layers": operation.affected_layers,
                    "elapsed_ms": round(elapsed_ms, 6),
                    "latency_budget_ms": budget_ms,
                    "persisted": False,
                    "error": str(operation.error),
                }
            )
        assert operation.result is not None
        payload = {
            "frame_type": "final",
            "status": operation.result.status,
            "business_codes": operation.result.business_codes,
            "command_id": operation.command_id,
            "command_type": operation.command_type,
            "affected_layers": operation.affected_layers,
            "elapsed_ms": round(elapsed_ms, 6),
            "latency_budget_ms": budget_ms,
            "persisted": operation.finalized,
            "result": operation.result.to_dict(),
        }
        return InteractiveFrame(payload)

    def _worker(self, operation: _Operation) -> None:
        try:
            result = self._execute_state(operation.candidate, operation.command_id)
            operation.result = result
            if result.status not in {"pending", "interrupted"}:
                self.store.save(operation.candidate)
                self.state = operation.candidate
                self.current_result = result
                operation.finalized = True
        except Exception as exc:  # pragma: no cover - defensive surface
            operation.error = exc
        finally:
            operation.event.set()

    def submit_command(
        self,
        command: dict[str, Any],
        *,
        latency_budget_ms: int | None = None,
    ) -> InteractiveFrame:
        request_start = time.monotonic()
        if self._active is not None and not self._active.event.is_set():
            raise RuntimeError("another interactive command is still running")
        possible_id = command.get("command_id") if isinstance(command, dict) else None
        self._sequence += 1
        command_id = (
            possible_id
            if isinstance(possible_id, str) and possible_id
            else f"cmd-{self._sequence:04d}"
        )
        command_type = str(command.get("type", "")) if isinstance(command, dict) else ""
        candidate = self.state.clone()
        try:
            affected = candidate.apply_command(command)
        except CommandError as exc:
            operation = _Operation(
                command_id=command_id,
                command_type=command_type,
                affected_layers=[],
                candidate=candidate,
                start_time=time.monotonic(),
                error=exc,
            )
            operation.event.set()
            self._active = operation
            return self._final_frame(operation, 0.0, 0)

        budget_ms = (
            self._latency_budget_ms()
            if latency_budget_ms is None
            else int(latency_budget_ms)
        )
        if budget_ms < 0:
            raise ValueError("latency_budget_ms must be non-negative")
        operation = _Operation(
            command_id=command_id,
            command_type=command_type,
            affected_layers=affected,
            candidate=candidate,
            start_time=request_start,
            budget_ms=budget_ms,
        )
        thread = threading.Thread(
            target=self._worker,
            args=(operation,),
            name=f"layerpack-{command_id}",
            daemon=True,
        )
        operation.thread = thread
        self._active = operation
        thread.start()
        remaining_budget_ms = budget_ms - (
            (time.monotonic() - request_start) * 1000.0
        )
        # Leave a small scheduling margin: CPython can spend a couple of
        # milliseconds reacquiring the GIL after a timed condition wait. An
        # early interim frame is valid, while exceeding the advertised budget
        # is not.
        safety_margin_ms = min(3.0, budget_ms * 0.25) if budget_ms > 5 else budget_ms
        wait_ms = remaining_budget_ms - safety_margin_ms
        if wait_ms > 0:
            operation.event.wait(wait_ms / 1000.0)
        elapsed_ms = (time.monotonic() - operation.start_time) * 1000.0
        if operation.event.is_set():
            return self._final_frame(operation, elapsed_ms, budget_ms)
        return self._interim_frame(operation, elapsed_ms, budget_ms)

    def wait_for_final(self, frame: InteractiveFrame | None = None) -> InteractiveFrame:
        if self._active is None:
            if frame is not None and frame.payload.get("frame_type") == "final":
                return frame
            raise RuntimeError("no active interactive command")
        self._active.event.wait()
        elapsed_ms = (time.monotonic() - self._active.start_time) * 1000.0
        return self._final_frame(
            self._active, elapsed_ms, self._active.budget_ms
        )

    def apply_commands(
        self,
        commands: Iterable[dict[str, Any]],
        *,
        latency_budget_ms: int | None = None,
    ) -> tuple[list[InteractiveFrame], InteractiveFrame]:
        frames: list[InteractiveFrame] = []
        final: InteractiveFrame | None = None
        for command in commands:
            frame = self.submit_command(
                command, latency_budget_ms=latency_budget_ms
            )
            frames.append(frame)
            if frame.payload.get("frame_type") == "interim":
                final = self.wait_for_final(frame)
                frames.append(final)
            else:
                final = frame
        if final is None:
            raise ValueError("at least one command is required")
        return frames, final

    def export_final_state(self, directory: str | Path) -> dict[str, str]:
        return self.state.export_to(directory)
