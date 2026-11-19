from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .diagnostics import (
    analyze_state,
    build_error_record,
    build_propagation_record,
    diagnostics_business_codes,
    fixed_operator_effect,
    float_reference_geometry,
)
from .errors import BusinessError, ErrorCode
from .mesh import MeshState
from .operators import apply_operator
from .persistence import SnapshotRecord, StateInspection, StateStore, atomic_write_json, utc_now
from .schema import MeshInput, RepairScript, make_version_id
from .topology import build_topology


EXIT_OK = 0
EXIT_SIGINT = 130
EXIT_SIGTERM = 143


@dataclass(slots=True)
class CancellationToken:
    _event: threading.Event
    exit_code: int = EXIT_OK
    signal_name: str | None = None

    def __init__(self) -> None:
        self._event = threading.Event()
        self.exit_code = EXIT_OK
        self.signal_name = None

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self, signum: int | None = None, name: str | None = None) -> None:
        if self._event.is_set():
            return
        if signum == signal.SIGINT:
            self.exit_code = EXIT_SIGINT
            self.signal_name = "SIGINT"
        elif signum == signal.SIGTERM:
            self.exit_code = EXIT_SIGTERM
            self.signal_name = "SIGTERM"
        else:
            self.exit_code = EXIT_SIGINT
            self.signal_name = name or "CANCEL"
        self._event.set()


@dataclass(slots=True)
class RunResult:
    version_id: str
    report: dict[str, Any]
    report_path: str | None
    exit_code: int
    completed_steps: int
    total_operators: int
    resumed_from_step: int
    state_inspection: StateInspection

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_OK


def _mesh_from_snapshot(payload: dict[str, Any]) -> MeshState:
    try:
        mesh_payload = payload["mesh"]
        state = MeshState.from_payload(mesh_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(
            ErrorCode.E_SNAPSHOT_INVALID,
            "snapshot mesh arrays are malformed",
            {"reason": str(exc)},
        ) from exc
    if state.fixed.ndim != 2 or state.fixed.shape[1] != 3:
        raise BusinessError(
            ErrorCode.E_SNAPSHOT_INVALID, "snapshot fixed coordinate shape is invalid"
        )
    if state.reference.shape != state.fixed.shape:
        raise BusinessError(
            ErrorCode.E_SNAPSHOT_INVALID, "snapshot coordinate arrays have different shapes"
        )
    if state.faces.ndim != 2 or state.faces.shape[1] != 3:
        raise BusinessError(ErrorCode.E_SNAPSHOT_INVALID, "snapshot face shape is invalid")
    if not np.all(np.isfinite(state.reference)):
        raise BusinessError(
            ErrorCode.E_SNAPSHOT_INVALID, "snapshot reference coordinates are non-finite"
        )
    if state.faces.size and (
        int(np.min(state.faces)) < 0 or int(np.max(state.faces)) >= state.vertex_count
    ):
        raise BusinessError(
            ErrorCode.E_SNAPSHOT_INVALID, "snapshot face indices are out of bounds"
        )
    return state


class RunEngine:
    def __init__(
        self,
        mesh: MeshInput,
        script: RepairScript,
        state_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.mesh = mesh
        self.script = script
        self.state_dir = Path(state_dir)
        self.output_dir = Path(output_dir)
        self.version_id = make_version_id(mesh.mesh_digest, script.script_digest)
        self.store = StateStore(self.state_dir)

    def _initial_snapshot_payload(self, state: MeshState) -> dict[str, Any]:
        _, diagnostics = analyze_state(
            state,
            self.script.area_tolerance,
            self.script.concurrency,
            self.script.memory_budget_nodes,
        )
        error = build_error_record(
            state,
            self.script.coordinate_tolerance,
            0,
            [],
        )
        business_codes = diagnostics_business_codes(diagnostics, "INITIAL")
        if error["exceeding_vertex_count"]:
            business_codes.append("W_QUANTIZATION_TOLERANCE_EXCEEDED")
        return {
            "snapshot_schema_version": 1,
            "version_id": self.version_id,
            "step": 0,
            "operator_index": 0,
            "operator_type": "input",
            "operator": None,
            "execution_count": 0,
            "execution_details": {"vertex_count": state.vertex_count, "face_count": state.face_count},
            "mesh": state.to_payload(),
            "diagnostics": diagnostics,
            "fixed_geometry": diagnostics["fixed_geometry"],
            "reference_geometry": float_reference_geometry(state),
            "operator_effect": fixed_operator_effect(None, state, diagnostics["fixed_geometry"]),
            "error": error,
            "propagation": {
                "operator_index": 0,
                "source_count": 0,
                "source_vertices": [],
                "trace_radius": self.script.trace_radius,
                "affected_vertex_count": 0,
                "affected_vertices": [],
                "max_reached_layer": 0,
                "decay_counts": {},
                "evaluation": {},
            },
            "propagation_sources": [],
            "business_codes": business_codes,
            "created_utc": utc_now(),
        }

    def _operator_snapshot_payload(
        self,
        step: int,
        operator: dict[str, Any],
        state: MeshState,
        previous_state: MeshState,
        sources: list[int],
        details: dict[str, Any],
        execution_count: int,
        diagnostics: dict[str, Any],
        topology: Any,
    ) -> dict[str, Any]:
        error = build_error_record(
            state,
            self.script.coordinate_tolerance,
            step,
            sources,
        )
        traces, propagation = build_propagation_record(
            state,
            topology,
            step,
            sources,
            self.script.trace_radius,
            self.script.coordinate_tolerance,
            self.script.concurrency,
            self.script.memory_budget_nodes,
        )
        business_codes = diagnostics_business_codes(diagnostics, f"STEP_{step}")
        if error["exceeding_vertex_count"]:
            business_codes.append("W_QUANTIZATION_TOLERANCE_EXCEEDED")
        return {
            "snapshot_schema_version": 1,
            "version_id": self.version_id,
            "step": step,
            "operator_index": step,
            "operator_type": str(operator["type"]),
            "operator": dict(operator),
            "execution_count": int(execution_count),
            "execution_details": details,
            "mesh": state.to_payload(),
            "diagnostics": diagnostics,
            "fixed_geometry": diagnostics["fixed_geometry"],
            "reference_geometry": float_reference_geometry(state),
            "operator_effect": fixed_operator_effect(
                previous_state, state, diagnostics["fixed_geometry"]
            ),
            "error": error,
            "propagation": propagation,
            "propagation_sources": traces,
            "business_codes": business_codes,
            "created_utc": utc_now(),
        }

    def _validate_completed_payload(
        self, record: SnapshotRecord, expected_step: int
    ) -> tuple[MeshState, dict[str, Any]]:
        if not record.valid or record.payload is None:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_INVALID,
                "attempted to use an invalid snapshot",
                {"step": record.step, "path": record.path, "reason": record.reason},
            )
        payload = record.payload
        if payload.get("version_id") != self.version_id:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_INVALID,
                "snapshot version does not match the current run",
                {"step": expected_step, "path": record.path},
            )
        if int(payload.get("step", -1)) != expected_step:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_INVALID,
                "snapshot step is not the expected step",
                {"expected": expected_step, "actual": payload.get("step")},
            )
        state = _mesh_from_snapshot(payload)
        if state.scale != self.script.fixed_scale:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_INVALID,
                "snapshot fixed-point scale does not match the script",
                {"expected": self.script.fixed_scale, "actual": state.scale},
            )
        return state, payload

    def _build_report(
        self,
        records: list[SnapshotRecord],
        resumed_from_step: int,
        signal_name: str | None,
    ) -> dict[str, Any]:
        if not records:
            raise BusinessError(ErrorCode.E_STATE, "cannot build a report without snapshots")
        payloads: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            _, payload = self._validate_completed_payload(record, index)
            payloads.append(payload)
        initial = payloads[0]
        final = payloads[-1]
        operator_payloads = payloads[1:]
        first_exceedance: dict[str, Any] | None = None
        for payload in payloads:
            candidate = payload.get("error", {}).get("first_tolerance_exceedance")
            if candidate is not None:
                first_exceedance = candidate
                break

        codes: set[str] = set()
        for payload in payloads:
            codes.update(str(code) for code in payload.get("business_codes", []))
        if first_exceedance is not None:
            codes.add("W_QUANTIZATION_TOLERANCE_EXCEEDED")
        codes.add("I_REPORT_COMPLETE")

        propagation_summary = []
        for payload in operator_payloads:
            propagation = payload["propagation"]
            propagation_summary.append(
                {
                    "operator_index": int(payload["operator_index"]),
                    "operator_type": payload["operator_type"],
                    "source_count": int(propagation.get("source_count", 0)),
                    "affected_vertex_count": int(propagation.get("affected_vertex_count", 0)),
                    "max_reached_layer": int(propagation.get("max_reached_layer", 0)),
                    "decay_counts": propagation.get("decay_counts", {}),
                    "evaluation": propagation.get("evaluation", {}),
                }
            )

        steps = []
        for payload in payloads:
            steps.append(
                {
                    "step": int(payload["step"]),
                    "operator_index": int(payload["operator_index"]),
                    "operator_type": payload["operator_type"],
                    "operator": payload.get("operator"),
                    "execution_count": int(payload.get("execution_count", 0)),
                    "execution_details": payload.get("execution_details", {}),
                    "diagnostics": payload["diagnostics"],
                    "error": payload["error"],
                    "propagation": payload["propagation"],
                    "operator_effect": payload.get("operator_effect"),
                    "business_codes": payload.get("business_codes", []),
                }
            )

        final_checks = final["diagnostics"]
        fixed_geometry = final["fixed_geometry"]
        finite_fixed = True
        finite_reference = True
        for payload in payloads:
            finite_fixed = finite_fixed and bool(
                np.all(np.isfinite(np.asarray(payload["mesh"]["fixed"], dtype=np.float64)))
            )
            finite_reference = finite_reference and bool(
                np.all(np.isfinite(np.asarray(payload["mesh"]["reference"], dtype=np.float64)))
            )
        fixed_ok = (
            finite_fixed
            and bool(final_checks.get("rescan_matches", False))
            and bool(
                final_checks.get("parallel_face_scan", {}).get(
                    "mask_matches_indexed_pass", False
                )
            )
            and first_exceedance is None
        )
        verdict = {
            "status": "PASS" if fixed_ok else "FAIL_FIXED_POINT_AUDIT",
            "decision_basis": "fixed_point_only",
            "checks": {
                "all_fixed_coordinates_finite": finite_fixed,
                "topology_rescan_matches": bool(final_checks.get("rescan_matches", False)),
                "parallel_face_scan_matches": bool(
                    final_checks.get("parallel_face_scan", {}).get(
                        "mask_matches_indexed_pass", False
                    )
                ),
                "coordinate_tolerance_respected": first_exceedance is None,
            },
            "float64_reference_used_for_verdict": False,
            "reference_coordinates_finite": finite_reference,
            "fixed_bbox_extent": fixed_geometry["bbox_extent"],
            "fixed_signed_volume": fixed_geometry["signed_volume"],
            "reference_bbox_extent": final["reference_geometry"]["bbox_extent"],
            "reference_signed_volume": final["reference_geometry"]["signed_volume"],
        }
        return {
            "schema_version": 1,
            "ok": True,
            "version_id": self.version_id,
            "mesh_digest": self.mesh.mesh_digest,
            "script_digest": self.script.script_digest,
            "input": {
                "path": self.mesh.source_path,
                "format": self.mesh.source_format,
                "initial_vertex_count": len(initial["mesh"]["fixed"]),
                "initial_face_count": len(initial["mesh"]["faces"]),
            },
            "normalized_script": self.script.normalized,
            "state_dir": str(self.state_dir),
            "output_dir": str(self.output_dir),
            "resume": {
                "resumed_from_step": resumed_from_step,
                "signal": signal_name,
                "state_archives": self.store.archive_entries(),
            },
            "summary": {
                "initial_diagnostics": initial["diagnostics"],
                "final_diagnostics": final["diagnostics"],
                "operator_count": len(self.script.operators),
                "completed_operator_count": len(operator_payloads),
                "first_tolerance_exceedance": first_exceedance,
                "final_max_abs_fixed_vs_reference_error": float(
                    final["error"]["max_abs_error"]
                ),
                "propagation": propagation_summary,
                "fixed_point_verdict": verdict,
                "reference_only": {
                    "role": "float64 is displayed only for comparison",
                    "final_geometry": final["reference_geometry"],
                },
            },
            "steps": steps,
            "business_codes": sorted(codes),
            "generated_utc": utc_now(),
        }

    def run(
        self,
        cancellation: CancellationToken | None = None,
        restart: bool = False,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> RunResult:
        token = cancellation or CancellationToken()
        manifest = self.store.initialize(
            version_id=self.version_id,
            mesh_digest=self.mesh.mesh_digest,
            script_digest=self.script.script_digest,
            total_operators=len(self.script.operators),
            restart=restart,
        )
        if int(manifest.get("total_operators", -1)) != len(self.script.operators):
            raise BusinessError(
                ErrorCode.E_STATE,
                "state manifest operator count does not match the script",
                {"manifest": manifest.get("total_operators"), "script": len(self.script.operators)},
            )
        inspection = self.store.inspect()
        if inspection.invalid_snapshot_count:
            invalid = [record.to_summary() for record in inspection.snapshots if not record.valid]
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_INVALID,
                "invalid snapshots are present; resume is refused and the state was not overwritten",
                {"invalid_snapshots": invalid},
            )

        initial_state = MeshState.from_input(self.mesh, self.script.fixed_scale)
        if inspection.latest_contiguous_valid_step is None:
            payload = self._initial_snapshot_payload(initial_state)
            self.store.write_snapshot(0, payload)
            inspection = self.store.inspect()
        records = self.store.valid_snapshots_for_resume(self.version_id)
        latest = records[-1]
        resumed_from_step = latest.step
        current_state, _ = self._validate_completed_payload(latest, latest.step)
        if latest.step > len(self.script.operators):
            raise BusinessError(
                ErrorCode.E_STATE,
                "state contains more steps than the script declares",
                {"latest_step": latest.step, "operator_count": len(self.script.operators)},
            )
        if progress is not None:
            progress(latest.step, len(self.script.operators), "loaded snapshot")

        if token.requested:
            return RunResult(
                version_id=self.version_id,
                report={},
                report_path=None,
                exit_code=token.exit_code,
                completed_steps=latest.step,
                total_operators=len(self.script.operators),
                resumed_from_step=resumed_from_step,
                state_inspection=self.store.inspect(),
            )

        for step in range(latest.step + 1, len(self.script.operators) + 1):
            if token.requested:
                break
            operator = self.script.operators[step - 1]
            topology = build_topology(
                current_state.fixed,
                current_state.faces,
                current_state.scale,
                self.script.area_tolerance,
            )
            next_state, sources, details, execution_count = apply_operator(
                current_state, topology, operator
            )
            _, diagnostics = analyze_state(
                next_state,
                self.script.area_tolerance,
                self.script.concurrency,
                self.script.memory_budget_nodes,
            )
            after_topology = build_topology(
                next_state.fixed,
                next_state.faces,
                next_state.scale,
                self.script.area_tolerance,
            )
            payload = self._operator_snapshot_payload(
                step,
                operator,
                next_state,
                current_state,
                sources,
                details,
                execution_count,
                diagnostics,
                after_topology,
            )
            self.store.write_snapshot(step, payload)
            current_state = next_state
            if progress is not None:
                progress(step, len(self.script.operators), str(operator["type"]))
            if token.requested:
                break

        inspection = self.store.inspect()
        records = self.store.valid_snapshots_for_resume(self.version_id)
        completed_steps = records[-1].step
        if token.requested:
            return RunResult(
                version_id=self.version_id,
                report={},
                report_path=None,
                exit_code=token.exit_code,
                completed_steps=completed_steps,
                total_operators=len(self.script.operators),
                resumed_from_step=resumed_from_step,
                state_inspection=inspection,
            )
        if completed_steps != len(self.script.operators):
            raise BusinessError(
                ErrorCode.E_STATE,
                "repair run stopped before all operators completed",
                {"completed_steps": completed_steps, "operator_count": len(self.script.operators)},
            )
        report = self._build_report(records, resumed_from_step, token.signal_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "report.json"
        atomic_write_json(report_path, report)
        return RunResult(
            version_id=self.version_id,
            report=report,
            report_path=str(report_path),
            exit_code=EXIT_OK,
            completed_steps=completed_steps,
            total_operators=len(self.script.operators),
            resumed_from_step=resumed_from_step,
            state_inspection=self.store.inspect(),
        )
