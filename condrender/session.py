from __future__ import annotations

from pathlib import Path
from typing import Any

from .bundle import apply_command, initial_state
from .engine import RenderEngine
from .errors import Codes
from .snapshot import SnapshotStore, restore_payload_into_engine


GLOBAL_OPS = {
    "set_weights", "set_denoise_steps", "set_tile_rows",
    "set_guidance", "set_quant_profile",
}


def affected_range(command: dict[str, Any], frames: int) -> list[int]:
    frame = command.get("frame")
    if isinstance(frame, int) and not isinstance(frame, bool) and 0 <= frame < frames:
        return [frame, frame]
    if command.get("op") in GLOBAL_OPS:
        return [0, frames - 1]
    return [0, frames - 1]


class InteractiveSession:
    def __init__(self, bundle: dict[str, Any], *, snapshot_dir: str | Path | None = None,
                 output_dir: str | Path | None = None, workers: int | None = None,
                 budget_ms: float | None = None, task_delay_ms: float = 0.0,
                 restore_snapshot: bool = True,
                 injections: list[dict[str, Any]] | None = None):
        self.bundle = bundle
        self.state = initial_state(bundle)
        self.workers = workers
        self.budget_ms = budget_ms
        self.task_delay_ms = task_delay_ms
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = SnapshotStore(snapshot_dir) if snapshot_dir else None
        self.cache: dict[str, dict[str, Any]] = {}
        self.ledger: list[dict[str, Any]] = []
        self.consumed_points = 0
        self.last_summary: dict[str, Any] | None = None
        self.command_reports: list[dict[str, Any]] = []
        self.failure_list: list[dict[str, Any]] = []
        self.injections = injections or []
        self.restore_decision = {"decision": "blank_full_render", "code": Codes.OK}
        self.last_engine: RenderEngine | None = None

        if self.store is not None and restore_snapshot:
            payload = self.store.load_current()
            if payload:
                self.cache = payload.get("cache", {})
                self.ledger = payload.get("ledger", [])
                self.consumed_points = int(payload.get("consumed_points", 0))
                self.last_summary = payload.get("last_summary")
                self.command_reports = payload.get("command_log", [])
                self.failure_list = payload.get("failures", [])
                self.restore_decision = {
                    "decision": "cache_restored",
                    "global_version": payload.get("global_version"),
                    "code": Codes.OK,
                }

    def _new_engine(self, output_dir: Path | None = None) -> RenderEngine:
        engine = RenderEngine(
            self.state,
            injections=self.injections,
            workers=self.workers,
            task_delay_ms=self.task_delay_ms,
            output_dir=output_dir or self.output_dir,
        )
        engine.cache = self.cache
        engine.ledger = self.ledger
        engine.consumed_points = self.consumed_points
        engine.last_summary = self.last_summary
        engine.command_log = self.command_reports
        engine.snapshot_failures = self.failure_list
        return engine

    def _transfer(self, engine: RenderEngine) -> None:
        self.cache = engine.cache
        self.ledger = engine.ledger
        self.consumed_points = engine.consumed_points
        if engine.last_summary is not None:
            self.last_summary = engine.last_summary

    def handle(self, command: dict[str, Any], *, budget_ms: float | None = None,
               output_dir: Path | None = None) -> dict[str, Any]:
        frames = int(self.state["config"]["frames"])
        affected = affected_range(command, frames)
        applied = apply_command(self.state, command)
        budget = self.budget_ms if budget_ms is None else budget_ms

        if not applied.get("ok"):
            failure = applied["failure"]
            self.failure_list.append(failure)
            report = {
                "command_id": command.get("id"),
                "op": command.get("op"),
                "affected_frame_range": affected,
                "status": "invalid",
                "code": failure["code"],
                "failure": failure,
                "summary": dict(self.last_summary or {}),
                "entries": [],
                "failures": [failure],
            }
            self.command_reports.append(report)
            return report

        engine = self._new_engine(output_dir=output_dir)
        checkpoint = self.store.commit if self.store is not None else None
        result = engine.render_all(
            budget_ms=budget, affected_range=affected, checkpoint=checkpoint
        )
        self._transfer(engine)
        self.last_engine = engine
        if result["failures"]:
            self.failure_list.extend(result["failures"])

        report = {
            "command_id": command.get("id"),
            "op": command.get("op"),
            "affected_frame_range": affected,
            "status": result["status"],
            "code": result.get("code", Codes.OK),
            "summary": result["summary"],
            "entries": result["entries"],
            "failures": result["failures"],
            "completed_frames": result["completed_frames"],
        }
        self.command_reports.append(report)
        if self.store is not None and report["status"] == "stale":
            self.store.commit(engine)
        return report

    def run_stream(self, commands: list[dict[str, Any]], *,
                   budget_ms: float | None = None) -> list[dict[str, Any]]:
        return [self.handle(command, budget_ms=budget_ms) for command in commands]

    def finish(self, *, output_dir: Path | None = None) -> dict[str, Any]:
        engine = self._new_engine(output_dir=output_dir)
        checkpoint = self.store.commit if self.store is not None else None
        result = engine.render_all(budget_ms=None, checkpoint=checkpoint)
        self._transfer(engine)
        self.last_engine = engine
        if result["failures"]:
            self.failure_list.extend(result["failures"])
        return result


def render_final_state(bundle: dict[str, Any], commands: list[dict[str, Any]], *,
                       workers: int | None = None, output_dir: str | Path | None = None,
                       task_delay_ms: float = 0.0,
                       injections: list[dict[str, Any]] | None = None
                       ) -> tuple[RenderEngine, dict[str, Any], dict[str, Any]]:
    state = initial_state(bundle)
    command_failures: list[dict[str, Any]] = []
    for command in commands:
        applied = apply_command(state, command)
        if not applied.get("ok"):
            command_failures.append(applied["failure"])
    engine = RenderEngine(
        state, injections=injections or [], workers=workers,
        task_delay_ms=task_delay_ms, output_dir=output_dir,
    )
    result = engine.render_all()
    result["command_failures"] = command_failures
    return engine, result, state


def field_report(engine: RenderEngine, result: dict[str, Any]) -> dict[str, Any]:
    pixels: dict[str, Any] = {}
    residuals: dict[str, Any] = {}
    tile_order: dict[str, Any] = {}
    lineage: dict[str, Any] = {}
    for frame in sorted(engine.frame_results):
        frame_result = engine.frame_results[frame]
        compose = engine.cache[frame_result["keys"]["frame_compose"]]["data"]
        pixels[str(frame)] = {
            "rgb": compose["rgb"],
            "png_sha256": compose["png_sha256"],
            "shape": compose["shape"],
        }
        residuals[str(frame)] = {
            "steps": [
                {
                    "key": key,
                    "update_residual": engine.cache[key]["data"]["update_residual"],
                    "update_direction": engine.cache[key]["data"]["update_direction"],
                    "guidance_residual": engine.cache[key]["data"]["guidance_residual"],
                    "residual_fingerprint": engine.cache[key]["data"]["residual_fingerprint"],
                }
                for key in frame_result["step_keys"]
            ],
            "residual_fps": frame_result["step_residual_fps"],
        }
        tile_order[str(frame)] = frame_result["tile_order"]
        lineage[str(frame)] = {
            "keys": frame_result["keys"],
            "step_keys": frame_result["step_keys"],
            "tile_keys": frame_result["tile_keys"],
            "condition_fingerprint": frame_result["condition_fingerprint"],
            "png_sha256": frame_result["png_sha256"],
        }
    return {
        "pixels": pixels,
        "denoise_residuals": residuals,
        "tile_completion_order": tile_order,
        "lineage": lineage,
        "failures": result.get("failures", []),
        "command_failures": result.get("command_failures", []),
    }


def canonical_ledger(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "command_id": report.get("command_id"),
            "status": report.get("status"),
            "code": report.get("code"),
            "affected_frame_range": report.get("affected_frame_range"),
            "entries": [
                {
                    "status": entry["status"],
                    "stage": entry["stage"],
                    "frame": entry["frame"],
                    "index": entry["index"],
                    "key": entry["key"],
                }
                for entry in report.get("entries", [])
            ],
            "failures": report.get("failures", []),
        }
        for report in reports
    ]
