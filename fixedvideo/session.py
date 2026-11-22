"""Reusable command-session and pure replay helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable

from .cache import ArtifactCache
from .config import RenderConfig
from .engine import InteractiveEngine
from .errors import RenderError
from .renderer import DeterministicRenderer
from .snapshot import SnapshotManager
from .state import RenderState, apply_command, initial_state, touch_rejected_command


def failure_record(
    *,
    sequence: int,
    command_index: int,
    command: dict[str, Any],
    error: RenderError,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "command_index": command_index,
        "command_id": command.get("command_id"),
        "op": command.get("op"),
        "code": error.code,
        "message": error.message,
        "details": error.details,
    }


def replay_final_state(
    config: RenderConfig, commands: list[dict[str, Any]]
) -> tuple[RenderState, list[dict[str, Any]]]:
    """Apply commands without rendering; invalid commands are recorded and state is left intact."""
    state = initial_state(config)
    failures: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        try:
            state, _ = apply_command(config, state, command, command_index=index)
        except RenderError as exc:
            state = touch_rejected_command(state, command, index)
            failures.append(
                failure_record(
                    sequence=len(failures) + 1,
                    command_index=index,
                    command=command,
                    error=exc,
                )
            )
    return state, failures


class SessionRunner:
    def __init__(
        self,
        *,
        config: RenderConfig,
        commands: list[dict[str, Any]],
        command_stream_hash: str,
        workdir: str | Path,
        resume: bool = False,
        expected_version: int | None = None,
    ) -> None:
        self.config = config
        self.commands = commands
        self.command_stream_hash = command_stream_hash
        self.workdir = Path(workdir).resolve()
        self.snapshot_manager = SnapshotManager(self.workdir)
        self.resumed = False
        resumed = None
        if resume:
            cache_probe = ArtifactCache()
            renderer_probe = DeterministicRenderer(config, cache_probe)
            resumed = self.snapshot_manager.load_for_resume(
                config=config,
                command_stream_hash=command_stream_hash,
                expected_version=expected_version,
                expected_projection_fp=renderer_probe.projection_fp,
            )
            self.resumed = True
            cache = resumed["cache"]
            state = resumed["state"]
            renderer = DeterministicRenderer(config, cache)
            self.engine = InteractiveEngine(
                config,
                renderer,
                state=state,
                attempt_index=resumed["attempt_index"],
                failures=resumed["failures"],
                command_reports=resumed["command_reports"],
                latest_preview=resumed["latest_preview"],
                last_result=None,
            )
        else:
            state = initial_state(config)
            renderer = DeterministicRenderer(config, ArtifactCache())
            self.engine = InteractiveEngine(config, renderer, state=state)
        if self.engine.state.command_cursor > len(commands):
            raise RenderError(
                "E_SNAPSHOT_STALE",
                "snapshot command cursor is beyond the supplied command stream",
                {
                    "command_cursor": self.engine.state.command_cursor,
                    "command_count": len(commands),
                },
            )
        self.immediate_responses: list[dict[str, Any]] = []
        self.stable_responses: list[dict[str, Any]] = []
        self.snapshot_commits: list[dict[str, Any]] = []
        self.stopped = False
        self.stop_reason: str | None = None
        self.last_pointer: dict[str, Any] | None = self.snapshot_manager.current_pointer()

    @property
    def renderer(self) -> DeterministicRenderer:
        return self.engine.renderer

    @property
    def state(self) -> RenderState:
        return self.engine.state

    def commit(self) -> dict[str, Any]:
        payload = self.engine.checkpoint_payload()
        pointer = self.snapshot_manager.commit(
            config=self.config,
            command_stream_hash=self.command_stream_hash,
            state=self.engine.state,
            cache=payload["cache"],
            renderer=payload["renderer"],
            attempt_index=payload["attempt_index"],
            failures=payload["failures"],
            command_reports=payload["command_reports"],
            latest_preview=payload["latest_preview"],
            render_result=payload["last_result"],
        )
        self.last_pointer = pointer
        self.snapshot_commits.append(copy.deepcopy(pointer))
        return pointer

    def run(
        self,
        *,
        stop_after: int | None = None,
        commit_periodic: bool = True,
        commit_final: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        start_cursor = self.engine.state.command_cursor
        processed = 0
        try:
            for index in range(start_cursor, len(self.commands)):
                if should_stop is not None and should_stop():
                    self.stopped = True
                    self.stop_reason = "signal"
                    break
                immediate = self.engine.submit(self.commands[index])
                self.immediate_responses.append(copy.deepcopy(immediate))
                stable = self.engine.wait_idle()[-1]
                self.stable_responses.append(copy.deepcopy(stable))
                processed += 1
                cursor = self.engine.state.command_cursor
                if (
                    commit_periodic
                    and cursor > 0
                    and cursor % self.config.snapshot_interval_commands == 0
                ):
                    self.commit()
                if stop_after is not None and processed >= stop_after:
                    self.stopped = True
                    self.stop_reason = "stop_after"
                    break
                if should_stop is not None and should_stop():
                    self.stopped = True
                    self.stop_reason = "signal"
                    break
            if commit_final and (processed > 0 or not self.resumed):
                self.commit()
            return {
                "processed": processed,
                "stopped": self.stopped,
                "stop_reason": self.stop_reason,
                "start_cursor": start_cursor,
                "final_cursor": self.engine.state.command_cursor,
                "snapshot": self.last_pointer,
            }
        finally:
            self.engine.shutdown()

    def diagnostics(self, *, status: str = "ok") -> dict[str, Any]:
        state = self.engine.state
        reports = self.engine.command_reports()
        last_result = self.engine.last_result
        immediate_pending = sum(
            1
            for response in self.immediate_responses
            if response.get("business_code") == "E_INTERACT_BUDGET_PENDING"
        )
        stats = self.engine.stats_summary()
        return {
            "schema_version": 1,
            "status": status,
            "workdir": str(self.workdir),
            "config_name": self.config.name,
            "source_config_hash": self.config.source_hash,
            "command_stream_hash": self.command_stream_hash,
            "resumed": self.resumed,
            "interaction_latency_budget_ms": self.config.interaction_latency_budget_ms,
            "worker_count": self.config.worker_count,
            "snapshot_interval_commands": self.config.snapshot_interval_commands,
            "final": {
                "version": state.version,
                "command_cursor": state.command_cursor,
                "config_fingerprint": reports[-1].get("config_fingerprint") if reports else None,
                "seed": state.seed,
                "steps": state.steps,
                "frame_count": state.frame_count,
                "chunk_rows": state.chunk_rows,
                "quantization_level": state.quantization_level,
                "trace_fingerprint": last_result.trace_fingerprint if last_result else None,
                "composition_fingerprint": last_result.composition_fp if last_result else None,
                "frame_fingerprints": list(last_result.frame_fps) if last_result else [],
            },
            "stats": stats,
            "immediate_pending_responses": immediate_pending,
            "immediate_responses": [
                {
                    "command_id": item.get("command_id"),
                    "command_index": item.get("command_index"),
                    "status": item.get("status"),
                    "business_code": item.get("business_code"),
                    "response_latency_ms": item.get("response_latency_ms"),
                    "version": item.get("version"),
                    "preview_version": item.get("preview_corresponds_to_version"),
                }
                for item in self.immediate_responses
            ],
            "commands": reports,
            "failures": copy.deepcopy(self.engine.failures),
            "pending_recompute": [],
            "last_recompute_keys": (
                [
                    record["key"]
                    for record in last_result.audit
                    if record["status"] == "recompute"
                ]
                if last_result
                else []
            ),
            "snapshot_commits": self.snapshot_commits,
            "current_snapshot": self.snapshot_manager.current_pointer(),
        }
