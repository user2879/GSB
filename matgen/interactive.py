"""Latency-bounded interactive responses backed by real queued builds."""

from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from .commands import apply_command
from .engine import MaterialEngine
from .errors import E_COMMAND_INVALID, E_RESPONSE_STALE, Issue, make_issue
from .hashing import fingerprint
from .models import BuildResult, Command, GenerationConfig, Material, Texture
from .snapshot import SnapshotManager, phase_hashes_from_result


@dataclass
class BuildRequest:
    revision: int
    command_index: int
    command_id: str
    config: GenerationConfig
    materials: tuple[Material, ...]


@dataclass
class BuildOutcome:
    revision: int
    command_index: int
    command_id: str
    config: GenerationConfig
    materials: tuple[Material, ...]
    result: BuildResult | None
    error: str = ""
    snapshot: dict[str, Any] | None = None


class InteractiveRunner:
    def __init__(
        self,
        *,
        engine: MaterialEngine,
        textures: dict[str, Texture],
        input_config_hash: str,
        commands: tuple[Command, ...],
        snapshot_manager: SnapshotManager | None = None,
        resume: bool = False,
        expected_snapshot_version: int | None = None,
    ) -> None:
        self.engine = engine
        self.textures = textures
        self.input_config_hash = input_config_hash
        self.commands = commands
        self.snapshot_manager = snapshot_manager
        self.resume = resume
        self.expected_snapshot_version = expected_snapshot_version
        self.commands_hash = fingerprint(
            [{"id": command.command_id, "op": command.op, "payload": command.payload} for command in commands]
        )
        self._condition = threading.Condition()
        self._outcomes: dict[int, BuildOutcome] = {}
        self._latest_completed_revision = -1
        self._queue: queue.Queue[BuildRequest | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self.resume_info: dict[str, Any] = {
            "requested": resume,
            "usable": False,
            "command_cursor": 0,
            "issues": [],
        }

    def _worker_loop(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return
            result: BuildResult | None = None
            error = ""
            try:
                result = self.engine.build(request.config, request.materials, self.textures)
            except Exception as exc:  # keep later commands alive
                error = f"build failed: {exc}"
            outcome = BuildOutcome(
                revision=request.revision,
                command_index=request.command_index,
                command_id=request.command_id,
                config=request.config,
                materials=request.materials,
                result=result,
                error=error,
            )
            with self._condition:
                self._outcomes[request.revision] = outcome
                self._latest_completed_revision = max(self._latest_completed_revision, request.revision)
                self._condition.notify_all()

            manager = self.snapshot_manager
            interval = request.config.snapshot_commit_interval
            if manager is not None and result is not None and (request.command_index + 1) % interval == 0:
                try:
                    outcome.snapshot = manager.commit(
                        cache=self.engine.cache,
                        config=request.config,
                        input_config_hash=self.input_config_hash,
                        commands_hash=self.commands_hash,
                        command_cursor=request.command_index + 1,
                        completed_batches_cursor=result.batches_consumed,
                        phase_hashes=phase_hashes_from_result(result),
                        ledger=result.ledger_entries,
                    )
                except Exception as exc:
                    outcome.snapshot = {"status": "error", "message": str(exc)}

    def _start_worker(self) -> None:
        self._worker = threading.Thread(target=self._worker_loop, name="matgen-interactive", daemon=True)
        self._worker.start()

    def _stop_worker(self) -> None:
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join()

    def _wait_for(self, revision: int, deadline: float) -> BuildOutcome | None:
        with self._condition:
            while self._latest_completed_revision < revision:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._outcomes.get(revision)

    def _latest_outcome(self) -> BuildOutcome | None:
        with self._condition:
            if self._latest_completed_revision < 0:
                return None
            return self._outcomes.get(self._latest_completed_revision)

    def _drain(self, revision: int) -> BuildOutcome | None:
        with self._condition:
            while self._latest_completed_revision < revision:
                self._condition.wait()
            return self._outcomes.get(revision)

    def _response_from_outcome(
        self,
        *,
        outcome: BuildOutcome | None,
        requested_revision: int,
        command: Command | None,
        affected_layers: set[int],
        affected_channels: set[str],
        stale: bool,
        elapsed_ms: float,
        latency_budget_ms: int,
        extra_issues: list[Issue] | None = None,
    ) -> dict[str, Any]:
        issues = list(extra_issues or [])
        codes: list[str] = []
        if stale:
            codes.append(E_RESPONSE_STALE)
        if outcome is None or outcome.result is None:
            preview: dict[str, Any] = {
                "completed_layers": 0,
                "total_layers": 0,
                "progress": 0.0,
                "final_channels": {},
                "revision_digest": None,
            }
            completed_layers = 0
            total_layers = 0
            progress = 0.0
            summary_available = False
        else:
            preview = outcome.result.preview_summary()
            completed_layers = int(preview["completed_layers"])
            total_layers = int(preview["total_layers"])
            progress = float(preview["progress"])
            summary_available = True
        if outcome is not None and outcome.result is not None:
            for issue in outcome.result.issues:
                if issue.code not in codes:
                    codes.append(issue.code)
            if outcome.error and "E_INTERNAL" not in codes:
                codes.append("E_INTERNAL")
        return {
            "command_id": command.command_id if command else "",
            "requested_revision": requested_revision,
            "summary_revision": outcome.revision if outcome else None,
            "stale": stale,
            "codes": codes,
            "within_budget": not stale,
            "elapsed_ms": round(elapsed_ms, 3),
            "latency_budget_ms": latency_budget_ms,
            "summary_available": summary_available,
            "completed_layers": completed_layers,
            "total_layers": total_layers,
            "progress": progress,
            "requested_progress": 1.0 if outcome is not None and outcome.revision == requested_revision and outcome.result is not None else 0.0,
            "affected_layers": sorted(affected_layers),
            "affected_channels": sorted(affected_channels),
            "final_channel_summary": preview.get("final_channels", {}),
            "revision_digest": preview.get("revision_digest"),
            "build_elapsed_ms": round(outcome.result.build_elapsed_ms, 3) if outcome and outcome.result else None,
            "issues": [issue.as_dict() for issue in issues],
            "snapshot": outcome.snapshot if outcome else None,
        }

    def run(
        self,
        config: GenerationConfig,
        materials: tuple[Material, ...],
        *,
        stop_after: int | None = None,
        commit_final_snapshot: bool = True,
    ) -> dict[str, Any]:
        cursor = 0
        if self.resume and self.snapshot_manager is not None:
            usable, manifest, snapshot_issues = self.snapshot_manager.check_resume(
                input_config_hash=self.input_config_hash,
                commands_hash=self.commands_hash,
                expected_version=self.expected_snapshot_version,
            )
            self.resume_info["issues"] = [issue.as_dict() for issue in snapshot_issues]
            self.resume_info["manifest"] = manifest
            if usable and manifest is not None:
                restore_issues = self.snapshot_manager.restore_cache(self.engine.cache)
                self.resume_info["restore_issues"] = [issue.as_dict() for issue in restore_issues]
                cursor = min(int(manifest.get("command_cursor", 0)), len(self.commands))
                self.resume_info["usable"] = True
                self.resume_info["command_cursor"] = cursor
                self.resume_info["global_version"] = manifest.get("global_version")
            else:
                self.resume_info["usable"] = False
                cursor = 0

        for index in range(cursor):
            config, materials, _, _, issues = apply_command(config, materials, self.commands[index])
            if issues:
                self.resume_info.setdefault("replay_issues", []).extend(issue.as_dict() for issue in issues)

        total_to_run = len(self.commands)
        end_index = total_to_run if stop_after is None else min(total_to_run, max(cursor, stop_after))
        baseline = self.engine.build(config, materials, self.textures)
        revision = 0
        outcomes_by_revision: dict[int, BuildOutcome] = {
            0: BuildOutcome(
                revision=0,
                command_index=cursor - 1,
                command_id="baseline",
                config=config,
                materials=materials,
                result=baseline,
            )
        }
        self._start_worker()
        responses: list[dict[str, Any]] = []
        final_outcome = outcomes_by_revision[0]
        try:
            for command_index in range(cursor, end_index):
                command = self.commands[command_index]
                touch = time.perf_counter()
                new_config, new_materials, affected_layers, affected_channels, issues = apply_command(
                    config,
                    materials,
                    command,
                )
                if issues:
                    elapsed_ms = (time.perf_counter() - touch) * 1000.0
                    response = self._response_from_outcome(
                        outcome=final_outcome,
                        requested_revision=revision,
                        command=command,
                        affected_layers=affected_layers,
                        affected_channels=affected_channels,
                        stale=False,
                        elapsed_ms=elapsed_ms,
                        latency_budget_ms=config.latency_budget_ms,
                        extra_issues=issues,
                    )
                    response["codes"] = [E_COMMAND_INVALID]
                    response["rejected"] = True
                    responses.append(response)
                    continue

                config, materials = new_config, new_materials
                revision += 1
                request = BuildRequest(
                    revision=revision,
                    command_index=command_index,
                    command_id=command.command_id,
                    config=copy.deepcopy(config),
                    materials=copy.deepcopy(materials),
                )
                self._queue.put(request)
                deadline = touch + config.latency_budget_ms / 1000.0
                outcome = self._wait_for(revision, deadline)
                elapsed_ms = (time.perf_counter() - touch) * 1000.0
                on_time = outcome is not None and elapsed_ms <= config.latency_budget_ms
                if on_time:
                    final_outcome = outcome
                    response_outcome = outcome
                    stale = False
                else:
                    response_outcome = self._latest_outcome() or final_outcome
                    stale = True
                    self.engine.injections.record_code_for_action(
                        E_RESPONSE_STALE,
                        action="delay_ms",
                        stage="layer_noise",
                    )
                response = self._response_from_outcome(
                    outcome=response_outcome,
                    requested_revision=revision,
                    command=command,
                    affected_layers=affected_layers,
                    affected_channels=affected_channels,
                    stale=stale,
                    elapsed_ms=elapsed_ms,
                    latency_budget_ms=config.latency_budget_ms,
                )
                responses.append(response)

            if revision > 0:
                final_outcome = self._drain(revision) or final_outcome
        finally:
            self._stop_worker()

        snapshot_result = None
        if commit_final_snapshot and self.snapshot_manager is not None and final_outcome.result is not None:
            try:
                snapshot_result = self.snapshot_manager.commit(
                    cache=self.engine.cache,
                    config=config,
                    input_config_hash=self.input_config_hash,
                    commands_hash=self.commands_hash,
                    command_cursor=end_index,
                    completed_batches_cursor=final_outcome.result.batches_consumed,
                    phase_hashes=phase_hashes_from_result(final_outcome.result),
                    ledger=final_outcome.result.ledger_entries,
                )
            except Exception as exc:
                snapshot_result = {"status": "error", "message": str(exc)}

        return {
            "commands_total": len(self.commands),
            "commands_started_at": cursor,
            "commands_processed": end_index - cursor,
            "responses": responses,
            "stale_responses": sum(1 for response in responses if response["stale"]),
            "within_budget_responses": sum(1 for response in responses if response["within_budget"]),
            "final_revision": revision,
            "final_result": final_outcome.result,
            "final_outcome": final_outcome,
            "resume": self.resume_info,
            "final_snapshot": snapshot_result,
            "config": config,
            "materials": materials,
        }
