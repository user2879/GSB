from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .engine import full_decode
from .errors import (
    AppError,
    E_COMMAND_INVALID,
    E_RESPONSE_BUDGET_PENDING,
    E_SESSION_NOT_FOUND,
    failure,
)
from .grid import effective_config_for_session
from .loader import (
    load_dataset,
    load_extra_scores,
)
from .models import Dataset, DecodeConfig, SessionArtifacts, SessionDataset, ShardRecord
from .reporting import build_summary, empty_summary, json_safe, public_report, write_report_atomic
from .snapshot import SnapshotManager


TAIL_ONLY_PARAMS = {
    "strategy",
    "lookahead_frames",
    "overlap_frames",
    "min_stable_frames",
    "stable_margin",
}

GRID_AFFECTING_PARAMS = {
    "lm_weight",
    "lm_backoff_score",
    "smoothing_score",
    "candidate_missing_below",
}

SLOT_AFFECTING_PARAMS = {"slot_min_frames", "slot_max_frames"}


class IncrementalApp:
    """Budget-aware interactive façade over the deterministic full decoder."""

    def __init__(
        self,
        sample_dir: str | Path,
        *,
        workspace_root: Optional[str | Path] = None,
        config: Optional[DecodeConfig] = None,
        injections: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.sample_dir = Path(sample_dir)
        self.workspace_root = Path(workspace_root or ".")
        self.lock = threading.RLock()
        self.version = 0
        self.command_counter = 0
        self.latest_report: Optional[Dict[str, Any]] = None
        self.latest_artifacts: Dict[str, SessionArtifacts] = {}
        self.latest_version = -1
        self.stale_compute_count = 0
        self.compute_events: List[Dict[str, Any]] = []
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self._inflight: Optional[Future] = None
        self._inflight_meta: Optional[Dict[str, Any]] = None
        self._background = ThreadPoolExecutor(max_workers=1, thread_name_prefix="streamparse-compute")
        self._closed = False

        dataset, loaded_config, loaded_injections, load_failures = load_dataset(self.sample_dir)
        self.dataset = dataset
        self.config = config or loaded_config
        self.injections = copy.deepcopy(dict(injections if injections is not None else loaded_injections))
        self.load_failures = load_failures
        report_root = self.workspace_root / self.config.snapshot_root
        self.snapshot_manager = SnapshotManager(
            report_root,
            version_provider=self._snapshot_version,
        )
        self.report_path = self.workspace_root / self.config.report_path

        # Initial materialisation is a full decode.  It is intentionally
        # completed before the first interactive budgeted command.
        initial = self._compute_snapshot(
            dataset=self.dataset.clone(),
            config=self.config,
            overrides=self._overrides_snapshot(),
            version=0,
            previous_artifacts={},
            affected_by_session=self._full_affected(self.dataset),
        )
        self._store_completed(0, initial, {})

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _snapshot_version(self) -> int:
        with self.lock:
            return int(self.version)

    def _overrides_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self.overrides.items()}

    def _effective_config(self, session_id: str) -> DecodeConfig:
        return effective_config_for_session(self.config, self.overrides.get(session_id))

    def _require_session(self, session_id: str) -> SessionDataset:
        session = self.dataset.sessions.get(session_id)
        if session is None:
            raise AppError(
                E_SESSION_NOT_FOUND,
                f"session does not exist: {session_id}",
                stage="command_validation",
                session_id=session_id,
            )
        return session

    def _target_session_ids(self, command: Mapping[str, Any], *, required: bool = False) -> List[str]:
        session_id = command.get("session_id")
        if session_id in (None, "*"):
            if required:
                raise AppError(
                    E_COMMAND_INVALID,
                    "command requires session_id",
                    stage="command_validation",
                )
            return self.dataset.sorted_session_ids()
        if not isinstance(session_id, str) or not session_id:
            raise AppError(E_COMMAND_INVALID, "session_id must be a non-empty string", stage="command_validation")
        self._require_session(session_id)
        return [session_id]

    def _range_for_session(self, session_id: str, reason: str) -> List[Dict[str, Any]]:
        session = self.dataset.sessions.get(session_id)
        if session is None:
            return []
        arrived = session.ordered_shards()
        if not arrived:
            return []
        return [
            {
                "start_frame": min(item.start_frame for item in arrived),
                "end_frame": max(item.end_frame for item in arrived),
                "reason": reason,
            }
        ]

    def _full_affected(self, dataset: Dataset) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {}
        for session_id in dataset.sorted_session_ids():
            result[session_id] = self._range_for_session(session_id, "full_decode")
        return result

    def _tail_affected(self, session_id: str, reason: str) -> List[Dict[str, Any]]:
        artifact = self.latest_artifacts.get(session_id)
        if artifact is None or not artifact.grid:
            return self._range_for_session(session_id, reason)
        start_index = artifact.decode.stable_length if artifact.decode is not None else 0
        start_index = min(start_index, len(artifact.grid) - 1)
        return [
            {
                "start_frame": artifact.grid[start_index].global_frame,
                "end_frame": artifact.grid[-1].global_frame,
                "reason": reason,
            }
        ]

    def _grid_affected(self, session_id: str, reason: str) -> List[Dict[str, Any]]:
        return self._range_for_session(session_id, reason)

    def summary(self, session_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        with self.lock:
            return build_summary(self.latest_report, session_ids)

    # ------------------------------------------------------------------
    # Score-row helpers
    # ------------------------------------------------------------------
    def _resolve_score_file(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        workspace_candidate = self.workspace_root / path
        if workspace_candidate.exists():
            return workspace_candidate
        sample_candidate = self.sample_dir / path
        if sample_candidate.exists():
            return sample_candidate
        return workspace_candidate

    def _rows_from_command(self, command: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if "score_file" in command:
            path = self._resolve_score_file(str(command["score_file"]))
            return load_extra_scores(path, self.dataset.vocab)
        raw = command.get("scores")
        if raw is None:
            raise AppError(
                E_COMMAND_INVALID,
                "command requires scores or score_file",
                stage="command_validation",
            )
        if not isinstance(raw, list):
            raise AppError(E_COMMAND_INVALID, "scores must be a list", stage="command_validation")
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(raw):
            if isinstance(item, dict):
                if "frame_in_shard" not in item or "scores" not in item:
                    raise AppError(
                        E_COMMAND_INVALID,
                        "inline score rows require frame_in_shard and scores",
                        stage="command_validation",
                    )
                rows.append(
                    {
                        "frame_in_shard": int(item["frame_in_shard"]),
                        "scores": list(item["scores"]),
                    }
                )
            elif isinstance(item, list):
                rows.append({"frame_in_shard": index, "scores": list(item)})
            else:
                raise AppError(E_COMMAND_INVALID, "invalid inline score row", stage="command_validation")
        return rows

    def _validate_rows_shape(
        self,
        rows: Sequence[Mapping[str, Any]],
        start_frame: int,
        end_frame: int,
        session_id: str,
        shard_seq: int,
    ) -> None:
        expected = end_frame - start_frame + 1
        frames = sorted(int(row["frame_in_shard"]) for row in rows)
        if frames != list(range(expected)):
            raise AppError(
                E_COMMAND_INVALID,
                "score rows do not exactly cover the shard's local frame range",
                stage="command_validation",
                session_id=session_id,
                shard_seq=shard_seq,
                details={"expected": list(range(expected)), "actual": frames},
            )
        for row in rows:
            values = list(row["scores"])
            if len(values) != len(self.dataset.vocab):
                raise AppError(
                    E_COMMAND_INVALID,
                    "score vector dimension does not match vocabulary",
                    stage="command_validation",
                    session_id=session_id,
                    shard_seq=shard_seq,
                )

    def _apply_shard_edit(self, command: Mapping[str, Any], *, replace_existing: bool) -> Dict[str, List[Dict[str, Any]]]:
        session_ids = self._target_session_ids(command, required=True)
        session_id = session_ids[0]
        session = self._require_session(session_id)
        try:
            shard_seq = int(command["shard_seq"])
            start_frame = int(command["start_frame"])
            end_frame = int(command["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError(
                E_COMMAND_INVALID,
                "shard_seq, start_frame and end_frame must be integers",
                stage="command_validation",
                session_id=session_id,
            ) from exc
        if shard_seq < 0 or start_frame < 0 or end_frame < start_frame:
            raise AppError(
                E_COMMAND_INVALID,
                "invalid shard sequence or frame bounds",
                stage="command_validation",
                session_id=session_id,
                shard_seq=shard_seq,
            )
        rows = self._rows_from_command(command)
        self._validate_rows_shape(rows, start_frame, end_frame, session_id, shard_seq)

        old = session.shards.get(shard_seq)
        if replace_existing:
            if old is None:
                raise AppError(
                    E_COMMAND_INVALID,
                    "replace_shard target does not exist",
                    stage="command_validation",
                    session_id=session_id,
                    shard_seq=shard_seq,
                )
            revision = old.revision + 1
        else:
            expected_seq = (max(session.shards) + 1) if session.shards else 0
            if shard_seq != expected_seq:
                raise AppError(
                    E_COMMAND_INVALID,
                    "append_shard must use the next contiguous shard sequence",
                    stage="command_validation",
                    session_id=session_id,
                    shard_seq=shard_seq,
                    details={"expected": expected_seq},
                )
            revision = 0

        session.shards[shard_seq] = ShardRecord(
            session_id=session_id,
            shard_seq=shard_seq,
            start_frame=start_frame,
            end_frame=end_frame,
            arrived=bool(command.get("arrived", True)),
            revision=revision,
        )
        if shard_seq not in session.ingest_order:
            session.ingest_order.append(shard_seq)
        session.scores[shard_seq] = {
            int(row["frame_in_shard"]): tuple(float(value) for value in row["scores"])
            for row in rows
        }

        if replace_existing and old is not None:
            earliest = min(old.start_frame, start_frame)
            latest = max(max((item.end_frame for item in session.shards.values()), default=end_frame), end_frame)
            affected = [
                {
                    "start_frame": earliest,
                    "end_frame": latest,
                    "reason": "replace_shard_and_downstream",
                }
            ]
        else:
            previous = [item for seq, item in session.shards.items() if seq < shard_seq]
            overlap_start = start_frame
            if previous:
                prev = max(previous, key=lambda item: item.shard_seq)
                overlap = min(max(0, self._effective_config(session_id).overlap_frames), prev.frame_count)
                overlap_start = max(prev.end_frame - overlap + 1, start_frame)
            affected = [
                {
                    "start_frame": overlap_start,
                    "end_frame": end_frame,
                    "reason": "append_new_and_overlap",
                }
            ]
        return {session_id: affected}

    # ------------------------------------------------------------------
    # Edit commands
    # ------------------------------------------------------------------
    def _apply_command(self, command: Mapping[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[List[str]]]:
        name = str(command.get("cmd", "")).lower()
        if name in ("set_strategy", "strategy"):
            strategy = command.get("strategy", command.get("value"))
            if strategy not in ("greedy", "lookahead"):
                raise AppError(
                    E_COMMAND_INVALID,
                    "strategy must be greedy or lookahead",
                    stage="command_validation",
                )
            return self._set_params({"strategy": strategy}, command), None
        if name in ("set_param", "set_parameter"):
            key = command.get("key")
            if not isinstance(key, str) or "value" not in command:
                raise AppError(E_COMMAND_INVALID, "set_param requires key and value", stage="command_validation")
            return self._set_params({key: command["value"]}, command), None
        if name in ("set_params", "set_parameters"):
            params = command.get("params")
            if not isinstance(params, dict):
                raise AppError(E_COMMAND_INVALID, "set_params requires a params object", stage="command_validation")
            return self._set_params(params, command), None
        if name == "append_shard":
            return self._apply_shard_edit(command, replace_existing=False), None
        if name == "replace_shard":
            return self._apply_shard_edit(command, replace_existing=True), None
        if name in ("append_shards", "replace_shards"):
            replace_existing = name == "replace_shards"
            shards = command.get("shards")
            if not isinstance(shards, list) or not shards:
                raise AppError(E_COMMAND_INVALID, f"{name} requires a non-empty shards list", stage="command_validation")
            affected: Dict[str, List[Dict[str, Any]]] = {}
            for item in shards:
                if not isinstance(item, dict):
                    raise AppError(E_COMMAND_INVALID, "each shard edit must be an object", stage="command_validation")
                nested = dict(item)
                nested.setdefault("session_id", command.get("session_id"))
                changes = self._apply_shard_edit(nested, replace_existing=replace_existing)
                for session_id, ranges in changes.items():
                    affected.setdefault(session_id, []).extend(ranges)
            return affected, None
        if name == "add_session":
            return self._add_session(command), None
        if name == "delete_session":
            session_ids = self._target_session_ids(command, required=True)
            for session_id in session_ids:
                self.dataset.sessions.pop(session_id, None)
                self.overrides.pop(session_id, None)
            return {}, session_ids
        if name == "reset":
            self._reload_state()
            return self._full_affected(self.dataset), None
        raise AppError(E_COMMAND_INVALID, f"unknown edit command: {name}", stage="command_validation")

    def _set_params(self, params: Mapping[str, Any], command: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        keys = set(params)
        unknown = keys - TAIL_ONLY_PARAMS - GRID_AFFECTING_PARAMS - SLOT_AFFECTING_PARAMS
        if unknown:
            raise AppError(
                E_COMMAND_INVALID,
                "unknown parameter(s): " + ", ".join(sorted(unknown)),
                stage="command_validation",
            )
        session_id = command.get("session_id")
        affected: Dict[str, List[Dict[str, Any]]] = {}
        if session_id in (None, "*"):
            # Global parameter edits update the base config.
            self.config = self.config.with_overrides(dict(params))
            targets = self.dataset.sorted_session_ids()
        else:
            targets = self._target_session_ids(command, required=True)
            for target in targets:
                merged = dict(self.overrides.get(target, {}))
                merged.update(params)
                effective_config_for_session(self.config, merged)
                self.overrides[target] = merged

        for target in targets:
            if keys <= TAIL_ONLY_PARAMS:
                affected[target] = self._tail_affected(target, "strategy_or_overlap_tail")
            elif keys <= SLOT_AFFECTING_PARAMS:
                affected[target] = self._tail_affected(target, "slot_bounds_tail")
            else:
                affected[target] = self._grid_affected(target, "grid_parameters_changed")
        return affected

    def _add_session(self, command: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        session_id = command.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise AppError(E_COMMAND_INVALID, "add_session requires session_id", stage="command_validation")
        if session_id in self.dataset.sessions:
            raise AppError(
                E_COMMAND_INVALID,
                "session already exists",
                stage="command_validation",
                session_id=session_id,
            )
        session = SessionDataset(session_id=session_id)
        self.dataset.sessions[session_id] = session
        shards = command.get("shards", [])
        if not isinstance(shards, list):
            raise AppError(E_COMMAND_INVALID, "add_session shards must be a list", stage="command_validation")
        for item in shards:
            if not isinstance(item, dict):
                raise AppError(E_COMMAND_INVALID, "add_session shard must be an object", stage="command_validation")
            nested = dict(item)
            nested["session_id"] = session_id
            self._apply_shard_edit(nested, replace_existing=False)
        return {session_id: self._range_for_session(session_id, "session_added")}

    def _reload_state(self) -> None:
        dataset, config, injections, failures = load_dataset(self.sample_dir)
        self.dataset = dataset
        self.config = config
        self.injections = copy.deepcopy(dict(injections))
        self.load_failures = failures
        self.overrides = {}

    # ------------------------------------------------------------------
    # Compute and budgets
    # ------------------------------------------------------------------
    def _compute_snapshot(
        self,
        *,
        dataset: Dataset,
        config: DecodeConfig,
        overrides: Mapping[str, Mapping[str, Any]],
        version: int,
        previous_artifacts: Mapping[str, SessionArtifacts],
        affected_by_session: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> Dict[str, Any]:
        if config.test_compute_delay_ms > 0:
            time.sleep(config.test_compute_delay_ms / 1000.0)
        report = full_decode(
            dataset,
            config,
            overrides=overrides,
            injections=self.injections,
            version=version,
            initial_failures=self.load_failures,
            previous_artifacts=previous_artifacts,
            affected_by_session=affected_by_session,
            snapshot_manager=self.snapshot_manager,
        )
        public = public_report(report)
        public["_artifacts_by_session"] = report.get("_artifacts_by_session", {})
        return public

    def _store_completed(
        self,
        version: int,
        report: Optional[Dict[str, Any]],
        meta: Mapping[str, Any],
    ) -> None:
        if report is None:
            return
        with self.lock:
            if version != self.version:
                self.stale_compute_count += 1
                self.compute_events.append(
                    {
                        "status": "stale_discarded",
                        "version": version,
                        "current_version": self.version,
                    }
                )
                return
            self.latest_report = report
            artifacts = report.get("_artifacts_by_session", {}) or {}
            self.latest_artifacts = dict(artifacts)
            self.latest_version = version
            self.compute_events.append(
                {
                    "status": "completed",
                    "version": version,
                    "command_id": meta.get("command_id"),
                }
            )
            event = self.latest_report.get("snapshot", {}).get("events", [])
            self.latest_report.setdefault("compute_events", []).extend(
                item for item in self.compute_events if item not in event
            )
            try:
                write_report_atomic(self.report_path, self.latest_report)
            except OSError as exc:
                self.compute_events.append({"status": "report_write_failed", "message": str(exc)})

    def _submit_compute(
        self,
        *,
        version: int,
        affected_by_session: Mapping[str, Sequence[Mapping[str, Any]]],
        command_id: str,
    ) -> Future:
        dataset_snapshot = self.dataset.clone()
        config_snapshot = self.config
        overrides_snapshot = self._overrides_snapshot()
        previous = dict(self.latest_artifacts)
        future = self._background.submit(
            self._compute_snapshot,
            dataset=dataset_snapshot,
            config=config_snapshot,
            overrides=overrides_snapshot,
            version=version,
            previous_artifacts=previous,
            affected_by_session={key: list(value) for key, value in affected_by_session.items()},
        )
        meta = {
            "version": version,
            "command_id": command_id,
            "affected_by_session": {key: list(value) for key, value in affected_by_session.items()},
        }

        def done_callback(completed: Future) -> None:
            try:
                report = completed.result()
            except Exception as exc:
                with self.lock:
                    self.compute_events.append(
                        {
                            "status": "background_failed",
                            "version": version,
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                return
            self._store_completed(version, report, meta)

        future.add_done_callback(done_callback)
        with self.lock:
            self._inflight = future
            self._inflight_meta = meta
        return future

    def _response_summary(self, session_ids: Optional[Sequence[str]]) -> Dict[str, Any]:
        with self.lock:
            return build_summary(self.latest_report, session_ids)

    def _response(
        self,
        *,
        command_id: str,
        command_name: str,
        status: str,
        budget_ms: int,
        session_ids: Optional[Sequence[str]],
        affected: Mapping[str, Sequence[Mapping[str, Any]]] | None,
        pending_version: Optional[int] = None,
        extra_codes: Optional[Sequence[str]] = None,
        error: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            latest_version = self.latest_version
            current_version = self.version
            summary = build_summary(self.latest_report, session_ids)
            codes = list(extra_codes or [])
            if status == "pending":
                codes.append(E_RESPONSE_BUDGET_PENDING)
            payload: Dict[str, Any] = {
                "command_id": command_id,
                "command": command_name,
                "status": status,
                "business_codes": sorted(set(codes)),
                "budget_ms": budget_ms,
                "global_version": current_version,
                "latest_completed_version": latest_version,
                "pending_version": pending_version,
                "summary": summary,
                "affected_intervals": {
                    key: list(value) for key, value in (affected or {}).items()
                },
                "compute_stale_count": self.stale_compute_count,
            }
            if error is not None:
                payload["error"] = dict(error)
            return json_safe(payload)

    def _wait_until(self, future: Future, *, deadline: float) -> bool:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            return future.done()
        try:
            future.result(timeout=remaining)
            return True
        except TimeoutError:
            return future.done()
        except Exception:
            return True

    def execute(self, command: Mapping[str, Any], *, no_budget: bool = False) -> Dict[str, Any]:
        started = time.monotonic()
        if not isinstance(command, Mapping):
            raise AppError(E_COMMAND_INVALID, "command must be an object", stage="command_validation")
        name = str(command.get("cmd", "")).lower()
        with self.lock:
            self.command_counter += 1
            command_id = str(command.get("command_id") or f"cmd-{self.command_counter:04d}")

        if name in ("quit", "exit"):
            return json_safe({"command_id": command_id, "command": name, "status": "closing"})

        if name in ("status", "wait", "report"):
            with self.lock:
                future = self._inflight
                meta = self._inflight_meta
                session_ids = None
                if command.get("session_id") not in (None, "*"):
                    target = str(command["session_id"])
                    if target in self.dataset.sessions:
                        session_ids = [target]
            if future is not None and not future.done() and name == "wait":
                try:
                    future.result()
                except Exception:
                    pass
            if future is not None and future.done() and meta is not None:
                try:
                    self._store_completed(int(meta["version"]), future.result(), meta)
                except Exception:
                    pass
            with self.lock:
                is_pending = self.latest_version != self.version
            if name == "report" and not is_pending:
                with self.lock:
                    return json_safe(public_report(self.latest_report or {}))
            return self._response(
                command_id=command_id,
                command_name=name,
                status="pending" if is_pending else "ok",
                budget_ms=self.config.response_budget_ms,
                session_ids=session_ids,
                affected=None,
                pending_version=self.version if is_pending else None,
            )

        # Edit command: mutate state under the lock, then compute a snapshot of
        # the new state in the background.  The caller only waits up to the
        # configured response budget.
        with self.lock:
            previous_version = self.version
            try:
                affected, deleted = self._apply_command(command)
            except AppError as exc:
                return self._response(
                    command_id=command_id,
                    command_name=name,
                    status="error",
                    budget_ms=self.config.response_budget_ms,
                    session_ids=None,
                    affected=None,
                    error=exc.as_failure(),
                )
            self.version = previous_version + 1
            target_ids = None
            requested_session = command.get("session_id")
            if requested_session not in (None, "*") and requested_session in self.dataset.sessions:
                target_ids = [str(requested_session)]
            elif deleted:
                target_ids = deleted
            future = self._submit_compute(
                version=self.version,
                affected_by_session=affected,
                command_id=command_id,
            )
            pending_version = self.version

        budget_ms = self.config.response_budget_ms
        # Keep a small serialization reserve so the response itself is emitted
        # inside the advertised budget rather than just the compute wait.
        deadline = started + max(0, budget_ms - 1) / 1000.0
        if no_budget:
            future.result()
            DONE = True
        else:
            DONE = self._wait_until(future, deadline=deadline)
        if DONE and future.done():
            try:
                report = future.result()
                meta = {"version": pending_version, "command_id": command_id}
                self._store_completed(pending_version, report, meta)
            except Exception as exc:
                return self._response(
                    command_id=command_id,
                    command_name=name,
                    status="error",
                    budget_ms=budget_ms,
                    session_ids=target_ids,
                    affected=affected,
                    error={"business_code": "E_INTERNAL", "message": f"{type(exc).__name__}: {exc}"},
                )
        with self.lock:
            completed = self.latest_version == pending_version
        return self._response(
            command_id=command_id,
            command_name=name,
            status="ok" if completed else "pending",
            budget_ms=budget_ms,
            session_ids=target_ids,
            affected=affected,
            pending_version=pending_version if not completed else None,
        )

    def wait(self) -> Dict[str, Any]:
        return self.execute({"cmd": "wait"}, no_budget=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self.lock:
            future = self._inflight
        if future is not None:
            try:
                future.result(timeout=10)
            except Exception:
                pass
        self._background.shutdown(wait=True, cancel_futures=False)
