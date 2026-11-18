from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable

from .config import config_hash, load_config
from .edits import apply_batch, load_edit_batches
from .engine import Engine
from .lexicon import Lexicon, load_lexicon
from .loader import LoadedInput, load_jsonl, normalize_records
from .snapshot import SnapshotManager
from .util import atomic_write_json, fingerprint


class WorkbenchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ConfigMismatchError(WorkbenchError):
    def __init__(self, message: str) -> None:
        super().__init__("E_CONFIG_MISMATCH", message)


class Workbench:
    def __init__(
        self,
        *,
        events_path: str | Path,
        lexicon_path: str | Path,
        config_path: str | Path,
        out_dir: str | Path,
        snapshot_dir: str | Path,
        edits_path: str | Path | None = None,
        fault_hook: Callable[[str, str, int], None] | None = None,
    ) -> None:
        self.events_path = Path(events_path)
        self.lexicon_path = Path(lexicon_path)
        self.config_path = Path(config_path)
        self.out_dir = Path(out_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.fault_hook = fault_hook
        self.lexicon: Lexicon = load_lexicon(self.lexicon_path)
        self.base_config = load_config(self.config_path)
        self.base_loaded: LoadedInput = load_jsonl(self.events_path, self.lexicon)
        self.batches = load_edit_batches(edits_path) if edits_path else []
        self.base_config_hash = config_hash(self.base_config)
        self.base_events_fingerprint = self.base_loaded.fingerprint
        self.edits_fingerprint = fingerprint(self.batches)
        self.snapshot_manager = SnapshotManager(self.snapshot_dir)
        self.current_engine: Engine | None = None
        self.current_result: dict[str, Any] | None = None
        self.stage_history: list[dict[str, Any]] = []
        self.resumed_from: int | None = None
        self._meta: dict[str, Any] | None = None
        self._last_snapshot: dict[str, Any] | None = None
        self._started_at = 0.0

    def run(
        self,
        *,
        resume: bool = False,
        stop_after_rounds: int | None = None,
        crash_after_snapshots: int | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        self._started_at = time.perf_counter()
        snapshot = self.snapshot_manager.load_current()
        if resume:
            if snapshot is None:
                raise WorkbenchError(
                    "E_SNAPSHOT_MISSING", "no complete snapshot is available to resume"
                )
            self._validate_snapshot(snapshot)
            self._last_snapshot = snapshot
            self.resumed_from = int(snapshot["version"])
            self._meta = copy.deepcopy(snapshot["state"]["workbench"])
            self.current_engine = self._engine_from_resume_state(
                snapshot["state"], stop_requested=stop_requested
            )
            first_round_limit = None
        else:
            if snapshot is not None:
                raise WorkbenchError(
                    "E_SNAPSHOT_EXISTS",
                    "CURRENT exists; pass resume=True or choose a fresh snapshot directory",
                )
            self._initialize_meta()
            self.current_engine = self._new_engine(
                seed_sessions=None,
                mode="full" if self._meta["stage"] == "baseline" else "incremental",
                initial_failures=[],
                stop_requested=stop_requested,
            )
            first_round_limit = stop_after_rounds

        while True:
            assert self.current_engine is not None
            result = self.current_engine.run(
                stop_after_rounds=first_round_limit,
                crash_after_snapshots=crash_after_snapshots,
            )
            first_round_limit = None
            if result.get("__crash_requested__"):
                return result
            self.current_result = result
            if result["status"] != "complete":
                self._write_outputs(result)
                return result

            self._record_stage_result(result)
            next_action = self._advance_stage()
            if next_action is None:
                self._write_outputs(result)
                return result

    def current_snapshot_info(self) -> dict[str, Any] | None:
        snapshot = self.snapshot_manager.load_current()
        if snapshot is None:
            return None
        return {
            "snapshot_version": snapshot["version"],
            "directory": snapshot["directory"],
            **snapshot["manifest"],
        }

    def full_rebuild(
        self,
        *,
        records: list[dict[str, Any]],
        config: dict[str, Any],
        loader_failures: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        loaded = normalize_records(
            copy.deepcopy(records),
            self.lexicon,
            parse_failures=self.base_loaded.parse_failures,
        )
        return Engine(
            config=config,
            lexicon=self.lexicon,
            loaded=loaded,
            mode="full",
            initial_failures=loader_failures,
            fault_hook=self.fault_hook,
        ).run()

    def _initialize_meta(self) -> None:
        max_order = max(
            (int(item.get("_order", 0)) for item in self.base_loaded.records),
            default=0,
        )
        self._meta = {
            "stage": "baseline",
            "batch_index": -1,
            "records": copy.deepcopy(self.base_loaded.records),
            "current_config": copy.deepcopy(self.base_config),
            "order_counter": max_order,
            "edit_failures": [],
            "base_config_hash": self.base_config_hash,
            "base_events_fingerprint": self.base_events_fingerprint,
            "base_vocab_fingerprint": self.lexicon.fingerprint,
            "edits_fingerprint": self.edits_fingerprint,
            "completed_stages": 0,
        }

    def _validate_snapshot(self, snapshot: dict[str, Any]) -> None:
        manifest = snapshot["manifest"]
        state = snapshot["state"]
        meta = state.get("workbench")
        if not isinstance(meta, dict):
            raise ConfigMismatchError("snapshot does not contain workbench metadata")
        if int(manifest.get("schema_version", -1)) != 1:
            raise ConfigMismatchError("unsupported snapshot schema version")
        current_config_hash = config_hash(meta.get("current_config", {}))
        accepted_config_hashes = {
            meta.get("base_config_hash"),
            current_config_hash,
            manifest.get("config_hash"),
        }
        if self.base_config_hash not in accepted_config_hashes:
            raise ConfigMismatchError(
                "config hash does not match either the base or current snapshot state"
            )
        if meta.get("base_vocab_fingerprint") != self.lexicon.fingerprint:
            raise ConfigMismatchError(
                "vocabulary fingerprint does not match the snapshot manifest"
            )
        if meta.get("base_events_fingerprint") != self.base_events_fingerprint:
            raise ConfigMismatchError(
                "base event fingerprint does not match the snapshot manifest"
            )
        if meta.get("edits_fingerprint") != self.edits_fingerprint:
            raise ConfigMismatchError(
                "edit stream fingerprint does not match the snapshot manifest"
            )
        if int(manifest.get("snapshot_version", -1)) != int(snapshot["version"]):
            raise ConfigMismatchError("snapshot version pointer is inconsistent")

    def _engine_from_resume_state(
        self,
        state: dict[str, Any],
        *,
        stop_requested: Callable[[], bool] | None,
    ) -> Engine:
        engine = Engine(
            config=state["config"],
            lexicon=self.lexicon,
            loaded=self.base_loaded,
            mode=str(state.get("mode", "full")),
            snapshot_manager=self.snapshot_manager,
            resume_state=state,
            resumed_from=self.resumed_from,
            stop_requested=stop_requested,
            fault_hook=self.fault_hook,
            state_extra=self._snapshot_meta,
        )
        return engine

    def _new_engine(
        self,
        *,
        seed_sessions: dict[str, dict[str, Any]] | None,
        mode: str,
        initial_failures: list[dict[str, Any]],
        stop_requested: Callable[[], bool] | None,
    ) -> Engine:
        assert self._meta is not None
        loaded = normalize_records(
            copy.deepcopy(self._meta["records"]),
            self.lexicon,
            parse_failures=self.base_loaded.parse_failures,
        )
        previous_versions = None
        previous_snapshot_version = self.snapshot_manager.current_version
        if self.current_engine is not None:
            previous_versions = dict(self.current_engine.artifact_versions)
        engine = Engine(
            config=self._meta["current_config"],
            lexicon=self.lexicon,
            loaded=loaded,
            mode=mode,
            snapshot_manager=self.snapshot_manager,
            seed_sessions=seed_sessions,
            initial_failures=initial_failures,
            fault_hook=self.fault_hook,
            stop_requested=stop_requested,
            state_extra=self._snapshot_meta,
            artifact_versions=previous_versions,
        )
        engine.snapshot_version = previous_snapshot_version
        engine.resumed_from = self.resumed_from
        return engine

    def _snapshot_meta(self) -> dict[str, Any]:
        assert self._meta is not None
        return copy.deepcopy(self._meta)

    def _advance_stage(self) -> str | None:
        assert self._meta is not None
        assert self.current_engine is not None
        if self._meta["stage"] == "baseline":
            if not self.batches:
                return None
            next_index = 0
        else:
            next_index = int(self._meta["batch_index"]) + 1
            if next_index >= len(self.batches):
                return None

        batch = self.batches[next_index]
        records, config, edit_failures, order_counter = apply_batch(
            self._meta["records"],
            self._meta["current_config"],
            batch,
            int(self._meta["order_counter"]),
        )
        self._meta.update(
            {
                "stage": "batch",
                "batch_index": next_index,
                "batch_id": batch["batch_id"],
                "records": records,
                "current_config": config,
                "order_counter": order_counter,
                "edit_failures": edit_failures,
            }
        )
        seed = self.current_engine.export_seed()
        previous_engine = self.current_engine
        self.current_engine = self._new_engine(
            seed_sessions=seed,
            mode="incremental",
            initial_failures=edit_failures,
            stop_requested=previous_engine.stop_requested,
        )
        return batch["batch_id"]

    def _record_stage_result(self, result: dict[str, Any]) -> None:
        assert self._meta is not None
        stage = copy.deepcopy(self._meta)
        stage.pop("records", None)
        self.stage_history.append(
            {
                "stage": stage,
                "result": result,
            }
        )
        self._meta["completed_stages"] = int(self._meta["completed_stages"]) + 1

    def _write_outputs(self, result: dict[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        output = copy.deepcopy(result)
        output["orchestration"] = {
            "stage": self._meta["stage"] if self._meta else None,
            "batch_id": self._meta.get("batch_id") if self._meta else None,
            "batch_index": self._meta.get("batch_index") if self._meta else None,
            "completed_stages": self._meta.get("completed_stages") if self._meta else 0,
            "total_batches": len(self.batches),
            "resumed_from": self.resumed_from,
        }
        report = build_report(
            output,
            elapsed_seconds=time.perf_counter() - self._started_at,
            snapshot_info=self.current_snapshot_info(),
        )
        atomic_write_json(self.out_dir / "result.json", output)
        atomic_write_json(self.out_dir / "report.json", report)


def build_report(
    result: dict[str, Any],
    *,
    elapsed_seconds: float | None = None,
    snapshot_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": result.get("schema_version"),
        "status": result.get("status"),
        "mode": result.get("mode"),
        "config_hash": result.get("config_hash"),
        "vocab_fingerprint": result.get("vocab_fingerprint"),
        "snapshot_version": result.get("recovery", {}).get("snapshot_version"),
        "snapshot": snapshot_info,
        "recovery": result.get("recovery"),
        "summary_lengths": result.get("summary_lengths", []),
        "counts": result.get("counts", {}),
        "stage_timing": result.get("stage_timing", {}),
        "observed_wall_clock_ms": result.get("runtime_metrics", {}).get(
            "observed_wall_clock_ms", {}
        ),
        "elapsed_seconds": round(elapsed_seconds, 6)
        if elapsed_seconds is not None
        else None,
        "failure_count": len(result.get("failures", [])),
        "failures": result.get("failures", []),
        "orchestration": result.get("orchestration"),
    }


def error_payload(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "error",
        "code": code,
        "message": message,
        **extra,
    }
