"""Workspace engine: operation application, incremental stages and commit flow."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

from .errors import DomainError, E_STAGE_STALE, E_STATE_CORRUPT
from .injection import injection_hash, validate_injection_config
from .model import Scene, scene_from_state
from .operations import apply_operation as apply_scene_operation
from .operations import validate_operation
from .persistence import PersistenceStore, RecoveryInfo, state_sha256
from .pipeline import (
    PIPELINE_VERSION,
    PipelineResult,
    build_pipeline,
    stage_content_hash,
    verify_against_full,
)


STATE_FORMAT = "spectral-workspace-state/v1"


def empty_state(*, injection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cleaned = validate_injection_config(injection)
    return {
        "schema": STATE_FORMAT,
        "revision": 0,
        "pipeline_version": PIPELINE_VERSION,
        "scene": None,
        "operations": [],
        "stages": {},
        "history": [],
        "run_context": {
            "injection": cleaned if injection is not None else None,
            "injection_hash": injection_hash(injection),
        },
    }


def _validate_state_shape(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or state.get("schema") != STATE_FORMAT:
        raise DomainError(
            E_STATE_CORRUPT,
            "workspace state has an unsupported schema",
            pointer="/state",
        )
    revision = state.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise DomainError(
            E_STATE_CORRUPT,
            "workspace revision must be a non-negative integer",
            pointer="/state/revision",
        )
    operations = state.get("operations")
    stages = state.get("stages")
    history = state.get("history")
    if not isinstance(operations, list):
        raise DomainError(
            E_STATE_CORRUPT, "state.operations must be an array", pointer="/state/operations"
        )
    if len(operations) != revision:
        raise DomainError(
            E_STATE_CORRUPT,
            "operation count does not match workspace revision",
            pointer="/state/operations",
            details={"revision": revision, "operation_count": len(operations)},
        )
    for index, operation in enumerate(operations, start=1):
        try:
            validate_operation(operation, expected_version=index)
        except DomainError as error:
            raise DomainError(
                E_STATE_CORRUPT,
                "persisted operation sequence is invalid",
                pointer=f"/state/operations/{index - 1}",
                details={"cause": error.to_dict()["error"]},
            ) from error
    if not isinstance(stages, Mapping):
        raise DomainError(
            E_STATE_CORRUPT, "state.stages must be an object", pointer="/state/stages"
        )
    for stage_id, stage in stages.items():
        if (
            not isinstance(stage, dict)
            or stage.get("stage_id") != stage_id
            or stage.get("content_hash") != stage_content_hash(stage)
        ):
            raise DomainError(
                E_STAGE_STALE,
                "persisted stage content hash is invalid",
                pointer=f"/state/stages/{stage_id}",
                details={"stage_id": stage_id},
            )
    if not isinstance(history, list):
        raise DomainError(
            E_STATE_CORRUPT, "state.history must be an array", pointer="/state/history"
        )
    if revision > 0 and state.get("scene") is None:
        raise DomainError(
            E_STATE_CORRUPT,
            "non-empty workspace state is missing its scene",
            pointer="/state/scene",
        )
    return state


class WorkspaceEngine:
    def __init__(
        self,
        store: PersistenceStore,
        *,
        state: dict[str, Any],
        scene: Scene | None,
        recovery: RecoveryInfo,
        snapshot_version: int,
        injection: Mapping[str, Any] | None,
    ) -> None:
        self.store = store
        self.state = state
        self.scene = scene
        self.recovery = recovery
        self.snapshot_version = snapshot_version
        self.injection = (
            validate_injection_config(injection) if injection is not None else None
        )
        self.pending_transaction: dict[str, Any] | None = None
        self.closed = False

    @property
    def revision(self) -> int:
        return int(self.state["revision"])

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        return self.state["stages"]

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.state["history"]

    @property
    def operations(self) -> list[dict[str, Any]]:
        return self.state["operations"]

    @property
    def state_hash(self) -> str:
        return state_sha256(self.state)

    def apply(
        self,
        operation: Mapping[str, Any],
        *,
        commit: bool = True,
        verify: bool = False,
    ) -> dict[str, Any]:
        validate_operation(operation, expected_version=self.revision + 1)
        if self.revision == 0 and operation["op"] != "initialize":
            raise DomainError(
                "E_OPERATION_INVALID",
                "first operation must be initialize",
                pointer="/op",
            )
        if self.revision > 0 and operation["op"] == "initialize":
            raise DomainError(
                "E_OPERATION_INVALID",
                "initialize is only valid at revision zero",
                pointer="/op",
            )

        new_scene = apply_scene_operation(self.scene, operation)
        pipeline = build_pipeline(
            new_scene,
            self.stages,
            revision=int(operation["version"]),
            injection=self.injection if self.injection else None,
        )
        stage_failure = (
            self.injection.get("stage_failure")
            if isinstance(self.injection, Mapping)
            else None
        )
        injected_stage_failure_active = bool(
            isinstance(stage_failure, Mapping)
            and stage_failure.get("operation_version") == int(operation["version"])
        )
        verification: dict[str, Any] | None = None
        if verify or injected_stage_failure_active:
            verification = verify_against_full(
                new_scene,
                pipeline.stages,
                revision=int(operation["version"]),
            )

        new_history = copy.deepcopy(self.history)
        new_history.append(
            pipeline.delta(revision=int(operation["version"]), operation=operation)
        )
        new_state = {
            "schema": STATE_FORMAT,
            "revision": int(operation["version"]),
            "pipeline_version": PIPELINE_VERSION,
            "scene": new_scene.to_dict(),
            "operations": copy.deepcopy(self.operations) + [copy.deepcopy(dict(operation))],
            "stages": pipeline.stages,
            "history": new_history,
            "run_context": {
                "injection": self.injection if self.injection else None,
                "injection_hash": injection_hash(self.injection),
            },
        }

        committed: dict[str, Any] | None = None
        transaction: dict[str, Any] | None = None
        if commit:
            target_version = self.snapshot_version + 1
            transaction = self.store.stage_transaction(
                new_state,
                target_snapshot_version=target_version,
                operation=operation,
            )
            committed = self.store.commit_staged(transaction, new_state)
            self.store.drop_transaction(transaction)
            self.snapshot_version = target_version
            self.pending_transaction = None
        else:
            target_version = self.snapshot_version + 1
            transaction = self.store.stage_transaction(
                new_state,
                target_snapshot_version=target_version,
                operation=operation,
            )
            self.pending_transaction = transaction

        self.state = new_state
        self.scene = new_scene
        result = {
            "operation": copy.deepcopy(dict(operation)),
            "committed": commit,
            "revision": self.revision,
            "snapshot_version": self.snapshot_version,
            "state_sha256": state_sha256(new_state),
            "delta": copy.deepcopy(new_history[-1]),
            "verification": verification,
            "transaction_id": transaction["transaction"]["transaction_id"]
            if transaction
            else None,
        }
        if committed is not None:
            result["commit"] = committed
        return result

    def commit_pending(self) -> dict[str, Any]:
        if self.pending_transaction is None:
            raise DomainError(
                "E_OPERATION_INVALID", "workspace has no pending transaction"
            )
        committed = self.store.commit_staged(self.pending_transaction, self.state)
        self.snapshot_version = int(committed["snapshot_version"])
        self.store.drop_transaction(self.pending_transaction)
        self.pending_transaction = None
        return committed

    def verify(self) -> dict[str, Any]:
        if self.scene is None:
            return {"status": "empty", "stage_count": 0, "max_abs_diff": 0.0}
        return verify_against_full(
            self.scene, self.stages, revision=self.revision
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.pending_transaction is not None:
            raise DomainError(
                "E_OPERATION_INVALID",
                "cannot close cleanly while a transaction is uncommitted",
            )
        self.store.write_close(
            snapshot_version=self.snapshot_version,
            state_hash=self.state_hash if self.revision > 0 else None,
            revision=self.revision,
        )
        self.closed = True


def open_workspace(
    root: str | os.PathLike[str],
    *,
    injection: Mapping[str, Any] | None = None,
    begin_session: bool = True,
) -> WorkspaceEngine:
    store = PersistenceStore(root)
    store.ensure_layout()
    recovery = store.recover()
    loaded = store.load_current()

    if loaded is None:
        state = empty_state(injection=injection)
        scene = None
        snapshot_version = 0
    else:
        state = _validate_state_shape(copy.deepcopy(loaded.state))
        scene = scene_from_state(state["scene"])
        snapshot_version = int(loaded.current["snapshot_version"])

        if scene is not None:
            reconciliation = build_pipeline(
                scene,
                state["stages"],
                revision=int(state["revision"]),
            )
            if reconciliation.recomputed:
                first = reconciliation.recomputed[0]["stage_id"]
                raise DomainError(
                    E_STAGE_STALE,
                    "stable snapshot requires stage recomputation during recovery",
                    pointer=f"/stages/{first}",
                    details={"stage_id": first, "recomputed_count": len(reconciliation.recomputed)},
                )
            verify_against_full(scene, state["stages"], revision=int(state["revision"]))

    engine = WorkspaceEngine(
        store,
        state=state,
        scene=scene,
        recovery=recovery,
        snapshot_version=snapshot_version,
        injection=injection,
    )
    if begin_session:
        store.begin_session(
            snapshot_version=snapshot_version,
            state_hash=engine.state_hash if engine.revision > 0 else None,
            revision=engine.revision,
        )
    return engine


def open_existing_workspace(
    root: str | os.PathLike[str], *, injection: Mapping[str, Any] | None = None
) -> WorkspaceEngine:
    if not Path(root).exists():
        raise DomainError(
            E_STATE_CORRUPT,
            "workspace does not exist",
            pointer=str(root),
        )
    return open_workspace(root, injection=injection)
