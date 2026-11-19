"""Atomic stable-state persistence and crash recovery."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_bytes,
    fsync_directory,
    qfloat,
    read_json,
    sha256_bytes,
    sha256_obj,
)
from .errors import DomainError, E_SNAPSHOT_VERSION, E_STATE_CORRUPT


CURRENT_FORMAT = "spectral-workspace-current/v1"
SNAPSHOT_FORMAT = "spectral-workspace-snapshot/v1"
TRANSACTION_FORMAT = "spectral-workspace-transaction/v1"
CLOSED_FORMAT = "spectral-workspace-closed/v1"
SESSION_FORMAT = "spectral-workspace-session/v1"


def state_sha256(state: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(state))


def _snapshot_id(version: int) -> str:
    return f"snap-{version:012d}"


def _stage_set_hash(stages: Mapping[str, Any]) -> str:
    return sha256_obj(
        {
            stage_id: stages[stage_id].get("content_hash")
            for stage_id in sorted(stages)
        }
    )


def _safe_stage_filename(index: int, stage: Mapping[str, Any]) -> str:
    kind = str(stage.get("kind", "stage")).replace(":", "_")
    digest = str(stage.get("content_hash", "sha256:")).split(":", 1)[-1][:20]
    return f"stage-{index:04d}-{kind}-{digest}.json"


@dataclass
class LoadedSnapshot:
    state: dict[str, Any]
    manifest: dict[str, Any]
    current: dict[str, Any]
    state_bytes: bytes


@dataclass
class RecoveryInfo:
    prior_exit_state: str
    current_snapshot_version: int
    current_revision: int
    current_state_sha256: str | None
    orphaned_transactions: list[dict[str, Any]]
    cleaned_committed_transactions: list[str]
    had_session_marker: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_exit_state": self.prior_exit_state,
            "current_snapshot_version": self.current_snapshot_version,
            "current_revision": self.current_revision,
            "current_state_sha256": self.current_state_sha256,
            "orphaned_transactions": list(self.orphaned_transactions),
            "cleaned_committed_transactions": list(
                self.cleaned_committed_transactions
            ),
            "had_session_marker": self.had_session_marker,
        }


class PersistenceStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.staging_dir = self.root / "staging"
        self.orphaned_dir = self.root / "orphaned"
        self.recovery_dir = self.root / "recovery"
        self.current_path = self.root / "CURRENT"
        self.closed_path = self.root / "CLOSED.json"
        self.session_path = self.root / "SESSION_OPEN.json"

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.orphaned_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)

    def read_current_doc(self) -> dict[str, Any] | None:
        if not self.current_path.exists():
            return None
        try:
            document = read_json(self.current_path)
        except Exception as error:  # noqa: BLE001 - mapped to a business error
            raise DomainError(
                E_STATE_CORRUPT,
                "CURRENT pointer is not valid JSON",
                pointer="/CURRENT",
                details={"error": str(error)},
            ) from error
        if not isinstance(document, dict) or document.get("format") != CURRENT_FORMAT:
            raise DomainError(
                E_STATE_CORRUPT,
                "CURRENT pointer has an unsupported format",
                pointer="/CURRENT",
            )
        return document

    def load_current(self) -> LoadedSnapshot | None:
        current = self.read_current_doc()
        if current is None:
            return None
        snapshot_id = current.get("snapshot")
        if not isinstance(snapshot_id, str):
            raise DomainError(
                E_STATE_CORRUPT,
                "CURRENT pointer is missing a snapshot id",
                pointer="/CURRENT/snapshot",
            )
        snapshot_dir = self.snapshots_dir / snapshot_id
        state_path = snapshot_dir / "state.json"
        manifest_path = snapshot_dir / "manifest.json"
        if not state_path.exists() or not manifest_path.exists():
            raise DomainError(
                E_STATE_CORRUPT,
                "committed snapshot files are missing",
                pointer=f"/snapshots/{snapshot_id}",
            )
        state_bytes = state_path.read_bytes()
        actual_hash = sha256_bytes(state_bytes)
        if actual_hash != current.get("state_sha256"):
            raise DomainError(
                E_STATE_CORRUPT,
                "committed state hash does not match CURRENT",
                pointer=f"/snapshots/{snapshot_id}/state.json",
                details={"expected": current.get("state_sha256"), "actual": actual_hash},
            )
        try:
            state = json.loads(state_bytes.decode("utf-8"))
            manifest = read_json(manifest_path)
        except Exception as error:  # noqa: BLE001
            raise DomainError(
                E_STATE_CORRUPT,
                "committed snapshot JSON is invalid",
                pointer=f"/snapshots/{snapshot_id}",
                details={"error": str(error)},
            ) from error
        if not isinstance(state, dict) or not isinstance(manifest, dict):
            raise DomainError(
                E_STATE_CORRUPT,
                "committed snapshot root must be an object",
                pointer=f"/snapshots/{snapshot_id}",
            )
        if manifest.get("format") != SNAPSHOT_FORMAT:
            raise DomainError(
                E_STATE_CORRUPT,
                "snapshot manifest format is invalid",
                pointer=f"/snapshots/{snapshot_id}/manifest.json",
            )
        if manifest.get("state_sha256") != actual_hash:
            raise DomainError(
                E_STATE_CORRUPT,
                "snapshot manifest state hash is invalid",
                pointer=f"/snapshots/{snapshot_id}/manifest.json",
            )
        if current.get("manifest_sha256") != sha256_obj(manifest):
            raise DomainError(
                E_STATE_CORRUPT,
                "snapshot manifest hash does not match CURRENT",
                pointer=f"/snapshots/{snapshot_id}/manifest.json",
            )
        if manifest.get("snapshot_version") != current.get("snapshot_version"):
            raise DomainError(
                E_STATE_CORRUPT,
                "CURRENT and snapshot manifest versions differ",
                pointer=f"/snapshots/{snapshot_id}/manifest.json",
            )
        return LoadedSnapshot(
            state=state,
            manifest=manifest,
            current=current,
            state_bytes=state_bytes,
        )

    def recover(self) -> RecoveryInfo:
        self.ensure_layout()
        loaded = self.load_current()
        current = loaded.current if loaded is not None else None
        closed = None
        if self.closed_path.exists():
            try:
                closed = read_json(self.closed_path)
            except Exception:
                closed = None
        had_session_marker = self.session_path.exists()

        controlled = bool(
            loaded is not None
            and isinstance(closed, dict)
            and closed.get("format") == CLOSED_FORMAT
            and closed.get("snapshot_version") == current.get("snapshot_version")
            and closed.get("state_sha256") == current.get("state_sha256")
        )
        if loaded is None:
            prior_exit_state = "new"
        elif controlled:
            prior_exit_state = "controlled_exit"
        else:
            prior_exit_state = "forced_termination_recovered"

        orphaned: list[dict[str, Any]] = []
        cleaned_committed: list[str] = []
        for transaction_dir in sorted(
            [path for path in self.staging_dir.iterdir() if path.is_dir()],
            key=lambda path: path.name,
        ):
            txn_path = transaction_dir / "txn.json"
            txn: dict[str, Any] = {}
            if txn_path.exists():
                try:
                    loaded_txn = read_json(txn_path)
                    if isinstance(loaded_txn, dict):
                        txn = loaded_txn
                except Exception:
                    txn = {}
            committed_leftover = bool(
                current is not None
                and txn.get("state_sha256") == current.get("state_sha256")
                and txn.get("target_snapshot") == current.get("snapshot")
            )
            if committed_leftover:
                shutil.rmtree(transaction_dir, ignore_errors=True)
                cleaned_committed.append(transaction_dir.name)
                continue

            destination = self.orphaned_dir / transaction_dir.name
            if destination.exists():
                suffix = sha256_obj(
                    {"name": transaction_dir.name, "txn": txn}
                ).split(":", 1)[-1][:12]
                destination = self.orphaned_dir / f"{transaction_dir.name}-{suffix}"
            os.replace(transaction_dir, destination)
            record = {
                "transaction_id": transaction_dir.name,
                "reason": "uncommitted_staging_after_unclean_exit",
                "revision": txn.get("revision"),
                "operation_hash": txn.get("operation_hash"),
                "state_sha256": txn.get("state_sha256"),
                "stage_count": txn.get("stage_count"),
                "stage_hashes": txn.get("stage_hashes", []),
                "orphan_path": destination.name,
                "transaction_metadata_readable": bool(txn),
            }
            orphaned.append(record)
            self._append_jsonl(self.recovery_dir / "orphan-records.jsonl", record)

        if loaded is not None and prior_exit_state == "forced_termination_recovered":
            event = {
                "event": "forced_termination_recovered",
                "snapshot": current.get("snapshot"),
                "snapshot_version": current.get("snapshot_version"),
                "state_sha256": current.get("state_sha256"),
                "revision": current.get("revision"),
                "orphan_transactions": [item["transaction_id"] for item in orphaned],
            }
            self._append_jsonl(self.recovery_dir / "events.jsonl", event)

        return RecoveryInfo(
            prior_exit_state=prior_exit_state,
            current_snapshot_version=int(current.get("snapshot_version", 0))
            if current
            else 0,
            current_revision=int(current.get("revision", 0)) if current else 0,
            current_state_sha256=current.get("state_sha256") if current else None,
            orphaned_transactions=orphaned,
            cleaned_committed_transactions=cleaned_committed,
            had_session_marker=had_session_marker,
        )

    def begin_session(self, *, snapshot_version: int, state_hash: str | None, revision: int) -> None:
        self.ensure_layout()
        try:
            self.closed_path.unlink(missing_ok=True)
        except OSError:
            pass
        atomic_write_json(
            self.session_path,
            {
                "format": SESSION_FORMAT,
                "snapshot_version": snapshot_version,
                "state_sha256": state_hash,
                "revision": revision,
            },
        )

    def write_close(self, *, snapshot_version: int, state_hash: str | None, revision: int) -> None:
        atomic_write_json(
            self.closed_path,
            {
                "format": CLOSED_FORMAT,
                "closed": True,
                "snapshot_version": snapshot_version,
                "state_sha256": state_hash,
                "revision": revision,
            },
        )
        try:
            self.session_path.unlink(missing_ok=True)
        except OSError:
            pass

    def next_snapshot_version(self) -> int:
        current = self.read_current_doc()
        if current is None:
            return 1
        return int(current["snapshot_version"]) + 1

    def stage_transaction(
        self,
        state: Mapping[str, Any],
        *,
        target_snapshot_version: int,
        operation: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a complete transaction image that is not yet committed."""

        self.ensure_layout()
        digest = state_sha256(state)
        operation_hash = sha256_obj(operation) if operation is not None else None
        base_name = f"txn-{target_snapshot_version:012d}-{digest.split(':', 1)[-1][:12]}"
        transaction_dir = self.staging_dir / base_name
        counter = 2
        while transaction_dir.exists():
            transaction_dir = self.staging_dir / f"{base_name}-{counter:02d}"
            counter += 1
        transaction_dir.mkdir(parents=True, exist_ok=False)

        stages = state.get("stages", {})
        if not isinstance(stages, Mapping):
            raise DomainError(E_STATE_CORRUPT, "state.stages must be an object")
        stage_hashes: list[dict[str, str]] = []
        stage_files: list[str] = []
        for index, stage_id in enumerate(sorted(stages)):
            stage = stages[stage_id]
            filename = _safe_stage_filename(index, stage)
            atomic_write_json(transaction_dir / filename, stage)
            stage_files.append(filename)
            stage_hashes.append(
                {"stage_id": stage_id, "content_hash": str(stage.get("content_hash"))}
            )
        atomic_write_json(transaction_dir / "state.json", state)
        txn = {
            "format": TRANSACTION_FORMAT,
            "transaction_id": transaction_dir.name,
            "target_snapshot": _snapshot_id(target_snapshot_version),
            "target_snapshot_version": target_snapshot_version,
            "revision": state.get("revision"),
            "operation_hash": operation_hash,
            "state_sha256": digest,
            "stage_count": len(stage_hashes),
            "stage_hashes": stage_hashes,
            "stage_files": stage_files,
        }
        atomic_write_json(transaction_dir / "txn.json", txn)
        fsync_directory(self.staging_dir)
        return {"path": str(transaction_dir), "transaction": txn}

    def commit_staged(
        self, transaction: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        txn = transaction["transaction"]
        target_version = int(txn["target_snapshot_version"])
        state_expected_hash = str(txn["state_sha256"])
        if state_sha256(state) != state_expected_hash:
            raise DomainError(
                E_STATE_CORRUPT,
                "state changed after its transaction image was staged",
                details={"expected": state_expected_hash, "actual": state_sha256(state)},
            )
        return self._commit_snapshot(
            state,
            snapshot_version=target_version,
            operations=state.get("operations", []),
        )

    def commit_state(
        self,
        state: Mapping[str, Any],
        *,
        snapshot_version: int | None = None,
    ) -> dict[str, Any]:
        target_version = (
            self.next_snapshot_version()
            if snapshot_version is None
            else int(snapshot_version)
        )
        transaction = self.stage_transaction(
            state, target_snapshot_version=target_version
        )
        committed = self.commit_staged(transaction, state)
        shutil.rmtree(Path(transaction["path"]), ignore_errors=True)
        return committed

    def _commit_snapshot(
        self,
        state: Mapping[str, Any],
        *,
        snapshot_version: int,
        operations: Any,
    ) -> dict[str, Any]:
        self.ensure_layout()
        snapshot_id = _snapshot_id(snapshot_version)
        snapshot_dir = self.snapshots_dir / snapshot_id
        digest = state_sha256(state)
        state_data = canonical_bytes(state)
        operations_list = list(operations) if isinstance(operations, list) else []
        operations_data = b"".join(
            canonical_bytes(operation) + b"\n" for operation in operations_list
        )
        stages = state.get("stages", {}) if isinstance(state, Mapping) else {}
        if not isinstance(stages, Mapping):
            raise DomainError(E_STATE_CORRUPT, "state.stages must be an object")
        manifest = {
            "format": SNAPSHOT_FORMAT,
            "snapshot": snapshot_id,
            "snapshot_version": snapshot_version,
            "revision": int(state.get("revision", 0)),
            "state_sha256": digest,
            "operation_count": len(operations_list),
            "last_operation_hash": sha256_obj(operations_list[-1])
            if operations_list
            else None,
            "stage_count": len(stages),
            "stage_set_sha256": _stage_set_hash(stages),
        }

        if snapshot_dir.exists():
            existing_manifest_path = snapshot_dir / "manifest.json"
            existing_state_path = snapshot_dir / "state.json"
            same = False
            if existing_manifest_path.exists() and existing_state_path.exists():
                try:
                    existing_manifest = read_json(existing_manifest_path)
                    same = (
                        existing_manifest.get("state_sha256") == digest
                        and existing_manifest.get("snapshot_version") == snapshot_version
                    )
                except Exception:
                    same = False
            if not same:
                raise DomainError(
                    E_SNAPSHOT_VERSION,
                    "snapshot version already exists with different content",
                    pointer=f"/snapshots/{snapshot_id}",
                    details={"snapshot_version": snapshot_version},
                )
        else:
            temporary = self.snapshots_dir / (
                f".tmp-{snapshot_id}-{digest.split(':', 1)[-1][:12]}"
            )
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True, exist_ok=False)
            try:
                atomic_write_bytes(temporary / "state.json", state_data)
                atomic_write_bytes(temporary / "operations.jsonl", operations_data)
                atomic_write_json(temporary / "manifest.json", manifest)
                fsync_directory(temporary)
                os.replace(temporary, snapshot_dir)
                fsync_directory(self.snapshots_dir)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        current_doc = {
            "format": CURRENT_FORMAT,
            "snapshot": snapshot_id,
            "snapshot_version": snapshot_version,
            "revision": int(state.get("revision", 0)),
            "state_sha256": digest,
            "manifest_sha256": sha256_obj(manifest),
        }
        atomic_write_json(self.current_path, current_doc)
        return {
            "snapshot": snapshot_id,
            "snapshot_version": snapshot_version,
            "revision": int(state.get("revision", 0)),
            "state_sha256": digest,
            "manifest": manifest,
            "current": current_doc,
        }

    def drop_transaction(self, transaction: Mapping[str, Any]) -> None:
        shutil.rmtree(Path(transaction["path"]), ignore_errors=True)

    def load_snapshot_bundle(self, bundle_path: str | os.PathLike[str]) -> dict[str, Any]:
        bundle = Path(bundle_path)
        state_path = bundle / "state.json"
        manifest_path = bundle / "manifest.json"
        if not state_path.exists() or not manifest_path.exists():
            raise DomainError(
                E_STATE_CORRUPT,
                "snapshot bundle must contain state.json and manifest.json",
                pointer=str(bundle),
            )
        state_bytes = state_path.read_bytes()
        try:
            state = json.loads(state_bytes.decode("utf-8"))
            manifest = read_json(manifest_path)
        except Exception as error:
            raise DomainError(
                E_STATE_CORRUPT,
                "snapshot bundle JSON is invalid",
                pointer=str(bundle),
                details={"error": str(error)},
            ) from error
        if not isinstance(state, dict) or not isinstance(manifest, dict):
            raise DomainError(
                E_STATE_CORRUPT, "snapshot bundle root must be an object", pointer=str(bundle)
            )
        actual = sha256_bytes(state_bytes)
        if manifest.get("state_sha256") != actual:
            raise DomainError(
                E_STATE_CORRUPT,
                "snapshot bundle manifest does not match state bytes",
                pointer=str(manifest_path),
            )
        return {"state": state, "manifest": manifest, "state_sha256": actual}

    def import_snapshot_bundle(self, bundle_path: str | os.PathLike[str]) -> dict[str, Any]:
        bundle = self.load_snapshot_bundle(bundle_path)
        incoming_version = bundle["manifest"].get("snapshot_version")
        if (
            isinstance(incoming_version, bool)
            or not isinstance(incoming_version, int)
            or incoming_version <= 0
        ):
            raise DomainError(
                E_SNAPSHOT_VERSION,
                "incoming snapshot has an invalid version",
                pointer="/manifest/snapshot_version",
                details={"raw": incoming_version},
            )
        current = self.read_current_doc()
        if current is not None:
            current_version = int(current.get("snapshot_version", 0))
            if incoming_version < current_version:
                raise DomainError(
                    E_SNAPSHOT_VERSION,
                    "refusing to replace a newer stable state with an older snapshot",
                    pointer="/manifest/snapshot_version",
                    details={
                        "incoming": incoming_version,
                        "current": current_version,
                    },
                )
            if (
                incoming_version == current_version
                and bundle["state_sha256"] != current.get("state_sha256")
            ):
                raise DomainError(
                    E_SNAPSHOT_VERSION,
                    "same snapshot version has conflicting state hashes",
                    pointer="/manifest/snapshot_version",
                    details={
                        "incoming": bundle["state_sha256"],
                        "current": current.get("state_sha256"),
                    },
                )
            if (
                incoming_version == current_version
                and bundle["state_sha256"] == current.get("state_sha256")
            ):
                return {"changed": False, "snapshot_version": incoming_version}

        committed = self._commit_snapshot(
            bundle["state"],
            snapshot_version=incoming_version,
            operations=bundle["state"].get("operations", []),
        )
        return {"changed": True, **committed}

    def _append_jsonl(self, path: Path, record: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_bytes(record) + b"\n"
        with path.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
