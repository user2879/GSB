"""Staged snapshot persistence with whole-directory atomic replacement.

Each pipeline phase (``load`` / ``advance`` / ``settle``) writes its artefacts
into a fresh temporary directory and then commits via a single
``os.replace`` of a directory symlink.  Artefacts are therefore never
half-overwritten file by file, and readers either see the previous complete
version or the next complete version.

The manifest records the format version, configuration / FSM fingerprints,
consumed global budget, pending sessions and per-artefact versions.  On
startup a mismatch between manifest and requested inputs yields
``E_SNAPSHOT_STALE`` and triggers a clean full replay.
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import digest
from .models import TurnCheckpoint

SNAPSHOT_FORMAT_VERSION = 1
STAGES = ("load", "advance", "settle")


@dataclass
class SnapshotState:
    format_version: int = SNAPSHOT_FORMAT_VERSION
    global_version: int = 0
    config_hash: str = ""
    fsm_hash: str = ""
    completed_batches: int = 0
    budget_consumed: int = 0
    pending_sessions: list = field(default_factory=list)
    artifact_versions: dict = field(default_factory=dict)
    checkpoints: list = field(default_factory=list)   # checkpoint dicts
    ledger: list = field(default_factory=list)
    stage: str = ""


class SnapshotStore:
    def __init__(self, root):
        self.root = Path(root)
        self.committed = self.root / "current"
        self.versions_dir = self.root / "versions"

    # ------------------------------------------------------------------ load
    def load(self):
        manifest_path = self.committed / "manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        checkpoint_path = self.committed / "checkpoints.json"
        checkpoints = []
        if checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as handle:
                checkpoints = json.load(handle)
        return SnapshotState(
            format_version=manifest["format_version"],
            global_version=manifest["global_version"],
            config_hash=manifest["config_hash"],
            fsm_hash=manifest["fsm_hash"],
            completed_batches=manifest["completed_batches"],
            budget_consumed=manifest["budget_consumed"],
            pending_sessions=list(manifest["pending_sessions"]),
            artifact_versions=dict(manifest["artifact_versions"]),
            checkpoints=checkpoints,
            ledger=list(manifest.get("ledger", [])),
            stage=manifest.get("stage", ""),
        )

    def resume_or_reset(self, config_hash, fsm_hash):
        """Decide whether the on-disk snapshot is usable.

        Returns ``(state, cache, stale)``.  ``stale`` carries the
        ``E_SNAPSHOT_STALE`` failure dict when a reset was forced.
        """
        state = self.load()
        if state is None:
            return None, {}, None
        stale_reason = None
        if state.format_version != SNAPSHOT_FORMAT_VERSION:
            stale_reason = (f"format version {state.format_version} != "
                            f"{SNAPSHOT_FORMAT_VERSION}")
        elif state.config_hash != config_hash:
            stale_reason = "pipeline config hash changed"
        elif state.fsm_hash != fsm_hash:
            stale_reason = "state machine definition hash changed"
        if stale_reason is not None:
            failure = {
                "session_id": "<snapshot>",
                "turn": None,
                "stage": "snapshot",
                "code": "E_SNAPSHOT_STALE",
                "message": stale_reason + "; discarding snapshot and replaying from scratch",
                "batch_id": "",
                "injection_id": None,
            }
            return None, {}, failure
        cache = {cp["key"]: _checkpoint_from_dict(cp) for cp in state.checkpoints}
        return state, cache, None

    # ----------------------------------------------------------------- write
    def commit(self, state: SnapshotState):
        """Write a new immutable version and atomically swap ``current``."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        version_id = state.global_version
        temp_dir = Path(tempfile.mkdtemp(prefix=f"v{version_id}-", dir=str(self.versions_dir)))
        try:
            checkpoint_dicts = [
                cp.to_dict() if isinstance(cp, TurnCheckpoint) else cp
                for cp in state.checkpoints
            ]
            _atomic_json(temp_dir / "checkpoints.json", checkpoint_dicts)
            manifest = {
                "format_version": state.format_version,
                "global_version": state.global_version,
                "config_hash": state.config_hash,
                "fsm_hash": state.fsm_hash,
                "completed_batches": state.completed_batches,
                "budget_consumed": state.budget_consumed,
                "pending_sessions": list(state.pending_sessions),
                "artifact_versions": dict(state.artifact_versions),
                "ledger": list(state.ledger),
                "stage": state.stage,
                "checkpoints_hash": digest(checkpoint_dicts),
            }
            _atomic_json(temp_dir / "manifest.json", manifest)
            self._swap(temp_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return version_id

    def reset(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def _swap(self, new_dir: Path):
        # os.replace on a directory is atomic on POSIX when target and source
        # live on the same filesystem.
        if self.committed.is_symlink() or self.committed.exists():
            if self.committed.is_symlink():
                os.unlink(self.committed)
            else:
                shutil.rmtree(self.committed)
        os.symlink(new_dir.resolve(), self.committed)


def _atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False,
                  indent=2, separators=(",", ": "))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _checkpoint_from_dict(data):
    from .models import SlotRecord
    slots = {}
    for name, raw in data["slots"].items():
        record = SlotRecord(
            name=raw["name"], type=raw["type"], strategy=raw["strategy"],
            value=raw["value"], confidence=raw["confidence"],
            weighted_sum=raw["weighted_sum"], weight_total=raw["weight_total"],
            history=list(raw["history"]),
        )
        slots[name] = record
    return TurnCheckpoint(
        session_id=data["session_id"], turn=data["turn"], key=data["key"],
        state=data["state"], slots=slots,
        transition_trace=list(data["transition_trace"]),
        rejected=list(data["rejected"]), failures=list(data["failures"]),
        consumed_entries=data["consumed_entries"],
    )
