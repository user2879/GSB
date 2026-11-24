"""Transaction-directory snapshots and recovery verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import canonical_dumps, fingerprint
from .model import RuntimeState, build_state
from .diagnostics import to_jsonable


STAGES = ("recall", "dedup", "topic", "attribution", "contribution", "scoring")


@dataclass
class SnapshotData:
    state: RuntimeState
    failures: list[dict[str, Any]]
    cache: dict[str, dict[str, Any]]
    history: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    manifest: dict[str, Any]
    recovery: dict[str, Any]


def state_to_document(state: RuntimeState) -> dict[str, Any]:
    declarations: list[dict[str, Any]] = []
    sessions = []
    for session_id in state.session_order:
        turns = []
        for turn in state.sorted_turns(session_id):
            hits = state.hits.get((session_id, turn.turn_id), [])
            turns.append(
                {
                    "turn_id": turn.turn_id,
                    "seq": turn.seq,
                    "query": turn.query,
                    "channel_hits": [
                        {
                            "channel_id": hit.channel_id,
                            "evidence_id": hit.evidence_id,
                            "raw_score": hit.raw_score,
                            "mentions": hit.mentions,
                        }
                        for hit in hits
                    ],
                }
            )
        sessions.append({"session_id": session_id, "turns": turns})
    return {
        "schema_version": state.schema_version,
        "seed": state.seed,
        "budget_points": state.budget_points,
        "workers": state.workers,
        "max_evidence_per_turn": state.max_evidence_per_turn,
        "evidence": [
            {"evidence_id": item.evidence_id, "text": item.text, "weight": item.weight}
            for item in sorted(state.evidence.values(), key=lambda item: item.evidence_id)
        ],
        "channels": [
            {"channel_id": item.channel_id, "weight": item.weight}
            for item in sorted(state.channels.values(), key=lambda item: item.channel_id)
        ],
        "sessions": sessions,
        "recall_declarations": declarations,
        "topic": {"similarity_threshold": state.topic_threshold, "equal_rule": state.equal_rule},
        "scoring": {
            "distance_decay": state.distance_decay,
            "cross_topic_penalty": state.cross_topic_penalty,
            "normalization_weight": state.normalization_weight,
            "answer_threshold": state.answer_threshold,
            "error_tolerance": state.error_tolerance,
        },
        "fixed_point": {"fixed_width": f"F{state.fixed_width}", "quant_gear": state.quant_gear, "rounding": state.rounding},
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.tmp_root = self.root / "tmp"
        self.commits_root = self.root / "commits"

    def prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.commits_root.mkdir(parents=True, exist_ok=True)
        for child in self.tmp_root.glob("txn-*"):
            shutil.rmtree(child, ignore_errors=True)

    def load(self) -> SnapshotData | None:
        self.prepare()
        if not self.manifest_path.exists():
            return None
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._rebuild(f"invalid_manifest:{type(exc).__name__}")
        snapshot_dir_value = manifest.get("snapshot_dir")
        if not isinstance(snapshot_dir_value, str):
            return self._rebuild("missing_snapshot_dir")
        snapshot_dir = self.root / snapshot_dir_value
        if not snapshot_dir.is_dir():
            return self._rebuild("missing_snapshot_directory")
        for relative_path, expected_hash in manifest.get("file_hashes", {}).items():
            path = snapshot_dir / relative_path
            if not path.is_file() or _sha256_file(path) != expected_hash:
                return self._rebuild(f"hash_mismatch:{relative_path}")
        state_path = snapshot_dir / manifest.get("state_file", "state.json")
        try:
            state_doc = json.loads(state_path.read_text(encoding="utf-8"), parse_float=__import__("decimal").Decimal)
            state, failures = build_state(state_doc)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return self._rebuild(f"invalid_state:{type(exc).__name__}")
        cache: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}
        for stage in STAGES:
            stage_file = snapshot_dir / "stages" / f"{stage}.json"
            if stage_file.exists():
                try:
                    cache[stage] = json.loads(stage_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return self._rebuild(f"invalid_stage:{stage}")
        try:
            history = json.loads((snapshot_dir / "history.json").read_text(encoding="utf-8"))
            pending = json.loads((snapshot_dir / "pending.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            history, pending = [], []
        try:
            failures = json.loads((snapshot_dir / "failures.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures = []
        recovery = {
            "mode": "resume",
            "code": "I_SNAPSHOT_RESUME",
            "global_version": manifest.get("global_version"),
            "command_cursor": manifest.get("command_cursor", 0),
            "reason": "manifest and all file hashes verified",
        }
        return SnapshotData(state, list(failures), cache, history, pending, manifest, recovery)

    def _rebuild(self, reason: str) -> SnapshotData:
        return SnapshotData(
            state=RuntimeState(),
            failures=[],
            cache={stage: {} for stage in STAGES},
            history=[],
            pending=[],
            manifest={},
            recovery={"mode": "rebuild", "code": "E_SNAPSHOT_REBUILD", "reason": reason},
        )

    def commit(
        self,
        *,
        state: RuntimeState,
        cache: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
        pending: list[dict[str, Any]],
        command_cursor: int,
        base_failures: list[dict[str, Any]],
        recovery: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.prepare()
        previous = {}
        previous_manifest: dict[str, Any] | None = None
        if self.manifest_path.exists():
            try:
                previous_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                previous_dir = self.root / previous_manifest["snapshot_dir"]
                previous = previous_manifest.get("stage_versions", {})
            except Exception:
                previous_manifest = None
                previous_dir = None
        global_version = int((previous_manifest or {}).get("global_version", 0)) + 1
        txn_dir = self.tmp_root / f"txn-{global_version:06d}"
        if txn_dir.exists():
            shutil.rmtree(txn_dir)
        (txn_dir / "stages").mkdir(parents=True)

        stage_versions: dict[str, str] = {}
        stage_changed: dict[str, bool] = {}
        for stage in STAGES:
            payload = cache.get(stage, {})
            version = fingerprint(payload)
            stage_versions[stage] = version
            stage_file = txn_dir / "stages" / f"{stage}.json"
            if previous.get(stage) == version and previous_manifest is not None:
                old_file = previous_dir / "stages" / f"{stage}.json"
                if old_file.exists():
                    try:
                        os.link(old_file, stage_file)
                    except OSError:
                        shutil.copy2(old_file, stage_file)
                    stage_changed[stage] = False
                    continue
            _write_json(stage_file, to_jsonable(payload))
            stage_changed[stage] = True

        state_doc = state_to_document(state)
        _write_json(txn_dir / "state.json", state_doc)
        _write_json(txn_dir / "history.json", to_jsonable(history))
        _write_json(txn_dir / "pending.json", to_jsonable(pending))
        _write_json(txn_dir / "failures.json", to_jsonable(base_failures))

        file_hashes: dict[str, str] = {}
        for path in sorted(txn_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(txn_dir).as_posix()
                file_hashes[relative] = _sha256_file(path)

        snapshot_dir_value = f"commits/v{global_version:06d}"
        manifest = {
            "manifest_format": "dialogue-rag-snapshot/1",
            "global_version": global_version,
            "config_hash": state.config_hash or fingerprint(state_doc),
            "evidence_fingerprint": state.evidence_fingerprint(),
            "command_cursor": command_cursor,
            "stage_versions": stage_versions,
            "stage_changed": stage_changed,
            "state_file": "state.json",
            "history_file": "history.json",
            "pending_file": "pending.json",
            "failures_file": "failures.json",
            "file_hashes": file_hashes,
            "snapshot_dir": snapshot_dir_value,
            "recovery": recovery or {"mode": "fresh"},
        }
        _write_json(txn_dir / "manifest.json", manifest)
        destination = self.root / snapshot_dir_value
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        txn_dir.replace(destination)

        root_manifest = self.manifest_path.with_name("manifest.json.tmp")
        _write_json(root_manifest, manifest)
        root_manifest.replace(self.manifest_path)
        return manifest
