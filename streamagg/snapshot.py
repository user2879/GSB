from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import digest_obj
from .engine import Engine
from .errors import FORMAT_VERSION, BusinessError
from .ledger import Ledger
from .logging_utils import JsonlLogger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SnapshotComparison:
    replacement_required: bool
    rules_version_matches: bool
    content_digest_matches: bool
    spec_digest_matches: bool
    current_rules_version: int | None
    snapshot_rules_version: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "replacement_required": self.replacement_required,
            "rules_version_matches": self.rules_version_matches,
            "content_digest_matches": self.content_digest_matches,
            "spec_digest_matches": self.spec_digest_matches,
            "current_rules_version": self.current_rules_version,
            "snapshot_rules_version": self.snapshot_rules_version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LoadedSnapshot:
    path: Path
    manifest: dict[str, Any]
    state: dict[str, Any]
    comparison: SnapshotComparison | None


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, engine: Engine, *, snapshot_id: str | None = None) -> LoadedSnapshot:
        state = engine.snapshot_state()
        content_digest = digest_obj(state)
        resolved_id = snapshot_id or f"snap-{content_digest[:20]}"
        manifest = {
            "snapshot_id": resolved_id,
            "format_version": FORMAT_VERSION,
            "created_at": _now(),
            "session_id": engine.session_id,
            "view_id": engine.view_id,
            "rules_version": engine.spec.rules_version,
            "fixed_scale": engine.spec.fixed_scale,
            "spec_digest": engine.spec.spec_digest,
            "content_digest": content_digest,
            "ledger_content_digest": engine.ledger_content_digest,
            "event_cursor": engine.event_cursor,
        }
        envelope = {"manifest": manifest, "state": state}
        path = self.root / f"{resolved_id}.json"
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
        if engine.ledger is not None:
            engine.ledger.record_snapshot(
                snapshot_id=resolved_id,
                session_id=engine.session_id,
                view_id=engine.view_id,
                spec_digest=engine.spec.spec_digest,
                content_digest=content_digest,
                ledger_content_digest=engine.ledger_content_digest,
                event_cursor=engine.event_cursor,
                rules_version=engine.spec.rules_version,
                file_path=str(path),
            )
        return LoadedSnapshot(path=path, manifest=manifest, state=state, comparison=None)

    def load(
        self,
        snapshot: str | Path,
        *,
        current_manifest: dict[str, Any] | Any | None = None,
    ) -> LoadedSnapshot:
        path = Path(snapshot)
        if not path.is_absolute():
            direct = self.root / path
            path = direct if direct.exists() else path
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot file not found",
                field="snapshot",
                details={"path": str(path)},
            )
        except json.JSONDecodeError as exc:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot file is not valid JSON",
                field="snapshot",
                details={"path": str(path), "line": exc.lineno, "column": exc.colno},
            )
        except OSError as exc:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot file cannot be read",
                field="snapshot",
                details={"path": str(path), "error": str(exc)},
            )
        if not isinstance(envelope, dict) or "manifest" not in envelope or "state" not in envelope:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot envelope is missing manifest or state",
                field="snapshot",
            )
        manifest = envelope["manifest"]
        state = envelope["state"]
        if not isinstance(manifest, dict) or not isinstance(state, dict):
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot manifest and state must be objects",
                field="snapshot",
            )
        if manifest.get("format_version") != FORMAT_VERSION:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="unsupported snapshot format version",
                field="manifest.format_version",
                details={"actual": manifest.get("format_version"), "expected": FORMAT_VERSION},
            )
        actual_digest = digest_obj(state)
        if actual_digest != manifest.get("content_digest"):
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot content digest mismatch",
                field="manifest.content_digest",
                details={
                    "expected": manifest.get("content_digest"),
                    "actual": actual_digest,
                    "path": str(path),
                },
            )
        if state.get("view_id") != manifest.get("view_id"):
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot view identifier mismatch",
                field="manifest.view_id",
            )
        if int(state.get("event_cursor", -1)) != int(manifest.get("event_cursor", -2)):
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot event cursor mismatch",
                field="manifest.event_cursor",
            )
        comparison = None
        if current_manifest is not None:
            comparison = compare_manifest(manifest, current_manifest)
        return LoadedSnapshot(path=path, manifest=manifest, state=state, comparison=comparison)


def _manifest_value(manifest: Any, field: str) -> Any:
    if isinstance(manifest, dict):
        return manifest.get(field)
    try:
        return manifest[field]
    except (KeyError, IndexError, TypeError):
        return None


def compare_manifest(snapshot_manifest: dict[str, Any], current_manifest: Any) -> SnapshotComparison:
    current_rules_version = _manifest_value(current_manifest, "rules_version")
    current_content_digest = _manifest_value(current_manifest, "content_digest")
    current_spec_digest = _manifest_value(current_manifest, "spec_digest")
    rules_match = current_rules_version == snapshot_manifest.get("rules_version")
    content_match = current_content_digest == snapshot_manifest.get("ledger_content_digest")
    spec_match = current_spec_digest == snapshot_manifest.get("spec_digest")
    differences: list[str] = []
    if not rules_match:
        differences.append("rules_version")
    if not content_match:
        differences.append("content_digest")
    if not spec_match:
        differences.append("spec_digest")
    if differences:
        reason = "snapshot differs from current session in " + ", ".join(differences)
    else:
        reason = "snapshot matches current session metadata"
    return SnapshotComparison(
        replacement_required=bool(differences),
        rules_version_matches=rules_match,
        content_digest_matches=content_match,
        spec_digest_matches=spec_match,
        current_rules_version=current_rules_version,
        snapshot_rules_version=int(snapshot_manifest.get("rules_version", 0)),
        reason=reason,
    )


def restore_engine(
    snapshot_path: str | Path,
    *,
    store: SnapshotStore,
    ledger: Ledger | None = None,
    current_manifest: Any | None = None,
    logger: JsonlLogger | None = None,
) -> tuple[Engine, LoadedSnapshot, list[dict[str, Any]]]:
    if current_manifest is None and ledger is not None:
        try:
            runtime = ledger.get_runtime(_peek_session_id(snapshot_path, store))
            current_manifest = ledger.get_view_manifest(
                runtime["session_id"], runtime["current_view_id"]
            )
        except (KeyError, BusinessError):
            current_manifest = None
    loaded = store.load(snapshot_path, current_manifest=current_manifest)
    if ledger is not None:
        try:
            snapshot_session = str(loaded.state["session_id"])
            snapshot_runtime = ledger.get_runtime(snapshot_session)
            ledger.verify_content_digest(snapshot_session, snapshot_runtime["current_view_id"])
        except KeyError:
            pass
    active_ledger = ledger
    ledger_attached = False
    if ledger is not None:
        try:
            runtime = ledger.get_runtime(loaded.state["session_id"])
            manifest_matches = loaded.comparison is None or not loaded.comparison.replacement_required
            if runtime["current_view_id"] == loaded.state["view_id"] and manifest_matches:
                active_ledger = ledger
                ledger_attached = True
            else:
                active_ledger = None
        except KeyError:
            active_ledger = None
    engine, unknown_rows = Engine.from_snapshot_state(
        loaded.state,
        ledger=active_ledger,
        logger=logger,
        mark_pending_unknown=True,
    )
    if not ledger_attached:
        engine.ledger = None
    if logger is not None:
        logger.info(
            "OK",
            "snapshot_restored",
            "snapshot restored by wholesale replacement",
            {
                "snapshot_id": loaded.manifest["snapshot_id"],
                "session_id": engine.session_id,
                "view_id": engine.view_id,
                "rules_version": engine.spec.rules_version,
                "replacement_required": (
                    loaded.comparison.replacement_required if loaded.comparison else False
                ),
                "ledger_attached": ledger_attached,
                "unknown_window_count": len(unknown_rows),
            },
        )
    return engine, loaded, unknown_rows


def _peek_session_id(snapshot_path: str | Path, store: SnapshotStore) -> str:
    path = Path(snapshot_path)
    if not path.is_absolute():
        direct = store.root / path
        path = direct if direct.exists() else path
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        return str(envelope["manifest"]["session_id"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BusinessError(
            code="E_SNAPSHOT",
            message="cannot read snapshot session identifier",
            field="manifest.session_id",
            details={"path": str(path), "error": str(exc)},
        )
