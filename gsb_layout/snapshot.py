"""Atomic run snapshots and startup staleness checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .cache import atomic_write_json
from .config import Config, canonical_hash, canonical_json

SNAPSHOT_FORMAT_VERSION = 1
PIPELINE_VERSION = "gsb-layout-1.0.0"


@dataclass
class SnapshotLoad:
    manifest: dict[str, Any] | None
    stale: bool
    stale_reason: str | None
    diagnostics: list[dict[str, Any]]


class SnapshotStore:
    def __init__(self, state_root: str | Path) -> None:
        self.state_root = Path(state_root)
        self.snapshot_root = self.state_root / "snapshot"
        self.stages_root = self.snapshot_root / "stages"
        self.manifest_path = self.snapshot_root / "manifest.json"

    def load(self, config: Config) -> SnapshotLoad:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return SnapshotLoad(None, False, None, [])
        except (OSError, json.JSONDecodeError) as exc:
            diag = self._stale_diag(f"清单损坏: {exc}")
            return SnapshotLoad(None, True, "manifest_unreadable", [diag])
        diagnostics: list[dict[str, Any]] = []
        stale_reason: str | None = None
        if manifest.get("format_version") != SNAPSHOT_FORMAT_VERSION:
            stale_reason = "format_version_mismatch"
        elif manifest.get("pipeline_version") != PIPELINE_VERSION:
            stale_reason = "pipeline_version_mismatch"
        elif manifest.get("config_hash") != config.fingerprint():
            stale_reason = "config_hash_mismatch"
        if stale_reason:
            diagnostics.append(self._stale_diag(stale_reason))
        return SnapshotLoad(manifest, stale_reason is not None, stale_reason, diagnostics)

    def _stale_diag(self, reason: str) -> dict[str, Any]:
        return {
            "code": "E_SNAPSHOT_STALE",
            "severity": "warning",
            "detail": reason,
            "action": "rebuild_snapshot_from_content_cache",
        }

    def commit(
        self,
        *,
        config: Config,
        global_version: int,
        completed_images: list[str],
        image_hashes: dict[str, str],
        stage_payloads: dict[str, Any],
        report_digest: str,
        budget: dict[str, Any],
        pending_count: int,
        previous_manifest: dict[str, Any] | None,
    ) -> dict[str, Any]:
        stage_versions: dict[str, str] = {}
        for stage in ("columns", "segments", "boxes", "matches"):
            payload = stage_payloads.get(stage, {"images": []})
            stage_versions[stage] = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            atomic_write_json(self.stages_root / f"{stage}.json", payload)
        manifest = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "global_version": int(global_version),
            "config_hash": config.fingerprint(),
            "completed_images": sorted(completed_images, key=lambda s: (s.casefold(), s)),
            "image_hashes": dict(sorted(image_hashes.items())),
            "stage_versions": stage_versions,
            "budget": budget,
            "pending_count": int(pending_count),
            "report_digest": report_digest,
            "committed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def write_checkpoint(self, name: str, payload: Any) -> None:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        atomic_write_json(self.snapshot_root / "checkpoints" / f"{safe}.json", payload)

    def stage_payload(self, stage: str) -> Any | None:
        try:
            return json.loads((self.stages_root / f"{stage}.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
