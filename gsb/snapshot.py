from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .errors import SnapshotUnavailable, SnapshotVersionMismatch


STAGE_FILES = {
    "validation": "01_validation.json",
    "hit_scan": "02_hit_scan.json",
    "scope_settlement": "03_scope_settlement.json",
    "composition": "04_composition.json",
    "error_report": "05_error_report.json",
    "queue": "queue.json",
}


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshot_root = self.root / "snapshots"
        self.current_link = self.root / "current"

    def inspect_current_version(self) -> dict[str, Any] | None:
        manifest_path = self.current_link / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return {
            "snapshot_id": manifest.get("snapshot_id"),
            "global_version": manifest.get("global_version"),
            "config_hash": manifest.get("config_hash"),
            "stage": manifest.get("stage"),
            "status": manifest.get("status"),
        }

    def commit_state(self, state: dict[str, Any], stage: str) -> dict[str, Any]:
        if stage not in STAGE_FILES and stage != "complete":
            raise ValueError(f"unknown snapshot stage {stage}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        global_version = int(state["global_version"])
        config_hash = str(state["config_hash"])
        previous = self.inspect_current_version()
        snapshot_id = f"snap-v{global_version:04d}-{config_hash[:12]}-{stage}-{uuid.uuid4().hex[:8]}"
        target = self.snapshot_root / snapshot_id
        staging = Path(tempfile.mkdtemp(prefix=f".staging-{snapshot_id}-", dir=self.root))
        try:
            products: dict[str, str] = {}
            if "bundle" in state:
                _write_json_atomic(staging / STAGE_FILES["validation"], {"bundle": state["bundle"]})
                products["validation"] = STAGE_FILES["validation"]
            if "scans" in state:
                _write_json_atomic(staging / STAGE_FILES["hit_scan"], {"scans": state["scans"]})
                products["hit_scan"] = STAGE_FILES["hit_scan"]
            if "settlements" in state:
                _write_json_atomic(staging / STAGE_FILES["scope_settlement"], {
                    "settlements": state["settlements"],
                    "consumed_budget_points": state.get("consumed_budget_points", 0),
                })
                products["scope_settlement"] = STAGE_FILES["scope_settlement"]
            if "conclusions" in state:
                _write_json_atomic(staging / STAGE_FILES["composition"], {
                    "conclusions": state["conclusions"],
                    "ordered_conclusions": state.get("ordered_conclusions", state["conclusions"]),
                })
                products["composition"] = STAGE_FILES["composition"]
            if "errors" in state:
                _write_json_atomic(staging / STAGE_FILES["error_report"], {
                    "errors": state["errors"],
                    "error_summary": state.get("error_summary"),
                })
                products["error_report"] = STAGE_FILES["error_report"]
            _write_json_atomic(staging / STAGE_FILES["queue"], {
                "pending_items": state.get("pending_items", []),
                "pending_tickets": state.get("pending_tickets", []),
                "consumed_budget_points": state.get("consumed_budget_points", 0),
            })
            products["queue"] = STAGE_FILES["queue"]

            manifest = {
                "snapshot_id": snapshot_id,
                "status": "committed",
                "stage": stage,
                "global_version": global_version,
                "config_hash": config_hash,
                "previous_snapshot": previous,
                "files": products,
            }
            _write_json_atomic(staging / "manifest.json", manifest)
            _fsync_dir(staging)
            os.replace(staging, target)
            _fsync_dir(self.snapshot_root)

            tmp_link = self.root / f".current-{uuid.uuid4().hex}.tmp"
            relative_target = os.path.relpath(target, self.root)
            os.symlink(relative_target, tmp_link)
            os.replace(tmp_link, self.current_link)
            _fsync_dir(self.root)
            return manifest
        except BaseException:
            # Staging directories are never pointed to by `current`; leftovers are safe to ignore.
            raise

    def load_current(
        self,
        *,
        expected_global_version: int | None = None,
        expected_config_hash: str | None = None,
    ) -> dict[str, Any]:
        if not self.root.exists():
            raise SnapshotUnavailable(f"snapshot directory does not exist: {self.root}")
        manifest_path = self.current_link / "manifest.json"
        if not self.current_link.exists() or not manifest_path.exists():
            raise SnapshotUnavailable(
                f"E_SNAPSHOT_UNAVAILABLE: no committed current snapshot under {self.root}; "
                "refusing to claim recovery"
            )
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("status") != "committed":
                raise SnapshotUnavailable("current snapshot manifest is not committed")
            if expected_global_version is not None and manifest.get("global_version") != expected_global_version:
                raise SnapshotVersionMismatch(
                    f"snapshot global_version={manifest.get('global_version')} != {expected_global_version}"
                )
            if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
                raise SnapshotVersionMismatch("snapshot config_hash does not match current configuration")

            state: dict[str, Any] = {"manifest": manifest}
            snapshot_dir = manifest_path.parent
            for stage, filename in manifest.get("files", {}).items():
                path = snapshot_dir / filename
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if stage == "validation":
                    state["bundle"] = payload["bundle"]
                elif stage == "hit_scan":
                    state["scans"] = payload["scans"]
                elif stage == "scope_settlement":
                    state["settlements"] = payload["settlements"]
                    state["consumed_budget_points"] = payload.get("consumed_budget_points", 0)
                elif stage == "composition":
                    state["conclusions"] = payload["conclusions"]
                    state["ordered_conclusions"] = payload.get("ordered_conclusions")
                elif stage == "error_report":
                    state["errors"] = payload["errors"]
                    state["error_summary"] = payload.get("error_summary")
                elif stage == "queue":
                    state["pending_items"] = payload.get("pending_items", [])
                    state["pending_tickets"] = payload.get("pending_tickets", [])
                    state["consumed_budget_points"] = payload.get("consumed_budget_points", 0)
            state["global_version"] = manifest["global_version"]
            state["config_hash"] = manifest["config_hash"]
            return state
        except SnapshotVersionMismatch:
            raise
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise SnapshotUnavailable(f"E_SNAPSHOT_UNAVAILABLE: committed snapshot is unreadable: {exc}") from exc


def canonical_result_for_compare(result: dict[str, Any]) -> dict[str, Any]:
    """Project away runtime-only scheduler metadata from field-wise comparisons."""
    return {
        "global_version": result.get("global_version"),
        "config_hash": result.get("config_hash"),
        "decimal_width": result.get("decimal_width"),
        "scans": result.get("scans"),
        "settlements": result.get("settlements"),
        "conclusions": result.get("conclusions"),
        "ordered_conclusions": result.get("ordered_conclusions"),
        "errors": result.get("errors"),
        "pending_items": result.get("pending_items", []),
        "pending_tickets": result.get("pending_tickets", []),
    }
