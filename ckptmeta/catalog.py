from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .canonical import (
    atomic_write_bytes,
    copy_atomic_readonly,
    digest_json,
    load_json,
    pretty_json_bytes,
    read_bytes,
    sha256_hex,
)
from .config import InteropConfig
from .errors import fail
from .formats import CURRENT_VERSION, Manifest, decode_manifest, decode_manifest_data, encode_manifest
from .ledger import Ledger, verify_manifest_against_ledger
from .view import view_digest


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_DIR = ".ckptmeta"
CHECKPOINT_NAME = "upgrade-checkpoint.json"


class ForcedTermination(BaseException):
    """Raised only by explicit crash-injection options; callers must not report success."""

    def __init__(self, exit_code: int, phase: str) -> None:
        super().__init__(f"forced termination requested during {phase}")
        self.exit_code = exit_code
        self.phase = phase


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def scan_catalog(directory: str | Path) -> list[Manifest]:
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        fail("E_IO", f"catalog directory does not exist: {root}", manifest_path=str(root))
    manifests: list[Manifest] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.name):
        if not path.is_file() or path.name.startswith("_") or path.name.endswith(".bak"):
            continue
        manifests.append(decode_manifest(path))
    return manifests


def validate_chain(manifests: list[Manifest]) -> None:
    by_id: dict[str, tuple[Manifest, str]] = {}
    for manifest in manifests:
        if manifest.version != CURRENT_VERSION:
            continue
        manifest_id = manifest.derived_id()
        if manifest_id in by_id:
            fail(
                "E_FORMAT",
                f"duplicate manifest_id {manifest_id}",
                path="manifest_id",
                manifest_path=str(manifest.source_path) if manifest.source_path else None,
                details={"manifest_id": manifest_id},
            )
        by_id[manifest_id] = (manifest, view_digest(manifest.to_view()))

    for manifest in manifests:
        if manifest.version != CURRENT_VERSION or manifest.parent_manifest is None:
            continue
        parent_ref = manifest.parent_manifest
        if parent_ref["manifest_id"] == manifest.derived_id():
            fail(
                "E_CHAIN",
                "a manifest cannot reference itself as its parent",
                path="parent_manifest.manifest_id",
                manifest_path=str(manifest.source_path) if manifest.source_path else None,
                details={
                    "field": "parent_manifest.manifest_id",
                    "parent_manifest_id": parent_ref["manifest_id"],
                    "global_step": manifest.global_step,
                },
            )
        parent = by_id.get(parent_ref["manifest_id"])
        if parent is None:
            fail(
                "E_CHAIN",
                f"parent manifest {parent_ref['manifest_id']} is absent",
                path="parent_manifest.manifest_id",
                manifest_path=str(manifest.source_path) if manifest.source_path else None,
                details={
                    "field": "parent_manifest.manifest_id",
                    "parent_manifest_id": parent_ref["manifest_id"],
                    "global_step": manifest.global_step,
                },
            )
        _, actual_parent_digest = parent
        if parent_ref["digest"] != actual_parent_digest:
            fail(
                "E_CHAIN",
                "parent manifest digest does not match the referenced manifest",
                path="parent_manifest.digest",
                manifest_path=str(manifest.source_path) if manifest.source_path else None,
                details={
                    "field": "parent_manifest.digest",
                    "parent_manifest_id": parent_ref["manifest_id"],
                    "expected": actual_parent_digest,
                    "actual": parent_ref["digest"],
                    "global_step": manifest.global_step,
                },
            )

    for start in sorted(by_id):
        seen: set[str] = set()
        current_id: str | None = start
        while current_id is not None:
            if current_id in seen:
                origin = by_id[start][0]
                fail(
                    "E_CHAIN",
                    "parent manifest chain contains a cycle",
                    path="parent_manifest.manifest_id",
                    manifest_path=str(origin.source_path) if origin.source_path else None,
                    details={"field": "parent_manifest.manifest_id", "cycle_start": start},
                )
            seen.add(current_id)
            current_manifest = by_id.get(current_id)
            if current_manifest is None:
                break
            parent = current_manifest[0].parent_manifest
            current_id = parent["manifest_id"] if parent is not None else None


def verify_catalog_truth(manifests: list[Manifest], ledger: Ledger) -> int:
    count = 0
    for manifest in manifests:
        count += verify_manifest_against_ledger(
            manifest, ledger, manifest_path=str(manifest.source_path) if manifest.source_path else None
        )
    return count


@dataclass(frozen=True)
class CatalogOpenReport:
    catalog: str
    manifests: tuple[dict[str, Any], ...]
    format_versions: tuple[int, ...]
    upgrade: dict[str, Any] | None
    verified_digest_count: int
    chain_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "operation": "open_catalog",
            "catalog": self.catalog,
            "manifests": list(self.manifests),
            "format_versions": list(self.format_versions),
            "upgrade": self.upgrade,
            "verified_digest_count": self.verified_digest_count,
            "chain_valid": self.chain_valid,
        }


@dataclass(frozen=True)
class Catalog:
    directory: Path
    config: InteropConfig
    ledger: Ledger

    def __init__(self, directory: str | Path, config: InteropConfig, ledger: Ledger):
        object.__setattr__(self, "directory", Path(directory).resolve())
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "ledger", ledger)

    @property
    def checkpoint_dir(self) -> Path:
        return self.directory / CHECKPOINT_DIR

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / CHECKPOINT_NAME

    def scan(self) -> list[Manifest]:
        return scan_catalog(self.directory)

    def manifest_summary(self, manifests: list[Manifest] | None = None) -> tuple[dict[str, Any], ...]:
        records = manifests if manifests is not None else self.scan()
        summary: list[dict[str, Any]] = []
        for manifest in records:
            summary.append(
                {
                    "path": manifest.source_path.relative_to(self.directory).as_posix()
                    if manifest.source_path
                    else None,
                    "manifest_id": manifest.derived_id(),
                    "format_version": manifest.version,
                    "global_step": manifest.global_step,
                }
            )
        return tuple(summary)

    def open(
        self,
        *,
        upgrade: bool = True,
        retry_unknown: bool = False,
        fault_hook: Callable[[str, dict[str, Any]], None] | None = None,
        terminate_after_completed: int | None = None,
        terminate_after_rewrite: int | None = None,
        terminate_exit_code: int = 137,
    ) -> CatalogOpenReport:
        initial = self.scan()
        validate_chain(initial)

        upgrade_report: dict[str, Any] | None = None
        if upgrade:
            has_low_version = any(manifest.version < CURRENT_VERSION for manifest in initial)
            if has_low_version or self.checkpoint_path.exists():
                manager = UpgradeManager(self)
                upgrade_report = manager.run(
                    initial,
                    retry_unknown=retry_unknown,
                    fault_hook=fault_hook,
                    terminate_after_completed=terminate_after_completed,
                    terminate_after_rewrite=terminate_after_rewrite,
                    terminate_exit_code=terminate_exit_code,
                )

        manifests = self.scan()
        validate_chain(manifests)
        verified = verify_catalog_truth(manifests, self.ledger)
        versions = tuple(sorted({manifest.version for manifest in manifests}))
        return CatalogOpenReport(
            catalog=str(self.directory),
            manifests=self.manifest_summary(manifests),
            format_versions=versions,
            upgrade=upgrade_report,
            verified_digest_count=verified,
        )


class UpgradeManager:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.root = catalog.directory

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            fail("E_UPGRADE", "checkpoint path escapes the catalog", path="path", details={"path": relative})
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            fail("E_UPGRADE", "checkpoint path escapes the catalog", path="path", details={"path": relative})
        return resolved

    def _save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        atomic_write_bytes(self.catalog.checkpoint_path, pretty_json_bytes(checkpoint))

    def _validate_identity_collisions(self, manifests: list[Manifest]) -> None:
        v3_ids: dict[str, str] = {}
        low_ids: dict[str, str] = {}
        for manifest in manifests:
            if manifest.version == CURRENT_VERSION:
                path = self._relative(manifest.source_path) if manifest.source_path else "<memory>"
                manifest_id = manifest.derived_id()
                v3_ids[manifest_id] = path
        for manifest in manifests:
            if manifest.version == CURRENT_VERSION:
                continue
            path = self._relative(manifest.source_path) if manifest.source_path else "<memory>"
            manifest_id = manifest.derived_id()
            if manifest_id in low_ids or manifest_id in v3_ids:
                fail(
                    "E_UPGRADE",
                    "upgrade would create a duplicate manifest_id",
                    path="manifest_id",
                    manifest_path=str(manifest.source_path) if manifest.source_path else None,
                    details={
                        "field": "manifest_id",
                        "manifest_id": manifest_id,
                        "existing_path": v3_ids.get(manifest_id, low_ids.get(manifest_id)),
                        "colliding_path": path,
                    },
                )
            low_ids[manifest_id] = path

    def _load_checkpoint(self) -> dict[str, Any]:
        raw = load_json(self.catalog.checkpoint_path, code="E_UPGRADE")
        if not isinstance(raw, dict):
            fail(
                "E_UPGRADE",
                "upgrade checkpoint root must be a JSON object",
                path="$",
                manifest_path=str(self.catalog.checkpoint_path),
            )
        if raw.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            fail(
                "E_UPGRADE",
                "unsupported upgrade checkpoint schema_version",
                path="schema_version",
                manifest_path=str(self.catalog.checkpoint_path),
            )
        if raw.get("target_version") != CURRENT_VERSION:
            fail(
                "E_UPGRADE",
                "upgrade checkpoint target_version does not match current format",
                path="target_version",
                manifest_path=str(self.catalog.checkpoint_path),
            )
        if raw.get("config_digest") != self.catalog.ledger.config_digest:
            fail(
                "E_UPGRADE",
                "upgrade checkpoint config does not match the active config",
                path="config_digest",
                manifest_path=str(self.catalog.checkpoint_path),
            )
        items = raw.get("items")
        if not isinstance(items, list) or not items:
            fail(
                "E_UPGRADE",
                "upgrade checkpoint has no items",
                path="items",
                manifest_path=str(self.catalog.checkpoint_path),
            )
        immutable_plan: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for index, item in enumerate(items):
            prefix = f"items[{index}]"
            if not isinstance(item, dict):
                fail("E_UPGRADE", "checkpoint item must be an object", path=prefix)
            required = {"path", "source_version", "source_sha256", "status"}
            missing = sorted(required - set(item))
            if missing:
                fail("E_UPGRADE", "checkpoint item is incomplete", path=f"{prefix}.{missing[0]}")
            if item["status"] not in {"pending", "processing", "done", "unknown"}:
                fail("E_UPGRADE", "unknown checkpoint item status", path=f"{prefix}.status")
            if item["path"] in seen_paths:
                fail("E_UPGRADE", "duplicate checkpoint path", path=f"{prefix}.path")
            seen_paths.add(item["path"])
            immutable_plan.append(
                {
                    "path": item["path"],
                    "source_version": item["source_version"],
                    "source_sha256": item["source_sha256"],
                }
            )
        if digest_json(immutable_plan) != raw.get("plan_digest"):
            fail(
                "E_UPGRADE",
                "upgrade checkpoint plan digest does not match its items",
                path="plan_digest",
                manifest_path=str(self.catalog.checkpoint_path),
                details={"expected": raw.get("plan_digest"), "actual": digest_json(immutable_plan)},
            )
        return raw

    def _new_checkpoint(self, low_manifests: list[Manifest]) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for manifest in low_manifests:
            assert manifest.source_path is not None
            relative = self._relative(manifest.source_path)
            items.append(
                {
                    "path": relative,
                    "source_version": manifest.version,
                    "source_sha256": sha256_hex(read_bytes(manifest.source_path)),
                    "status": "pending",
                    "backup_path": None,
                    "target_sha256": None,
                    "updated_at": None,
                }
            )
        immutable_plan = [
            {
                "path": item["path"],
                "source_version": item["source_version"],
                "source_sha256": item["source_sha256"],
            }
            for item in items
        ]
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "target_version": CURRENT_VERSION,
            "config_digest": self.catalog.ledger.config_digest,
            "plan_digest": digest_json(immutable_plan),
            "created_at": _utc_now(),
            "run_state": "active",
            "items": items,
        }
        self._save_checkpoint(checkpoint)
        return checkpoint

    def _backup_path(self, item: dict[str, Any]) -> Path:
        flat = item["path"].replace("/", "__")
        return self.catalog.checkpoint_dir / "backups" / f"{flat}.v{item['source_version']}.bak"

    def _ensure_backup(
        self,
        item: dict[str, Any],
        low_source_path: Path | None,
    ) -> Path:
        backup = self._backup_path(item)
        if backup.exists():
            backup_digest = sha256_hex(read_bytes(backup))
            if backup_digest != item["source_sha256"]:
                fail(
                    "E_UPGRADE",
                    "read-only backup does not match the upgrade checkpoint",
                    path="backup_path",
                    manifest_path=str(backup),
                    details={"expected": item["source_sha256"], "actual": backup_digest},
                )
        else:
            if low_source_path is None:
                fail(
                    "E_UPGRADE",
                    "read-only backup is missing for an already-rewritten manifest",
                    path="backup_path",
                    manifest_path=str(backup),
                )
            source_bytes = read_bytes(low_source_path)
            if sha256_hex(source_bytes) != item["source_sha256"]:
                fail(
                    "E_UPGRADE",
                    "source manifest no longer matches the upgrade checkpoint",
                    path="source_sha256",
                    manifest_path=str(low_source_path),
                    details={"expected": item["source_sha256"], "actual": sha256_hex(source_bytes)},
                )
            copy_atomic_readonly(low_source_path, backup)
        item["backup_path"] = self._relative(backup)
        return backup

    def _read_low_source(self, item: dict[str, Any]) -> tuple[Manifest, Path, bool]:
        original_path = self._resolve(item["path"])
        if not original_path.exists():
            fail(
                "E_UPGRADE",
                "checkpoint source manifest is missing",
                path="path",
                manifest_path=str(original_path),
            )
        current_digest = sha256_hex(read_bytes(original_path))
        current = decode_manifest(original_path)
        if current.version < CURRENT_VERSION and current_digest == item["source_sha256"]:
            return current, original_path, True
        if item.get("backup_path"):
            backup_path = self._resolve(item["backup_path"])
            if not backup_path.exists():
                fail(
                    "E_UPGRADE",
                    "checkpoint backup is missing for an already-rewritten item",
                    path="backup_path",
                    manifest_path=str(backup_path),
                )
            backup_digest = sha256_hex(read_bytes(backup_path))
            if backup_digest != item["source_sha256"]:
                fail(
                    "E_UPGRADE",
                    "checkpoint backup digest mismatch",
                    path="backup_path",
                    manifest_path=str(backup_path),
                )
            backup_manifest = decode_manifest(backup_path)
            if backup_manifest.version != item["source_version"]:
                fail(
                    "E_UPGRADE",
                    "checkpoint backup version mismatch",
                    path="source_version",
                    manifest_path=str(backup_path),
                )
            return backup_manifest, original_path, False
        fail(
            "E_UPGRADE",
            "manifest was rewritten without a resumable checkpoint backup",
            path="backup_path",
            manifest_path=str(original_path),
            details={"current_version": current.version, "checkpoint_status": item["status"]},
        )

    def _upgraded_manifest(self, source: Manifest, original_path: Path) -> tuple[Manifest, bytes]:
        verify_manifest_against_ledger(source, self.catalog.ledger, manifest_path=str(original_path))
        truth = self.catalog.ledger.entry(source.global_step)
        record = {"format_version": CURRENT_VERSION, **truth.view}
        upgraded = decode_manifest_data(record, source_path=original_path)
        return upgraded, encode_manifest(upgraded)

    def _validate_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        for index, item in enumerate(checkpoint["items"]):
            try:
                source_path = self._resolve(item["path"])
            except BaseException as exc:
                if isinstance(exc, Exception) and getattr(exc, "code", None):
                    raise
                raise
            if item["status"] == "done":
                if not source_path.exists():
                    fail(
                        "E_UPGRADE",
                        "completed checkpoint item is missing on disk",
                        path=f"items[{index}].path",
                        manifest_path=str(source_path),
                    )
                actual_digest = sha256_hex(read_bytes(source_path))
                if actual_digest != item.get("target_sha256"):
                    fail(
                        "E_UPGRADE",
                        "completed checkpoint target digest does not match disk",
                        path=f"items[{index}].target_sha256",
                        manifest_path=str(source_path),
                        details={"expected": item.get("target_sha256"), "actual": actual_digest},
                    )
                target = decode_manifest(source_path)
                if target.version != CURRENT_VERSION:
                    fail(
                        "E_UPGRADE",
                        "completed checkpoint item is not at the target format version",
                        path=f"items[{index}].status",
                        manifest_path=str(source_path),
                    )

    def _schedule_checkpoint(self, checkpoint: dict[str, Any], low_manifests: list[Manifest]) -> None:
        planned = {item["path"] for item in checkpoint["items"]}
        for manifest in low_manifests:
            assert manifest.source_path is not None
            relative = self._relative(manifest.source_path)
            if relative not in planned:
                fail(
                    "E_UPGRADE",
                    "a low-version manifest exists outside the persisted upgrade plan",
                    path="items",
                    manifest_path=str(manifest.source_path),
                    details={"path": relative},
                )

    def _trace(self, checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for item in checkpoint["items"]:
            trace.append(
                {
                    "manifest_path": item["path"],
                    "from_version": item["source_version"],
                    "to_version": CURRENT_VERSION,
                    "status": item["status"],
                    "backup_path": item.get("backup_path"),
                }
            )
        return trace

    def run(
        self,
        initial_manifests: list[Manifest] | None = None,
        *,
        retry_unknown: bool = False,
        fault_hook: Callable[[str, dict[str, Any]], None] | None = None,
        terminate_after_completed: int | None = None,
        terminate_after_rewrite: int | None = None,
        terminate_exit_code: int = 137,
    ) -> dict[str, Any]:
        manifests = initial_manifests if initial_manifests is not None else self.catalog.scan()
        low_manifests = [manifest for manifest in manifests if manifest.version < CURRENT_VERSION]
        self._validate_identity_collisions(manifests)
        reused_checkpoint = self.catalog.checkpoint_path.exists()

        if reused_checkpoint:
            checkpoint = self._load_checkpoint()
            previous_run_report = (
                "unavailable"
                if checkpoint.get("run_state") == "forced_termination_requested"
                else None
            )
            self._validate_checkpoint_state(checkpoint)
            self._schedule_checkpoint(checkpoint, low_manifests)
        else:
            if not low_manifests:
                return {
                    "ok": True,
                    "operation": "upgrade_catalog",
                    "catalog": str(self.root),
                    "status": "noop",
                    "target_version": CURRENT_VERSION,
                    "format_versions": sorted({manifest.version for manifest in manifests}),
                    "trace": [],
                    "processed": [],
                    "skipped_done": [],
                    "unknown": [],
                    "completed": 0,
                    "pending": 0,
                    "checkpoint": None,
                    "reused_checkpoint": False,
                    "previous_run_report": None,
                }
            # Verify every source before changing any file.
            for manifest in low_manifests:
                verify_manifest_against_ledger(
                    manifest,
                    self.catalog.ledger,
                    manifest_path=str(manifest.source_path) if manifest.source_path else None,
                )
            checkpoint = self._new_checkpoint(low_manifests)
            previous_run_report = None

        marked_unknown: list[str] = []
        for item in checkpoint["items"]:
            if item["status"] == "processing":
                item["status"] = "unknown"
                item["updated_at"] = _utc_now()
                marked_unknown.append(item["path"])
        if marked_unknown:
            self._save_checkpoint(checkpoint)

        unknown_items = [item for item in checkpoint["items"] if item["status"] == "unknown"]
        if unknown_items and not retry_unknown:
            fail(
                "E_UPGRADE",
                "upgrade was interrupted before a checkpoint commit; rerun with retry_unknown",
                path="items",
                manifest_path=str(self.catalog.checkpoint_path),
                details={
                    "unknown_items": [item["path"] for item in unknown_items],
                    "requires_retry": True,
                    "previous_run_report": previous_run_report,
                },
            )

        processed: list[dict[str, Any]] = []
        skipped_done: list[str] = []
        completed_this_run = 0
        for item in checkpoint["items"]:
            if item["status"] == "done":
                skipped_done.append(item["path"])
                continue
            if item["status"] not in {"pending", "unknown"}:
                fail("E_UPGRADE", "checkpoint item cannot be resumed", path="items", details={"path": item["path"]})

            source, original_path, source_is_original_low = self._read_low_source(item)
            upgraded, target_bytes = self._upgraded_manifest(source, original_path)
            self._ensure_backup(
                item,
                original_path if source_is_original_low else None,
            )

            # The processing status is durably persisted before any rewrite. A
            # forced stop in the rewrite window becomes unknown on restart.
            item["status"] = "processing"
            item["updated_at"] = _utc_now()
            self._save_checkpoint(checkpoint)

            atomic_write_bytes(original_path, target_bytes)
            target_digest = sha256_hex(target_bytes)
            completed_this_run += 1

            if fault_hook is not None:
                fault_hook("after_rewrite", dict(item))
            if terminate_after_rewrite is not None and completed_this_run >= terminate_after_rewrite:
                sys.stdout.flush()
                sys.stderr.flush()
                raise ForcedTermination(terminate_exit_code, "after_rewrite")

            item["status"] = "done"
            item["target_sha256"] = target_digest
            item["updated_at"] = _utc_now()
            self._save_checkpoint(checkpoint)
            processed.append(
                {
                    "manifest_path": item["path"],
                    "from_version": item["source_version"],
                    "to_version": CURRENT_VERSION,
                    "status": "done",
                    "target_sha256": target_digest,
                }
            )

            if fault_hook is not None:
                fault_hook("after_checkpoint", dict(item))
            if terminate_after_completed is not None and completed_this_run >= terminate_after_completed:
                checkpoint["run_state"] = "forced_termination_requested"
                self._save_checkpoint(checkpoint)
                sys.stdout.flush()
                sys.stderr.flush()
                raise ForcedTermination(terminate_exit_code, "after_checkpoint")

        pending = [item for item in checkpoint["items"] if item["status"] in {"pending", "unknown", "processing"}]
        if pending:
            fail(
                "E_UPGRADE",
                "upgrade plan is incomplete",
                path="items",
                details={"pending_items": [item["path"] for item in pending]},
            )

        remaining_unknown = [item for item in checkpoint["items"] if item["status"] == "unknown"]
        checkpoint["run_state"] = "complete"
        self._save_checkpoint(checkpoint)
        return {
            "ok": True,
            "operation": "upgrade_catalog",
            "catalog": str(self.root),
            "status": "complete" if not reused_checkpoint else "resumed_complete",
            "target_version": CURRENT_VERSION,
            "format_versions": sorted({item["source_version"] for item in checkpoint["items"]} | {CURRENT_VERSION}),
            "trace": self._trace(checkpoint),
            "processed": processed,
            "skipped_done": skipped_done,
            "unknown": remaining_unknown,
            "completed": sum(1 for item in checkpoint["items"] if item["status"] == "done"),
            "pending": 0,
            "checkpoint": self._relative(self.catalog.checkpoint_path),
            "reused_checkpoint": reused_checkpoint,
            "previous_run_report": previous_run_report,
        }


def open_catalog(
    directory: str | Path,
    config: InteropConfig,
    ledger: Ledger,
    **kwargs: Any,
) -> CatalogOpenReport:
    return Catalog(directory, config, ledger).open(**kwargs)
