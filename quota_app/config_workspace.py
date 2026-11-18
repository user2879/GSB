from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .codec import CURRENT_FORMAT_VERSION, load_document, peek_version, write_document
from .derive import resolve_effective
from .errors import AppError, ConfigFormatError, E_FORMAT, E_UPGRADE, UpgradeError
from .models import ConfigDocument
from .utils import (
    atomic_write_json,
    make_read_only,
    sha256_file,
    sha256_text,
    utc_now,
)


CHECKPOINT_FILENAME = "upgrade_checkpoint.json"
BACKUP_DIRNAME = "backups"


@dataclass
class UpgradeEntryReport:
    filename: str
    status: str
    from_version: int | None = None
    to_version: int = CURRENT_FORMAT_VERSION
    backup: str | None = None
    message: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "filename": self.filename,
            "status": self.status,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "backup": self.backup,
        }
        if self.message:
            result["message"] = self.message
        if self.error_code:
            result["error_code"] = self.error_code
        return result


@dataclass
class UpgradeReport:
    checkpoint_path: str
    checkpoint_present: bool
    target_version: int = CURRENT_FORMAT_VERSION
    entries: list[UpgradeEntryReport] = field(default_factory=list)
    errors: list[AppError] = field(default_factory=list)
    checkpoint: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> list[UpgradeEntryReport]:
        return [entry for entry in self.entries if entry.status == "upgraded"]

    @property
    def skipped(self) -> list[UpgradeEntryReport]:
        return [
            entry
            for entry in self.entries
            if entry.status in {"already_completed", "no_upgrade_needed", "recovered_completion"}
        ]

    @property
    def failed(self) -> list[UpgradeEntryReport]:
        return [entry for entry in self.entries if entry.status == "failed"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_present": self.checkpoint_present,
            "target_version": self.target_version,
            "entries": [entry.as_dict() for entry in self.entries],
            "errors": [error.as_dict() for error in self.errors],
        }


@dataclass
class ConfigWorkspaceResult:
    directory: str
    documents: list[ConfigDocument]
    active: ConfigDocument | None
    upgrade: UpgradeReport
    format_errors: list[AppError] = field(default_factory=list)

    @property
    def errors(self) -> list[AppError]:
        return list(self.upgrade.errors) + list(self.format_errors)

    def require_active(self) -> ConfigDocument:
        if self.active is None:
            if self.format_errors:
                raise self.format_errors[0]
            raise ConfigFormatError(
                "no usable configuration document was found",
                field_path=self.directory,
            )
        return self.active


class ConfigWorkspace:
    def __init__(
        self,
        directory: str | Path,
        *,
        target_version: int = CURRENT_FORMAT_VERSION,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.target_version = target_version
        self.checkpoint_path = self.directory / CHECKPOINT_FILENAME
        self.backup_directory = self.directory / BACKUP_DIRNAME
        self.checkpoint_callback = checkpoint_callback
        self.report = UpgradeReport(
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_present=self.checkpoint_path.exists(),
        )
        self.checkpoint: dict[str, Any] = {}
        self.documents: list[ConfigDocument] = []
        self.active: ConfigDocument | None = None

    def open(self) -> ConfigWorkspaceResult:
        if not self.directory.exists():
            raise ConfigFormatError(
                "configuration directory does not exist",
                field_path=str(self.directory),
            )
        if not self.directory.is_dir():
            raise ConfigFormatError(
                "configuration path is not a directory",
                field_path=str(self.directory),
            )

        config_paths = self._config_paths()
        if not config_paths:
            raise ConfigFormatError(
                "configuration directory contains no JSON entries",
                field_path=str(self.directory),
            )

        versions: dict[Path, int] = {}
        format_errors: list[AppError] = []
        for path in config_paths:
            try:
                versions[path] = peek_version(path)
            except ConfigFormatError as exc:
                if exc.field_path in {"$.format_version", "$.version", "$.schema_version"}:
                    # Version identification is a format-level failure even
                    # while scanning a directory.  It must not be downgraded to
                    # a skippable legacy-entry problem.
                    self.report.errors.append(exc)
                    format_errors.append(exc)
                    self.report.entries.append(
                        UpgradeEntryReport(
                            filename=path.name,
                            status="failed",
                            message=exc.message,
                            error_code=E_FORMAT,
                        )
                    )
                    continue
                # The explicit single-file loader reports E_FORMAT.  During a
                # directory upgrade scan, an unreadable legacy entry is an
                # upgrade-level problem (E_UPGRADE) and must not abort others.
                upgrade_error = UpgradeError(
                    f"cannot inspect configuration entry: {exc.message}",
                    field_path=str(path),
                    details={"filename": path.name, "original_code": exc.code},
                )
                self.report.errors.append(upgrade_error)
                self.report.entries.append(
                    UpgradeEntryReport(
                        filename=path.name,
                        status="failed",
                        message=upgrade_error.message,
                        error_code=E_UPGRADE,
                    )
                )

        pending_low = sorted(
            path for path, version in versions.items() if version < self.target_version
        )
        checkpoint_state = self._load_checkpoint(required=bool(pending_low))
        if checkpoint_state == "missing":
            self.report.checkpoint_present = False
            self.report.errors.append(
                UpgradeError(
                    "upgrade checkpoint is missing while legacy entries remain",
                    field_path=str(self.checkpoint_path),
                    details={"pending": [path.name for path in pending_low]},
                )
            )
        elif checkpoint_state == "invalid":
            self.report.checkpoint_present = True
        elif checkpoint_state == "ok":
            self.report.checkpoint_present = True

        checkpoint_usable = checkpoint_state == "ok"
        if pending_low and checkpoint_usable:
            for path in pending_low:
                self._upgrade_entry(path, versions[path])
        elif pending_low and not checkpoint_usable:
            for path in pending_low:
                self.report.entries.append(
                    UpgradeEntryReport(
                        filename=path.name,
                        status="blocked",
                        from_version=versions[path],
                        message="upgrade blocked because checkpoint is missing or invalid",
                        error_code=E_UPGRADE,
                    )
                )

        # Successful in-place upgrades use the same path but are now v3.  The
        # pre-upgrade version map must be refreshed before loading documents.
        for path in pending_low:
            try:
                if peek_version(path) == self.target_version:
                    versions[path] = self.target_version
            except ConfigFormatError:
                pass

        self._reconcile_interrupted_upgrades(versions)

        # Surface already-completed checkpoint entries on a later open.  They
        # are intentionally not loaded or rewritten again.
        reported_names = {entry.filename for entry in self.report.entries}
        checkpoint_entries = self.checkpoint.get("entries", {}) if self.checkpoint else {}
        for path in config_paths:
            if versions.get(path) != self.target_version or path.name in reported_names:
                continue
            completed = checkpoint_entries.get(path.name)
            if not isinstance(completed, dict) or completed.get("status") != "completed":
                continue
            expected_hash = completed.get("upgraded_sha256")
            if expected_hash and sha256_file(path) == expected_hash:
                self.report.entries.append(
                    UpgradeEntryReport(
                        filename=path.name,
                        status="already_completed",
                        from_version=completed.get("from_version"),
                        backup=completed.get("backup"),
                    )
                )

        if checkpoint_state == "missing" and not pending_low:
            self._initialize_checkpoint()
            self.report.checkpoint_present = True

        # Load current-version documents only after all safe upgrades have
        # completed.  A malformed v3 document invalidates the active view.
        for path in config_paths:
            current_version = versions.get(path)
            if current_version is None:
                continue
            if current_version < self.target_version:
                continue
            try:
                document = load_document(path)
            except ConfigFormatError as exc:
                self.report.errors.append(exc)
                format_errors.append(exc)
                continue
            self.documents.append(document)

        self.documents.sort(key=lambda doc: (doc.revision, doc.source_path or doc.name))
        if not format_errors and self.documents:
            self.active = max(
                self.documents,
                key=lambda doc: (doc.revision, doc.source_path or doc.name),
            )

        result = ConfigWorkspaceResult(
            directory=str(self.directory),
            documents=list(self.documents),
            active=self.active,
            upgrade=self.report,
            format_errors=format_errors,
        )
        return result

    def _config_paths(self) -> list[Path]:
        paths: list[Path] = []
        for path in sorted(self.directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() != ".json":
                continue
            if path.name == CHECKPOINT_FILENAME:
                continue
            if path.name.startswith("."):
                continue
            paths.append(path)
        return paths

    def _load_checkpoint(self, *, required: bool) -> str:
        if not self.checkpoint_path.exists():
            if required:
                self.checkpoint = {}
                return "missing"
            return "missing"
        try:
            raw = json.loads(self.checkpoint_path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or not isinstance(raw.get("entries", {}), dict):
                raise ValueError("checkpoint must contain an entries object")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.checkpoint = {}
            self.report.errors.append(
                UpgradeError(
                    f"upgrade checkpoint is unreadable: {exc}",
                    field_path=str(self.checkpoint_path),
                )
            )
            return "invalid"
        raw.setdefault("entries", {})
        raw.setdefault("checkpoint_version", 1)
        raw.setdefault("target_schema_version", self.target_version)
        self.checkpoint = raw
        return "ok"

    def _initialize_checkpoint(self) -> None:
        self.checkpoint = {
            "checkpoint_version": 1,
            "target_schema_version": self.target_version,
            "entries": {},
            "updated_at": utc_now(),
        }
        self._write_checkpoint()

    def _write_checkpoint(self) -> None:
        self.checkpoint["updated_at"] = utc_now()
        atomic_write_json(self.checkpoint_path, self.checkpoint)
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(dict(self.checkpoint))

    def _backup_path(self, path: Path, version: int) -> Path:
        return self.backup_directory / f"{path.stem}.v{version}.bak"

    def _upgrade_entry(self, path: Path, from_version: int) -> None:
        entries = self.checkpoint.setdefault("entries", {})
        completed = entries.get(path.name)
        if isinstance(completed, dict) and completed.get("status") == "completed":
            expected_hash = completed.get("upgraded_sha256")
            actual_hash = sha256_file(path)
            if expected_hash and actual_hash == expected_hash:
                self.report.entries.append(
                    UpgradeEntryReport(
                        filename=path.name,
                        status="already_completed",
                        from_version=from_version,
                        backup=completed.get("backup"),
                    )
                )
                return
            error = UpgradeError(
                "checkpoint says entry is upgraded but its digest differs",
                field_path=str(path),
                details={"filename": path.name, "expected": expected_hash, "actual": actual_hash},
            )
            self.report.errors.append(error)
            self.report.entries.append(
                UpgradeEntryReport(
                    filename=path.name,
                    status="failed",
                    from_version=from_version,
                    message=error.message,
                    error_code=E_UPGRADE,
                )
            )
            return

        try:
            original_bytes = path.read_bytes()
            document = load_document(path)
        except (OSError, ConfigFormatError) as exc:
            message = exc.message if isinstance(exc, ConfigFormatError) else str(exc)
            upgrade_error = UpgradeError(
                f"cannot upgrade entry {path.name}: {message}",
                field_path=str(path),
                details={
                    "filename": path.name,
                    "original_code": exc.code if isinstance(exc, AppError) else E_FORMAT,
                },
            )
            self.report.errors.append(upgrade_error)
            self.report.entries.append(
                UpgradeEntryReport(
                    filename=path.name,
                    status="failed",
                    from_version=from_version,
                    message=upgrade_error.message,
                    error_code=E_UPGRADE,
                )
            )
            entries[path.name] = {
                "status": "failed",
                "failed_at": utc_now(),
                "message": message,
            }
            self._write_checkpoint()
            return

        if document.revision < 1:
            upgrade_error = UpgradeError(
                f"cannot upgrade entry {path.name}: invalid revision",
                field_path=str(path),
            )
            self.report.errors.append(upgrade_error)
            return

        self.backup_directory.mkdir(parents=True, exist_ok=True)
        backup = self._backup_path(path, from_version)
        original_hash = sha256_text(original_bytes.decode("utf-8-sig"))
        if backup.exists():
            existing_hash = sha256_file(backup)
            if existing_hash != sha256_text(original_bytes.decode("utf-8-sig")):
                upgrade_error = UpgradeError(
                    "existing read-only backup does not match legacy entry",
                    field_path=str(backup),
                    details={"filename": path.name, "backup": str(backup)},
                )
                self.report.errors.append(upgrade_error)
                self.report.entries.append(
                    UpgradeEntryReport(
                        filename=path.name,
                        status="failed",
                        from_version=from_version,
                        message=upgrade_error.message,
                        error_code=E_UPGRADE,
                    )
                )
                return
        else:
            atomic_write_bytes_compat(backup, original_bytes)
            make_read_only(backup)

        write_document(path, document)
        self._maybe_force_exit(path.name)
        upgraded_hash = sha256_file(path)
        relative_backup = str(backup.relative_to(self.directory)).replace(os.sep, "/")
        entries[path.name] = {
            "status": "completed",
            "from_version": from_version,
            "to_version": self.target_version,
            "original_sha256": original_hash,
            "upgraded_sha256": upgraded_hash,
            "backup": relative_backup,
            "completed_at": utc_now(),
        }
        self._write_checkpoint()
        self.report.entries.append(
            UpgradeEntryReport(
                filename=path.name,
                status="upgraded",
                from_version=from_version,
                backup=relative_backup,
            )
        )

    def _reconcile_interrupted_upgrades(self, versions: dict[Path, int]) -> None:
        """Resume when a process died after atomic file replacement."""

        entries = self.checkpoint.get("entries", {}) if self.checkpoint else {}
        changed = False
        for path, version in sorted(versions.items(), key=lambda item: item[0].name):
            if version != self.target_version or path.name in entries:
                continue
            # A legacy backup with the same stem is evidence that the file
            # replacement completed before the checkpoint update.
            candidates = sorted(self.backup_directory.glob(f"{path.stem}.v*.bak"))
            if not candidates:
                continue
            backup = candidates[-1]
            try:
                backup_raw = json.loads(backup.read_text(encoding="utf-8-sig"))
                backup_version, _ = _version_from_raw(backup_raw)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            entries[path.name] = {
                "status": "completed",
                "from_version": backup_version,
                "to_version": self.target_version,
                "original_sha256": sha256_file(backup),
                "upgraded_sha256": sha256_file(path),
                "backup": str(backup.relative_to(self.directory)).replace(os.sep, "/"),
                "completed_at": utc_now(),
                "recovered_after_interruption": True,
            }
            changed = True
            self.report.entries.append(
                UpgradeEntryReport(
                    filename=path.name,
                    status="recovered_completion",
                    from_version=backup_version,
                    backup=str(backup.relative_to(self.directory)).replace(os.sep, "/"),
                    message="recovered completion from atomic file before checkpoint update",
                )
            )
        if changed:
            self.checkpoint["entries"] = entries
            self._write_checkpoint()

    def _maybe_force_exit(self, filename: str) -> None:
        requested = os.environ.get("QUOTA_UPGRADE_KILL_AFTER")
        if not requested:
            return
        if requested in {"1", "all", filename}:
            os._exit(86)

    def next_revision(self, minimum: int = 0) -> int:
        revisions = [document.revision for document in self.documents]
        return max([minimum, *revisions]) + 1

    def save_active(self, document: ConfigDocument, *, minimum_revision: int = 0) -> ConfigDocument:
        active = self.active
        if active is None:
            raise ConfigFormatError("there is no active configuration to replace", field_path="active")
        if active.source_path is None:
            raise ConfigFormatError("active configuration has no local source path", field_path="active")
        if document.revision <= max(active.revision, minimum_revision):
            raise ConfigFormatError(
                "configuration revision must strictly increase",
                field_path="$.revision",
                details={
                    "revision": document.revision,
                    "active_revision": active.revision,
                    "ledger_revision": minimum_revision,
                },
            )

        resolve_effective(document)

        target = Path(active.source_path)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        previous_backup = self.backup_directory / f"{target.stem}.rev{active.revision}.bak"
        if not previous_backup.exists() and target.exists():
            atomic_write_bytes_compat(previous_backup, target.read_bytes())
            make_read_only(previous_backup)
        write_document(target, document)

        replacements = [doc for doc in self.documents if doc.source_path != str(target)]
        replacements.append(document)
        replacements.sort(key=lambda doc: (doc.revision, doc.source_path or doc.name))
        self.documents = replacements
        self.active = document
        return document


def atomic_write_bytes_compat(path: Path, data: bytes) -> None:
    # Local import avoids making the public module signature depend on utils in
    # callers that only need workspace discovery.
    from .utils import atomic_write_bytes

    atomic_write_bytes(path, data)


def _version_from_raw(raw: Any) -> tuple[int, str]:
    for key in ("format_version", "version", "schema_version"):
        if isinstance(raw, dict) and key in raw:
            value = raw[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("version is not an integer")
            return value, key
    raise ValueError("version is missing")


def open_workspace(
    directory: str | Path,
    *,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ConfigWorkspaceResult:
    return ConfigWorkspace(directory, checkpoint_callback=checkpoint_callback).open()
