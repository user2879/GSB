"""Versioned snapshots, content verification and whole-state replacement."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, digest
from .errors import SnapshotError
from .generator import DeterministicUdpGenerator
from .ledger import DecisionLedger
from .models import ErrorPropagationResult, PolicySet
from .policy import (
    cascade_digest,
    load_policy,
    policy_as_dict,
    policy_fingerprint,
    shaper_digest,
)


SNAPSHOT_SCHEMA_VERSION = 1
REPLACED_FIELDS = (
    "cascade_graph",
    "error_conclusions",
    "ledger",
    "policy_set",
    "scale",
    "shaper_set",
)


def _without_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("content_digest", None)
    return result


def snapshot_content_digest(payload: Mapping[str, Any]) -> str:
    return digest(_without_digest(payload))


@dataclass(frozen=True)
class SnapshotDocument:
    path: Path | None
    payload: dict[str, Any]

    @property
    def policy_version(self) -> str:
        return str(self.payload["policy_version"])

    @property
    def content_digest(self) -> str:
        return str(self.payload["content_digest"])

    @property
    def scale(self) -> int:
        return int(self.payload["scale"])

    @property
    def policy(self) -> PolicySet:
        return load_policy(self.payload["policy"])

    @property
    def generator_config(self) -> dict[str, Any]:
        return dict(self.payload["generator"])

    @property
    def packet_sequence(self) -> list[dict[str, Any]]:
        return list(self.payload["packet_sequence"])

    @property
    def ledger(self) -> dict[str, Any]:
        return dict(self.payload["ledger"])

    @property
    def errors(self) -> dict[str, Any]:
        return dict(self.payload["error_conclusions"])

    @property
    def unknown_packet_ids(self) -> tuple[str, ...]:
        return tuple(self.payload.get("unknown_packet_ids", []))

    def verify(self) -> bool:
        if int(self.payload.get("schema_version", -1)) != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotError("unsupported snapshot schema version")
        actual = snapshot_content_digest(self.payload)
        if actual != self.content_digest:
            raise SnapshotError("snapshot content digest mismatch")
        sequence = self.packet_sequence
        if digest(sequence) != self.payload.get("packet_sequence_digest"):
            raise SnapshotError("snapshot packet-sequence digest mismatch")
        return True

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotDocument":
        file_path = Path(path)
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read snapshot {file_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SnapshotError("snapshot top level must be an object")
        document = cls(path=file_path, payload=payload)
        document.verify()
        return document

    def descriptor(self) -> dict[str, Any]:
        ledger = self.ledger
        return {
            "policy_version": self.policy_version,
            "scale": self.scale,
            "shaper_digest": self.payload["shaper_digest"],
            "cascade_digest": self.payload["cascade_digest"],
            "generator_digest": self.payload["packet_sequence_digest"],
            "ledger_digest": ledger.get("content_digest"),
            "content_digest": self.content_digest,
        }

    def replacement(self) -> dict[str, Any]:
        return {
            "policy_set": dict(self.payload["policy"]),
            "shaper_set": list(self.payload["policy"].get("shapers", [])),
            "cascade_graph": list(self.payload["policy"].get("edges", [])),
            "scale": self.scale,
            "ledger": dict(self.payload["ledger"]),
            "error_conclusions": dict(self.payload["error_conclusions"]),
        }

    def restore_ledger(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Materialize the snapshot's ledger into a new SQLite file atomically.

        The target session signature is derived entirely from the snapshot. With
        ``overwrite=True`` the old database file is replaced as a whole; rows are
        never merged into it. Callers should close any open handle on the target.
        """

        target = Path(path)
        if target.exists() and not overwrite:
            raise SnapshotError(f"target ledger already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".restore.tmp")
        for suffix in ("", "-wal", "-shm"):
            stale = Path(str(temporary) + suffix)
            if stale.exists():
                stale.unlink()
        policy = self.policy
        generator = DeterministicUdpGenerator(self.generator_config)
        try:
            with DecisionLedger(temporary) as ledger:
                ledger.initialize(policy=policy, scale=self.scale, generator=generator)
                ledger.restore_from_snapshot(self.ledger)
                ledger.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            os.replace(temporary, target)
            for suffix in ("-wal", "-shm"):
                stale = Path(str(target) + suffix)
                if stale.exists():
                    stale.unlink()
        except Exception:
            for suffix in ("", "-wal", "-shm"):
                stale = Path(str(temporary) + suffix)
                if not stale.exists():
                    continue
                try:
                    stale.unlink()
                except OSError:
                    pass
            raise
        return target


@dataclass(frozen=True)
class SnapshotRestoreResult:
    document: SnapshotDocument
    applied: bool
    version_matches: bool
    content_matches: bool
    replaced_fields: tuple[str, ...]
    backup_path: Path | None
    replacement: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "version_matches": self.version_matches,
            "content_matches": self.content_matches,
            "replaced_fields": list(self.replaced_fields),
            "backup_path": None if self.backup_path is None else str(self.backup_path),
            "descriptor": self.document.descriptor(),
        }


def current_session_descriptor(
    *,
    policy: PolicySet,
    scale: int,
    generator: DeterministicUdpGenerator,
    ledger: DecisionLedger | None = None,
) -> dict[str, Any]:
    ledger_digest = None
    if ledger is not None:
        ledger_digest = ledger.progress().content_digest
    return {
        "policy_version": policy.policy_version,
        "scale": scale,
        "shaper_digest": shaper_digest(policy),
        "cascade_digest": cascade_digest(policy),
        "generator_digest": generator.sequence_digest,
        "policy_fingerprint": policy_fingerprint(policy),
        "ledger_digest": ledger_digest,
    }


class SnapshotManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.backup_directory = self.directory / "backups"
        self.backup_directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        filename: str,
        policy: PolicySet,
        scale: int,
        generator: DeterministicUdpGenerator,
        ledger: DecisionLedger,
        errors: ErrorPropagationResult,
        indistinguishability: Iterable[Mapping[str, Any]] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> SnapshotDocument:
        ledger.verify()
        progress = ledger.progress()
        ledger_decisions = ledger.export_decisions()
        payload: dict[str, Any] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "policy_version": policy.policy_version,
            "scale": scale,
            "shaper_digest": shaper_digest(policy),
            "cascade_digest": cascade_digest(policy),
            "policy_fingerprint": policy_fingerprint(policy),
            "policy": policy_as_dict(policy),
            "generator": generator.config.as_dict(),
            "packet_sequence_digest": generator.sequence_digest,
            "packet_sequence": list(generator.manifest),
            "ledger": {
                "format_version": ledger.meta_dict().get("format_version"),
                "cursor_seq": progress.cursor_seq,
                "committed_batches": progress.committed_batches,
                "next_event_seq": progress.next_event_seq,
                "state": progress.state,
                "summary": progress.summary,
                "content_digest": progress.content_digest,
                "decisions": ledger_decisions,
                "unknown_unresolved": list(ledger.unknown_packets(unresolved_only=True)),
            },
            "error_conclusions": errors.as_dict(),
            "indistinguishability": [dict(item) for item in indistinguishability],
            "unknown_packet_ids": list(generator_unknowns(ledger)),
            "extra": dict(extra or {}),
        }
        payload["content_digest"] = snapshot_content_digest(payload)
        path = Path(filename)
        if not path.is_absolute():
            path = self.directory / path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
        document = SnapshotDocument(path=path, payload=payload)
        document.verify()
        return document

    def list_snapshots(self) -> tuple[dict[str, Any], ...]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                document = SnapshotDocument.load(path)
            except SnapshotError:
                continue
            items.append(
                {
                    "filename": path.name,
                    "policy_version": document.policy_version,
                    "scale": document.scale,
                    "content_digest": document.content_digest,
                    "cursor_seq": int(document.ledger.get("cursor_seq", 0)),
                    "committed_batches": int(
                        document.ledger.get("committed_batches", 0)
                    ),
                }
            )
        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item["policy_version"],
                    item["content_digest"],
                    item["filename"],
                ),
            )
        )

    def open(
        self,
        path: str | Path,
        *,
        current: Mapping[str, Any] | None = None,
    ) -> SnapshotRestoreResult:
        document = SnapshotDocument.load(path)
        if current is None:
            version_matches = False
            content_matches = False
        else:
            version_matches = current.get("policy_version") == document.policy_version
            if "content_digest" in current:
                content_matches = current.get("content_digest") == document.content_digest
            else:
                content_matches = all(
                    current.get(key) == document.descriptor().get(key)
                    for key in (
                        "policy_version",
                        "scale",
                        "shaper_digest",
                        "cascade_digest",
                        "generator_digest",
                        "ledger_digest",
                    )
                )
        applied = not (version_matches and content_matches)
        backup_path: Path | None = None
        if applied and current is not None:
            backup_path = self._backup_current(current, document)
        return SnapshotRestoreResult(
            document=document,
            applied=applied,
            version_matches=version_matches,
            content_matches=content_matches,
            replaced_fields=REPLACED_FIELDS if applied else (),
            backup_path=backup_path,
            replacement=document.replacement() if applied else {},
        )

    def _backup_current(
        self,
        current: Mapping[str, Any],
        replacement: SnapshotDocument,
    ) -> Path:
        descriptor = canonical_json(dict(current))
        current_digest = digest(dict(current))
        current_path_value = current.get("snapshot_path")
        if current_path_value:
            current_path = Path(str(current_path_value))
            if current_path.exists() and current_path.resolve() != (
                replacement.path.resolve() if replacement.path is not None else None
            ):
                name = (
                    "backup-"
                    + str(current.get("policy_version", "unknown"))
                    + "-"
                    + current_digest[:16]
                    + ".json"
                )
                target = self.backup_directory / name
                if not target.exists():
                    shutil.copyfile(current_path, target)
                    os.chmod(target, 0o444)
                return target
        target = self.backup_directory / f"backup-session-{current_digest[:16]}.json"
        if not target.exists():
            target.write_text(descriptor + "\n", encoding="utf-8")
            os.chmod(target, 0o444)
        return target


def generator_unknowns(ledger: DecisionLedger) -> tuple[str, ...]:
    return tuple(item["packet_id"] for item in ledger.unknown_packets(unresolved_only=True))
