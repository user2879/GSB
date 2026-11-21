"""Record-level auto upgrade of legacy lease records with crash resumption."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .codec import LeaseRecord, decode_record, upgrade_record
from .errors import E_UPGRADE, MiddlewareError
from .spec import Spec, CURRENT_FORMAT
from .storage import Storage


@dataclass
class UpgradeReport:
    upgraded_ids: list[str]
    skipped_ids: list[str]
    checkpoint_path: str

    def to_dict(self) -> dict:
        return {
            "upgraded": self.upgraded_ids,
            "already_current": self.skipped_ids,
            "checkpoint": self.checkpoint_path,
        }


class UpgradeManager:
    """Upgrades every lease record older than r3, one atomic record at a time.

    Each successful upgrade immediately refreshes the checkpoint file, so a
    forced termination can only resume from fully durable work.  The old record
    is retained as a read-only backup inside the same SQLite row.
    """

    def __init__(self, storage: Storage, spec: Spec):
        self.storage = storage
        self.spec = spec

    def run(self) -> UpgradeReport:
        checkpoint = self.storage.load_checkpoint()
        done: list[str] = list(checkpoint.get("upgraded", []))
        hashes: dict[str, str] = dict(checkpoint.get("hashes", {}))
        self._verify_checkpoint(done, hashes)

        upgraded: list[str] = []
        skipped: list[str] = []
        for row in self.storage.list_lease_rows():
            lease_id = row["lease_id"]
            record_json = row["record_json"]
            current_hash = self.storage.record_hash(record_json)

            if lease_id in done:
                # Already upgraded before the crash; the stored row must match
                # the hash captured in the checkpoint, otherwise stop instead
                # of guessing the resume point.
                if hashes.get(lease_id) != current_hash:
                    raise MiddlewareError(
                        E_UPGRADE, "checkpoint/ledger hash mismatch; refusing to resume",
                        "checkpoint.hashes", lease_id=lease_id,
                        context={"checkpoint_hash": hashes.get(lease_id),
                                 "ledger_hash": current_hash},
                    )
                skipped.append(lease_id)
                continue

            raw = json.loads(record_json)
            if raw.get("format") == CURRENT_FORMAT:
                # Native current record: track it but do not rewrite.
                done.append(lease_id)
                hashes[lease_id] = current_hash
                self.storage.save_checkpoint({"upgraded": done, "hashes": hashes})
                skipped.append(lease_id)
                continue

            known = {r["lease_id"] for r in self.storage.list_lease_rows()}
            normalized: LeaseRecord = decode_record(raw, self.spec, known_ids=known)
            upgraded_record = upgrade_record(normalized)
            if upgraded_record.lease_version != normalized.lease_version:
                raise MiddlewareError(E_UPGRADE, "upgrade changed lease version",
                                      "lease_version", lease_id=lease_id)

            with self.storage.transaction():
                self.storage.upsert_lease(
                    upgraded_record, state=row["state"],
                    mono_deadline=row["mono_deadline"], upgraded=True,
                    backup_json=record_json,
                )
                new_row = self.storage.get_lease_row(lease_id)
                new_hash = self.storage.record_hash(new_row["record_json"])
                done.append(lease_id)
                hashes[lease_id] = new_hash
                self.storage.save_checkpoint({"upgraded": done, "hashes": hashes})
            upgraded.append(lease_id)

        return UpgradeReport(upgraded_ids=upgraded, skipped_ids=skipped,
                             checkpoint_path=str(self.storage.checkpoint_path))

    def upgrade_next(self) -> str | None:
        """Upgrade exactly one pending record; return its id, or None if done."""
        checkpoint = self.storage.load_checkpoint()
        done = list(checkpoint.get("upgraded", []))
        hashes = dict(checkpoint.get("hashes", {}))
        self._verify_checkpoint(done, hashes)
        rows = self.storage.list_lease_rows()
        known = {r["lease_id"] for r in rows}
        for row in rows:
            lease_id = row["lease_id"]
            record_json = row["record_json"]
            current_hash = self.storage.record_hash(record_json)
            if lease_id in done:
                if hashes.get(lease_id) != current_hash:
                    raise MiddlewareError(
                        E_UPGRADE, "checkpoint/ledger hash mismatch; refusing to resume",
                        "checkpoint.hashes", lease_id=lease_id,
                        context={"checkpoint_hash": hashes.get(lease_id),
                                 "ledger_hash": current_hash})
                continue
            raw = json.loads(record_json)
            if raw.get("format") == CURRENT_FORMAT:
                done.append(lease_id)
                hashes[lease_id] = current_hash
                self.storage.save_checkpoint({"upgraded": done, "hashes": hashes})
                continue
            normalized = decode_record(raw, self.spec, known_ids=known)
            upgraded_record = upgrade_record(normalized)
            with self.storage.transaction():
                self.storage.upsert_lease(
                    upgraded_record, state=row["state"],
                    mono_deadline=row["mono_deadline"], upgraded=True,
                    backup_json=record_json)
                new_row = self.storage.get_lease_row(lease_id)
                hashes[lease_id] = self.storage.record_hash(new_row["record_json"])
                done.append(lease_id)
                self.storage.save_checkpoint({"upgraded": done, "hashes": hashes})
            return lease_id
        return None

    def _verify_checkpoint(self, done: list[str], hashes: dict[str, str]) -> None:
        if sorted(done) != sorted(hashes.keys()):
            raise MiddlewareError(
                E_UPGRADE, "checkpoint upgraded list and hash table disagree",
                "checkpoint", context={"upgraded": done, "hash_keys": sorted(hashes)})
