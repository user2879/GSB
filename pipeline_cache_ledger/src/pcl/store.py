"""Directory-backed cache store with atomic v1/v2 -> v3 open-time upgrade."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .digest import digest_file
from .entries import (
    CURRENT_ENTRY_VERSION,
    Entry,
    decode_entry,
    encode_entry,
    normalize_to_v3,
)
from .errors import CorruptEntryError, SpecError

CHECKPOINT_NAME = "upgrade_checkpoint.json"
ENTRY_SUFFIX = ".pcl"
TMP_SUFFIX = ".tmp"


class UpgradeInterrupted(Exception):
    """Raised by the crash simulation; the checkpoint is already durable."""


class CacheStore:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.cache_dir / CHECKPOINT_NAME
        self.source_versions: dict[str, int] = {}
        self.backfilled: dict[str, tuple] = {}
        self.upgrade_log: list[dict] = []
        self.checkpoint = self._load_checkpoint()

    # ------------------------------------------------------------------ paths
    def _entry_path(self, key) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)
        return self.cache_dir / f"{safe}{ENTRY_SUFFIX}"

    def scan_entry_files(self):
        return sorted(
            p for p in self.cache_dir.iterdir()
            if p.is_file() and p.suffix == ENTRY_SUFFIX
        )

    def keys(self):
        return sorted(p.name[: -len(ENTRY_SUFFIX)] for p in self.scan_entry_files())

    # -------------------------------------------------------------- checkpoint
    def _load_checkpoint(self):
        if not self.checkpoint_path.exists():
            return None
        try:
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise CorruptEntryError(
                "upgrade checkpoint unreadable", CHECKPOINT_NAME, 0
            )

    def _write_checkpoint(self, all_keys, processed, updated):
        data = {
            "checkpoint_version": 1,
            "target_version": CURRENT_ENTRY_VERSION,
            "all_keys": sorted(all_keys),
            "processed": sorted(processed),
            "updated": updated,
        }
        tmp = self.checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.checkpoint_path)
        self.checkpoint = data

    # ------------------------------------------------------------- raw access
    def write_entry(self, entry: Entry):
        """New writes always use the current format version (v3)."""
        v3 = Entry(
            key=entry.key,
            digest=entry.digest,
            version=entry.version,
            format_version=CURRENT_ENTRY_VERSION,
            size=entry.size,
            stage_id=entry.stage_id,
            refs=entry.normalized_refs(),
            compressed=entry.compressed,
            source_format=CURRENT_ENTRY_VERSION,
        )
        blob = encode_entry(v3, CURRENT_ENTRY_VERSION)
        path = self._entry_path(v3.key)
        tmp = path.with_suffix(TMP_SUFFIX)
        tmp.write_bytes(blob)
        os.replace(tmp, path)
        self.source_versions[v3.key] = CURRENT_ENTRY_VERSION
        return path

    def read_raw_entry(self, key) -> Entry:
        path = self._entry_path(key)
        if not path.exists():
            raise KeyError(key)
        return decode_entry(path.read_bytes(), key)

    def read_entry(self, key, *, size=None, stage_id=None) -> Entry:
        """Decode and normalize any historical version to in-memory v3."""
        entry = self.read_raw_entry(key)
        normalized = normalize_to_v3(entry, size=size, stage_id=stage_id)
        self.source_versions.setdefault(key, normalized.source_format)
        if normalized.backfilled:
            self.backfilled[key] = normalized.backfilled
        return normalized

    def delete_entry(self, key):
        path = self._entry_path(key)
        if path.exists():
            path.unlink()

    def verify_against_file(self, key, artifact_path):
        entry = self.read_raw_entry(key)
        actual = digest_file(artifact_path)
        if actual != entry.digest:
            from .errors import DigestMismatchError

            raise DigestMismatchError(
                "artifact digest disagrees with cache entry",
                key,
                entry.digest,
                actual,
            )
        return actual

    # ---------------------------------------------------------------- upgrade
    def open_and_upgrade(
        self, manifest=None, *, simulate_kill_after=None
    ) -> dict:
        """Upgrade every legacy entry, resuming strictly from the checkpoint."""
        scanned = {p.name[: -len(ENTRY_SUFFIX)]: p for p in self.scan_entry_files()}
        all_keys = set(scanned)

        processed: set[str] = set()
        updated: list[dict] = []
        if self.checkpoint:
            if self.checkpoint.get("target_version") != CURRENT_ENTRY_VERSION:
                raise SpecError(
                    "checkpoint targets an unknown format version",
                    CHECKPOINT_NAME,
                )
            cp_all = set(self.checkpoint.get("all_keys", ()))
            if not cp_all.issubset(all_keys):
                raise CorruptEntryError(
                    "checkpoint lists entries missing on disk",
                    CHECKPOINT_NAME,
                    0,
                )
            processed |= set(self.checkpoint.get("processed", ()))
            updated = list(self.checkpoint.get("updated", []))
            all_keys |= cp_all
            for item in updated:
                self.source_versions[item["key"]] = item["to"]
                if item.get("backfilled"):
                    self.backfilled[item["key"]] = tuple(item["backfilled"])

        pending = sorted(k for k in all_keys if k not in processed)
        upgrades_done = 0
        this_run_updates: list[dict] = []
        before_versions: dict[str, int] = {}

        for key in pending:
            path = scanned.get(key) or self._entry_path(key)
            entry = decode_entry(path.read_bytes(), key)
            before_versions[key] = entry.format_version
            self.source_versions[key] = entry.format_version
            ctx_size = None
            ctx_stage = None
            if manifest is not None and key in manifest.artifacts:
                decl = manifest.artifacts[key]
                ctx_size = decl.abs_path.stat().st_size
                ctx_stage = decl.stage_id
            normalized = normalize_to_v3(
                entry, size=ctx_size, stage_id=ctx_stage
            )
            if entry.format_version < CURRENT_ENTRY_VERSION:
                blob = encode_entry(normalized, CURRENT_ENTRY_VERSION)
                tmp = path.with_suffix(TMP_SUFFIX)
                tmp.write_bytes(blob)
                os.replace(tmp, path)
                if not any(u["key"] == key for u in updated):
                    record = {
                        "key": key,
                        "from": entry.format_version,
                        "to": CURRENT_ENTRY_VERSION,
                        "backfilled": list(normalized.backfilled),
                    }
                    updated.append(record)
                    this_run_updates.append(record)
            if normalized.backfilled:
                self.backfilled[key] = normalized.backfilled
            processed.add(key)
            self._write_checkpoint(all_keys, processed, updated)
            upgrades_done += 1
            if (
                simulate_kill_after is not None
                and upgrades_done >= simulate_kill_after
            ):
                raise UpgradeInterrupted(
                    f"simulated kill after {upgrades_done} upgrade(s)"
                )

        already_current = sorted(
            k for k in all_keys
            if decode_entry(self._entry_path(k).read_bytes(), k).format_version
            == CURRENT_ENTRY_VERSION
            and not any(u["key"] == k for u in updated)
        )
        report = {
            "total": len(all_keys),
            "processed": sorted(processed),
            "upgraded": updated,
            "already_current": already_current,
            "checkpoint": self.checkpoint_path.name,
        }
        self.upgrade_log = this_run_updates
        return report

    def upgrade_verification(self):
        """Re-scan disk and prove the processed set is fully current."""
        versions = {}
        for path in self.scan_entry_files():
            key = path.name[: -len(ENTRY_SUFFIX)]
            entry = decode_entry(path.read_bytes(), key)
            versions[key] = entry.format_version
        cp = self._load_checkpoint() or {}
        return {
            "disk_versions": versions,
            "checkpoint_processed": sorted(cp.get("processed", ())),
            "all_current": all(
                v == CURRENT_ENTRY_VERSION for v in versions.values()
            ),
            "processed_covers_disk": set(cp.get("processed", ())) == set(versions),
        }
