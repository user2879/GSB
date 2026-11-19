"""Manifest format-version handling, full replacement and versioned audit."""
from __future__ import annotations

import json
import os
import stat
from typing import Any, Optional

from .bitmap import Bitmap
from .errors import DomainError, E_INPUT
from .manager import manifest_digest, now_iso
from .volume import block_digest


class ManifestService:
    def __init__(self, manager: Any) -> None:
        self.m = manager

    # -- full replacement ----------------------------------------------
    def replace_manifest(
        self,
        volume_name: str,
        new_format_version: int,
        entries: list[dict[str, Any]],
        *,
        claimed_digest: Optional[str] = None,
    ) -> dict[str, Any]:
        """Atomically replace every snapshot entry and bitmap.

        ``entries`` is the full new entry set (never a delta).  Each entry has
        ``snapshot_id, name, version_no, parent_id, bitmap_hex, refs`` where
        ``refs`` maps block number -> block hash and the block objects already
        exist in the content store.
        """
        if not isinstance(new_format_version, int) or new_format_version <= 0:
            raise DomainError(E_INPUT, "format_version must be a positive integer", field="format_version")
        row = self.m._require_volume_row(volume_name)
        block_count = int(row["block_count"])
        old_manifest = self.m.store.get_manifest(volume_name)
        old_rows = self.m.store.list_snapshots(volume_name)
        old_refs = {r["snapshot_id"]: self.m.store.ref_map(volume_name, r["snapshot_id"]) for r in old_rows}
        old_digest = old_manifest["content_digest"] if old_manifest else None
        old_version = int(old_manifest["format_version"]) if old_manifest else None

        normalized = self._normalize_entries(volume_name, block_count, entries)
        recomputed = manifest_digest(normalized, {r["snapshot_id"]: r["_refs"] for r in normalized})
        if claimed_digest is not None and claimed_digest != recomputed:
            raise DomainError(
                E_INPUT,
                "claimed content digest does not match the supplied entry set",
                field="content_digest",
                extra={"claimed": claimed_digest, "recomputed": recomputed},
            )

        backup_path = self._backup_manifest(volume_name, old_version, old_rows, old_refs)
        conn = self.m.store.conn
        try:
            replaced = self.m.store.delete_all_snapshots(volume_name)
            for entry in normalized:
                self.m.store.insert_snapshot(
                    volume_name, entry["snapshot_id"], entry["name"], int(entry["version_no"]),
                    entry["parent_id"], bytes(entry["bitmap"]), now_iso(),
                )
                for block_no, block_hash in sorted(entry["_refs"].items()):
                    self.m.store.insert_ref(volume_name, entry["snapshot_id"], block_no, block_hash)
            self.m.store.put_manifest(volume_name, new_format_version, recomputed, now_iso())
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        result = {
            "replaced_entries": replaced,
            "new_entries": len(normalized),
            "old_format_version": old_version,
            "new_format_version": new_format_version,
            "version_changed": old_version != new_format_version,
            "old_digest": old_digest,
            "new_digest": recomputed,
            "digest_changed": old_digest != recomputed,
            "backup_path": backup_path,
        }
        self.m.logger.log("INFO", "manifest_replaced",
                          manifest_version=new_format_version, **result)
        return result

    def _normalize_entries(
        self, volume_name: str, block_count: int, entries: list[dict[str, Any]]
    ) -> list[Any]:
        if not isinstance(entries, list):
            raise DomainError(E_INPUT, "entries must be a full list, not a delta", field="entries")
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        normalized: list[dict[str, Any]] = []
        expected_versions = list(range(1, len(entries) + 1))
        actual_versions = sorted(int(e.get("version_no", 0)) for e in entries)
        if actual_versions != expected_versions:
            raise DomainError(E_INPUT, "entry version numbers must be monotonic starting at 1",
                              field="version_no")
        for entry in entries:
            sid = entry.get("snapshot_id")
            name = entry.get("name")
            if not sid or sid in seen_ids:
                raise DomainError(E_INPUT, f"bad or duplicate snapshot_id {sid!r}", field="snapshot_id")
            if not name or name in seen_names:
                raise DomainError(E_INPUT, f"bad or duplicate snapshot name {name!r}", field="name")
            seen_ids.add(sid)
            seen_names.add(name)
            try:
                raw = bytes.fromhex(entry["bitmap_hex"])
            except (KeyError, ValueError):
                raise DomainError(E_INPUT, f"bitmap_hex invalid for {sid!r}", field="bitmap_hex")
            bitmap = Bitmap(block_count, raw)
            refs = entry.get("refs", {})
            if not isinstance(refs, dict):
                raise DomainError(E_INPUT, f"refs must be a map for {sid!r}", field="refs")
            ref_map = {int(b): h for b, h in refs.items()}
            for block_no, block_hash in ref_map.items():
                if not 0 <= block_no < block_count:
                    raise DomainError(E_INPUT, f"ref block {block_no} out of range", field="refs")
                if not self.m.blocks.exists(block_hash):
                    raise DomainError(E_INPUT,
                                      f"block object {block_hash[:12]}... missing in store",
                                      field="refs")
            # The base snapshot keeps an empty bitmap yet holds full initial
            # block refs; later snapshots' refs must match their bitmap set.
            bitmap_blocks = set(bitmap.changed_blocks())
            if not bitmap_blocks.issubset(set(ref_map)):
                raise DomainError(
                    E_INPUT,
                    f"bitmap blocks and refs disagree for {sid!r}",
                    field="bitmap_hex",
                    snapshot_id=sid,
                )
            if bitmap_blocks and set(ref_map) != bitmap_blocks:
                raise DomainError(
                    E_INPUT,
                    f"non-base snapshot {sid!r} refs must equal its bitmap blocks",
                    field="refs",
                    snapshot_id=sid,
                )
            normalized.append({
                "snapshot_id": sid,
                "name": name,
                "version_no": int(entry["version_no"]),
                "parent_id": entry.get("parent_id"),
                "bitmap": bitmap.to_bytes(),
                "_refs": ref_map,
            })
        return normalized

    def _backup_manifest(
        self, volume_name: str, old_version: Optional[int], rows: list[Any],
        refs: dict[str, dict[int, str]],
    ) -> str:
        payload = {
            "volume_name": volume_name,
            "format_version": old_version,
            "content_digest": manifest_digest(rows, refs),
            "entries": [
                {
                    "snapshot_id": r["snapshot_id"],
                    "name": r["name"],
                    "version_no": int(r["version_no"]),
                    "parent_id": r["parent_id"],
                    "bitmap_hex": bytes(r["bitmap"]).hex(),
                    "refs": {str(b): h for b, h in refs.get(r["snapshot_id"], {}).items()},
                }
                for r in rows
            ],
            "backed_up_at": now_iso(),
        }
        path = os.path.join(
            self.m.backup_dir,
            f"manifest-{volume_name}-{old_version}-{now_iso().replace(':', '')}.json",
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        # keep the containing directory writable so future backups and clean
        # workspace resets are not blocked by a read-only sibling
        os.chmod(self.m.backup_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        return os.path.abspath(path)

    def detect_drift(self, volume_name: str) -> dict[str, Any]:
        """Return whether the stored manifest still matches its entry set."""
        view = self.m.manifest_view(volume_name)
        manifest = self.m.store.get_manifest(volume_name)
        return {
            "drift": not view["digest_match"],
            "format_version": view["format_version"],
            "stored_digest": view["stored_digest"],
            "recomputed_digest": view["recomputed_digest"],
            "expected_format": int(manifest["format_version"]) if manifest else None,
        }

    # -- versioned audit -----------------------------------------------
    def audit(self, volume_name: str) -> dict[str, Any]:
        row = self.m._require_volume_row(volume_name)
        block_count = int(row["block_count"])
        manifest = self.m.store.get_manifest(volume_name)
        manifest_version = int(manifest["format_version"]) if manifest else 0
        rows = self.m.store.list_snapshots(volume_name)
        inconsistent: list[dict[str, Any]] = []
        first_reason: Optional[str] = None
        ok = True

        ids = {r["snapshot_id"] for r in rows}
        cumulative: set[int] = set()
        for index, snap in enumerate(rows, start=1):
            sid = snap["snapshot_id"]
            if int(snap["version_no"]) != index:
                ok = False
                first_reason = first_reason or f"{sid}: version number {snap['version_no']} != {index}"
                inconsistent.append({"snapshot_id": sid, "reason": "version_number"})
            parent = snap["parent_id"]
            if index == 1:
                if parent is not None:
                    ok = False
                    first_reason = first_reason or f"{sid}: base snapshot must have no parent"
                    inconsistent.append({"snapshot_id": sid, "reason": "base_parent"})
            elif parent not in ids:
                ok = False
                first_reason = first_reason or f"{sid}: dangling parent {parent}"
                inconsistent.append({"snapshot_id": sid, "reason": "dangling_parent"})
            try:
                bitmap = Bitmap(block_count, snap["bitmap"])
            except DomainError as err:
                ok = False
                first_reason = first_reason or f"{sid}: {err.message}"
                inconsistent.append({"snapshot_id": sid, "reason": "bitmap_shape"})
                continue
            refs = self.m.store.ref_map(volume_name, sid)
            bitmap_blocks = set(bitmap.changed_blocks())
            if not bitmap_blocks.issubset(set(refs)):
                ok = False
                first_reason = first_reason or f"{sid}: bitmap/ref block sets differ"
                inconsistent.append({"snapshot_id": sid, "reason": "bitmap_ref_mismatch"})
            elif bitmap_blocks and set(refs) != bitmap_blocks:
                ok = False
                first_reason = first_reason or f"{sid}: non-base refs exceed bitmap blocks"
                inconsistent.append({"snapshot_id": sid, "reason": "bitmap_ref_mismatch"})
            for block_no, block_hash in refs.items():
                if not self.m.blocks.exists(block_hash) or block_digest(self.m.blocks.get(block_hash)) != block_hash:
                    ok = False
                    first_reason = first_reason or f"{sid}: corrupt block {block_no}"
                    inconsistent.append({"snapshot_id": sid, "reason": "corrupt_block", "block_no": block_no})
            cumulative.update(bitmap.changed_blocks())

        # chain must replay back to the initial state: every block reachable
        state = {}
        try:
            state = self.m.reconstruct_state(volume_name, rows[-1]["snapshot_id"]) if rows else {}
        except DomainError as err:
            ok = False
            first_reason = first_reason or err.message
            inconsistent.append({"snapshot_id": rows[-1]["snapshot_id"] if rows else None,
                                 "reason": "chain_not_replayable"})
        if rows and len(state) != block_count:
            ok = False
            first_reason = first_reason or (
                f"chain replay covers {len(state)}/{block_count} blocks, initial state unreachable"
            )
            inconsistent.append({"snapshot_id": rows[-1]["snapshot_id"], "reason": "incomplete_initial_state"})

        if manifest is not None:
            view = self.m.manifest_view(volume_name)
            if not view["digest_match"]:
                ok = False
                first_reason = first_reason or "manifest content digest drift"
                inconsistent.append({"snapshot_id": None, "reason": "manifest_digest"})

        payload = {
            "ok": ok,
            "manifest_version": manifest_version,
            "snapshot_count": len(rows),
            "inconsistent": inconsistent,
            "first_reason": first_reason,
            "audited_at": now_iso(),
        }
        self._replace_stale_verdict(volume_name, manifest_version, ok, payload)
        self.m.logger.log("INFO" if ok else "ERROR", "audit_finished",
                          manifest_version=manifest_version,
                          **{"ok": ok, "inconsistent_count": len(inconsistent)})
        return payload

    def _replace_stale_verdict(
        self, volume_name: str, manifest_version: int, ok: bool, payload: dict[str, Any]
    ) -> None:
        previous = self.m.store.get_audit(volume_name)
        if previous is not None and int(previous["manifest_version"]) != manifest_version:
            self.m.logger.log(
                "WARN", "audit_verdict_replaced",
                manifest_version=manifest_version,
                **{"old_manifest_version": int(previous["manifest_version"])},
            )
        self.m.store.put_audit(volume_name, manifest_version, ok, payload, now_iso())
        self.m.store.commit()

    def audit_verdict(self, volume_name: str) -> Optional[dict[str, Any]]:
        row = self.m.store.get_audit(volume_name)
        if row is None:
            return None
        return {
            "manifest_version": int(row["manifest_version"]),
            "ok": bool(row["ok"]),
            "payload": json.loads(row["payload"]),
            "updated_at": row["updated_at"],
        }
