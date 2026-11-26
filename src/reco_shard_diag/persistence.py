from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json


class SnapshotVersionMismatch(Exception):
    def __init__(self, reason: str, current: dict[str, Any] | None):
        super().__init__(reason)
        self.reason = reason
        self.current = current


class SnapshotStore:
    def __init__(self, workdir: Path, *, global_version: int, base_config_hash: str):
        self.workdir = workdir
        self.root = workdir / "ledger"
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self.global_version = global_version
        self.base_config_hash = base_config_hash
        self.pointer = self.root / "CURRENT"

    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.pointer.exists():
            return None
        try:
            return json.loads(self.pointer.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SnapshotVersionMismatch("CURRENT pointer corrupt", None) from exc

    def load(self) -> dict[str, Any] | None:
        pointer = self._read_pointer()
        if pointer is None:
            return None
        if pointer.get("global_version") != self.global_version:
            raise SnapshotVersionMismatch("global_version changed", pointer)
        if pointer.get("base_config_hash") != self.base_config_hash:
            raise SnapshotVersionMismatch("base_config_hash changed", pointer)
        snapshot_dir = self.root / pointer["snapshot_dir"]
        checksum_path = snapshot_dir / "checksums.json"
        ledger_path = snapshot_dir / "ledger.json"
        if not checksum_path.exists() or not ledger_path.exists():
            raise SnapshotVersionMismatch("committed snapshot files missing", pointer)
        try:
            checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SnapshotVersionMismatch("checksums corrupt", pointer) from exc
        ledger_bytes = ledger_path.read_bytes()
        actual = "sha256:" + hashlib.sha256(ledger_bytes).hexdigest()
        if checksums.get("ledger.json") != actual:
            raise SnapshotVersionMismatch("ledger checksum mismatch", pointer)
        ledger = json.loads(ledger_bytes.decode("utf-8"))
        ledger["_snapshot_manifest"] = pointer
        return ledger

    def commit(self, ledger: dict[str, Any]) -> dict[str, Any]:
        pointer = self._read_pointer()
        previous_sequence = int(pointer.get("sequence", 0)) if pointer else 0
        sequence = previous_sequence + 1
        snapshot_id = f"{sequence:08d}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        temp_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.snapshots))
        try:
            (temp_dir / "ledger.json").write_bytes(canonical_json(ledger).encode("utf-8"))
            (temp_dir / "stage-products.json").write_bytes(
                canonical_json(ledger.get("products", {})).encode("utf-8")
            )
            (temp_dir / "queues.json").write_bytes(
                canonical_json(
                    {
                        "pending": ledger.get("pending_queue", []),
                        "completed_cursor": ledger.get("completed_cursor", []),
                    }
                ).encode("utf-8")
            )
            (temp_dir / "leases.json").write_bytes(
                canonical_json(ledger.get("lease_chains", {})).encode("utf-8")
            )
            files = ["ledger.json", "stage-products.json", "queues.json", "leases.json"]
            checksums = {
                name: "sha256:"
                + hashlib.sha256((temp_dir / name).read_bytes()).hexdigest()
                for name in files
            }
            (temp_dir / "checksums.json").write_bytes(
                canonical_json(checksums).encode("utf-8")
            )
            manifest = {
                "snapshot_id": snapshot_id,
                "snapshot_dir": f"snapshots/{snapshot_id}",
                "sequence": sequence,
                "global_version": self.global_version,
                "base_config_hash": self.base_config_hash,
                "event_stream_hash": ledger.get("event_stream_hash"),
                "phase": ledger.get("phase"),
                "committed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            (temp_dir / "manifest.json").write_bytes(
                canonical_json(manifest).encode("utf-8")
            )
            for file_path in temp_dir.iterdir():
                with file_path.open("rb") as handle:
                    os.fsync(handle.fileno())
            final_dir = self.snapshots / snapshot_id
            os.replace(temp_dir, final_dir)
            pointer_tmp = self.root / f".CURRENT-{os.getpid()}-{sequence}"
            pointer_tmp.write_bytes(canonical_json(manifest).encode("utf-8"))
            with pointer_tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(pointer_tmp, self.pointer)
            return manifest
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def current_manifest(self) -> dict[str, Any] | None:
        return self._read_pointer()


def termination_classification(ledger: dict[str, Any] | None) -> dict[str, Any]:
    if ledger is None:
        return {
            "kind": "NO_COMMITTED_SNAPSHOT",
            "clean": False,
            "message": "没有可恢复的已提交快照",
        }
    phase = ledger.get("phase")
    if phase in {"COMPLETE", "CLEAN_EXIT"}:
        return {
            "kind": "CLEAN_EXIT",
            "clean": True,
            "message": "上一进程完成正常提交并退出",
            "completed_cursor": ledger.get("completed_cursor", []),
        }
    return {
        "kind": "RECOVERED_AFTER_FORCED_TERMINATION",
        "clean": False,
        "message": "上一进程停在已提交的 RUNNING 快照，按强制终止恢复",
        "completed_cursor": ledger.get("completed_cursor", []),
    }
