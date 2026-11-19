"""Atomic edit ledger, undo support and crash recovery.

On-disk layout under ``<workspace>/ledger``:

    ledger.log        append-only committed edit log (atomic JSONL records)
    manifest.json     canonical head pointer + per-version content hashes
    cache/v<N>/       committed snapshot side data (hashes only by default)
    cache/tmp-*/      uncommitted scratch, reaped as orphans on startup
    exit_report.json  written only on clean shutdown; never fabricated
"""
from __future__ import annotations

import glob
import os

from .hashing import atomic_write_bytes, blake2b_hex, read_json, write_canonical_json

LEDGER_DIR = "ledger"
LOG_NAME = "ledger.log"
MANIFEST_NAME = "manifest.json"
CACHE_DIR = "cache"
EXIT_REPORT = "exit_report.json"
COMMIT_MARK = "committed"


class Ledger:
    def __init__(self, workspace: str):
        self.root = os.path.join(workspace, LEDGER_DIR)
        self.cache_root = os.path.join(self.root, CACHE_DIR)
        os.makedirs(self.cache_root, exist_ok=True)
        self.log_path = os.path.join(self.root, LOG_NAME)
        self.manifest_path = os.path.join(self.root, MANIFEST_NAME)
        self.exit_report_path = os.path.join(self.root, EXIT_REPORT)

    # ------------------------------------------------------------- lifecycle
    def recover(self) -> dict:
        """Reap orphans and return the last committed manifest (or a fresh one)."""
        orphans = self._reap_orphans()
        if os.path.exists(self.manifest_path):
            manifest = read_json(self.manifest_path)
        else:
            manifest = {"format": "pcrecon-ledger", "format_version": 1, "head_version": -1, "versions": []}
        manifest["recovered_orphans"] = orphans
        manifest["exit_report_present"] = os.path.exists(self.exit_report_path)
        return manifest

    def _reap_orphans(self) -> list[str]:
        orphans = sorted(glob.glob(os.path.join(self.cache_root, "tmp-*")))
        reaped = []
        for path in orphans:
            self._remove_path(path)
            reaped.append(os.path.basename(path))
        return reaped

    @staticmethod
    def _remove_path(path: str) -> None:
        import shutil

        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)

    # -------------------------------------------------------------- commits
    def append_commit(self, version: int, edit: dict, hashes: dict, audit: dict) -> None:
        record = {
            "record": "commit",
            "version": version,
            "edit": edit,
            "hashes": hashes,
            "audit": audit,
        }
        from .hashing import canonical_json_bytes

        payload = canonical_json_bytes(record) + b"\n"
        # Append + fsync: each line is self-contained canonical JSON.
        with open(self.log_path, "ab") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        manifest = self._read_manifest_unchecked()
        manifest["head_version"] = version
        manifest["versions"].append(
            {"version": version, "hashes": hashes, "edit_type": edit.get("type")}
        )
        write_canonical_json(self.manifest_path, manifest)
        self._stamp_committed(version)

    def _stamp_committed(self, version: int) -> None:
        vdir = os.path.join(self.cache_root, f"v{version}")
        os.makedirs(vdir, exist_ok=True)
        atomic_write_bytes(os.path.join(vdir, COMMIT_MARK), b"1\n")

    def append_undo(self, version: int, restored_version: int, hashes: dict) -> None:
        from .hashing import canonical_json_bytes

        record = {
            "record": "undo",
            "version": version,
            "restored_version": restored_version,
            "hashes": hashes,
        }
        with open(self.log_path, "ab") as fh:
            fh.write(canonical_json_bytes(record) + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        manifest = self._read_manifest_unchecked()
        manifest["head_version"] = restored_version
        manifest["versions"].append(
            {"version": version, "undo_to": restored_version, "hashes": hashes}
        )
        write_canonical_json(self.manifest_path, manifest)

    def reset_to_initial(self) -> None:
        """Clear committed run state so a deterministic run starts clean.

        Uncommitted tmp-* scratch is reaped first; the initial commit is
        re-created by the application.  Crash recovery semantics are covered
        separately by ``recover()`` / the crash-demo command.
        """
        self._reap_orphans()
        for path in (self.log_path, self.manifest_path, self.exit_report_path):
            if os.path.exists(path):
                os.remove(path)
        import shutil

        if os.path.isdir(self.cache_root):
            shutil.rmtree(self.cache_root, ignore_errors=True)
        os.makedirs(self.cache_root, exist_ok=True)

    def read_commits(self) -> list[dict]:
        if not os.path.exists(self.log_path):
            return []
        out = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(read_json_text(line))
        return out

    def write_exit_report(self, report: dict) -> str:
        # Only call this on a clean shutdown.
        return write_canonical_json(self.exit_report_path, report)

    def remove_exit_report(self) -> None:
        if os.path.exists(self.exit_report_path):
            os.remove(self.exit_report_path)

    def _read_manifest_unchecked(self) -> dict:
        if os.path.exists(self.manifest_path):
            return read_json(self.manifest_path)
        return {"format": "pcrecon-ledger", "format_version": 1, "head_version": -1, "versions": []}


def read_json_text(text: str):
    import json

    return json.loads(text)
