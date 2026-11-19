"""Transactional, content-addressed snapshot persistence."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_hash,
    file_records,
    pretty_json_bytes,
    read_json,
    sha256_file,
)


class SnapshotError(RuntimeError):
    pass


def _payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in snapshot.items()
        if key not in {"version", "integrity"}
    }


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.staging = self.root / "staging"
        self.runs = self.root / "runs"
        self.views = self.root / "views"
        self.current_path = self.root / "CURRENT.json"
        self.latest_report_path = self.root / "latest-report.json"
        self._create_layout()

    def _create_layout(self) -> None:
        for path in (self.root, self.staging, self.runs, self.views):
            path.mkdir(parents=True, exist_ok=True)

    def _remove_path(self, path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _identity(self, payload: dict[str, Any], version_prefix: str) -> tuple[str, str]:
        content_hash = canonical_hash(payload)
        version = f"{version_prefix}-{content_hash[:16]}"
        return content_hash, version

    def _materialize(self, payload: dict[str, Any], version_prefix: str) -> dict[str, Any]:
        content_hash, version = self._identity(payload, version_prefix)
        complete = dict(payload)
        complete["version"] = version
        complete["integrity"] = {
            "algorithm": "sha256",
            "content_sha256": content_hash,
            "version": version,
            "hash_scope": "canonical JSON excluding version and integrity",
        }
        return complete

    def start_staging(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="run-", dir=str(self.staging)))

    def recover(self) -> dict[str, Any]:
        """Recover exactly the latest committed run and recycle orphaning writes."""

        self._create_layout()
        recovered: list[str] = []
        for entry in sorted(self.staging.iterdir()):
            self._remove_path(entry)
            recovered.append(f"staging/{entry.name}")

        current = None
        if self.current_path.exists():
            current = read_json(self.current_path)
            version = str(current.get("version", ""))
            run_dir = self.root / str(current.get("run_dir", ""))
            snapshot_path = self.root / str(current.get("snapshot", ""))
            if not snapshot_path.is_file():
                raise SnapshotError(f"committed snapshot is missing: {snapshot_path}")
            snapshot = read_json(snapshot_path)
            payload = _payload(snapshot)
            content_hash = canonical_hash(payload)
            if content_hash != current.get("content_sha256") or snapshot["integrity"].get(
                "content_sha256"
            ) != content_hash:
                raise SnapshotError("committed snapshot content hash mismatch")
            if snapshot.get("version") != version:
                raise SnapshotError("committed snapshot version mismatch")
            keep_run = run_dir.name
        else:
            keep_run = None

        for entry in sorted(self.runs.iterdir()):
            if keep_run is not None and entry.name == keep_run:
                continue
            self._remove_path(entry)
            recovered.append(f"runs/{entry.name}")

        keep_view = keep_run
        for entry in sorted(self.views.iterdir()):
            if keep_view is not None and entry.name == keep_view:
                continue
            self._remove_path(entry)
            recovered.append(f"views/{entry.name}")

        for entry in sorted(self.root.glob(".*.tmp-*")):
            self._remove_path(entry)
            recovered.append(entry.name)
        return {"current": current, "recovered_orphans": recovered}

    def commit(
        self,
        staging_dir: Path,
        payload: dict[str, Any],
        version_prefix: str,
    ) -> dict[str, Any]:
        staging_dir = Path(staging_dir)
        complete = self._materialize(payload, version_prefix)
        version = complete["version"]
        content_hash = complete["integrity"]["content_sha256"]
        snapshot_path = staging_dir / "snapshot.json"
        atomic_write_bytes(snapshot_path, pretty_json_bytes(complete))

        final_dir = self.runs / version
        if final_dir.exists():
            existing_path = final_dir / "snapshot.json"
            if not existing_path.is_file():
                raise SnapshotError(f"existing snapshot directory is incomplete: {final_dir}")
            existing = read_json(existing_path)
            existing_hash = canonical_hash(_payload(existing))
            if existing_hash != content_hash:
                raise SnapshotError(
                    f"version collision for {version}: {existing_hash} != {content_hash}"
                )
            shutil.rmtree(staging_dir)
        else:
            os.replace(staging_dir, final_dir)

        current = {
            "schema": "spatial-diagnostic-current/1",
            "version": version,
            "content_sha256": content_hash,
            "run_dir": f"runs/{version}",
            "snapshot": f"runs/{version}/snapshot.json",
        }
        atomic_write_json(self.current_path, current)
        self.recover()
        return complete

    def load_current(self) -> dict[str, Any]:
        if not self.current_path.is_file():
            raise SnapshotError("no committed snapshot")
        current = read_json(self.current_path)
        snapshot_path = self.root / current["snapshot"]
        if not snapshot_path.is_file():
            raise SnapshotError("current snapshot file is missing")
        snapshot = read_json(snapshot_path)
        payload = _payload(snapshot)
        content_hash = canonical_hash(payload)
        if content_hash != current["content_sha256"]:
            raise SnapshotError("current snapshot content hash mismatch")
        if snapshot.get("version") != current["version"]:
            raise SnapshotError("current snapshot version mismatch")
        return snapshot

    def run_dir(self, snapshot: dict[str, Any]) -> Path:
        return self.runs / str(snapshot["version"])

    def verify_current(self) -> dict[str, Any]:
        snapshot = self.load_current()
        run_dir = self.run_dir(snapshot)
        mismatches: list[dict[str, Any]] = []
        verified: list[dict[str, Any]] = []
        for record in snapshot.get("artifacts", []):
            path = run_dir / record["path"]
            if not path.is_file():
                mismatches.append({"path": record["path"], "reason": "missing"})
                continue
            size = path.stat().st_size
            digest = sha256_file(path)
            if size != int(record["bytes"]) or digest != record["sha256"]:
                mismatches.append(
                    {
                        "path": record["path"],
                        "reason": "content mismatch",
                        "expected_bytes": record["bytes"],
                        "actual_bytes": size,
                        "expected_sha256": record["sha256"],
                        "actual_sha256": digest,
                    }
                )
            else:
                verified.append(record)
        return {
            "version": snapshot["version"],
            "content_sha256": snapshot["integrity"]["content_sha256"],
            "artifact_count": len(snapshot.get("artifacts", [])),
            "verified_count": len(verified),
            "mismatches": mismatches,
            "ok": len(mismatches) == 0,
        }

    def collect_files(self, base: Path) -> list[dict[str, Any]]:
        paths = [path for path in Path(base).rglob("*") if path.is_file()]
        return file_records(paths, Path(base))

    def publish_views(self, version: str, staging_view: Path) -> Path:
        final = self.views / version
        if final.exists():
            self._remove_path(staging_view)
            return final
        os.replace(staging_view, final)
        for entry in sorted(self.views.iterdir()):
            if entry.name != version:
                self._remove_path(entry)
        return final

    def write_latest_report(self, report: dict[str, Any]) -> None:
        atomic_write_json(self.latest_report_path, report)

