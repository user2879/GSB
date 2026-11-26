from __future__ import annotations

import fcntl
import json
import os
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import fingerprint
from .errors import E_SNAPSHOT_CORRUPT, PipelineError


COMPONENT_FILES = (
    "manifest.json",
    "cleaning.json",
    "dedup.json",
    "sampling.json",
    "summary.json",
    "ledger.json",
    "queue.json",
    "budget.json",
    "error.json",
)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.tmp_dir = self.root / "tmp"
        self.reports_dir = self.root / "reports"
        self.current_link = self.root / "current"
        self.lock_path = self.root / "commit.lock"
        for path in (self.root, self.snapshots_dir, self.tmp_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    def load_current(self) -> dict[str, Any] | None:
        if not self.current_link.exists():
            return None
        return self._load_snapshot_dir(self.current_link.resolve())

    def load_version(self, global_version: int) -> dict[str, Any] | None:
        candidate = self.snapshots_dir / f"snapshot-{global_version:06d}"
        if not candidate.exists():
            return None
        return self._load_snapshot_dir(candidate)

    def _load_snapshot_dir(self, snapshot_dir: Path) -> dict[str, Any]:
        try:
            state: dict[str, Any] = {}
            for filename in COMPONENT_FILES:
                path = snapshot_dir / filename
                if filename == "error.json" and not path.exists():
                    state["error"] = None
                    continue
                with path.open("r", encoding="utf-8") as handle:
                        state[filename.removesuffix(".json")] = json.load(handle)
            files = state["manifest"].get("files", {})
            for filename, expected_hash in files.items():
                if filename == "manifest.json":
                    continue
                actual_hash = fingerprint(_read_json(snapshot_dir / filename))
                if actual_hash != expected_hash:
                    raise PipelineError(
                        E_SNAPSHOT_CORRUPT,
                        "snapshot component hash does not match manifest",
                        field=f"snapshot.{filename}",
                    )
            state["_snapshot_dir"] = str(snapshot_dir)
            return state
        except FileNotFoundError as exc:
            raise PipelineError(
                E_SNAPSHOT_CORRUPT,
                "committed snapshot is missing a required component",
                field=str(snapshot_dir),
            ) from exc
        except json.JSONDecodeError as exc:
            raise PipelineError(
                E_SNAPSHOT_CORRUPT,
                f"committed snapshot contains invalid JSON: {exc.msg}",
                field=str(snapshot_dir),
            ) from exc

    def commit(self, state: dict[str, Any], *, crash_before_replace: bool = False) -> Path:
        manifest = state["manifest"]
        version = manifest["global_version"]
        final_dir = self.snapshots_dir / f"snapshot-{version:06d}"
        staging_dir = Path(tempfile.mkdtemp(prefix=f"snapshot-{version:06d}-", dir=self.tmp_dir))

        components = {
            "cleaning.json": state.get("cleaning", {"status": "pending", "records": []}),
            "dedup.json": state.get("dedup", {"status": "pending", "shards": []}),
            "sampling.json": state.get("sampling"),
            "summary.json": state.get("summary"),
            "ledger.json": state.get("ledger", {"entries": []}),
            "queue.json": state.get("queue", {"items": []}),
            "budget.json": state.get("budget", {}),
            "error.json": state.get("error"),
        }
        file_hashes: dict[str, str] = {}
        for filename, value in components.items():
            path = staging_dir / filename
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            path.write_text(payload + "\n", encoding="utf-8")
            _fsync_file(path)
            file_hashes[filename] = fingerprint(value)

        manifest["files"] = file_hashes
        staging_manifest = staging_dir / "manifest.json"
        staging_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _fsync_file(staging_manifest)

        if crash_before_replace:
            _fsync_directory(staging_dir)
            os.kill(os.getpid(), signal.SIGKILL)
            os._exit(137)

        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                current = self.load_current()
                expected_parent = manifest.get("parent_global_version")
                actual_version = current["manifest"]["global_version"] if current else None
                if expected_parent != actual_version:
                    raise PipelineError(
                        E_VERSION_CONFLICT,
                        "current global snapshot version does not match expected parent",
                        field="manifest.parent_global_version",
                        details={"expected": expected_parent, "actual": actual_version},
                    )
                if final_dir.exists():
                    raise PipelineError(
                        E_VERSION_CONFLICT,
                        "snapshot directory for global_version already exists",
                        field="manifest.global_version",
                        details={"global_version": version},
                    )
                os.rename(staging_dir, final_dir)
                _fsync_directory(self.snapshots_dir)

                tmp_link = self.root / f".current-link-{os.getpid()}"
                if tmp_link.exists() or tmp_link.is_symlink():
                    tmp_link.unlink()
                os.symlink(f"snapshots/snapshot-{version:06d}", tmp_link)
                os.replace(tmp_link, self.current_link)
                _fsync_directory(self.root)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return final_dir

    def write_closeout_report(self, report: dict[str, Any]) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(report)
        payload.setdefault("written_at", datetime.now(timezone.utc).isoformat())
        latest = self.reports_dir / "closeout-latest.json"
        tmp = self.reports_dir / ".closeout-latest.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _fsync_file(tmp)
        os.replace(tmp, latest)
        timestamped = self.reports_dir / f"closeout-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
        timestamped.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return latest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
