from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_RECOVERY, E_VERSION_CONFLICT


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def safe_artifact_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "__")


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bucket_dir = self.root / "buckets"
        self.field_tf_dir = self.root / "field_tf"
        self.fusion_dir = self.root / "fusion"
        self.ranking_dir = self.root / "ranking"
        for directory in (
            self.bucket_dir,
            self.field_tf_dir,
            self.fusion_dir,
            self.ranking_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _path(self, directory: Path, key: str) -> Path:
        return directory / f"{safe_artifact_name(key)}.json"

    def has_bucket(self, bucket_id: str, fingerprint: str | None = None) -> bool:
        path = self._path(self.bucket_dir, bucket_id)
        if not path.exists():
            return False
        if fingerprint is None:
            return True
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("bucket_fingerprint") == fingerprint
        except (json.JSONDecodeError, OSError):
            return False

    def write_bucket(self, bucket: dict[str, Any]) -> None:
        atomic_write_json(self._path(self.bucket_dir, bucket["bucket_id"]), bucket)

    def read_bucket(self, bucket_id: str) -> dict[str, Any] | None:
        path = self._path(self.bucket_dir, bucket_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def remove_bucket(self, bucket_id: str) -> None:
        path = self._path(self.bucket_dir, bucket_id)
        path.unlink(missing_ok=True)

    def has_field_tf(self, record_id: str, cache_key: str | None = None) -> bool:
        path = self._path(self.field_tf_dir, record_id)
        if not path.exists():
            return False
        if cache_key is None:
            return True
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("cache_key") == cache_key
        except (json.JSONDecodeError, OSError):
            return False

    def write_field_tf(self, record: dict[str, Any]) -> None:
        atomic_write_json(self._path(self.field_tf_dir, record["record_id"]), record)

    def remove_field_tf(self, record_id: str) -> None:
        self._path(self.field_tf_dir, record_id).unlink(missing_ok=True)

    def has_fusion(self, cache_key: str) -> bool:
        return self._path(self.fusion_dir, cache_key).exists()

    def write_fusion(self, result: dict[str, Any]) -> None:
        atomic_write_json(self._path(self.fusion_dir, result["fusion_cache_key"]), result)

    def read_fusion(self, cache_key: str) -> dict[str, Any] | None:
        path = self._path(self.fusion_dir, cache_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def has_ranking(self, cache_key: str) -> bool:
        return self._path(self.ranking_dir, cache_key).exists()

    def write_ranking(self, query_id: str, cache_key: str, ranking: dict[str, Any]) -> None:
        payload = {"query_id": query_id, "ranking_cache_key": cache_key, "ranking": ranking}
        atomic_write_json(self._path(self.ranking_dir, cache_key), payload)

    def read_ranking(self, cache_key: str) -> dict[str, Any] | None:
        path = self._path(self.ranking_dir, cache_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


class SnapshotStore:
    PHASES = ("01_validation", "02_index", "03_statistics", "04_scores", "05_ranking")

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.staging = self.root / "staging"
        self.releases = self.root / "releases"
        self.pointer_path = self.root / "CURRENT.json"
        self.staging.mkdir(parents=True, exist_ok=True)
        self.releases.mkdir(parents=True, exist_ok=True)

    def new_transaction(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="txn-", dir=str(self.staging)))

    def write_phase(self, transaction_dir: Path, phase: str, payload: Any) -> Path:
        if phase not in self.PHASES and phase not in {"scheduler", "report", "manifest"}:
            raise ValueError(f"unknown snapshot phase: {phase}")
        path = transaction_dir / f"{phase}.json"
        atomic_write_json(path, payload)
        return path

    def current_pointer(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists():
            return None
        return json.loads(self.pointer_path.read_text(encoding="utf-8"))

    def commit(
        self,
        transaction_dir: Path,
        *,
        global_version: int,
        config_hash: str,
        state_hash: str,
        expected_global_version: int | None = None,
        expected_config_hash: str | None = None,
    ) -> dict[str, Any]:
        current = self.current_pointer()
        if expected_global_version is not None:
            current_version = int(current["global_version"]) if current else 0
            if current_version != expected_global_version:
                raise E_VERSION_CONFLICT(
                    f"global version changed: expected {expected_global_version}, found {current_version}"
                )
        if expected_config_hash is not None and current is not None:
            if current.get("config_hash") != expected_config_hash:
                raise E_VERSION_CONFLICT("config hash changed before snapshot commit")

        manifest = {
            "status": "prepared",
            "global_version": global_version,
            "config_hash": config_hash,
            "state_hash": state_hash,
            "phases": list(self.PHASES),
        }
        self.write_phase(transaction_dir, "manifest", manifest)
        release_name = f"snapshot-v{global_version}-{config_hash[:12]}-{state_hash[:12]}"
        release_dir = self.releases / release_name
        if release_dir.exists():
            shutil.rmtree(release_dir)
        shutil.move(str(transaction_dir), str(release_dir))
        pointer = {
            "release": release_name,
            "global_version": global_version,
            "config_hash": config_hash,
            "state_hash": state_hash,
        }
        pointer_tmp = self.root / ".CURRENT.json.tmp"
        atomic_write_json(pointer_tmp, pointer)
        os.replace(pointer_tmp, self.pointer_path)
        return pointer

    def recover(self, *, promote_prepared: bool = True, clean_incomplete: bool = False) -> dict[str, Any]:
        pointer = self.current_pointer()
        last_good: dict[str, Any] | None = None
        if pointer:
            release_dir = self.releases / pointer["release"]
            manifest_path = release_dir / "manifest.json"
            if release_dir.is_dir() and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                missing = [
                    phase for phase in self.PHASES if not (release_dir / f"{phase}.json").exists()
                ]
                if not missing and manifest.get("global_version") == pointer["global_version"]:
                    last_good = pointer

        orphan_transactions: list[dict[str, Any]] = []
        promoted: list[str] = []
        for transaction_dir in sorted(self.staging.glob("txn-*")):
            manifest_path = transaction_dir / "manifest.json"
            phases_complete = all(
                (transaction_dir / f"{phase}.json").exists() for phase in self.PHASES
            )
            if not manifest_path.exists() or not phases_complete:
                item = {
                    "path": str(transaction_dir),
                    "classification": "incomplete",
                    "code": E_SNAPSHOT_RECOVERY,
                }
                orphan_transactions.append(item)
                if clean_incomplete:
                    shutil.rmtree(transaction_dir, ignore_errors=True)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            item = {
                "path": str(transaction_dir),
                "classification": "prepared_complete",
                "global_version": manifest.get("global_version"),
                "config_hash": manifest.get("config_hash"),
            }
            orphan_transactions.append(item)
            if promote_prepared and last_good is not None:
                if int(manifest["global_version"]) > int(last_good["global_version"]):
                    pointer = self.commit(
                        transaction_dir,
                        global_version=int(manifest["global_version"]),
                        config_hash=manifest["config_hash"],
                        state_hash=manifest.get("state_hash", "recovered"),
                        expected_global_version=int(last_good["global_version"]),
                        expected_config_hash=last_good.get("config_hash"),
                    )
                    promoted.append(str(transaction_dir))
                    last_good = pointer

        residue = list(self.staging.glob("txn-*"))
        return {
            "last_good_pointer": last_good,
            "orphan_transactions": orphan_transactions,
            "promoted": promoted,
            "staging_residue_count": len(residue),
            "decision": "promoted_prepared" if promoted else "use_last_good",
            "cleanup_status": "complete" if not residue else "staging_residue_present",
            "can_exit_cleanly": not residue,
        }

    def shutdown_status(self) -> dict[str, Any]:
        residue = sorted(str(path) for path in self.staging.glob("txn-*"))
        return {
            "staging_residue": residue,
            "staging_residue_count": len(residue),
            "cleanup_status": "complete" if not residue else "staging_residue_present",
            "can_exit": True,
            "can_process_exit": True,
        }
