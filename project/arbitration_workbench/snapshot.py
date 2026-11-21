"""Atomic generation snapshots with pointer-based publication and recovery checks."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .constants import ERROR_SNAPSHOT_INVALID, ERROR_SNAPSHOT_STALE, ERROR_TEMP_VOLUME_MISMATCH
from .errors import SnapshotInvalidError, SnapshotStaleError
from .numeric import (
    atomic_write_text,
    fingerprint,
    jsonable,
    read_json,
    same_volume,
    sha256_bytes,
    write_json,
)


SNAPSHOT_SCHEMA = "arbitration-snapshot/v1"
CURRENT_SCHEMA = "arbitration-snapshot-pointer/v1"


class SnapshotStore:
    def __init__(self, snapshot_dir: str | Path, temp_dir: str | Path):
        self.snapshot_dir = Path(snapshot_dir)
        self.temp_dir = Path(temp_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir = self.snapshot_dir / "generations"
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.snapshot_dir / "CURRENT"
        if not same_volume(self.snapshot_dir, self.temp_dir):
            raise SnapshotInvalidError(
                ERROR_TEMP_VOLUME_MISMATCH,
                "snapshot_dir and temp_dir must be on the same filesystem volume",
            )

    def _read_current(self) -> dict | None:
        if not self.current_path.exists():
            return None
        try:
            pointer = read_json(self.current_path)
        except (json.JSONDecodeError, OSError) as exc:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, f"snapshot CURRENT pointer is invalid: {exc}") from exc
        if not isinstance(pointer, dict) or pointer.get("schema") != CURRENT_SCHEMA:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, "snapshot CURRENT pointer schema is invalid")
        return pointer

    def latest_manifest(self) -> dict | None:
        pointer = self._read_current()
        if pointer is None:
            return None
        generation_name = pointer.get("generation")
        if not isinstance(generation_name, str) or not generation_name or "/" in generation_name or "\\" in generation_name:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, "snapshot generation name is invalid")
        generation_dir = self.generations_dir / generation_name
        manifest_path = generation_dir / "manifest.json"
        if not manifest_path.exists():
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, "snapshot manifest is missing")
        try:
            manifest_payload = manifest_path.read_bytes()
            manifest = json.loads(manifest_payload.decode("utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, f"snapshot manifest is invalid: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != SNAPSHOT_SCHEMA:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, "snapshot manifest schema is invalid")
        if manifest.get("generation") != generation_name:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, "snapshot manifest generation mismatch")
        expected_hash = pointer.get("manifest_sha256")
        if expected_hash and fingerprint(manifest) != expected_hash:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, "snapshot manifest hash mismatch")
        return manifest

    def inspect(
        self,
        *,
        expected_config_hash: str | None = None,
        expected_version: int | None = None,
    ) -> dict:
        try:
            manifest = self.latest_manifest()
        except SnapshotInvalidError as exc:
            return {"ok": False, "code": exc.code, "message": exc.message, "manifest": None}
        if manifest is None:
            return {"ok": False, "code": "E_SNAPSHOT_ABSENT", "message": "no snapshot CURRENT pointer", "manifest": None}
        stale_reasons: list[str] = []
        if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
            stale_reasons.append("config_hash")
        if expected_version is not None and int(manifest.get("global_version", -1)) != expected_version:
            stale_reasons.append("global_version")
        if stale_reasons:
            return {
                "ok": False,
                "code": ERROR_SNAPSHOT_STALE,
                "message": "snapshot is stale for the requested resume point",
                "stale_reasons": stale_reasons,
                "manifest": manifest,
            }
        return {"ok": True, "code": "OK", "manifest": manifest}

    def load_manifest(self, *, require_compatible: bool = True, expected_config_hash: str | None = None) -> dict | None:
        status = self.inspect(expected_config_hash=expected_config_hash)
        if status.get("ok"):
            return status["manifest"]
        if status.get("code") == "E_SNAPSHOT_ABSENT":
            return None
        if require_compatible:
            raise SnapshotStaleError(status["code"], status["message"], status)
        return status.get("manifest")

    def generation_path(self, manifest: dict) -> Path:
        return self.generations_dir / str(manifest["generation"])

    def load_artifact(self, manifest: dict, name: str) -> Any:
        artifacts = manifest.get("artifacts", {})
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, f"snapshot artifact {name} is not recorded")
        path = self.generation_path(manifest) / artifact["file"]
        if not path.exists():
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, f"snapshot artifact {name} is missing")
        payload = path.read_bytes()
        expected_hash = artifact.get("sha256")
        if expected_hash and sha256_bytes(payload) != expected_hash:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, f"snapshot artifact {name} hash mismatch")
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SnapshotInvalidError(ERROR_SNAPSHOT_INVALID, f"snapshot artifact {name} is not JSON") from exc

    def load_cache(self, manifest: dict | None) -> dict | None:
        if manifest is None:
            return None
        return self.load_artifact(manifest, "cache")

    def recovery_info(self, manifest: dict | None = None) -> dict:
        manifest = manifest if manifest is not None else self.latest_manifest()
        if manifest is None:
            return {"available": False, "code": "E_SNAPSHOT_ABSENT"}
        return {
            "available": True,
            "generation": manifest.get("generation"),
            "global_version": manifest.get("global_version"),
            "config_hash": manifest.get("config_hash"),
            "rules_fingerprint": manifest.get("rules_fingerprint"),
            "completed_ticket_count": len(manifest.get("completed_ticket_ids", [])),
            "partial": bool(manifest.get("partial", False)),
            "stage_versions": manifest.get("stage_versions", {}),
            "published_at_sequence": manifest.get("publish_sequence"),
        }

    def commit(
        self,
        *,
        global_version: int,
        config_hash: str,
        rules_fingerprint: str,
        cache_payload: dict,
        decisions: list[dict],
        failures: list[dict],
        summary: dict,
        conversions: dict,
        completed_ticket_ids: list[str],
        stage_versions: dict[str, str],
        partial: bool,
        publish_sequence: int,
        extra_manifest: dict | None = None,
    ) -> dict:
        if not same_volume(self.snapshot_dir, self.temp_dir):
            raise SnapshotInvalidError(
                ERROR_TEMP_VOLUME_MISMATCH,
                "snapshot_dir and temp_dir must be on the same filesystem volume",
            )
        artifacts: dict[str, Any] = {
            "cache": ("cache.json", jsonable(cache_payload)),
            "decisions": ("decisions.json", jsonable(decisions)),
            "failures": ("failures.json", jsonable(failures)),
            "summary": ("summary.json", jsonable(summary)),
            "conversions": ("conversions.json", jsonable(conversions)),
        }
        staging = Path(tempfile.mkdtemp(prefix="snapshot-stage-", dir=str(self.temp_dir)))
        try:
            artifact_manifest: dict[str, dict] = {}
            for name, (filename, value) in artifacts.items():
                payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
                path = staging / filename
                with path.open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                artifact_manifest[name] = {"file": filename, "sha256": sha256_bytes(payload), "size": len(payload)}
            manifest = {
                "schema": SNAPSHOT_SCHEMA,
                "global_version": global_version,
                "config_hash": config_hash,
                "rules_fingerprint": rules_fingerprint,
                "completed_ticket_ids": sorted(completed_ticket_ids),
                "stage_versions": {key: stage_versions[key] for key in sorted(stage_versions)},
                "partial": bool(partial),
                "publish_sequence": publish_sequence,
                "artifacts": artifact_manifest,
            }
            if extra_manifest:
                for key, value in extra_manifest.items():
                    if key not in manifest:
                        manifest[key] = jsonable(value)
            content_fp = fingerprint(manifest)
            generation_name = f"gen-{global_version:08d}-{content_fp[:12]}"
            manifest["generation"] = generation_name
            manifest_fp = fingerprint(manifest)
            manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            manifest_path = staging / "manifest.json"
            with manifest_path.open("wb") as handle:
                handle.write(manifest_text.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            final_dir = self.generations_dir / generation_name
            if final_dir.exists():
                shutil.rmtree(staging, ignore_errors=True)
            else:
                os.replace(staging, final_dir)
            pointer = {
                "schema": CURRENT_SCHEMA,
                "generation": generation_name,
                "manifest_sha256": manifest_fp,
                "global_version": global_version,
            }
            atomic_write_text(
                self.current_path,
                json.dumps(pointer, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                temp_dir=self.temp_dir,
            )
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def list_generations(self) -> list[dict]:
        result = []
        for path in sorted(self.generations_dir.iterdir() if self.generations_dir.exists() else []):
            if not path.is_dir():
                continue
            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                result.append({"generation": path.name, "complete": False})
                continue
            try:
                manifest = read_json(manifest_path)
                result.append(
                    {
                        "generation": path.name,
                        "complete": True,
                        "global_version": manifest.get("global_version"),
                        "partial": manifest.get("partial"),
                    }
                )
            except Exception:
                result.append({"generation": path.name, "complete": False})
        return result
