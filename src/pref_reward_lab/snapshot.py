from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, fingerprint, read_json
from .errors import SnapshotError

MANIFEST_SCHEMA_VERSION = 1
REQUIRED_FILES = ("features.json", "scoring.json", "conversion.json", "cache.json")


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.generations_dir = self.root / "generations"
        self.generations_dir.mkdir(parents=True, exist_ok=True)

    def latest_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            manifest = read_json(self.manifest_path)
        except (OSError, ValueError) as exc:
            raise SnapshotError(
                "E_SNAPSHOT_INCOMPLETE",
                f"snapshot manifest is not readable: {exc}",
            ) from exc
        if not isinstance(manifest, dict):
            raise SnapshotError("E_SNAPSHOT_INCOMPLETE", "snapshot manifest is not an object")
        return manifest

    def recover(
        self,
        *,
        config_hash: str,
        expected_global_version: int | None = None,
        expected_source_version: str | None = None,
    ) -> dict[str, Any]:
        manifest = self.latest_manifest()
        if manifest is None:
            return {"status": "empty", "recovery_point": None}
        reasons: list[str] = []
        if manifest.get("config_hash") != config_hash:
            reasons.append("config_hash differs")
        if (
            expected_global_version is not None
            and manifest.get("global_version") != expected_global_version
        ):
            reasons.append("global_version differs")
        if (
            expected_source_version is not None
            and manifest.get("source_version") != expected_source_version
        ):
            reasons.append("source_version differs")
        if reasons:
            raise SnapshotError(
                "E_CONFIG_MISMATCH",
                "snapshot cannot be resumed because " + ", ".join(reasons),
                manifest_config_hash=manifest.get("config_hash"),
                requested_config_hash=config_hash,
                manifest_global_version=manifest.get("global_version"),
                expected_global_version=expected_global_version,
                reasons=reasons,
            )
        generation = manifest.get("generation")
        if not isinstance(generation, str):
            raise SnapshotError("E_SNAPSHOT_INCOMPLETE", "manifest has no generation")
        generation_dir = self.generations_dir / generation
        self._validate_generation(generation_dir)
        return {
            "status": "restored",
            "recovery_point": {
                "generation": generation,
                "global_version": manifest.get("global_version"),
                "completed_round_cursor": manifest.get("completed_round_cursor"),
                "config_hash": manifest.get("config_hash"),
                "data_fingerprint": manifest.get("data_fingerprint"),
                "source_version": manifest.get("source_version"),
            },
            "manifest": manifest,
            "payload": self.load_payload(manifest),
        }

    def load_latest_unchecked(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            manifest = self.latest_manifest()
        except SnapshotError as exc:
            return None, exc.code
        if manifest is None:
            return None, None
        generation = manifest.get("generation")
        if not isinstance(generation, str):
            return None, "E_SNAPSHOT_INCOMPLETE"
        generation_dir = self.generations_dir / generation
        try:
            self._validate_generation(generation_dir)
        except SnapshotError:
            return None, "E_SNAPSHOT_INCOMPLETE"
        try:
            return self.load_payload(manifest), None
        except (OSError, ValueError, KeyError) as exc:
            return None, f"E_SNAPSHOT_INCOMPLETE:{exc}"

    def load_payload(self, manifest: dict[str, Any]) -> dict[str, Any]:
        generation_dir = self.generations_dir / str(manifest["generation"])
        payload: dict[str, Any] = {"manifest": manifest}
        for name in REQUIRED_FILES:
            payload[name.removesuffix(".json")] = read_json(generation_dir / name)
        return payload

    def commit(
        self,
        stage_payloads: dict[str, Any],
        *,
        global_version: int,
        config_hash: str,
        data_fingerprint: str,
        source_version: str,
        completed_round_cursor: int,
        commit_interval: int,
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        generation_base = f"gen-{global_version:08d}-{config_hash[:12]}"
        generation = generation_base
        final_dir = self.generations_dir / generation
        revision = 1
        while final_dir.exists():
            generation = f"{generation_base}-r{revision:02d}"
            final_dir = self.generations_dir / generation
            revision += 1
        temp_dir = Path(
            tempfile.mkdtemp(prefix=".tmp-snapshot-", dir=str(self.generations_dir))
        )
        try:
            for name in REQUIRED_FILES:
                stage_name = name.removesuffix(".json")
                if stage_name not in stage_payloads:
                    raise SnapshotError(
                        "E_SNAPSHOT_INCOMPLETE",
                        f"missing stage payload: {stage_name}",
                    )
                atomic_write_json(temp_dir / name, stage_payloads[stage_name])
            files = {
                name: fingerprint(read_json(temp_dir / name), prefix=f"{name}:")
                for name in REQUIRED_FILES
            }
            complete = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "generation": generation,
                "files": files,
            }
            atomic_write_json(temp_dir / "COMPLETE", complete)
            os.rename(temp_dir, final_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generation": generation,
            "global_version": global_version,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "source_version": source_version,
            "completed_round_cursor": completed_round_cursor,
            "snapshot_commit_interval": commit_interval,
            "checkpoint_due": completed_round_cursor % commit_interval == 0,
            "required_files": list(REQUIRED_FILES),
            "stage_fingerprints": files,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        atomic_write_json(self.manifest_path, manifest)
        self._cleanup_old_generations(generation)
        return manifest

    def write_uncommitted_crash_temp(self, name: str, payload: Any) -> Path:
        temp_dir = Path(
            tempfile.mkdtemp(prefix=".tmp-crash-", dir=str(self.generations_dir))
        )
        atomic_write_json(temp_dir / f"{name}.json", payload)
        return temp_dir

    @staticmethod
    def _validate_generation(generation_dir: Path) -> None:
        complete_marker = generation_dir / "COMPLETE"
        if not complete_marker.is_file():
            raise SnapshotError(
                "E_SNAPSHOT_INCOMPLETE",
                "manifest points to a generation without a COMPLETE marker",
                generation=generation_dir.name,
            )
        missing = [name for name in REQUIRED_FILES if not (generation_dir / name).is_file()]
        if missing:
            raise SnapshotError(
                "E_SNAPSHOT_INCOMPLETE",
                "snapshot generation is missing required files",
                generation=generation_dir.name,
                missing=missing,
            )
        try:
            complete = read_json(complete_marker)
        except (OSError, ValueError) as exc:
            raise SnapshotError(
                "E_SNAPSHOT_INCOMPLETE",
                f"snapshot COMPLETE marker is not readable: {exc}",
                generation=generation_dir.name,
            ) from exc
        expected = complete.get("files") if isinstance(complete, dict) else None
        if not isinstance(expected, dict):
            raise SnapshotError(
                "E_SNAPSHOT_INCOMPLETE",
                "snapshot COMPLETE marker has no file fingerprints",
                generation=generation_dir.name,
            )
        for name in REQUIRED_FILES:
            try:
                actual = fingerprint(read_json(generation_dir / name), prefix=f"{name}:")
            except (OSError, ValueError) as exc:
                raise SnapshotError(
                    "E_SNAPSHOT_INCOMPLETE",
                    f"snapshot stage file is not readable: {name}",
                    generation=generation_dir.name,
                ) from exc
            if expected.get(name) != actual:
                raise SnapshotError(
                    "E_SNAPSHOT_INCOMPLETE",
                    f"snapshot stage fingerprint mismatch: {name}",
                    generation=generation_dir.name,
                )

    def _cleanup_old_generations(self, current: str) -> None:
        generations = sorted(
            (
                path
                for path in self.generations_dir.iterdir()
                if path.is_dir() and path.name.startswith("gen-")
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for old in generations[2:]:
            if old.name != current:
                shutil.rmtree(old, ignore_errors=True)
