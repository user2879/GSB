from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .engine import Engine, EngineOptions
from .model import STABLE_SCHEMA_VERSION


@dataclass
class SnapshotLoadResult:
    engine: Engine | None
    manifest: dict[str, Any] | None
    stale: bool = False
    stale_reason: str | None = None
    command_cursor: int = 0
    artifacts_valid: bool = True


def command_prefix_hash(commands: list[dict[str, Any]], cursor: int) -> str:
    prefix = commands[: max(0, min(cursor, len(commands)))]
    return hashlib.sha256(canonical_json(prefix).encode("utf-8")).hexdigest()


class SnapshotManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.stages_dir = self.output_dir / "stages"
        self.manifest_path = self.output_dir / "manifest.json"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"__corrupt__": True}
        return data if isinstance(data, dict) else {"__corrupt__": True}

    def veto_if_stale(
        self,
        manifest: dict[str, Any] | None,
        *,
        input_hashes: dict[str, str],
        commands: list[dict[str, Any]],
    ) -> SnapshotLoadResult:
        if manifest is None:
            return SnapshotLoadResult(engine=None, manifest=None)
        if manifest.get("__corrupt__"):
            return SnapshotLoadResult(
                engine=None,
                manifest=manifest,
                stale=True,
                stale_reason="manifest is not valid JSON",
            )
        if manifest.get("schema_version") != STABLE_SCHEMA_VERSION:
            return SnapshotLoadResult(
                engine=None,
                manifest=manifest,
                stale=True,
                stale_reason="snapshot schema version mismatch",
            )
        manifest_inputs = manifest.get("input_hashes", {})
        for name in ("lexicon", "config", "params", "sentences"):
            expected = manifest_inputs.get(name)
            actual = input_hashes.get(name)
            if expected is not None and actual != expected:
                return SnapshotLoadResult(
                    engine=None,
                    manifest=manifest,
                    stale=True,
                    stale_reason=f"{name} hash mismatch",
                )
        cursor = int(manifest.get("command_cursor", 0))
        if cursor < 0 or cursor > len(commands):
            return SnapshotLoadResult(
                engine=None,
                manifest=manifest,
                stale=True,
                stale_reason="command cursor is outside command stream",
            )
        expected_prefix = manifest.get("command_prefix_hash")
        actual_prefix = command_prefix_hash(commands, cursor)
        if expected_prefix is not None and expected_prefix != actual_prefix:
            return SnapshotLoadResult(
                engine=None,
                manifest=manifest,
                stale=True,
                stale_reason="command prefix hash mismatch",
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "alignment",
            "decode",
            "ledger",
        }:
            return SnapshotLoadResult(
                engine=None,
                manifest=manifest,
                stale=True,
                stale_reason="artifact manifest is incomplete",
                artifacts_valid=False,
            )
        version_dir = self.stages_dir / str(manifest.get("version_dir", ""))
        for name, artifact in artifacts.items():
            path = version_dir / artifact["file"]
            if not path.exists():
                return SnapshotLoadResult(
                    engine=None,
                    manifest=manifest,
                    stale=True,
                    stale_reason=f"artifact {name} is missing",
                    artifacts_valid=False,
                )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact.get("sha256"):
                return SnapshotLoadResult(
                    engine=None,
                    manifest=manifest,
                    stale=True,
                    stale_reason=f"artifact {name} hash mismatch",
                    artifacts_valid=False,
                )
        return SnapshotLoadResult(
            engine=None,
            manifest=manifest,
            command_cursor=cursor,
            artifacts_valid=True,
        )

    def restore(
        self,
        load_result: SnapshotLoadResult,
        *,
        options: EngineOptions | None = None,
    ) -> Engine | None:
        if (
            load_result.manifest is None
            or load_result.stale
            or not load_result.artifacts_valid
        ):
            return None
        manifest = load_result.manifest
        version_dir = self.stages_dir / str(manifest["version_dir"])
        try:
            alignment = json.loads(
                (version_dir / manifest["artifacts"]["alignment"]["file"]).read_text(
                    encoding="utf-8"
                )
            )
            decode = json.loads(
                (version_dir / manifest["artifacts"]["decode"]["file"]).read_text(
                    encoding="utf-8"
                )
            )
            ledger = json.loads(
                (version_dir / manifest["artifacts"]["ledger"]["file"]).read_text(
                    encoding="utf-8"
                )
            )
            engine_data = ledger["engine"]
            engine_data["caches"] = {
                "alignment": alignment["alignment"],
                "decode": decode["decode"],
                "confidence": decode["confidence"],
            }
            engine = Engine.from_snapshot_dict(engine_data, options=options)
            return engine
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def commit(
        self,
        engine: Engine,
        *,
        input_hashes: dict[str, str],
        commands: list[dict[str, Any]],
        command_cursor: int,
        artifact_version: int | None = None,
    ) -> dict[str, Any]:
        previous = self.read_manifest()
        if previous and previous.get("__corrupt__"):
            previous = None
        previous_version = int(previous.get("global_version", 0)) if previous else 0
        version = (
            artifact_version
            if artifact_version is not None
            else previous_version + 1
        )
        temp_dir = self.stages_dir / f".tmp-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        engine_snapshot = engine.to_snapshot_dict()
        caches = engine_snapshot.pop("caches")
        payloads = {
            "alignment": {
                "schema_version": STABLE_SCHEMA_VERSION,
                "alignment": caches["alignment"],
            },
            "decode": {
                "schema_version": STABLE_SCHEMA_VERSION,
                "decode": caches["decode"],
                "confidence": caches["confidence"],
            },
            "ledger": {
                "schema_version": STABLE_SCHEMA_VERSION,
                "engine": engine_snapshot,
            },
        }
        artifact_files = {
            "alignment": "alignment.json",
            "decode": "decode.json",
            "ledger": "ledger.json",
        }
        artifact_manifest: dict[str, Any] = {}
        for stage, payload in payloads.items():
            file_name = artifact_files[stage]
            path = temp_dir / file_name
            encoded = canonical_json(payload).encode("utf-8")
            with path.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            artifact_manifest[stage] = {
                "file": file_name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
                "stage_version": version,
            }

        target_dir = self.stages_dir / f"v{version}"
        suffix = 0
        while target_dir.exists():
            suffix += 1
            target_dir = self.stages_dir / f"v{version}-{suffix}"
        os.replace(temp_dir, target_dir)

        manifest = {
            "schema_version": STABLE_SCHEMA_VERSION,
            "global_version": version,
            "version_dir": target_dir.name,
            "lexicon_hash": engine.state.lexicon_hash(),
            "input_lexicon_hash": input_hashes.get("lexicon"),
            "params_fingerprint": engine.params.fingerprint(),
            "config_fingerprint": engine.config.fingerprint(),
            "state_fingerprint": engine.state.state_fingerprint(),
            "completed_batch_cursor": engine.summary()["completed_batch_cursor"],
            "pending_ids": sorted(engine.pending_ids),
            "command_cursor": command_cursor,
            "command_prefix_hash": command_prefix_hash(commands, command_cursor),
            "input_hashes": dict(sorted(input_hashes.items())),
            "artifacts": artifact_manifest,
        }
        manifest_temp = self.output_dir / f".manifest-{uuid.uuid4().hex}.tmp"
        with manifest_temp.open("wb") as handle:
            handle.write(canonical_json(manifest).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(manifest_temp, self.manifest_path)
        self._cleanup_old_versions(target_dir.name)
        return manifest

    def _cleanup_old_versions(self, current_name: str) -> None:
        version_dirs = sorted(
            [
                path
                for path in self.stages_dir.iterdir()
                if path.is_dir() and path.name.startswith("v")
            ],
            key=lambda path: path.stat().st_mtime_ns,
        )
        keep = {current_name}
        if version_dirs:
            for path in version_dirs[:-2]:
                if path.name not in keep:
                    shutil.rmtree(path, ignore_errors=True)
