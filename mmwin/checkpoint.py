from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_json, read_json, sha256_bytes
from .errors import CheckpointError


CHECKPOINT_SCHEMA = "mmwin-checkpoint-v1"
ENGINE_STATE_VERSION = 1
STAGE_FILES = {
    "replay": "replay.json",
    "membership": "membership.json",
    "visual": "visual.json",
    "fusion": "fusion.json",
    "resettle": "resettle.json",
    "pending": "pending.json",
    "runtime": "runtime.json",
    "cache": "cache.json",
}


@dataclass
class LoadedCheckpoint:
    manifest: dict[str, Any]
    payloads: dict[str, Any]
    generation_dir: Path


class CheckpointStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "MANIFEST.json"
        self.generations_dir = self.root / "generations"

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self) -> LoadedCheckpoint | None:
        if not self.manifest_path.exists():
            return None
        try:
            manifest = read_json(self.manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read checkpoint manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise CheckpointError("checkpoint manifest is not an object")
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise CheckpointError("unsupported checkpoint schema")
        generation_name = manifest.get("generation")
        if not isinstance(generation_name, str) or Path(generation_name).name != generation_name:
            raise CheckpointError("invalid checkpoint generation")
        generation_dir = self.generations_dir / generation_name
        if not generation_dir.is_dir():
            raise CheckpointError("checkpoint generation is missing")
        stages = manifest.get("stages")
        if not isinstance(stages, dict):
            raise CheckpointError("checkpoint manifest has no stage table")
        payloads: dict[str, Any] = {}
        for stage in STAGE_FILES:
            if stage not in stages:
                raise CheckpointError(f"checkpoint manifest is missing stage {stage}")
            entry = stages[stage]
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise CheckpointError(f"invalid stage entry {stage}")
            filename = entry["file"]
            if Path(filename).name != filename:
                raise CheckpointError(f"invalid stage filename {stage}")
            path = generation_dir / filename
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise CheckpointError(f"cannot read stage {stage}: {exc}") from exc
            if sha256_bytes(data) != entry.get("sha256"):
                raise CheckpointError(f"checkpoint stage hash mismatch: {stage}")
            try:
                payloads[stage] = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CheckpointError(f"invalid checkpoint stage {stage}: {exc}") from exc
        return LoadedCheckpoint(manifest=manifest, payloads=payloads, generation_dir=generation_dir)

    def commit(
        self,
        *,
        global_version: int,
        config_hash: str,
        plan_hash: str,
        source_hash: str,
        processed_arrival_seq: int | None,
        processed_arrival_count: int,
        pending_arrival_seqs: list[int],
        watermark_ms: int | None,
        payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Write every stage into a new generation, then atomically publish it."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        generation_name = f"g{global_version:08d}-{uuid.uuid4().hex[:12]}"
        staging = self.root / f".staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            stage_table: dict[str, Any] = {}
            for stage, filename in STAGE_FILES.items():
                payload = payloads[stage]
                data = (canonical_json(payload) + "\n").encode("utf-8")
                path = staging / filename
                with path.open("wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                stage_table[stage] = {
                    "file": filename,
                    "sha256": sha256_bytes(data),
                    "version": global_version,
                    "bytes": len(data),
                }
            manifest = {
                "schema": CHECKPOINT_SCHEMA,
                "engine_state_version": ENGINE_STATE_VERSION,
                "global_version": global_version,
                "generation": generation_name,
                "config_hash": config_hash,
                "plan_hash": plan_hash,
                "source_hash": source_hash,
                "processed_arrival_seq": processed_arrival_seq,
                "processed_arrival_count": processed_arrival_count,
                "pending_arrival_seqs": pending_arrival_seqs,
                "watermark_ms": watermark_ms,
                "stages": stage_table,
            }
            manifest_temp = staging / "MANIFEST.json"
            with manifest_temp.open("w", encoding="utf-8") as handle:
                handle.write(canonical_json(manifest) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            generation_dir = self.generations_dir / generation_name
            os.replace(staging, generation_dir)
            temp_manifest = self.root / f".MANIFEST.tmp-{uuid.uuid4().hex}"
            with temp_manifest.open("w", encoding="utf-8") as handle:
                handle.write(canonical_json(manifest) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_manifest, self.manifest_path)
            self._prune_generations(generation_name)
            return manifest
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _prune_generations(self, keep: str) -> None:
        if not self.generations_dir.exists():
            return
        generations = sorted(
            (path for path in self.generations_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        retained = {keep}
        for path in generations:
            if path.name in retained:
                continue
            if len(retained) < 2:
                retained.add(path.name)
                continue
            shutil.rmtree(path, ignore_errors=True)


def checkpoint_compat(manifest: dict[str, Any], *, config_hash: str, engine_version: int) -> tuple[bool, str | None]:
    if manifest.get("engine_state_version") != engine_version:
        return False, "E_CHECKPOINT_VERSION_MISMATCH"
    manifest_config = manifest.get("config_hash")
    if manifest_config != config_hash:
        return False, "E_CHECKPOINT_VERSION_MISMATCH"
    return True, None
