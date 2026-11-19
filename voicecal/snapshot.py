"""Atomic, generation-based snapshot storage."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    E_CONFIG_MISMATCH,
    E_SNAPSHOT_INVALID,
    PRODUCT_VERSION,
    SNAPSHOT_ARTIFACT_VERSION,
    STAGE_ARTIFACT_VERSION,
    STAGES,
)
from .errors import VoicecalError
from .util import atomic_write_json, canonical_json, json_hash, read_json


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.generations_dir = root / "generations"

    def has_snapshot(self) -> bool:
        return self.manifest_path.is_file()

    def load_manifest(self) -> dict[str, Any]:
        try:
            manifest = read_json(self.manifest_path)
        except FileNotFoundError as exc:
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot manifest is missing", file=str(self.manifest_path)) from exc
        except (ValueError, OSError) as exc:
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot manifest is unreadable", file=str(self.manifest_path)) from exc
        if not isinstance(manifest, dict):
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot manifest must be an object")
        return manifest

    def validate_manifest(self, expected_base_config_hash: str) -> dict[str, Any]:
        manifest = self.load_manifest()
        if manifest.get("artifact_version") != SNAPSHOT_ARTIFACT_VERSION:
            raise VoicecalError(
                E_CONFIG_MISMATCH,
                "snapshot artifact version does not match runtime",
                expected=SNAPSHOT_ARTIFACT_VERSION,
                actual=manifest.get("artifact_version"),
            )
        if manifest.get("product_version") != PRODUCT_VERSION:
            raise VoicecalError(
                E_CONFIG_MISMATCH,
                "snapshot product version does not match runtime",
                expected=PRODUCT_VERSION,
                actual=manifest.get("product_version"),
            )
        if manifest.get("base_config_hash") != expected_base_config_hash:
            raise VoicecalError(
                E_CONFIG_MISMATCH,
                "snapshot configuration hash does not match current configuration",
                expected=expected_base_config_hash,
                actual=manifest.get("base_config_hash"),
            )
        stages = manifest.get("stages")
        if not isinstance(stages, dict) or any(stage not in stages for stage in STAGES):
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot manifest is missing stage metadata")
        return manifest

    def load_state(self, manifest: dict[str, Any]) -> dict[str, Any]:
        generation_id = manifest.get("active_generation")
        if not isinstance(generation_id, str) or not generation_id:
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot manifest has no active generation")
        state_path = self.generations_dir / generation_id / "state.json"
        try:
            state = read_json(state_path)
        except FileNotFoundError as exc:
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot state payload is missing", file=str(state_path)) from exc
        except (ValueError, OSError) as exc:
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot state payload is unreadable", file=str(state_path)) from exc
        if not isinstance(state, dict):
            raise VoicecalError(E_SNAPSHOT_INVALID, "snapshot state payload must be an object")
        return state

    def commit(
        self,
        state: dict[str, Any],
        stage_pools: dict[str, dict[str, Any]],
        base_config_hash: str,
        effective_config_hash: str,
        injection_hash: str,
        fixed_point_bits: int,
        turn_cursor: int,
        processed_turn_count: int,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        state_json = canonical_json(state)
        generation_id = f"g-{turn_cursor:08d}-{json_hash(state_json)[:16]}"
        generation_dir = self.generations_dir / generation_id
        if generation_dir.exists():
            generation_id = f"{generation_id}-{uuid.uuid4().hex[:8]}"
            generation_dir = self.generations_dir / generation_id

        temp_dir = Path(tempfile.mkdtemp(prefix=".tmp-generation-", dir=self.generations_dir))
        try:
            for stage in STAGES:
                payload = {
                    "stage": stage,
                    "artifact_version": STAGE_ARTIFACT_VERSION,
                    "config_hash": effective_config_hash,
                    "injection_hash": injection_hash,
                    "fixed_point_bits": fixed_point_bits,
                    "records": stage_pools.get(stage, {}),
                }
                atomic_write_json(temp_dir / f"{stage}.json", payload)
            atomic_write_json(temp_dir / "state.json", state)
            os.replace(temp_dir, generation_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        stages_manifest: dict[str, Any] = {}
        for stage in STAGES:
            stages_manifest[stage] = {
                "stage": stage,
                "config_hash": effective_config_hash,
                "injection_hash": injection_hash,
                "turn_cursor": turn_cursor,
                "fixed_point_bits": fixed_point_bits,
                "artifact_version": STAGE_ARTIFACT_VERSION,
                "payload": str((self.generations_dir / generation_id / f"{stage}.json").relative_to(self.root)),
                "payload_hash": json_hash(stage_pools.get(stage, {})),
            }
        manifest = {
            "artifact_version": SNAPSHOT_ARTIFACT_VERSION,
            "product_version": PRODUCT_VERSION,
            "base_config_hash": base_config_hash,
            "config_hash": effective_config_hash,
            "injection_hash": injection_hash,
            "active_generation": generation_id,
            "state_path": str((self.generations_dir / generation_id / "state.json").relative_to(self.root)),
            "turn_cursor": turn_cursor,
            "processed_turn_count": processed_turn_count,
            "stages": stages_manifest,
            "committed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(self.manifest_path, manifest)
