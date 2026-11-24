"""Transactional stage snapshots, resume identity, and deterministic manifests."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import E_SNAPSHOT_CORRUPT, E_SNAPSHOT_STALE
from .util import (
    atomic_write_json,
    pretty_json,
    read_json,
    recover_directory,
    replace_directory,
    sha256_json,
)


SCHEMA_VERSION = 1
STAGES = (
    "parse_validation",
    "calibration",
    "quantization",
    "error_analysis",
    "distillation",
)


@dataclass
class RecoveryState:
    cache_allowed: bool
    status: str
    previous_status: str | None
    stale: bool = False
    corrupt: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


class SnapshotStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.stages_dir = self.output_dir / "stages"
        self.manifest_path = self.output_dir / "manifest.json"
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            recover_directory(self.stages_dir / stage)

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            value = read_json(self.manifest_path)
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def determine_recovery(self, identity: Mapping[str, Any]) -> RecoveryState:
        manifest = self.read_manifest()
        if manifest is None:
            if self.manifest_path.exists():
                return RecoveryState(
                    cache_allowed=False,
                    status="full",
                    previous_status="corrupt",
                    corrupt=True,
                    events=[
                        {
                            "business_code": E_SNAPSHOT_CORRUPT,
                            "message": "snapshot manifest is not valid JSON; rebuilding from scratch",
                        }
                    ],
                )
            return RecoveryState(cache_allowed=False, status="new", previous_status=None)
        previous_status = str(manifest.get("status", "unknown"))
        if not self._manifest_structurally_valid(manifest):
            return RecoveryState(
                cache_allowed=False,
                status="full",
                previous_status=previous_status,
                corrupt=True,
                events=[
                    {
                        "business_code": E_SNAPSHOT_CORRUPT,
                        "message": "snapshot manifest is incomplete or corrupt; rebuilding from scratch",
                    }
                ],
            )
        if previous_status in {"running", "interrupted"}:
            mismatched = [
                key
                for key in (
                    "config_hash",
                    "weight_pack_fingerprint",
                    "calibration_hash",
                    "quantization_profile",
                    "fractional_bits",
                )
                if manifest.get(key) != identity.get(key)
            ]
            if mismatched:
                return RecoveryState(
                    cache_allowed=False,
                    status="full",
                    previous_status=previous_status,
                    stale=True,
                    events=[
                        {
                            "business_code": E_SNAPSHOT_STALE,
                            "message": "unfinished snapshot identity differs from the current input; full rebuild required",
                            "mismatched_fields": mismatched,
                        }
                    ],
                )
            return RecoveryState(
                cache_allowed=True,
                status="resumed",
                previous_status=previous_status,
                events=[],
            )
        return RecoveryState(
            cache_allowed=True,
            status="incremental",
            previous_status=previous_status,
            events=[],
        )

    @staticmethod
    def _manifest_structurally_valid(manifest: Mapping[str, Any]) -> bool:
        required = (
            "global_version",
            "status",
            "config_hash",
            "weight_pack_fingerprint",
            "calibration_hash",
            "quantization_profile",
            "completed_layers",
            "stage_versions",
        )
        return all(key in manifest for key in required)

    def load_stage_results(self, stage: str) -> tuple[dict[str, dict[str, Any]], str | None]:
        stage_dir = self.stages_dir / stage
        data_path = stage_dir / "data.json"
        if not data_path.exists():
            return {}, None
        try:
            payload = read_json(data_path)
        except Exception:
            return {}, None
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
            return {}, None
        results: dict[str, dict[str, Any]] = {}
        for layer, value in payload["results"].items():
            if isinstance(layer, str) and isinstance(value, dict):
                results[layer] = value
        version = payload.get("stage_version")
        return results, str(version) if version is not None else None

    def save_stage(
        self,
        stage: str,
        *,
        identity: Mapping[str, Any],
        results: Mapping[str, Mapping[str, Any]],
        result_keys: Mapping[str, str],
    ) -> str:
        body = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "identity": dict(identity),
            "results": {name: copy.deepcopy(dict(results[name])) for name in sorted(results)},
            "result_keys": {name: result_keys[name] for name in sorted(result_keys) if name in results},
        }
        version = sha256_json(body)
        stage_dir = self.stages_dir / stage
        data_path = stage_dir / "data.json"
        if data_path.exists():
            try:
                current = read_json(data_path)
                if isinstance(current, dict) and current.get("stage_version") == version:
                    return version
            except Exception:
                pass
        payload = dict(body)
        payload["stage_version"] = version
        replace_directory(stage_dir, payload)
        return version

    def read_stage_versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        for stage in STAGES:
            path = self.stages_dir / stage / "data.json"
            if not path.exists():
                continue
            try:
                payload = read_json(path)
            except Exception:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("stage_version"), str):
                versions[stage] = payload["stage_version"]
        return versions

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        atomic_write_json(self.manifest_path, dict(manifest), pretty=True)

    def write_report(self, report: Mapping[str, Any]) -> None:
        atomic_write_json(self.output_dir / "report.json", dict(report), pretty=True)


def snapshot_global_version(identity: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "config_hash": identity.get("config_hash"),
            "weight_pack_fingerprint": identity.get("weight_pack_fingerprint"),
            "calibration_hash": identity.get("calibration_hash"),
            "quantization_profile": identity.get("quantization_profile"),
            "fractional_bits": identity.get("fractional_bits"),
        }
    )[:32]
