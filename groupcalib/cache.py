"""Immutable stage snapshots, cache loading, and stale-snapshot recovery."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .errors import E_SNAPSHOT_STALE, BusinessEvent
from .hashing import canonical_json, sha256_bytes, sha256_value
from .io_utils import atomic_write_text, read_json
from .version import SCHEMA_VERSION


STAGES = (
    "quantile_calibration",
    "fixed_rescale",
    "error_attribution",
    "sensitivity",
)


def _event(stage: str, reason: str, *, details: dict[str, Any] | None = None) -> BusinessEvent:
    payload = {"stage": stage, "reason": reason, "from_stage": stage}
    if details:
        payload.update(details)
    return BusinessEvent(
        code=E_SNAPSHOT_STALE,
        message=f"snapshot stage {stage!r} is stale: {reason}",
        sort_key=(stage, reason),
        details=payload,
    )


def _stage_version(files: dict[str, bytes]) -> str:
    digest_payload = [
        {"name": name, "sha256": sha256_bytes(data)} for name, data in sorted(files.items())
    ]
    return sha256_value(digest_payload)


def _stage_filename(record: dict[str, Any]) -> str:
    group_id = str(record["group_id"])
    bit_width = int(record["bit_width"])
    safe_group = sha256_value(group_id)[:16]
    return f"group-{safe_group}__w{bit_width:03d}.json"


class StageCache:
    def __init__(self, out_dir: str | os.PathLike[str]) -> None:
        self.out_dir = Path(out_dir)
        self.stages_dir = self.out_dir / "stages"
        self.manifest_path = self.out_dir / "manifest.json"
        self.records: dict[str, dict[str, dict[str, Any]]] = {stage: {} for stage in STAGES}
        self.manifest: dict[str, Any] | None = None
        self.events: list[BusinessEvent] = []
        self.loaded_stage_versions: dict[str, str] = {}
        self.available_stages: set[str] = set()

    def load(self, *, config_hash: str, full_rebuild: bool = False) -> None:
        if full_rebuild:
            return
        if not self.manifest_path.exists():
            return
        try:
            manifest = read_json(self.manifest_path)
        except Exception as exc:  # noqa: BLE001 - recovery must not crash startup
            self.events.append(_event("all", f"manifest cannot be read: {exc}"))
            return
        if not isinstance(manifest, dict):
            self.events.append(_event("all", "manifest root is not an object"))
            return
        self.manifest = manifest
        if manifest.get("global_version") != SCHEMA_VERSION:
            self.events.append(
                _event(
                    "all",
                    "global version mismatch",
                    details={"expected": SCHEMA_VERSION, "actual": manifest.get("global_version")},
                )
            )
            return
        if manifest.get("config_hash") != config_hash:
            self.events.append(
                _event(
                    "all",
                    "configuration hash mismatch",
                    details={"expected": config_hash, "actual": manifest.get("config_hash")},
                )
            )
            return

        manifest_stage_versions = manifest.get("stage_versions", {})
        if not isinstance(manifest_stage_versions, dict):
            self.events.append(_event("all", "stage_versions is not an object"))
            return

        for stage in STAGES:
            expected_version = manifest_stage_versions.get(stage)
            stage_root = self.stages_dir / stage
            current_path = stage_root / "CURRENT"
            if not isinstance(expected_version, str) or not expected_version:
                self.events.append(_event(stage, "stage version is missing from manifest"))
                continue
            if not current_path.exists():
                self.events.append(_event(stage, "CURRENT pointer is missing"))
                continue
            try:
                current_version = current_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                self.events.append(_event(stage, f"CURRENT pointer cannot be read: {exc}"))
                continue
            if current_version != expected_version:
                self.events.append(
                    _event(
                        stage,
                        "CURRENT pointer differs from manifest",
                        details={"expected": expected_version, "actual": current_version},
                    )
                )
                continue
            version_dir = stage_root / current_version
            if not version_dir.is_dir():
                self.events.append(_event(stage, "snapshot directory is missing"))
                continue
            try:
                file_payloads: dict[str, bytes] = {}
                for path in sorted(version_dir.glob("*.json")):
                    file_payloads[path.name] = path.read_bytes()
                actual_version = _stage_version(file_payloads)
            except OSError as exc:
                self.events.append(_event(stage, f"snapshot cannot be read: {exc}"))
                continue
            if actual_version != current_version:
                self.events.append(
                    _event(
                        stage,
                        "snapshot content hash mismatch",
                        details={"expected": current_version, "actual": actual_version},
                    )
                )
                continue
            try:
                loaded: dict[str, dict[str, Any]] = {}
                for filename, payload in sorted(file_payloads.items()):
                    record = json.loads(payload.decode("utf-8"))
                    if not isinstance(record, dict) or not isinstance(record.get("cache_key"), str):
                        raise ValueError(f"invalid stage record {filename}")
                    loaded[record["cache_key"]] = record
            except Exception as exc:  # noqa: BLE001
                self.events.append(_event(stage, f"snapshot JSON is invalid: {exc}"))
                continue
            self.records[stage] = loaded
            self.loaded_stage_versions[stage] = current_version
            self.available_stages.add(stage)

        # A missing upstream stage invalidates every dependent stage in this run.
        if "quantile_calibration" not in self.available_stages:
            for stage in STAGES[1:]:
                self.available_stages.discard(stage)
        if "fixed_rescale" not in self.available_stages:
            for stage in STAGES[2:]:
                self.available_stages.discard(stage)
        if "error_attribution" not in self.available_stages:
            self.available_stages.discard("sensitivity")

    def get(self, stage: str, cache_key: str) -> dict[str, Any] | None:
        if stage not in self.available_stages:
            return None
        return self.records.get(stage, {}).get(cache_key)

    def put(self, stage: str, record: dict[str, Any]) -> None:
        self.records[stage][record["cache_key"]] = record
        self.available_stages.add(stage)

    def commit_stage(self, stage: str, records: Iterable[dict[str, Any]]) -> str:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        by_filename: dict[str, bytes] = {}
        for record in records:
            filename = _stage_filename(record)
            if filename in by_filename:
                raise ValueError(f"duplicate stage record filename {filename}")
            by_filename[filename] = (canonical_json(record, pretty=True) + "\n").encode("utf-8")
        version = _stage_version(by_filename)
        stage_root = self.stages_dir / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        final_dir = stage_root / version
        if not final_dir.exists():
            temp_dir = Path(
                tempfile.mkdtemp(prefix=f".staging-{stage}-{version[:12]}-", dir=stage_root)
            )
            try:
                for filename, payload in sorted(by_filename.items()):
                    with open(temp_dir / filename, "wb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                try:
                    os.replace(temp_dir, final_dir)
                except OSError:
                    if not final_dir.exists():
                        raise
                    shutil.rmtree(temp_dir, ignore_errors=True)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
        current_temp = stage_root / f".CURRENT.{os.getpid()}.tmp"
        atomic_write_text(current_temp, version + "\n")
        os.replace(current_temp, stage_root / "CURRENT")
        self.loaded_stage_versions[stage] = version
        return version
