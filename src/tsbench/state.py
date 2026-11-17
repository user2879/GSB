"""On-disk cache, atomic snapshots and manifest-based recovery."""

from __future__ import annotations

import os

import json

from .errors import E_CONFIG_MISMATCH, ConfigError
from .util import atomic_write_json, content_fingerprint, read_json

MANIFEST_NAME = "manifest.json"


class RunState:
    """Owns all cache directories for one workbench output root."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.preprocess_dir = os.path.join(self.output_dir, "preprocess")
        self.feature_dir = os.path.join(self.output_dir, "features")
        self.model_dir = os.path.join(self.output_dir, "models")
        self.backtest_dir = os.path.join(self.output_dir, "backtests")
        self.report_dir = os.path.join(self.output_dir, "reports")
        self.log_path = os.path.join(self.output_dir, "update_log.jsonl")
        self.manifest_path = os.path.join(self.output_dir, MANIFEST_NAME)
        for directory in (
            self.preprocess_dir,
            self.feature_dir,
            self.model_dir,
            self.backtest_dir,
            self.report_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    # ---- generic cache helpers -------------------------------------------------

    @staticmethod
    def _path(base: str, series_id: str, key: str) -> str:
        safe = series_id.replace("/", "_")
        return os.path.join(base, f"{safe}__{key}.json")

    def load(self, base: str, series_id: str, key: str):
        path = self._path(base, series_id, key)
        if not os.path.exists(path):
            return None
        return read_json(path)

    def store(self, base: str, series_id: str, key: str, payload: dict) -> str:
        path = self._path(base, series_id, key)
        atomic_write_json(path, payload)
        return path

    # ---- manifest --------------------------------------------------------------

    def load_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        return read_json(self.manifest_path)

    def check_recoverable(self, config_hash: str, mode: str) -> dict | None:
        """Refuse recovery when the persisted config hash no longer matches."""
        manifest = self.load_manifest()
        if manifest is None:
            return None
        if manifest.get("config_hash") != config_hash:
            raise ConfigError(
                "stored manifest config hash does not match the current "
                f"configuration (stored={manifest.get('config_hash')[:12]}..., "
                f"current={config_hash[:12]}...)",
                E_CONFIG_MISMATCH,
            )
        return manifest

    def write_manifest(self, payload: dict) -> None:
        atomic_write_json(self.manifest_path, payload)

    def append_log(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def recovery_point(self) -> dict:
        manifest = self.load_manifest()
        if manifest is None:
            return {"recovered": False, "reason": "no manifest present"}
        return {
            "recovered": True,
            "global_version": manifest.get("global_version"),
            "last_batch": manifest.get("last_batch"),
            "completed_series": sorted(manifest.get("completed_series", [])),
            "stage_versions": manifest.get("stage_versions", {}),
            "data_fingerprint": manifest.get("data_fingerprint"),
        }
