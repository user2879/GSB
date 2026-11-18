"""Staged snapshots written atomically and validated on startup."""

from __future__ import annotations

import os

from . import SNAPSHOT_VERSION
from .errors import E_CONFIG_MISMATCH, PipelineError
from .util import atomic_write_json, fingerprint, read_json

STAGES = ("features", "scoring", "hallucination", "metrics")


class SnapshotStore:
    def __init__(self, directory: str, config_fp: str, rules_fp: str, tier_id: str):
        self.directory = directory
        self.config_fp = config_fp
        self.rules_fp = rules_fp
        self.tier_id = tier_id
        os.makedirs(directory, exist_ok=True)

    def _path(self, stage: str) -> str:
        return os.path.join(self.directory, f"snapshot_{stage}.json")

    def write(self, stage: str, cursor: int, artifacts: dict, data_fp: str = "") -> dict:
        snapshot = {
            "snapshot_version": SNAPSHOT_VERSION,
            "stage": stage,
            "config_fp": self.config_fp,
            "rules_fp": self.rules_fp,
            "tier_id": self.tier_id,
            "data_fp": data_fp,
            "cursor": int(cursor),
            "artifact_digest": fingerprint(artifacts),
            "artifacts": artifacts,
        }
        atomic_write_json(self._path(stage), snapshot)
        return snapshot

    def load(self, stage: str, strict: bool = False):
        path = self._path(stage)
        if not os.path.exists(path):
            return None
        snapshot = read_json(path)
        mismatch = (
            snapshot.get("snapshot_version") != SNAPSHOT_VERSION
            or snapshot.get("config_fp") != self.config_fp
            or snapshot.get("rules_fp") != self.rules_fp
            or snapshot.get("tier_id") != self.tier_id
        )
        if mismatch:
            if strict:
                raise PipelineError(
                    E_CONFIG_MISMATCH,
                    f"snapshot {stage} version/config/rules/tier mismatch; resume refused",
                    stage=stage,
                )
            return None
        return snapshot

    def reset(self) -> None:
        for stage in STAGES:
            path = self._path(stage)
            if os.path.exists(path):
                os.unlink(path)

    def versions(self) -> dict:
        result = {}
        for stage in STAGES:
            path = self._path(stage)
            if os.path.exists(path):
                result[stage] = read_json(path)["snapshot_version"]
        return result
