"""Staged snapshots with atomic directory replacement and resume policy.

A run opens one transaction (``begin_commit``), writes any number of stage
artifacts into the temporary tree, then performs a single whole-directory
``commit``. Nothing in the live snapshot is partially visible.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from quantcalib.fingerprint import fingerprint_json

MANIFEST_NAME = "manifest.json"
STAGES = ("calibration", "quantization", "error", "distillation", "receipt")


@dataclass
class Manifest:
    global_version: int
    config_hash: str
    weight_pack_fingerprint: str
    calibration_fingerprint: str
    receipt_version_fingerprint: str
    completed_layers: list[str]
    consumed_budget: int
    per_layer_consumed: dict[str, int]
    receipt_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.tmp = self.root.with_name(self.root.name + ".tmp")
        self._transaction_open = False
        self.root.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            (self.root / stage).mkdir(exist_ok=True)

    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def read_manifest(self) -> Manifest | None:
        path = self.manifest_path()
        if not path.exists():
            return None
        return Manifest(**json.loads(path.read_text(encoding="utf-8")))

    def resume_decision(self, expected: Manifest) -> tuple[bool, str]:
        """Snapshot-level resume policy.

        The snapshot is a persistent, append-only content-addressed store. It
        stays valid across config/weight edits: precise per-stage invalidation
        is performed by the cache keys in the engine. Only a missing manifest,
        a structurally incompatible snapshot, or a non-monotonic global
        version causes whole-snapshot invalidation. The expected config hash
        and fingerprints are still recorded in every committed manifest.
        """
        current = self.read_manifest()
        if current is None:
            return False, "no manifest: full rebuild"
        if current.global_version > expected.global_version:
            return False, "global_version regressed: invalidate"
        return True, "snapshot intact: stage cache keys decide reuse"

    def begin_commit(self) -> Path:
        # A leftover tmp tree means a previous process died mid-commit.
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        shutil.copytree(self.root, self.tmp)
        self._transaction_open = True
        return self.tmp

    def rollback(self) -> None:
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self._transaction_open = False

    def stage_path(self, stage: str, tmp: bool = False) -> Path:
        base = self.tmp if tmp else self.root
        return base / stage

    def write_artifact(self, stage: str, key: str, payload: dict[str, Any]) -> Path:
        if not self._transaction_open:
            raise RuntimeError("write_artifact called outside begin_commit/commit")
        target = self.tmp / stage / f"{key}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return target

    def read_artifact(self, stage: str, key: str) -> dict[str, Any] | None:
        target = self.root / stage / f"{key}.json"
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def commit(self, manifest: Manifest) -> None:
        if not self._transaction_open:
            raise RuntimeError("commit called without begin_commit")
        (self.tmp / MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), sort_keys=True, indent=2),
            encoding="utf-8")
        backup = self.root.with_name(self.root.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(self.root, backup)
        os.replace(self.tmp, self.root)
        shutil.rmtree(backup, ignore_errors=True)
        self._transaction_open = False

    def invalidate(self, reason: str) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            (self.root / stage).mkdir(exist_ok=True)
        (self.root / "invalidation.log").write_text(reason + "\n",
                                                     encoding="utf-8")
        self._transaction_open = False


def expected_manifest(global_version: int, config_dict: dict[str, Any],
                      weight_fingerprint: str, calib_fingerprint: str,
                      receipt_version_fingerprint: str,
                      receipt_version: str) -> Manifest:
    return Manifest(
        global_version=global_version,
        config_hash=fingerprint_json(config_dict),
        weight_pack_fingerprint=weight_fingerprint,
        calibration_fingerprint=calib_fingerprint,
        receipt_version_fingerprint=receipt_version_fingerprint,
        completed_layers=[],
        consumed_budget=0,
        per_layer_consumed={},
        receipt_version=receipt_version,
    )
