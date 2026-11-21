"""Staged snapshot commits with atomic replacement (requirement 8).

Stage products are written to a temporary directory first, then committed
by wholesale replacement.  The manifest records the global version, the
config hash and the version/fingerprint of every artifact.  On startup
the manifest is compared against the current config; a mismatch is
reported as E_SNAPSHOT_STALE so the caller can decide to rerun.
"""

from __future__ import annotations

import json
import os

from . import codes
from .fingerprint import fingerprint

ARTIFACTS = ("normalize_stats", "quantized", "weighted_sum", "clip", "report")


class Snapshotter:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.tmp_dir = os.path.join(run_dir, "tmp")
        self.snap_dir = os.path.join(run_dir, "snapshots")
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.snap_dir, exist_ok=True)

    @property
    def manifest_path(self):
        return os.path.join(self.snap_dir, "manifest.json")

    def _artifact_path(self, name):
        return os.path.join(self.snap_dir, f"{name}.json")

    def commit(self, version, config_hash, artifacts):
        """Write artifacts to tmp, then replace the snapshot wholesale."""
        staged = {}
        for name, payload in artifacts.items():
            tmp_path = os.path.join(self.tmp_dir, f"{name}.json")
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=1)
            staged[name] = tmp_path
        manifest = {
            "global_version": version,
            "config_hash": config_hash,
            "artifacts": {
                name: {"version": version, "fingerprint": fingerprint(payload)}
                for name, payload in artifacts.items()
            },
        }
        tmp_manifest = os.path.join(self.tmp_dir, "manifest.json")
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=1)
        # Wholesale replacement: move staged files into the snapshot dir.
        for name, tmp_path in staged.items():
            os.replace(tmp_path, self._artifact_path(name))
        os.replace(tmp_manifest, self.manifest_path)
        return manifest

    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def check_stale(self, config_hash):
        """Return a list of staleness findings (empty when up to date)."""
        manifest = self.load_manifest()
        if manifest is None:
            return []
        findings = []
        if manifest.get("config_hash") != config_hash:
            findings.append({
                "code": codes.E_SNAPSHOT_STALE,
                "detail": "config hash changed since last snapshot commit",
                "snapshot_version": manifest.get("global_version"),
                "snapshot_config_hash": manifest.get("config_hash"),
                "current_config_hash": config_hash,
            })
        return findings


def snapshot_artifacts(report):
    """Build the per-stage snapshot artifacts from a report."""
    quantized = {}
    weighted_sum = {}
    clip = {}
    for turn in report["turns"]:
        tid = turn["turn_id"]
        if turn["tier_config"] is None:
            continue
        quantized[tid] = turn["tier_config"]["quantize"]
        weighted_sum[tid] = turn["tier_config"]["weighted_sum"]
        clip[tid] = turn["tier_config"]["clip"]
    return {
        "normalize_stats": report.get("normalize_stats", {}),
        "quantized": quantized,
        "weighted_sum": weighted_sum,
        "clip": clip,
        "report": report,
    }
