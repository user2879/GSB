from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .errors import Failure
from .fingerprint import canonical_json, digest
from .models import Config, Spec

STAGES = ("bucketing", "metrics", "drift", "attribution")
SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class SnapshotStore:
    """Content-addressed, stage-wise snapshots with atomic current pointers."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / "MANIFEST.json"

    def inspect(self, config: Config, specs: dict[str, Spec]) -> dict[str, Any]:
        expected = {
            "global_version": config.snapshot_version,
            "config_fingerprint": config.fingerprint(),
            "spec_versions": sorted(specs),
        }
        if not self.manifest_path.exists():
            return {"status": "new", "failures": [], "expected": expected, "manifest": None}
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failure = Failure(
                "E_SNAPSHOT_STALE",
                f"snapshot manifest is unreadable: {exc}",
                field_path="MANIFEST.json",
            )
            return {"status": "stale", "failures": [failure], "expected": expected, "manifest": None}

        differences: list[str] = []
        if manifest.get("global_version") != expected["global_version"]:
            differences.append("global_version")
        if manifest.get("config_fingerprint") != expected["config_fingerprint"]:
            differences.append("config_fingerprint")
        if manifest.get("spec_versions") != expected["spec_versions"]:
            differences.append("spec_versions")
        if differences:
            failure = Failure(
                "E_SNAPSHOT_STALE",
                "snapshot does not match current version/config/spec catalogue; rebuilding",
                field_path=",".join(differences),
                context={
                    "expected": expected,
                    "actual": {
                        "global_version": manifest.get("global_version"),
                        "config_fingerprint": manifest.get("config_fingerprint"),
                        "spec_versions": manifest.get("spec_versions"),
                    },
                },
            )
            return {"status": "stale", "failures": [failure], "expected": expected, "manifest": manifest}
        return {"status": "resumed", "failures": [], "expected": expected, "manifest": manifest}

    def load_entries(self) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}
        for stage in STAGES:
            current = self.root / "stages" / stage / "current"
            if not current.exists():
                continue
            try:
                payload = json.loads((current / "entries.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                entries[stage] = payload
        return entries

    def _commit_stage(self, stage: str, entries: dict[str, Any]) -> str:
        product_version = digest(entries)
        stage_root = self.root / "stages" / stage
        versions = stage_root / "versions"
        version_dir = versions / product_version
        versions.mkdir(parents=True, exist_ok=True)
        if not version_dir.exists():
            temporary = versions / f".tmp-{stage}-{os.getpid()}"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            _atomic_write_json(temporary / "entries.json", entries)
            os.replace(temporary, version_dir)
        link_tmp = stage_root / f".current.tmp-{os.getpid()}"
        try:
            link_tmp.unlink()
        except FileNotFoundError:
            pass
        os.symlink(Path("versions") / product_version, link_tmp)
        os.replace(link_tmp, stage_root / "current")
        return product_version

    def commit(
        self,
        entries: dict[str, dict[str, Any]],
        config: Config,
        specs: dict[str, Spec],
        *,
        committed_batch: int | None = None,
    ) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        stage_meta: dict[str, Any] = {}
        for stage in STAGES:
            stage_entries = entries.get(stage, {})
            version = self._commit_stage(stage, stage_entries)
            stage_meta[stage] = {
                "product_version": version,
                "entries_count": len(stage_entries),
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "global_version": config.snapshot_version,
            "config_fingerprint": config.fingerprint(),
            "spec_versions": sorted(specs),
            "active_spec_version": config.active_spec_version,
            "stages": stage_meta,
            "committed_batch": committed_batch,
        }
        _atomic_write_json(self.manifest_path, manifest)
        return manifest

    def report(self, status: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "enabled": True,
            "path": str(self.root),
            "status": status,
            "global_version": manifest.get("global_version") if manifest else None,
            "config_fingerprint": manifest.get("config_fingerprint") if manifest else None,
            "stage_versions": {
                name: details.get("product_version")
                for name, details in (manifest or {}).get("stages", {}).items()
            },
        }
