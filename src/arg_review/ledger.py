from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import fingerprint


class FingerprintLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"version": 1, "entries": {}, "artifacts": {}}
        self.data.setdefault("entries", {})
        self.data.setdefault("artifacts", {})
        self.old_artifacts = dict(self.data["artifacts"])
        self.reuse_hits: list[dict[str, Any]] = []
        self.invalidated: list[dict[str, Any]] = []
        self.recomputed: list[str] = []

    def has_content_key(self, cache_key: str) -> bool:
        return cache_key in self.data["entries"]

    def get_result(self, cache_key: str) -> Any:
        return self.data["entries"][cache_key]["result"]

    def inspect(self, artifact_id: str, cache_key: str) -> dict[str, Any] | None:
        old = self.old_artifacts.get(artifact_id)
        entry = self.data["entries"].get(cache_key)
        if entry is None:
            return None
        basis = ["content_cache_key_exists"]
        if old and old.get("cache_key") == cache_key:
            basis.append("stable_artifact_id_and_cache_key_match")
        elif old:
            basis.append("cache_key_reappeared_from_content_addressed_history")
        return {
            "artifact_id": artifact_id,
            "cache_key": cache_key,
            "result_hash": entry.get("result_hash"),
            "basis": basis,
        }

    def record_reuse(self, hit: dict[str, Any]) -> None:
        self.reuse_hits.append(hit)

    def change_reasons(
        self, artifact_id: str, dependency_fingerprints: dict[str, str]
    ) -> list[str]:
        old = self.old_artifacts.get(artifact_id)
        if not old:
            return ["new_artifact"]
        reasons: list[str] = []
        old_deps = old.get("dependency_fingerprints", {})
        for name in sorted(set(old_deps) | set(dependency_fingerprints)):
            if old_deps.get(name) != dependency_fingerprints.get(name):
                reasons.append(f"dependency_changed:{name}")
        if not reasons and old.get("cache_key"):
            reasons.append("cache_key_changed")
        return reasons

    def store_entry(
        self,
        artifact_id: str,
        stage: str,
        cache_key: str,
        result: Any,
        dependencies: list[str] | None = None,
        dependency_fingerprints: dict[str, str] | None = None,
    ) -> None:
        result_hash = fingerprint(result, 16)
        self.data["entries"][cache_key] = {
            "artifact_id": artifact_id,
            "stage": stage,
            "cache_key": cache_key,
            "result_hash": result_hash,
            "result": result,
            "dependencies": dependencies or [],
            "dependency_fingerprints": dependency_fingerprints or {},
        }
        self.recomputed.append(artifact_id)

    def commit_manifest(self, artifacts: dict[str, dict[str, Any]]) -> None:
        current_ids = set(artifacts)
        for artifact_id, old in sorted(self.old_artifacts.items()):
            if artifact_id not in current_ids:
                self.invalidated.append(
                    {
                        "artifact_id": artifact_id,
                        "old_cache_key": old.get("cache_key"),
                        "status": "removed",
                        "reasons": ["artifact_no_longer_in_final_state"],
                    }
                )
        self.data["artifacts"] = artifacts
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if Path(temporary_name).exists():
                Path(temporary_name).unlink()

    def run_lists(self) -> dict[str, Any]:
        reuse_hits = sorted(self.reuse_hits, key=lambda item: item["artifact_id"])
        invalidated = sorted(self.invalidated, key=lambda item: item["artifact_id"])
        recomputed = sorted(set(self.recomputed))
        return {
            "reuse_hits": reuse_hits,
            "invalidated": invalidated,
            "recomputed": recomputed,
            "restricted_recompute_set": recomputed,
            "reuse_artifact_ids": [item["artifact_id"] for item in reuse_hits],
            "invalidated_artifact_ids": [item["artifact_id"] for item in invalidated],
        }
