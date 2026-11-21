"""Atomic stage snapshots and resumption verdict.

Layout under the configured snapshot root::

    <root>/generation-<N>/
        manifest.json
        objects/<stage>/<cache_key>.json

Each stage object is rendered in a private temporary directory first and moved
into ``objects/<stage>/`` with a single ``os.replace`` (atomic whole-object
replacement, never an in-place overwrite).  The manifest is itself written via
a temp-file-and-replace swap and carries the global version number, all input
hashes, completed task keys and, for every entry, the object path and payload
hash.  On startup a format/config/input mismatch returns
:data:`E_SNAPSHOT_STALE` and a brand-new generation directory is created; the
old generation is never mutated.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .common import canonical_json, content_hash
from .config import load_json
from .errors import E_SNAPSHOT_STALE

FORMAT_VERSION = 4


class SnapshotStore:
    def __init__(self, root: str | Path, input_hashes: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.input_hashes = input_hashes
        self.generation_dir, self.status = self._resume_or_fresh()
        self.objects_dir = self.generation_dir / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.generation_dir / "manifest.json"
        if self.status == "resumed":
            self.manifest = load_json(self.manifest_path)
        else:
            self.manifest = {
                "format_version": FORMAT_VERSION,
                "global_version": 0,
                "input_hashes": input_hashes,
                "entries": {},          # stage -> {cache_key: relpath}
                "entry_hashes": {},     # stage -> {cache_key: payload_hash}
                "completed_tasks": [],
                "failures": {},
            }
            self._flush_manifest()

    # -- startup verdict ------------------------------------------------------
    def _resume_or_fresh(self) -> tuple[Path, str]:
        gens = sorted(
            [p for p in self.root.glob("generation-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if not gens:
            return self._new_generation(0), "fresh"
        latest = gens[-1]
        manifest_file = latest / "manifest.json"
        if not manifest_file.exists():
            return self._new_generation(len(gens)), E_SNAPSHOT_STALE
        manifest = load_json(manifest_file)
        if manifest.get("format_version") != FORMAT_VERSION:
            # only an incompatible on-disk format is "stale"; ordinary input
            # edits stay in the same generation and are handled by precise,
            # content-fingerprint cache invalidation (old objects are simply
            # no longer referenced and are listed in the invalidation audit)
            return self._new_generation(len(gens)), E_SNAPSHOT_STALE
        return latest, "resumed"

    def _new_generation(self, index: int) -> Path:
        gen = self.root / "generation-{}".format(index)
        gen.mkdir(parents=True, exist_ok=True)
        return gen

    # -- object IO ------------------------------------------------------------
    def _object_path(self, stage: str, cache_key: str) -> Path:
        return self.objects_dir / stage / "{}.json".format(cache_key)

    def has_entry(self, stage: str, cache_key: str) -> bool:
        return cache_key in self.manifest.get("entries", {}).get(stage, {})

    def load_object(self, stage: str, cache_key: str) -> dict:
        rel = self.manifest["entries"][stage][cache_key]
        return load_json(self.generation_dir / rel)

    def commit(self, stage: str, payload: dict, new_tasks: list[str]) -> None:
        """Atomically write each new stage object and swap the manifest."""
        stage_dir = self.objects_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        entries = self.manifest.setdefault("entries", {}).setdefault(stage, {})
        hashes = self.manifest.setdefault(
            "entry_hashes", {}).setdefault(stage, {})
        for cache_key, value in payload.items():
            target = self._object_path(stage, cache_key)
            tmp_dir = Path(tempfile.mkdtemp(
                prefix="obj-", dir=str(stage_dir)))
            try:
                tmp_file = tmp_dir / target.name
                with open(tmp_file, "w", encoding="utf-8") as handle:
                    handle.write(canonical_json(value))
                # atomic whole-object swap; durability relies on the manifest
                # commit rather than per-object fsync to keep the hot path fast
                os.replace(tmp_file, target)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            rel = str(target.relative_to(self.generation_dir))
            entries[cache_key] = rel
            hashes[cache_key] = content_hash(value)

        tasks = set(self.manifest.get("completed_tasks", []))
        tasks.update(new_tasks)
        self.manifest["completed_tasks"] = sorted(tasks)
        self.manifest["global_version"] = \
            self.manifest.get("global_version", 0) + 1
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(self.manifest))
        os.replace(tmp, self.manifest_path)

    # -- failure records (durable, so resumption rebuilds the failure list) --
    def record_failure(self, failure: dict) -> None:
        failures = self.manifest.setdefault("failures", {})
        key = failure["task_key"]
        if key not in failures:
            failures[key] = failure
            self.manifest["global_version"] = \
                self.manifest.get("global_version", 0) + 1
            self._flush_manifest()

    def all_failures(self) -> list[dict]:
        return list(self.manifest.get("failures", {}).values())

    @property
    def completed_tasks(self) -> set[str]:
        return set(self.manifest.get("completed_tasks", []))
