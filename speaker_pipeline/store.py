"""Content-addressed product store with atomic snapshot commits.

Products are written to a temporary directory and moved into place with a
single ``os.replace`` (atomic within the same filesystem).  The snapshot
manifest is always the last file replaced, so a manifest that exists and is
marked ``complete`` is proof of a fully committed snapshot; a process killed
mid-commit leaves at most a stale temp directory and no new manifest.
"""

import json
import os
import shutil
import tempfile

from .fingerprint import canonical_bytes, content_fingerprint
from .errors import SnapshotError

MANIFEST_NAME = "manifest.json"


class ProductStore:
    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self.work_dir = os.path.join(self.root_dir, "work_audio")
        self.product_dir = os.path.join(self.root_dir, "products")
        self.snapshot_dir = os.path.join(self.root_dir, "snapshots")
        self.tmp_dir = os.path.join(self.root_dir, "tmp")
        for directory in (
            self.work_dir,
            self.product_dir,
            self.snapshot_dir,
            self.tmp_dir,
        ):
            os.makedirs(directory, exist_ok=True)
        self.reused = []
        self.recomputed = []

    def reset_audit(self):
        self.reused = []
        self.recomputed = []

    # ---- content addressed products -------------------------------------
    def _product_path(self, key):
        safe = key.replace(":", "__")
        return os.path.join(self.product_dir, safe + ".json")

    def has_product(self, key):
        return os.path.exists(self._product_path(key))

    def get_product(self, key):
        with open(self._product_path(key), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def put_product(self, key, payload):
        path = self._product_path(key)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".prod-", suffix=".tmp", dir=self.tmp_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except OSError as exc:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise SnapshotError(
                f"failed to commit product {key}: {exc}", stage="snapshot"
            ) from exc

    def get_or_compute(self, key, stage, producer, references=None):
        """Reuse a cached product or compute+store it; audit both paths."""
        if self.has_product(key):
            self.reused.append(
                {
                    "stage": stage,
                    "cache_key": key,
                    "references": sorted(references or []),
                    "action": "reused",
                }
            )
            return self.get_product(key), True
        payload = producer()
        if isinstance(payload, tuple):
            payload, extra_refs = payload
            references = (references or []) + (extra_refs or [])
        if not isinstance(payload, dict):
            payload = {"value": payload}
        payload.setdefault("_cache_key", key)
        payload.setdefault("_stage", stage)
        payload.setdefault("_content_fingerprint", content_fingerprint(payload))
        payload.setdefault("_references", sorted(set(references or [])))
        self.put_product(key, payload)
        self.recomputed.append(
            {
                "stage": stage,
                "cache_key": key,
                "references": sorted(set(references or [])),
                "action": "recomputed",
            }
        )
        return payload, False

    # ---- snapshots -------------------------------------------------------
    def load_manifest(self):
        path = os.path.join(self.snapshot_dir, MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not manifest.get("complete"):
            return None
        return manifest

    def commit_snapshot(self, manifest, crash_before_commit=False):
        """Stage the manifest in a temp dir, then atomically replace it."""
        manifest = dict(manifest)
        manifest["complete"] = True
        staging = tempfile.mkdtemp(
            prefix=".snap-", dir=self.tmp_dir
        )
        tmp_manifest = os.path.join(staging, MANIFEST_NAME)
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(manifest, sort_keys=True, ensure_ascii=False)
            )
            handle.flush()
            os.fsync(handle.fileno())
        if crash_before_commit:
            # Simulate SIGKILL: leave the staged temp dir in place but do
            # NOT perform the atomic replace, so no manifest is published.
            raise SnapshotError(
                "simulated forced termination before atomic replace",
                stage="snapshot",
            )
        target = os.path.join(self.snapshot_dir, MANIFEST_NAME)
        os.replace(tmp_manifest, target)
        shutil.rmtree(staging, ignore_errors=True)
        return manifest

    def cleanup_temp(self):
        for name in os.listdir(self.tmp_dir):
            path = os.path.join(self.tmp_dir, name)
            if name.startswith((".snap-", ".prod-")):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def manifest_fingerprint(self, manifest):
        return content_fingerprint(
            {
                key: manifest[key]
                for key in (
                    "global_version",
                    "config_hash",
                    "command_cursor",
                    "threshold",
                    "override_ledger",
                    "stage_products",
                )
                if key in manifest
            }
        )
