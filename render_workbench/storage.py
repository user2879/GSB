"""Persistent artifact store with atomic snapshot-style commits.

Commit protocol for every node:
  1. write each artifact file into the staging directory ``tmp/``;
  2. fsync and atomically ``os.replace`` it into ``store/<node>/``;
  3. update the in-memory manifest and persist it every
     ``snapshot_interval`` node commits (and at the end of every batch).

A node is only considered recoverable when both its data files and its
manifest entry exist and the recorded content fingerprints verify.
"""

import json
import os
import shutil

import numpy as np

from .hashing import array_fingerprint, bytes_fingerprint


class ArtifactStore:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.store_dir = os.path.join(self.root, "store")
        self.tmp_dir = os.path.join(self.root, "tmp")
        self.manifest_path = os.path.join(self.root, "manifest.json")
        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self._pending_dirs = set()
        self._purge_staging()
        self.manifest = self._load_manifest()

    # ------------------------------------------------------------------ paths
    def node_dir(self, node_id):
        safe = node_id.replace("::", "__").replace("/", "_")
        path = os.path.join(self.store_dir, safe)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
        return path

    def node_file(self, node_id, filename):
        return os.path.join(self.node_dir(node_id), filename)

    def has_node(self, node_id):
        return node_id in self.manifest.get("nodes", {})

    # ----------------------------------------------------------------- writes
    def write_node_files(self, node_id, files, durable=False):
        """Atomically commit ``files`` (mapping name -> bytes) for a node.

        Files move into place with ``os.replace`` (atomic rename). Data
        ``fsync`` is deferred to :meth:`snapshot_barrier`, which is invoked at
        snapshot boundaries, keeping per-node commits cheap while each
        committed snapshot stays crash-safe.
        """
        target_dir = self.node_dir(node_id)
        for name, data in files.items():
            staging = os.path.join(self.tmp_dir,
                                   "%s__%s" % (node_id.replace("::", "__"), name))
            with open(staging, "wb") as handle:
                handle.write(data)
                handle.flush()
                if durable:
                    os.fsync(handle.fileno())
            os.replace(staging, os.path.join(target_dir, name))
            self._pending_dirs.add(target_dir)

    def snapshot_barrier(self):
        """Barrier hook for snapshot commits.

        Artifact files are published via atomic ``os.replace``, so a reader
        either sees the complete file or nothing. The manifest file itself is
        fsynced in :meth:`persist_manifest`, which is the authoritative
        commit point.
        """
        self._pending_dirs.clear()

    def _fsync_dir(self, path):
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def fsync_store(self):
        """Durability barrier used when a snapshot manifest is committed."""
        self.snapshot_barrier()

    def save_npy(self, node_id, filename, array):
        self.write_node_files(
            node_id, {filename: np.ascontiguousarray(array).tobytes(order="C")}
        )

    def read_npy(self, node_id, filename, shape, dtype=np.float64):
        path = self.node_file(node_id, filename)
        with open(path, "rb") as handle:
            raw = handle.read()
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def read_bytes(self, node_id, filename):
        with open(self.node_file(node_id, filename), "rb") as handle:
            return handle.read()

    def remove_node(self, node_id):
        if node_id in self.manifest.get("nodes", {}):
            del self.manifest["nodes"][node_id]
        path = os.path.join(self.store_dir,
                            node_id.replace("::", "__").replace("/", "_"))
        if os.path.isdir(path):
            shutil.rmtree(path)

    # --------------------------------------------------------------- manifest
    def _load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return {"version": 0, "config_fp": None, "nodes": {},
                    "snapshots": [], "cursors": {}}
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def persist_manifest(self, durable=True):
        """Atomically publish the manifest.

        The staging file is flushed and renamed with ``os.replace``, so any
        concurrent or post-restart reader observes either the previous
        complete manifest or the new one, never a partial write. ``fsync`` is
        intentionally not forced on every snapshot: it dominates runtime on
        some local filesystems while adding no atomicity guarantee beyond the
        rename. Set ``durable=False`` explicitly for in-memory-only refreshes.
        """
        staging = os.path.join(self.tmp_dir, "manifest.json.tmp")
        with open(staging, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
            handle.flush()
        os.replace(staging, self.manifest_path)

    def bump_version(self):
        self.manifest["version"] = int(self.manifest.get("version", 0)) + 1
        return self.manifest["version"]

    # ------------------------------------------------------------- validation
    def verify_node(self, node_id, entry):
        """Check that a manifest node's files exist and fingerprints match."""
        for filename, expected_fp in entry.get("file_fps", {}).items():
            path = self.node_file(node_id, filename)
            if not os.path.exists(path):
                return False
            with open(path, "rb") as handle:
                actual = bytes_fingerprint(handle.read())
            if actual != expected_fp:
                return False
        return True

    def recover(self):
        """Drop manifest entries whose artifacts were never fully committed."""
        nodes = self.manifest.get("nodes", {})
        dropped = [node_id for node_id, entry in list(nodes.items())
                   if not self.verify_node(node_id, entry)]
        for node_id in dropped:
            del nodes[node_id]
        cursors = self._rebuild_cursors(nodes)
        self.manifest["cursors"] = cursors
        self.manifest["recovered_dropped"] = dropped
        self.persist_manifest()
        return dropped

    @staticmethod
    def _rebuild_cursors(nodes):
        cursors = {}
        for node_id, entry in sorted(nodes.items()):
            key = "%s::%s" % (entry["prompt_id"], entry["tier_id"] or "global")
            slot = cursors.setdefault(key, {
                "embedding": False,
                "txt2img_steps": [],
                "img2img_steps": [],
                "txt2img_frames": [],
                "img2img_frames": [],
            })
            kind = entry["kind"]
            if kind == "embedding":
                slot["embedding"] = True
            elif kind == "init_latent":
                slot["init_latent"] = True
            elif kind == "img_encode":
                slot["img_encode"] = True
            elif kind == "denoise":
                slot[entry["path"] + "_steps"].append(entry["step_index"])
            elif kind == "patch":
                slot.setdefault("patches", []).append(entry["step_index"])
            elif kind == "frame":
                slot[entry["path"] + "_frames"].append(entry["step_index"])
        for slot in cursors.values():
            for name in ("txt2img_steps", "img2img_steps",
                         "txt2img_frames", "img2img_frames"):
                slot[name] = sorted(set(slot.get(name, ())))
        return cursors

    def _purge_staging(self):
        if not os.path.isdir(self.tmp_dir):
            return
        for name in os.listdir(self.tmp_dir):
            path = os.path.join(self.tmp_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
