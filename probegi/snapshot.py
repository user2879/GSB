"""Atomic, versioned snapshot persistence.

Layout under the store root::

    snapshots/v000001/            committed snapshot (immutable)
    snapshots/v000002.partial/    in-flight commit (orphan until promoted)
    CURRENT.json                  {"version": N, "content_sha256": ...}
    orphans/<ts-seq>/             recovered uncommitted payloads
    recovery.log                  JSON lines, sequence-numbered (no wall clock)

Commit protocol: write every file into ``vNNNNNN.partial`` (each file via
tmp-write + fsync + os.replace), write the manifest with per-file content
hashes, fsync, rename the directory to ``vNNNNNN`` and finally replace
``CURRENT.json`` atomically.  A crash at any point before the rename leaves
only a ``.partial`` directory, which recovery reclaims as an orphan; a crash
after the rename but before ``CURRENT.json`` is replaced leaves the previous
committed state authoritative.  Recovery never fabricates an exit report.
"""
import json
import os
import shutil

from .errors import SnapshotStaleError
from .hashing import canonical_json, sha256_bytes, sha256_json

CURRENT = "CURRENT.json"
RECOVERY_LOG = "recovery.log"


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class SnapshotStore:
    def __init__(self, root: str):
        self.root = root
        self.snapshots = os.path.join(root, "snapshots")
        self.orphans = os.path.join(root, "orphans")
        os.makedirs(self.snapshots, exist_ok=True)
        os.makedirs(self.orphans, exist_ok=True)
        self._log_seq = self._recover()

    # -- recovery ------------------------------------------------------
    def _recover(self) -> int:
        """Reclaim uncommitted .partial payloads; returns next log seq."""
        seq = 0
        if os.path.exists(os.path.join(self.root, RECOVERY_LOG)):
            with open(os.path.join(self.root, RECOVERY_LOG),
                      encoding="utf-8") as fh:
                seq = sum(1 for _ in fh)
        reclaimed = []
        for name in sorted(os.listdir(self.snapshots)):
            if name.endswith(".partial"):
                src = os.path.join(self.snapshots, name)
                dst = os.path.join(self.orphans, name)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
                reclaimed.append(name)
        for name in reclaimed:
            self._log(seq, "orphan_reclaimed", {"payload": name})
            seq += 1
        return seq

    def _log(self, seq: int, event: str, payload: dict) -> None:
        line = canonical_json({"seq": seq, "event": event, **payload})
        with open(os.path.join(self.root, RECOVERY_LOG), "a",
                  encoding="utf-8") as fh:
            fh.write(line + "\n")

    # -- state ---------------------------------------------------------
    def current(self):
        path = os.path.join(self.root, CURRENT)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _verify_snapshot(self, version: int) -> dict:
        sdir = os.path.join(self.snapshots, f"v{version:06d}")
        mpath = os.path.join(sdir, "manifest.json")
        if not os.path.exists(mpath):
            raise SnapshotStaleError("snapshot manifest missing",
                                     version=version)
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
        for rel, digest in sorted(manifest["files"].items()):
            with open(os.path.join(sdir, rel), "rb") as fh:
                if sha256_bytes(fh.read()) != digest:
                    raise SnapshotStaleError("snapshot content hash mismatch",
                                             version=version, file=rel)
        return manifest

    # -- commit --------------------------------------------------------
    def commit(self, files: dict, base_version=None) -> dict:
        """Atomically commit {relative_path: bytes|dict}.  Optimistic
        concurrency: ``base_version`` must equal the current version."""
        cur = self.current()
        cur_version = cur["version"] if cur else 0
        if base_version is not None and base_version != cur_version:
            raise SnapshotStaleError("stale base version for commit",
                                     expected=cur_version, got=base_version)
        version = cur_version + 1
        payload = {}
        for rel, data in sorted(files.items()):
            if isinstance(data, (dict, list)):
                data = canonical_json(data).encode("utf-8")
            payload[rel] = data
        content_hash = sha256_json({rel: sha256_bytes(d)
                                    for rel, d in sorted(payload.items())})
        if cur and cur["content_sha256"] == content_hash:
            raise SnapshotStaleError(
                "identical content already committed; refusing cache reuse",
                version=cur_version)
        partial = os.path.join(self.snapshots, f"v{version:06d}.partial")
        if os.path.exists(partial):
            shutil.rmtree(partial)
        os.makedirs(partial)
        for rel, data in payload.items():
            dest = os.path.join(partial, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _atomic_write(dest, data)
        manifest = {"version": version, "content_sha256": content_hash,
                    "files": {rel: sha256_bytes(d)
                              for rel, d in sorted(payload.items())}}
        _atomic_write(os.path.join(partial, "manifest.json"),
                      canonical_json(manifest).encode("utf-8"))
        final = os.path.join(self.snapshots, f"v{version:06d}")
        os.replace(partial, final)  # commit point
        _atomic_write(os.path.join(self.root, CURRENT), canonical_json(
            {"version": version, "content_sha256": content_hash}
        ).encode("utf-8"))
        self._log(self._log_seq, "commit", {"version": version,
                                            "content_sha256": content_hash})
        self._log_seq += 1
        return manifest

    # -- load ----------------------------------------------------------
    def load(self) -> dict:
        """Load the last committed state; verifies version + content hash."""
        cur = self.current()
        if cur is None:
            raise SnapshotStaleError("no committed snapshot")
        manifest = self._verify_snapshot(cur["version"])
        if manifest["content_sha256"] != cur["content_sha256"]:
            raise SnapshotStaleError("CURRENT content hash mismatch",
                                     version=cur["version"])
        sdir = os.path.join(self.snapshots, f"v{cur['version']:06d}")
        out = {"version": cur["version"],
               "content_sha256": cur["content_sha256"], "files": {}}
        for rel in manifest["files"]:
            with open(os.path.join(sdir, rel), "rb") as fh:
                out["files"][rel] = fh.read()
        return out


def content_hash_for(files: dict) -> str:
    """Content hash a commit would produce for {relative_path: bytes|dict}."""
    payload = {}
    for rel, data in sorted(files.items()):
        if isinstance(data, (dict, list)):
            data = canonical_json(data).encode("utf-8")
        payload[rel] = data
    return sha256_json({rel: sha256_bytes(d)
                        for rel, d in sorted(payload.items())})
