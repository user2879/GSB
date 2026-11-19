"""Cache-session snapshots: versioned, content-addressed, full replacement."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .digest import digest_bytes
from .errors import SnapshotError

SNAPSHOT_FORMAT = 1
SNAPSHOT_MANIFEST = "snapshot.json"
SNAPSHOT_PAYLOAD = "session.json"
READONLY_MARKER = "READONLY"


class SnapshotLibrary:
    def __init__(self, snapshots_dir):
        self.dir = Path(snapshots_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_dir(self, name) -> Path:
        return self.dir / name

    def create(self, name, session_doc):
        snap_dir = self._snapshot_dir(name)
        if snap_dir.exists():
            raise SnapshotError("snapshot name already exists", {"name": name})
        tmp_dir = self.dir / f".{name}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        payload = json.dumps(session_doc, sort_keys=True, indent=2).encode("utf-8")
        payload_digest = digest_bytes(payload)
        (tmp_dir / SNAPSHOT_PAYLOAD).write_bytes(payload)
        manifest = {
            "snapshot_format": SNAPSHOT_FORMAT,
            "payload_digest": payload_digest,
            "payload_len": len(payload),
            "pipeline_version": session_doc["pipeline_version"],
            "entry_digest_set": session_doc["entry_digest_set"],
            "graph_version": session_doc["graph_version"],
            "hit_count": len(session_doc["hits"]),
            "invalidation_roots": sorted(session_doc["invalidations"]),
            "state_signature": _state_signature(session_doc),
        }
        (tmp_dir / SNAPSHOT_MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        os.replace(tmp_dir, snap_dir)
        return manifest

    def open(self, name):
        snap_dir = self._snapshot_dir(name)
        manifest_path = snap_dir / SNAPSHOT_MANIFEST
        payload_path = snap_dir / SNAPSHOT_PAYLOAD
        if not manifest_path.exists() or not payload_path.exists():
            raise SnapshotError("snapshot not found", {"name": name})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("snapshot_format") != SNAPSHOT_FORMAT:
            raise SnapshotError(
                "unsupported snapshot format",
                {"found": manifest.get("snapshot_format"),
                 "supported": SNAPSHOT_FORMAT},
            )
        payload = payload_path.read_bytes()
        if len(payload) != manifest["payload_len"]:
            raise SnapshotError("snapshot length mismatch", {"name": name})
        actual_digest = digest_bytes(payload)
        if actual_digest != manifest["payload_digest"]:
            raise SnapshotError(
                "snapshot digest mismatch",
                {"expected": manifest["payload_digest"],
                 "actual": actual_digest},
            )
        session_doc = json.loads(payload.decode("utf-8"))
        return manifest, session_doc

    def list_snapshots(self):
        out = []
        for child in sorted(self.dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            manifest_path = child / SNAPSHOT_MANIFEST
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            out.append(
                {
                    "name": child.name,
                    "readonly": (child / READONLY_MARKER).exists(),
                    "snapshot_format": manifest["snapshot_format"],
                    "payload_digest": manifest["payload_digest"],
                    "pipeline_version": manifest["pipeline_version"],
                    "graph_version": manifest["graph_version"],
                    "entry_digest_set": manifest["entry_digest_set"],
                }
            )
        return out

    def mark_readonly(self, name):
        (self._snapshot_dir(name) / READONLY_MARKER).write_text(
            "frozen\n", encoding="utf-8"
        )

    def compare(self, name, current_doc):
        manifest, _ = self.open(name)
        current_set = sorted(current_doc["entry_digest_set"])
        stored_set = sorted(manifest["entry_digest_set"])
        current_signature = _state_signature(current_doc)
        state_equal = manifest.get("state_signature") == current_signature
        return {
            "name": name,
            "snapshot_pipeline_version": manifest["pipeline_version"],
            "current_pipeline_version": current_doc["pipeline_version"],
            "snapshot_graph_version": manifest["graph_version"],
            "current_graph_version": current_doc["graph_version"],
            "digest_set_equal": current_set == stored_set,
            "state_signature_equal": state_equal,
            "only_in_snapshot": sorted(set(stored_set) - set(current_set)),
            "only_on_disk": sorted(set(current_set) - set(stored_set)),
            "versions_equal": (
                manifest["pipeline_version"] == current_doc["pipeline_version"]
                and manifest["graph_version"] == current_doc["graph_version"]
                and current_set == stored_set
                and state_equal
            ),
        }


def _state_signature(session_doc):
    """Stable fingerprint of hit history and invalidation trails."""
    hits = tuple(
        (h["key"], h["outcome"], h["reason"], h["seq"])
        for h in session_doc.get("hits", [])
    )
    invalidations = tuple(
        (root, tuple(session_doc["invalidations"][root]))
        for root in sorted(session_doc.get("invalidations", {}))
    )
    joined = json.dumps(
        {"hits": hits, "invalidations": invalidations}, sort_keys=True
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
