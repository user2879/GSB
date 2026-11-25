"""Content-addressed cache, append-only ledger, and staged snapshot commits.
Every run writes artifacts into a staging directory and atomically swaps it
into place, so a crash never leaves a half-written store."""
from __future__ import annotations

import hashlib
import json
import os
import shutil


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fp(obj):
    """Stable content fingerprint used for cache keys and manifest hashes."""
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()[:24]


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


class Store:
    """Persistent run state. Nothing is visible until commit()."""

    def __init__(self, root):
        self.root = root
        self.state = _read_json(os.path.join(root, "state.json"),
                                {"requests": [], "config": None, "receipts": {},
                                 "pending": [], "run_seq": 0})
        self.cache = _read_json(os.path.join(root, "cache.json"), {})
        self.manifest = _read_json(os.path.join(root, "manifest.json"), None)
        self.ledger_path = os.path.join(root, "ledger.jsonl")
        self.new_ledger = []
        self.outputs = _read_json(os.path.join(root, "outputs.json"), None)
        self.report = _read_json(os.path.join(root, "report.json"), None)

    def log(self, entry):
        self.new_ledger.append(entry)

    def commit(self):
        staging = self.root + ".staging"
        old = self.root + ".old"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)
        _write_json(os.path.join(staging, "state.json"), self.state)
        _write_json(os.path.join(staging, "cache.json"), self.cache)
        if self.outputs is not None:
            _write_json(os.path.join(staging, "outputs.json"), self.outputs)
        if self.report is not None:
            _write_json(os.path.join(staging, "report.json"), self.report)
        _write_json(os.path.join(staging, "manifest.json"), self.manifest)
        with open(os.path.join(staging, "ledger.jsonl"), "w", encoding="utf-8") as fh:
            if os.path.exists(self.ledger_path):
                with open(self.ledger_path, "r", encoding="utf-8") as prev:
                    fh.write(prev.read())
            for entry in self.new_ledger:
                fh.write(canon(entry) + "\n")
        if os.path.exists(old):
            shutil.rmtree(old)
        if os.path.exists(self.root):
            os.rename(self.root, old)
        os.rename(staging, self.root)
        if os.path.exists(old):
            shutil.rmtree(old)
        self.ledger_path = os.path.join(self.root, "ledger.jsonl")
        self.new_ledger = []

    def read_ledger(self):
        if not os.path.exists(self.ledger_path):
            return []
        with open(self.ledger_path, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
