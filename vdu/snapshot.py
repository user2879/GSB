"""Staged snapshot commits with atomic whole-version replacement."""

import json
import os
import shutil

from .errors import SnapshotError
from .fingerprint import (config_fingerprint, label_fingerprint,
                          structure_fingerprint)

MANIFEST = "manifest.json"
LEDGER = "ledger.json"
STATE = "state.json"


class SnapshotStore:
    def __init__(self, root):
        self.root = root
        self.current = os.path.join(root, "current")
        self.tmp = os.path.join(root, "tmp")
        os.makedirs(root, exist_ok=True)

    def begin(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)
        os.makedirs(self.tmp, exist_ok=False)
        return self.tmp

    def write_json(self, name, payload):
        path = os.path.join(self.tmp, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path

    def commit(self, manifest):
        self.write_json(MANIFEST, manifest)
        staged = os.path.join(self.root, "staged")
        if os.path.exists(staged):
            shutil.rmtree(staged)
        os.replace(self.tmp, staged)
        if os.path.islink(self.current) or os.path.exists(self.current):
            if os.path.islink(self.current) or os.path.isfile(self.current):
                os.remove(self.current)
            else:
                shutil.rmtree(self.current)
        os.symlink(os.path.basename(staged), self.current)
        return self.current

    def current_path(self):
        if not os.path.islink(self.current) and not os.path.exists(self.current):
            return None
        return self.current

    def read_json(self, name):
        base = self.current_path()
        if base is None:
            return None
        path = os.path.join(base, name)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def detect_mode(self, config, transcript):
        manifest = self.read_json(MANIFEST)
        if manifest is None:
            return "rebuild", None
        cfg_fp = config_fingerprint(config)
        label_fp = label_fingerprint(transcript)
        struct_fp = structure_fingerprint(transcript)
        if manifest.get("config_fp") != cfg_fp:
            return "rebuild", manifest
        if (manifest.get("label_fp") != label_fp or
                manifest.get("structure_fp") != struct_fp):
            return "incremental", manifest
        return "resume", manifest

    def require_resumable(self, config):
        manifest = self.read_json(MANIFEST)
        if manifest is None:
            raise SnapshotError("no snapshot available for recovery")
        if manifest.get("config_fp") != config_fingerprint(config):
            raise SnapshotError(
                "config hash mismatch; cannot resume, rebuild required")
        return manifest

    def write_exit_report(self, reason, extra=None):
        base = self.current_path()
        if base is None:
            return None
        report = {"reason": reason}
        if extra:
            report.update(extra)
        path = os.path.join(base, "exit_report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        return path
