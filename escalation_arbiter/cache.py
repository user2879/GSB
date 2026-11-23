"""Persistent per-stage content-addressed cache inside the workspace."""

import json
import os
import threading

from .fingerprint import fingerprint
from .models import STAGES, TicketDecision


class StageCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._lock = threading.RLock()
        self.index_path = os.path.join(root, "index.json")
        self.index = self._load_index()

    def _load_index(self):
        with self._lock:
            if os.path.exists(self.index_path):
                with open(self.index_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            return {"version": 1, "entries": {}}

    def _save_index(self):
        with self._lock:
            os.makedirs(self.root, exist_ok=True)
            tmp = self.index_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.index, fh, ensure_ascii=False, indent=2,
                          sort_keys=True)
            os.replace(tmp, self.index_path)

    @staticmethod
    def entry_id(tid, stage, key):
        return f"{tid}|{stage}|{fingerprint(key)[:16]}"

    def get(self, tid, stage, key):
        with self._lock:
            entry_id = self.entry_id(tid, stage, key)
        entry = self.index["entries"].get(entry_id)
        if entry is None:
            return None
        path = os.path.join(self.root, entry["file"])
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("stage_key") != key or data.get("tid") != tid \
                or data.get("stage") != stage:
            return None
        return data

    def put(self, tid, stage, key, payload):
        with self._lock:
            entry_id = self.entry_id(tid, stage, key)
        file_name = entry_id.replace("|", "_") + ".json"
        path = os.path.join(self.root, file_name)
        record = {"tid": tid, "stage": stage, "stage_key": key,
                  "payload": payload}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
        self.index["entries"][entry_id] = {"file": file_name,
                                           "stage": stage, "tid": tid}
        self._save_index()

    def load_decision(self, tid, stage, key):
        data = self.get(tid, stage, key)
        if data is None:
            return None
        return TicketDecision.from_dict(data["payload"])

    def store_decision(self, decision, stage, key):
        self.put(decision.tid, stage, key, decision.to_dict())

    def stage_keys(self):
        return STAGES
