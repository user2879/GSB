"""Structured JSON-lines logging with monotonic timestamps."""

import json
import threading
import time

from .clock import mono_ms


class StructuredLogger:
    """Appends one JSON object per line; safe for concurrent writers."""

    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.Lock()

    def emit(self, event, *, shard_id=None, code=None, **fields):
        record = {
            "ts_mono_ms": mono_ms(),
            "ts_wall": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "event": event,
            "shard_id": shard_id,
            "code": code,
        }
        record.update(fields)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record

    def emit_error(self, err, *, event="error"):
        return self.emit(
            event,
            shard_id=getattr(err, "shard_id", None),
            code=getattr(err, "code", "E_INTERNAL"),
            message=str(err),
            field=getattr(err, "field", None),
            details=getattr(err, "details", {}),
        )


class MemoryLogger(StructuredLogger):
    """Test helper: keeps records in memory instead of a file."""

    def __init__(self):
        self.records = []
        self._lock = threading.Lock()

    def emit(self, event, *, shard_id=None, code=None, **fields):
        record = {
            "ts_mono_ms": mono_ms(),
            "event": event,
            "shard_id": shard_id,
            "code": code,
        }
        record.update(fields)
        with self._lock:
            self.records.append(record)
        return record
