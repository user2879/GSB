"""Reviewable ledger and content-addressed artifact store.

The ledger directory holds:

* ``business.jsonl``  - deterministic per-shard conclusions, one JSON per line
* ``failures.jsonl``  - every rejectable business outcome with a code
* ``algebra.jsonl``   - per-lease generation events observed from nodes
* ``artifacts/``      - reusable keyed artifacts (summaries, receipts,
  conclusions, serializations, ledger entries)
* ``reuse.jsonl``     - invalidation reasons, reuse hits and recompute scope
* ``exits.jsonl``     - processable termination records with business codes

Business entries are keyed by shard index, so re-running a batch replaces a
shard's entry in place while the on-disk order stays identical to a serial
full rebuild.  Reuse events live in their own audit stream and never alter
the deterministic business ledger.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import Failure

LEDGER_FILES = ("business.jsonl", "failures.jsonl", "algebra.jsonl", "reuse.jsonl", "exits.jsonl")


class Ledger:
    def __init__(self, root: str) -> None:
        self.root = root
        self.artifact_dir = os.path.join(root, "artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)
        for name in LEDGER_FILES:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8"):
                    pass
        self._business: dict[int, dict[str, Any]] = {}
        self._failures: list[dict[str, Any]] = []
        self._algebra: list[dict[str, Any]] = []
        self._reuse: list[dict[str, Any]] = []
        self._exits: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        for entry in self._read_jsonl("business.jsonl"):
            self._business[int(entry["index"])] = entry
        self._failures = self._read_jsonl("failures.jsonl")
        self._algebra = self._read_jsonl("algebra.jsonl")
        self._reuse = self._read_jsonl("reuse.jsonl")
        self._exits = self._read_jsonl("exits.jsonl")

    @staticmethod
    def _read_jsonl(path: str) -> list[dict[str, Any]]:
        if not os.path.exists(path):
            return []
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _append_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
        with open(path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    @staticmethod
    def _write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    # -- business ledger ---------------------------------------------------
    def upsert_business(self, entry: dict[str, Any]) -> None:
        self._business[int(entry["index"])] = entry

    def business_rows(self) -> list[dict[str, Any]]:
        return [self._business[key] for key in sorted(self._business)]

    def flush_business(self) -> None:
        self._write_jsonl(os.path.join(self.root, "business.jsonl"), self.business_rows())

    # -- failures ----------------------------------------------------------
    def add_failure(self, failure: Failure) -> None:
        self._failures.append(failure.to_dict())

    def add_failures(self, failures: Iterable[Failure]) -> None:
        for failure in failures:
            self.add_failure(failure)

    def failure_rows(self) -> list[dict[str, Any]]:
        return list(self._failures)

    def flush_failures(self) -> None:
        self._append_jsonl(os.path.join(self.root, "failures.jsonl"), self._failures)
        self._failures = []

    # -- algebra -----------------------------------------------------------
    def add_algebra(self, event: dict[str, Any]) -> None:
        self._algebra.append(event)

    def add_algebra_many(self, events: Iterable[dict[str, Any]]) -> None:
        self._algebra.extend(events)

    def algebra_rows(self) -> list[dict[str, Any]]:
        return list(self._algebra)

    def flush_algebra(self) -> None:
        self._append_jsonl(os.path.join(self.root, "algebra.jsonl"), self._algebra)
        self._algebra = []

    # -- reuse audit -------------------------------------------------------
    def record_reuse(self, kind: str, key: str, detail: dict[str, Any]) -> None:
        self._reuse.append({"kind": kind, "key": key, "detail": detail})

    def reuse_rows(self) -> list[dict[str, Any]]:
        return list(self._reuse)

    def flush_reuse(self) -> None:
        self._append_jsonl(os.path.join(self.root, "reuse.jsonl"), self._reuse)
        self._reuse = []

    # -- exits -------------------------------------------------------------
    def add_exit(self, record: dict[str, Any]) -> None:
        self._exits.append(record)

    def exit_rows(self) -> list[dict[str, Any]]:
        return list(self._exits)

    def flush_exits(self) -> None:
        self._append_jsonl(os.path.join(self.root, "exits.jsonl"), self._exits)
        self._exits = []

    # -- artifacts ---------------------------------------------------------
    def artifact_path(self, key: str) -> str:
        return os.path.join(self.artifact_dir, key + ".json")

    def has_artifact(self, key: str) -> bool:
        return os.path.exists(self.artifact_path(key))

    def load_artifact(self, key: str) -> dict[str, Any] | None:
        path = self.artifact_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def store_artifact(self, key: str, payload: dict[str, Any]) -> None:
        path = self.artifact_path(key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        os.replace(tmp, path)

    def all_artifacts(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not os.path.isdir(self.artifact_dir):
            return out
        for name in sorted(os.listdir(self.artifact_dir)):
            if name.endswith(".json"):
                with open(os.path.join(self.artifact_dir, name), "r", encoding="utf-8") as handle:
                    out[name[:-5]] = json.load(handle)
        return out


@dataclass
class ReuseReport:
    invalidated: list[str]
    reused: list[str]
    recomputed: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "invalidated": list(self.invalidated),
            "reused": list(self.reused),
            "recomputed": list(self.recomputed),
        }
