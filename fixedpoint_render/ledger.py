"""Content-fingerprint ledger and precise invalidation (business rule 6).

Every pipeline product has a stable cache key derived from the snapshot
version, stage name, upstream fingerprints, parameter fingerprint and the
quantization mode.  The ledger records which upstream artifacts each entry
references, so an edit invalidates exactly the downstream subtree.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .fingerprints import combine, fingerprint


@dataclass
class LedgerEntry:
    key: str
    stage: str
    frame: int
    index: int                     # step / tile row index, -1 when n/a
    upstream: list[str]
    param_fp: str
    qmode: str
    frac_bits: int
    version: int
    artifact_path: str
    reused: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key, "stage": self.stage, "frame": self.frame,
            "index": self.index, "upstream": list(self.upstream),
            "param_fp": self.param_fp, "qmode": self.qmode,
            "frac_bits": self.frac_bits, "version": self.version,
            "artifact_path": self.artifact_path, "reused": self.reused,
        }


class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.entries: dict[str, LedgerEntry] = {}
        self.reused_count = 0
        self.recomputed_count = 0
        if os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for item in raw.get("entries", []):
            entry = LedgerEntry(**item)
            self.entries[entry.key] = entry

    def save(self) -> None:
        payload = {
            "reused_count": self.reused_count,
            "recomputed_count": self.recomputed_count,
            "entries": [e.to_dict() for e in self.entries.values()],
        }
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=1,
                      ensure_ascii=False)

    @staticmethod
    def make_key(version: int, stage: str, frame: int, index: int,
                 upstream_fps: list[str], param_obj, qmode: str,
                 frac_bits: int) -> str:
        param_fp = fingerprint(param_obj, f"param-{stage}")
        material = [
            str(version), stage, f"frame={frame}", f"index={index}",
            f"qmode={qmode}", f"frac_bits={frac_bits}",
            f"param_fp={param_fp}",
            "upstream=" + combine(upstream_fps, "upstream"),
        ]
        return "k:" + combine(material, f"key-{stage}")

    def lookup(self, key: str) -> LedgerEntry | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        if not os.path.exists(entry.artifact_path):
            # A referenced artifact must physically exist to be reusable.
            return None
        return entry

    def record(self, key: str, stage: str, frame: int, index: int,
               upstream: list[str], param_obj, qmode: str, frac_bits: int,
               version: int, artifact_path: str) -> LedgerEntry:
        param_fp = fingerprint(param_obj, f"param-{stage}")
        entry = LedgerEntry(key=key, stage=stage, frame=frame, index=index,
                            upstream=list(upstream), param_fp=param_fp,
                            qmode=qmode, frac_bits=frac_bits, version=version,
                            artifact_path=artifact_path)
        self.entries[key] = entry
        self.recomputed_count += 1
        if hasattr(self, "run_recomputed_keys"):
            self.run_recomputed_keys.add(key)
        return entry

    def mark_reused(self, entry: LedgerEntry) -> None:
        entry.reused = True
        self.reused_count += 1
        if hasattr(self, "run_reused_keys"):
            self.run_reused_keys.add(entry.key)

    def reset_counters(self) -> None:
        self.reused_count = 0
        self.recomputed_count = 0
        for entry in self.entries.values():
            entry.reused = False

    def entries_for_stage(self, stage: str) -> list[LedgerEntry]:
        return sorted((e for e in self.entries.values() if e.stage == stage),
                      key=lambda e: (e.frame, e.index))
