"""Per-frame deterministic diagnostics report.

Reports are canonical JSON (sorted keys, fixed float rounding) so identical
runs produce byte-identical output; the run hash chains per-frame hashes.
"""

from __future__ import annotations

import hashlib
import json


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FrameReport:
    def __init__(self, frame: int):
        self.frame = frame
        self.data: dict = {"frame": frame}

    def set(self, **kwargs):
        self.data.update(kwargs)
        return self

    def finalize(self) -> dict:
        self.data["sha"] = sha256_text(canonical_json(self.data))
        return self.data


class RunReport:
    """Accumulates frame reports, computes deltas and the chained run hash."""

    def __init__(self):
        self.frames: list[dict] = []
        self._chain = hashlib.sha256()

    def add(self, report: dict) -> dict:
        prev = self.frames[-1] if self.frames else None
        if prev is not None:
            report["delta"] = {
                "visible_blocks": report["visible_blocks"] - prev["visible_blocks"],
                "triangles": report["triangles"] - prev["triangles"],
                "degraded_blocks": report["degraded_blocks"] - prev["degraded_blocks"],
                "max_screen_error": round(
                    report["max_screen_error"] - prev["max_screen_error"], 6
                ),
                "evictions": report["evictions_total"] - prev["evictions_total"],
            }
        else:
            report["delta"] = None
        report["sha"] = sha256_text(canonical_json(report))
        self.frames.append(report)
        self._chain.update(report["sha"].encode("ascii"))
        return report

    @property
    def run_hash(self) -> str:
        return self._chain.hexdigest()

    def to_jsonl(self) -> str:
        return "".join(canonical_json(f) + "\n" for f in self.frames)
