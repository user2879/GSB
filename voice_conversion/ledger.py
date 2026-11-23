"""Content-fingerprint cache keys and the per-segment reuse ledger."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig, canonical_json, fingerprint_json, sha256_text
from .model import Segment

STAGES = ("analysis", "mapping", "synthesis", "assembly")


def segment_content_fingerprint(seg: Segment) -> str:
    """Fingerprint of the canonical fields that drive recomputation.

    Passthrough metadata is intentionally excluded: it never affects PCM.
    """
    payload = {
        "index": seg.index,
        "start_frame": seg.start_frame,
        "end_frame": seg.end_frame,
        "intensity": seg.intensity,
        "timbre": seg.timbre,
        "missing": seg.missing,
    }
    return fingerprint_json(payload)


def derive_key(upstream: str, config_fp: str, quant_levels: int,
               stage: str) -> str:
    """Derive a stage cache key.

    The exchange version is carried inside the segment content fingerprint
    (``source_version`` is one of its fields), so an edit that only changes the
    wrapper version keeps identical canonical segments on identical keys while
    a genuine content change still moves the key.
    """
    material = canonical_json({
        "upstream": upstream,
        "config": config_fp,
        "quant_levels": quant_levels,
        "stage": stage,
    })
    return f"{stage}-{sha256_text(material)[:16]}"


def analysis_key(seg: Segment, config: PipelineConfig,
                 audio_fp: str) -> str:
    upstream = sha256_text(segment_content_fingerprint(seg) + "|" + audio_fp)
    return derive_key(upstream, config.fingerprint(), config.quant_levels,
                      "analysis")


def mapping_key(seg: Segment, config: PipelineConfig, analysis_fp: str) -> str:
    return derive_key(analysis_fp, config.fingerprint(), config.quant_levels,
                      "mapping")


def synthesis_key(seg: Segment, config: PipelineConfig, mapping_fp: str) -> str:
    return derive_key(mapping_fp, config.fingerprint(), config.quant_levels,
                      "synthesis")


def assembly_key(segments: list[Segment], config: PipelineConfig,
                 synthesis_fps: list[str]) -> str:
    material = canonical_json({
        "ordered_segments": [segment_content_fingerprint(s) for s in
                             sorted(segments, key=lambda s: s.index)],
        "synthesis": sorted(synthesis_fps),
        "config": config.fingerprint(),
        "quant_levels": config.quant_levels,
        "stage": "assembly",
    })
    return f"assembly-{sha256_text(material)[:16]}"


@dataclass
class LedgerEntry:
    segment: int
    analysis_key: str
    mapping_key: str
    synthesis_key: str
    source_frame_set: list
    parameter_set: list
    content_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "segment": self.segment,
            "analysis_key": self.analysis_key,
            "mapping_key": self.mapping_key,
            "synthesis_key": self.synthesis_key,
            "source_frame_set": self.source_frame_set,
            "parameter_set": self.parameter_set,
            "content_fingerprint": self.content_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LedgerEntry":
        return cls(
            segment=int(data["segment"]),
            analysis_key=data["analysis_key"],
            mapping_key=data["mapping_key"],
            synthesis_key=data["synthesis_key"],
            source_frame_set=list(data["source_frame_set"]),
            parameter_set=list(data["parameter_set"]),
            content_fingerprint=data["content_fingerprint"],
        )


class ReuseLedger:
    """Persists stage keys and answers reuse/invalidation decisions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[int, LedgerEntry] = {}
        self.assembly_key = ""

    def load(self) -> bool:
        if not self.path.exists():
            return False
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = {
            int(k): LedgerEntry.from_dict(v) for k, v in data["entries"].items()
        }
        self.assembly_key = data.get("assembly_key", "")
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": {str(k): v.to_dict()
                        for k, v in sorted(self.entries.items())},
            "assembly_key": self.assembly_key,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def can_reuse(self, segment: int, analysis: str, mapping: str,
                  synthesis: str) -> bool:
        old = self.entries.get(segment)
        if old is None:
            return False
        return (old.analysis_key == analysis and old.mapping_key == mapping
                and old.synthesis_key == synthesis)

    def remember(self, entry: LedgerEntry) -> None:
        self.entries[entry.segment] = entry

    def prune(self, live_segments: set[int]) -> None:
        for dead in [k for k in self.entries if k not in live_segments]:
            del self.entries[dead]
