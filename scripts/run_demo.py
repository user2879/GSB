"""End-to-end deterministic demonstration used by the acceptance run."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fxrender.app import build_engine
from fxrender.config import RenderConfig
from fxrender.formats import (read_v1, read_v2, restore_v2_from_sidecar,
                              v2_to_v1, write_v1, write_v2)
from fxrender.pipeline import render_all
from fxrender.samplegen import generate_sample_bundle


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="fx-demo-"))
    bundle = generate_sample_bundle(work / "samples", frame_count=6)
    engine, checks = build_engine(bundle, work / "cache")
    result = render_all(engine)
    records = engine.cache.frame_records()
    write_v2(work / "frames.v2.jsonl", records)
    write_v1(work / "frames.v1.txt", records)
    sidecars = {}
    for record in records:
        _, _, sidecar = v2_to_v1(record)
        sidecars[record.frame_id] = sidecar
    v1_records = read_v1(work / "frames.v1.txt")
    write_v2(work / "frames.roundtrip.jsonl",
             [restore_v2_from_sidecar(record, sidecars[record.frame_id])
              for record in v1_records])
    print(json.dumps({
        "workdir": str(work),
        "checks_rejected": [c.as_dict() for c in checks if not c.accepted],
        "completed_frames": result.completed_frames,
        "consumed_budget": result.consumed_budget,
        "pending_events": result.pending_events,
    }, indent=2))


if __name__ == "__main__":
    main()
