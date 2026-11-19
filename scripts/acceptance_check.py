"""Acceptance driver: full tests + incremental/full-rebuild equivalence + report."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detbox.config import DetectionParams, MatchConfig, load_injections
from detbox.models import ClassTable
from detbox.pipeline import Pipeline
from detbox.report import build_report, write_report
from detbox.samplegen import HEIGHT, WIDTH, generate_dataset
from detbox.util import read_json


def signature(result):
    frames = {}
    for sequence in sorted(result["frames"]):
        for fno in sorted(result["frames"][sequence]):
            frame = result["frames"][sequence][fno]
            frames[f"{sequence}:{fno}"] = [
                (c.class_code, c.int_box_tuple(), c.int_score_value())
                for c in frame.candidates
            ]
    tracks = {
        sequence: [
            (t.track_id, t.class_code, list(t.frames),
             [list(b) for b in t.boxes], list(t.scores))
            for t in seq_tracks
        ]
        for sequence, seq_tracks in result["tracks"].items()
    }
    failures = sorted(
        (f["sequence"], f["frame"], f["stage"], f["code"], f["injection_id"])
        for f in result["failures"]
    )
    return {"frames": frames, "tracks": tracks, "failures": failures}


def make_pipeline(dataset, workdir, fmt, overrides=None):
    return Pipeline(
        dataset_root=dataset,
        workdir=workdir,
        fmt=fmt,
        detection_params=DetectionParams.load(dataset / "detection_params.json"),
        match_config=MatchConfig.load(dataset / "match_config.json"),
        classes=ClassTable.from_dict(read_json(dataset / "ground_truth.json")["class_table"]),
        width=WIDTH,
        height=HEIGHT,
        injections=load_injections(dataset / "injections.json"),
        overrides=overrides or {},
    )


def main() -> int:
    acceptance = ROOT / "work" / "acceptance"
    if acceptance.exists():
        shutil.rmtree(acceptance)
    dataset = acceptance / "dataset"
    generate_dataset(dataset)

    # Edit sequence: switch versions, frame replacement, image replacement,
    # IoU change and pool capacity change.
    inc = acceptance / "incremental"
    make_pipeline(dataset, inc, "v1").run()

    from PIL import Image

    replacement = acceptance / "replacement.png"
    Image.new("RGB", (WIDTH, HEIGHT), (120, 60, 30)).save(replacement)
    from detbox.cli import apply_edit

    override_text = (
        "seqA".ljust(10) + "2".rjust(5) + "1".rjust(3)
        + "12".rjust(5) + "14".rjust(5) + "30".rjust(5) + "32".rjust(5)
        + "700".rjust(7) + "0".rjust(1) + "\n"
    )
    apply_edit(dataset, {
        "kind": "replace_frame_candidates", "sequence": "seqA", "frame": 2,
        "payload": {"format": "v1", "text": override_text},
    })
    apply_edit(dataset, {
        "kind": "replace_image", "sequence": "seqB", "frame": 0,
        "source_png": str(replacement),
    })
    apply_edit(dataset, {"kind": "update_iou", "iou_threshold": 0.45})
    apply_edit(dataset, {"kind": "update_pool_capacity", "pool_capacity": 12})
    apply_edit(dataset, {"kind": "switch_format", "format": "v2"})

    overrides_path = dataset / "frame_overrides.json"
    overrides = read_json(overrides_path)
    incremental = make_pipeline(dataset, inc, "v2", overrides).run()

    full = make_pipeline(dataset, acceptance / "full_rebuild", "v2", overrides).run()
    equal = signature(incremental) == signature(full)

    classes = ClassTable.from_dict(read_json(dataset / "ground_truth.json")["class_table"])
    report = build_report(incremental, classes, WIDTH, HEIGHT, extra={
        "acceptance": {
            "incremental_full_rebuild_equal": equal,
            "incremental_reused": incremental["reused"],
            "incremental_recomputed": incremental["recomputed"],
            "injection_hit_count": len(incremental["injection_hits"]),
        }
    })
    report_path = acceptance / "acceptance_report.json"
    write_report(report_path, report)

    print(json.dumps({
        "equal": equal,
        "report": str(report_path),
        "tracks": len(report["tracks"]),
        "injection_hits": len(report["injection_hits"]),
        "failed_frames": report["failed_frames"],
        "coord_histogram": report["round_trip"]["coord_delta_histogram"],
        "max_score_delta": report["round_trip"]["max_score_delta"],
    }, indent=2))
    return 0 if equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
