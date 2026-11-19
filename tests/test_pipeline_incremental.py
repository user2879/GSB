"""Requirement 4/5: incremental edits must equal a clean full rebuild."""
from __future__ import annotations

import copy
import shutil

import pytest

from detbox.config import DetectionParams, MatchConfig, load_injections
from detbox.models import ClassTable
from detbox.pipeline import Pipeline
from detbox.samplegen import generate_dataset, WIDTH, HEIGHT
from detbox.util import read_json, atomic_write_json


@pytest.fixture()
def dataset(tmp_path):
    root = tmp_path / "dataset"
    generate_dataset(root)
    return root


def _classes(dataset):
    return ClassTable.from_dict(read_json(dataset / "ground_truth.json")["class_table"])


def _make_pipeline(dataset, workdir, fmt, injections=None, overrides=None,
                   params_name="detection_params.json", config_name="match_config.json"):
    return Pipeline(
        dataset_root=dataset,
        workdir=workdir,
        fmt=fmt,
        detection_params=DetectionParams.load(dataset / params_name),
        match_config=MatchConfig.load(dataset / config_name),
        classes=_classes(dataset),
        width=WIDTH,
        height=HEIGHT,
        injections=load_injections(injections) if injections else [],
        overrides=overrides or {},
    )


def _signature(result):
    frames = result["frames"]
    frame_sig = {}
    for sequence in sorted(frames):
        for fno in sorted(frames[sequence]):
            frame = frames[sequence][fno]
            frame_sig[f"{sequence}:{fno}"] = [
                (
                    candidate.class_code,
                    candidate.int_box_tuple(),
                    candidate.int_score_value(),
                )
                for candidate in frame.candidates
            ]
    tracks = {}
    for sequence, seq_tracks in result["tracks"].items():
        tracks[sequence] = [
            (track.track_id, track.class_code, list(track.frames),
             [list(b) for b in track.boxes], list(track.scores))
            for track in seq_tracks
        ]
    failures = sorted(
        (item.sequence, item.frame, item.stage, item.code, item.injection_id)
        for item in [type("R", (), f)() for f in result["failures"]]
    ) if False else sorted(
        (f["sequence"], f["frame"], f["stage"], f["code"], f["injection_id"])
        for f in result["failures"]
    )
    return {"frames": frame_sig, "tracks": tracks, "failures": failures}


def _edit(dataset, workdir, edit: dict):
    from detbox.cli import apply_edit

    apply_edit(dataset, edit)
    journal = workdir / "edit_journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with open(journal, "a", encoding="utf-8") as handle:
        import json

        handle.write(json.dumps(edit, sort_keys=True) + "\n")


def test_v1_v2_switch_incremental_equals_rebuild(dataset, tmp_path):
    injections = dataset / "injections.json"
    inc_work = tmp_path / "inc"
    pipeline = _make_pipeline(dataset, inc_work, "v1", injections=injections)
    first = pipeline.run()
    assert first["reused"] == []
    assert first["recomputed"]

    # Switch version: format fingerprint changes; serialized artifacts rebuild,
    # frame detection is reused where the cache key still matches.
    pipeline_v2 = _make_pipeline(dataset, inc_work, "v2", injections=injections)
    incremental = pipeline_v2.run()
    assert any(item.startswith("detect") for item in incremental["reused"])
    assert "serialize" in incremental["invalidated"]

    full_work = tmp_path / "full"
    full = _make_pipeline(dataset, full_work, "v2", injections=injections).run()
    assert _signature(incremental) == _signature(full)


def test_edit_chain_matches_clean_rebuild(dataset, tmp_path):
    inc_work = tmp_path / "inc"
    _make_pipeline(dataset, inc_work, "v1").run()

    # 1. single-frame candidate replacement
    override = {
        "format": "v1",
        "text": (
            "seqA".ljust(10) + "2".rjust(5) + "1".rjust(3)
            + "12".rjust(5) + "14".rjust(5) + "30".rjust(5) + "32".rjust(5)
            + "700".rjust(7) + "0".rjust(1) + "\n"
        ),
    }
    payload = {"kind": "replace_frame_candidates", "sequence": "seqA",
               "frame": 2, "payload": override}
    _edit(dataset, inc_work, payload)
    overrides_path = dataset / "frame_overrides.json"

    # 2. IoU threshold change
    _edit(dataset, inc_work, {"kind": "update_iou", "iou_threshold": 0.45})
    # 3. pool capacity change
    _edit(dataset, inc_work, {"kind": "update_pool_capacity", "pool_capacity": 12})
    # 4. image replacement (regenerate another deterministic frame bytes)
    source = dataset / "images" / "seqB" / "frame_0001.png"
    _edit(dataset, inc_work, {
        "kind": "replace_image", "sequence": "seqB", "frame": 0,
        "source_png": str(source),
    })
    # 5. weight change
    _edit(dataset, inc_work, {
        "kind": "update_weights",
        "score_weights": {"color": 0.8, "edge": 0.2},
    })
    # 6. switch to v2 then back to v1
    _edit(dataset, inc_work, {"kind": "switch_format", "format": "v2"})

    overrides = read_json(overrides_path) if overrides_path.exists() else {}
    incremental = _make_pipeline(
        dataset, inc_work, "v2", overrides=overrides
    ).run()
    assert incremental["reused"], "unchanged artifacts must be reused"
    assert incremental["invalidated"], "edits must invalidate downstream artifacts"

    full_work = tmp_path / "full"
    full = _make_pipeline(
        dataset, full_work, "v2", overrides=overrides
    ).run()
    assert _signature(incremental) == _signature(full)


def test_image_replacement_only_invalidates_affected_frames(dataset, tmp_path):
    work = tmp_path / "inc"
    _make_pipeline(dataset, work, "v2").run()
    # Build a genuinely different replacement PNG (existing frames are
    # content-addressed, so copying another frame would legally hit cache).
    from PIL import Image

    source = tmp_path / "replacement.png"
    image = Image.new("RGB", (WIDTH, HEIGHT), (12, 34, 56))
    image.save(source, format="PNG")
    _edit(dataset, work, {
        "kind": "replace_image", "sequence": "seqA", "frame": 0,
        "source_png": str(source),
    })
    result = _make_pipeline(dataset, work, "v2").run()
    affected = {item for item in result["recomputed"] if item.startswith("detect")}
    assert any("seqA:0" in item for item in affected)
    assert not any("seqB" in item for item in affected)
    assert any("seqB" in item for item in result["reused"])
