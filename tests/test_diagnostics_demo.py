import json
import os

import pytest

from terrain_culling.config import DEFAULT_CONFIG
from terrain_culling.camera import Camera, generate_track
from terrain_culling.terrain import TerrainDataset
from terrain_culling.engine import CullingEngine
from terrain_culling.diagnostics import (
    canonical_bytes, content_hash, find_first_divergence, frame_digest,
)
from terrain_culling.errors import TrackDivergenceError
from terrain_culling.gl_context import probe_egl


def _reports(seed_size=512, frames=4, perturb=False):
    camera = Camera(192, 128)
    track = generate_track(frames, seed_size)
    engine = CullingEngine(TerrainDataset.generate(seed_size), camera,
                           DEFAULT_CONFIG)
    reports = []
    for i, frame in enumerate(track):
        if perturb and i == 2:
            # Mutate one eye coordinate to force a different decision.
            eye = list(frame.eye)
            eye[0] += 400.0
            frame = camera.frame(i, eye, frame.target)
        reports.append(engine.run_frame(i, frame))
    return reports


def test_canonical_bytes_stable():
    obj = {"b": 1, "a": [1, 2, {"c": 3}], "z": None}
    assert canonical_bytes(obj) == canonical_bytes(json.loads(json.dumps(obj)))
    assert len(content_hash(obj)) == 64


def test_identical_tracks_do_not_diverge():
    a = _reports()
    b = _reports()
    result = find_first_divergence(a, b)
    assert result == {"diverged": False, "frames_compared": 4}


def test_first_divergence_locates_frame_and_block():
    baseline = _reports()
    perturbed = _reports(perturb=True)
    with pytest.raises(TrackDivergenceError) as exc:
        find_first_divergence(baseline, perturbed)
    assert exc.value.code.value == "E_TRACK_DIVERGE"
    ctx = exc.value.context
    assert ctx["diverged"] is True
    assert ctx["frame"] == 2
    assert "block_id" in ctx and "camera_a" in ctx and "camera_b" in ctx


def test_same_count_different_assignment_still_diverges():
    a = _reports()
    b = _reports(perturb=True)
    # Counts can coincidentally match; per-block signatures must catch it.
    with pytest.raises(TrackDivergenceError):
        find_first_divergence(a, b)


def test_egl_probe_reports_status():
    result = probe_egl()
    assert "available" in result and "backend" in result
    if result["available"]:
        assert result["backend"] == "egl_gles2"
        assert "egl_version" in result


def test_demo_end_to_end_and_byte_reproducible(tmp_path):
    from terrain_culling.demo import run_size
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    info_a = run_size(512, 4, str(out_a), DEFAULT_CONFIG)
    info_b = run_size(512, 4, str(out_b), DEFAULT_CONFIG)
    assert info_a["frames_report_sha256"] == info_b["frames_report_sha256"]
    assert info_a["summary_sha256"] == info_b["summary_sha256"]
    raw_a = (out_a / "frames_512.json").read_bytes()
    raw_b = (out_b / "frames_512.json").read_bytes()
    assert raw_a == raw_b
    summary = json.loads((out_a / "summary_512.json").read_bytes())
    assert summary["self_divergence_check"]["diverged"] is False
