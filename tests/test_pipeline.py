import pytest

from terrain_workbench.camera import CameraTrack
from terrain_workbench.errors import E_TRACK_DIVERGE, TerrainError
from terrain_workbench.pipeline import run_pipeline

TRACK_KF = [
    {"t": 0.0, "pos": [32.0, 8.0, 87.5], "target": [56.0, 62.0, 11.0]},
    {"t": 1.0, "pos": [90.0, 40.0, 70.0], "target": [140.0, 120.0, 15.0]},
    {"t": 2.0, "pos": [160.0, 200.0, 80.0], "target": [128.0, 128.0, 12.0]},
]


def _track(frames=8):
    return CameraTrack(TRACK_KF, frames, (1280, 720), 60.0)


def test_pipeline_run_twice_byte_identical():
    r1 = run_pipeline(257, 99, _track(), 60000, 1 << 20, 4.0, prefer_gl=False)
    r2 = run_pipeline(257, 99, _track(), 60000, 1 << 20, 4.0, prefer_gl=False)
    assert r1["report_jsonl"] == r2["report_jsonl"]
    assert r1["summary"]["run_hash"] == r2["summary"]["run_hash"]
    assert r1["manifests"] == r2["manifests"]
    assert r1["ledger"] == r2["ledger"]


def test_pipeline_reports_and_deltas():
    r = run_pipeline(257, 99, _track(frames=6), 60000, 1 << 20, 4.0, prefer_gl=False)
    frames = r["frames"]
    assert len(frames) == 6
    assert frames[0]["delta"] is None
    for prev, cur in zip(frames, frames[1:]):
        assert cur["delta"]["visible_blocks"] == (
            cur["visible_blocks"] - prev["visible_blocks"])
    for f in frames:
        assert f["triangles"] <= 60000
        assert f["crack_max_gap"] == 0.0
        assert f["gl_status"] == "E_GL_CONTEXT"
    assert r["summary"]["residency_audit"]["resident_bytes"] <= (1 << 20)


def test_pipeline_tight_memory_forces_evictions():
    r = run_pipeline(257, 99, _track(frames=12), 120000, 40000, 4.0, prefer_gl=False)
    assert r["summary"]["residency_audit"]["evictions"] > 0
    assert r["summary"]["residency_audit"]["resident_bytes"] <= 40000


def test_track_diverge_raises():
    bad = CameraTrack(
        [
            {"t": 0.0, "pos": [10.0, 10.0, 50.0], "target": [50.0, 50.0, 10.0]},
            {"t": 1.0, "pos": [99999.0, 10.0, 50.0], "target": [50.0, 50.0, 10.0]},
        ],
        4, (1280, 720), 60.0,
    )
    with pytest.raises(TerrainError) as exc:
        run_pipeline(257, 99, bad, 60000, 1 << 20, 4.0, prefer_gl=False)
    assert exc.value.code == E_TRACK_DIVERGE
    assert exc.value.context["frame"] is not None
