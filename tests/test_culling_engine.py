import numpy as np
import pytest

from terrain_culling.config import DEFAULT_CONFIG, CullingConfig
from terrain_culling.camera import Camera, generate_track
from terrain_culling.terrain import TerrainDataset
from terrain_culling.depth import DepthRenderer, groundtruth_block_visible
from terrain_culling.culling import (
    OcclusionCuller, expand_aabb, sample_hiz as terrain_culling_culling_sample_hiz,
)
from terrain_culling.engine import CullingEngine
from terrain_culling.streaming import ResidencyLedger
from terrain_culling.errors import (
    BlockMissingError, BudgetDeniedError, ResidentLeakError,
)


@pytest.fixture(scope="module")
def dataset():
    return TerrainDataset.generate(512)


def _render(dataset, frame, resident_ids):
    cfg = DEFAULT_CONFIG
    renderer = DepthRenderer(cfg.depth_width, cfg.depth_height,
                             cfg.raymarch_steps, cfg.max_hiz_levels)
    return renderer, *renderer.build_hiz(
        renderer.render_depth(dataset, frame, resident_ids),
        fov_rad=DEFAULT_CONFIG.fov_rad)


def test_block_missing_error(dataset):
    ledger = ResidencyLedger(dataset, 10**9, 10**9)
    with pytest.raises(BlockMissingError) as exc:
        ledger.load("L00_X9999_Y9999", "test")
    assert exc.value.code.value == "E_BLOCK_MISSING"


def test_budget_denied_error(dataset):
    block = dataset.leaf_blocks()[0]
    ledger = ResidencyLedger(dataset, block.payload_bytes - 1, 10**12)
    with pytest.raises(BudgetDeniedError) as exc:
        ledger.load(block.block_id, "test")
    assert exc.value.code.value == "E_BUDGET_DENIED"
    assert ledger.memory_used == 0


def test_resident_ledger_reload_identical(dataset):
    ledger = ResidencyLedger(dataset, 10**9, 10**12)
    block = dataset.leaf_blocks()[0]
    ledger.load(block.block_id, "first")
    payload_before = ledger.memory_used
    ledger.evict(block.block_id, "away")
    assert not ledger.contains(block.block_id)
    ledger.load(block.block_id, "reload")
    assert ledger.memory_used == payload_before
    ledger.verify()


def test_evict_non_resident_raises(dataset):
    ledger = ResidencyLedger(dataset, 10**9, 10**12)
    with pytest.raises(ResidentLeakError) as exc:
        ledger.evict(dataset.leaf_blocks()[0].block_id, "x")
    assert exc.value.code.value == "E_RESIDENT_LEAK"


def test_uncertain_decision_is_kept(dataset, monkeypatch):
    """A block whose farthest corner sits inside the Hi-Z error band is kept."""
    import terrain_culling.culling as culling_mod
    from terrain_culling.terrain import Block
    camera = Camera(192, 128)
    frame = generate_track(6, 512)[2]
    renderer, levels, infos = _render(
        dataset, frame, {b.block_id for b in dataset.leaf_blocks()})
    culler = OcclusionCuller(camera, DEFAULT_CONFIG)

    # Pick a real in-frustum block, then force the Hi-Z sample to sit between
    # its nearest and farthest depths with an error band covering the far side.
    target = None
    for candidate in dataset.leaf_blocks():
        decision = culler.cull_block(
            candidate, frame, levels, infos, 0.09, 0, heights=dataset.heights)
        if decision.state != "frustum_out":
            target = candidate
            break
    assert target is not None
    surface = culling_mod.block_surface_points(target, dataset.heights)
    depths = camera.world_depth(frame.view, surface)
    nearest = float(depths.min())
    farthest = float(depths.max())

    # A zero angular field makes the total error independent of depth, so the
    # nearest-face guard passes while the far corner stays inside the band.
    for info in infos:
        object.__setattr__(info, "pixel_half_angle_rad", 0.0)
    # With the nearest-point single-threshold rule, uncertain means
    # |nearest - hiz| < total_error. Put the stored depth exactly at the
    # nearest depth and inject a large Hi-Z band so the delta falls inside it.
    stored = nearest + 0.25
    probe = culler.cull_block(
        target, frame, levels, infos, 0.09, 0, heights=dataset.heights)
    fixed = (probe.error_components["abs_error"]
             + probe.error_components["expansion_error_world"]
             + probe.error_components["angular_aliasing_error"])
    injected = 10.0  # well above the 0.25 gap, dominated by the explicit band
    assert fixed + injected > abs(nearest - stored)
    monkeypatch.setattr(culling_mod, "sample_hiz",
                        lambda *a, **k: (stored, injected))
    decision = culler.cull_block(
        target, frame, levels, infos, 0.09, 0, heights=dataset.heights)
    assert decision.state == "uncertain"
    assert decision.hiz_sample_depth == stored
    assert "within_error_band_kept_conservative" in decision.reasons


def test_conservative_no_false_culling(dataset):
    """Every block the oracle sees must be visible/uncertain, never occluded."""
    camera = Camera(192, 128)
    track = generate_track(6, 512)
    culler = OcclusionCuller(camera, DEFAULT_CONFIG)
    renderer = DepthRenderer(192, 128, 160, 6)
    resident = {b.block_id for b in dataset.leaf_blocks()}
    for frame in track:
        depth = renderer.render_depth(dataset, frame, resident)
        levels, infos = renderer.build_hiz(depth, fov_rad=DEFAULT_CONFIG.fov_rad)
        for block in dataset.leaf_blocks():
            decision = culler.cull_block(block, frame, levels, infos,
                                         expansion_margin=0.09, hiz_level=0)
            if groundtruth_block_visible(dataset, camera, frame, block):
                assert decision.state in ("visible", "uncertain"), (
                    frame.index, block.block_id, decision.state)


def test_engine_frame_reports_and_ledger(dataset):
    camera = Camera(192, 128)
    engine = CullingEngine(dataset, camera, DEFAULT_CONFIG)
    track = generate_track(5, 512)
    total = len(dataset.leaf_blocks())
    for frame in track:
        report = engine.run_frame(frame.index, frame)
        counts = report["counts"]
        assert sum(counts[k] for k in
                   ("visible", "occluded", "uncertain", "frustum_out")) == total
        for key in ("error_summary", "streaming", "budget", "hiz", "degradation"):
            assert key in report
        # Every visible block is resident after the streaming pass.
        visible_ids = [d["block_id"] for d in report["decisions"]
                       if d["state"] in ("visible", "uncertain")]
        assert all(engine.ledger.contains(bid) for bid in visible_ids)
        engine.ledger.verify()


def test_engine_deterministic_across_instances(dataset):
    from terrain_culling.diagnostics import find_first_divergence
    camera = Camera(192, 128)
    track = generate_track(5, 512)
    engine_a = CullingEngine(TerrainDataset.generate(512), camera, DEFAULT_CONFIG)
    engine_b = CullingEngine(TerrainDataset.generate(512), Camera(192, 128),
                             DEFAULT_CONFIG)
    reports_a = [engine_a.run_frame(i, f) for i, f in enumerate(track)]
    reports_b = [engine_b.run_frame(i, f) for i, f in enumerate(
        generate_track(5, 512))]
    result = find_first_divergence(reports_a, reports_b)
    assert result["diverged"] is False


def test_aabb_expansion_is_conservative():
    bbox = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
    box = expand_aabb(bbox, 1.0)
    assert box[0].tolist() == [-1.0, -1.0, -1.0]
    assert box[1].tolist() == [11.0, 11.0, 11.0]
