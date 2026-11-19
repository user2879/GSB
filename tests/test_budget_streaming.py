import pytest

from terrain_culling.config import CullingConfig
from terrain_culling.camera import Camera, generate_track
from terrain_culling.terrain import TerrainDataset
from terrain_culling.engine import CullingEngine
from terrain_culling.streaming import ResidencyLedger
from terrain_culling.budget import DegradationPolicy, LADDER, base_expansion_margin
from terrain_culling.errors import BudgetDeniedError, ResidentLeakError


def test_degradation_ladder_order_and_determinism():
    ds = TerrainDataset.generate(512)
    cfg = CullingConfig()
    cam = Camera(cfg.depth_width, cfg.depth_height)
    track = generate_track(8, 512)

    def run():
        engine = CullingEngine(TerrainDataset.generate(512),
                               Camera(cfg.depth_width, cfg.depth_height), cfg)
        seq = []
        for frame in track:
            report = engine.run_frame(frame.index, frame)
            seq.append((report["frame"], report["degradation"]["step"]))
        return seq, engine.policy.sequence_log

    seq_a, log_a = run()
    seq_b, log_b = run()
    assert seq_a == seq_b
    assert log_a == log_b
    steps = [s for _, s in seq_a]
    # Ladder is monotonic non-decreasing and within cap.
    assert all(b >= a for a, b in zip(steps, steps[1:]))
    assert max(steps) <= cfg.degradation_cap
    for record in log_a:
        assert LADDER[record["to_step"]].step == record["to_step"]


def test_budget_denied_when_visible_set_exceeds_budget():
    ds = TerrainDataset.generate(512)
    # A budget smaller than a single leaf payload makes any load impossible.
    leaf = ds.leaf_blocks()[0]
    cfg = CullingConfig(memory_budget_bytes=leaf.payload_bytes - 10,
                        vram_budget_bytes=leaf.vram_bytes)
    engine = CullingEngine(ds, Camera(cfg.depth_width, cfg.depth_height), cfg)
    frame = generate_track(4, 512)[0]
    with pytest.raises(BudgetDeniedError) as exc:
        engine.run_frame(0, frame)
    assert exc.value.code.value == "E_BUDGET_DENIED"
    # No partial residency leaked after the denial.
    assert engine.ledger.memory_used == 0
    assert len(engine.ledger.resident) == 0


def test_ledger_recompute_detects_corruption():
    ds = TerrainDataset.generate(512)
    ledger = ResidencyLedger(ds, 10**9, 10**12)
    block = ds.leaf_blocks()[0]
    ledger.load(block.block_id, "x")
    ledger.memory_used += 999
    with pytest.raises(ResidentLeakError) as exc:
        ledger.verify()
    assert exc.value.code.value == "E_RESIDENT_LEAK"


def test_loads_precede_evictions_and_visible_never_missing():
    ds = TerrainDataset.generate(512)
    cfg = CullingConfig()
    engine = CullingEngine(ds, Camera(cfg.depth_width, cfg.depth_height), cfg)
    for frame in generate_track(10, 512):
        report = engine.run_frame(frame.index, frame)
        visible = [d["block_id"] for d in report["decisions"]
                   if d["state"] in ("visible", "uncertain")]
        assert all(engine.ledger.contains(bid) for bid in visible)
        # Streaming report never reports an eviction of a visible block.
        evicted = {r["block_id"] for r in report["streaming"]["requests"]
                   if r["kind"] == "evict"}
        assert evicted.isdisjoint(visible)
        engine.ledger.verify()


def test_reload_after_eviction_restores_byte_ledger():
    ds = TerrainDataset.generate(512)
    ledger = ResidencyLedger(ds, 10**9, 10**12)
    a, b = ds.leaf_blocks()[0].block_id, ds.leaf_blocks()[1].block_id
    ledger.load(a, "on")
    ledger.load(b, "on")
    snapshot = (ledger.memory_used, ledger.vram_used)
    ledger.evict(a, "away")
    ledger.evict(b, "away")
    assert ledger.memory_used == 0 and ledger.vram_used == 0
    ledger.load(a, "back")
    ledger.load(b, "back")
    assert (ledger.memory_used, ledger.vram_used) == snapshot
    ledger.verify()


def test_base_expansion_is_deterministic():
    assert base_expansion_margin() > 0
    assert base_expansion_margin() == base_expansion_margin()
