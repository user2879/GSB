import pytest

from terrain_workbench.errors import E_TEXTURE_RESIDENT, TerrainError
from terrain_workbench.residency import ResidencyManager


def test_lru_eviction_and_ledger_audit():
    mgr = ResidencyManager(capacity_bytes=100)
    mgr.require("a", 40, "sha-a", frame=0)
    mgr.require("b", 40, "sha-b", frame=0)
    mgr.require("a", 40, "sha-a", frame=1)          # hit: a is now MRU
    mgr.require("c", 40, "sha-c", frame=2)          # evicts b (LRU)
    ops = [(e.op, e.key) for e in mgr.ledger]
    assert ("evict", "b") in ops
    assert mgr.resident_bytes == 80
    audit = mgr.audit()
    assert audit["uploads"] == 3 and audit["evictions"] == 1 and audit["hits"] == 1


def test_reupload_after_eviction_identical_hash():
    mgr = ResidencyManager(capacity_bytes=80)
    mgr.require("tile", 60, "sha-x", frame=0)
    mgr.require("other", 60, "sha-y", frame=1)      # evicts tile
    mgr.require("tile", 60, "sha-x", frame=2)       # re-upload, same content
    uploads = [e for e in mgr.ledger if e.op == "upload" and e.key == "tile"]
    assert len(uploads) == 2
    assert uploads[0].content_sha == uploads[1].content_sha
    mgr.audit()


def test_oversized_tile_raises():
    mgr = ResidencyManager(capacity_bytes=16)
    with pytest.raises(TerrainError) as exc:
        mgr.require("huge", 17, "sha", frame=0)
    assert exc.value.code == E_TEXTURE_RESIDENT
    assert exc.value.block == "huge"


def test_hash_mismatch_on_resident_raises():
    mgr = ResidencyManager(capacity_bytes=100)
    mgr.require("a", 10, "sha-1", frame=0)
    with pytest.raises(TerrainError) as exc:
        mgr.require("a", 10, "sha-2", frame=1)
    assert exc.value.code == E_TEXTURE_RESIDENT
