from __future__ import annotations

import numpy as np
import pytest

from glyphforge.atlas import AtlasAsset, AtlasBudget, AtlasBudgetManager, AtlasManager
from glyphforge.errors import ErrorCode, GlyphForgeError
from glyphforge.raster import bitmap_hash


def _asset(key: str, width: int, height: int, value: int, size: int = 12) -> AtlasAsset:
    pixels = bytes([value]) * (width * height)
    return AtlasAsset(
        key=key,
        width=width,
        height=height,
        pixels=pixels,
        content_hash=bitmap_hash(width, height, pixels),
        size=size,
        fill_rule="nonzero",
    )


def _region_bytes(manager: AtlasManager, key: str) -> bytes:
    region = manager.regions[key]
    page = manager._page_by_index(region.page_index)
    return page.bitmap[
        region.y : region.y + region.height,
        region.x : region.x + region.width,
    ].tobytes()


def test_pack_allocates_distinct_regions_and_tracks_free_space():
    manager = AtlasManager(
        AtlasBudget(page_size=64, max_pages=1, max_occupied_pixels=4096, padding=1)
    )
    first = manager.pack(_asset("a", 20, 10, 10))
    second = manager.pack(_asset("b", 15, 12, 20))
    assert first.page_index == second.page_index == 0
    assert (first.x, first.y) != (second.x, second.y)
    manager.validate_region("a")
    manager.validate_region("b")
    ledger = manager.ledger()
    assert ledger["pages"][0]["occupied_pixels"] == 20 * 10 + 15 * 12
    assert ledger["pages"][0]["free_rectangles"]


def test_same_size_replacement_reuses_slot_without_touching_other_regions():
    manager = AtlasManager(
        AtlasBudget(page_size=64, max_pages=1, max_occupied_pixels=4096, padding=1)
    )
    first = manager.pack(_asset("a", 20, 10, 10))
    manager.pack(_asset("b", 10, 10, 30))
    other_bytes = _region_bytes(manager, "b")
    replacement = manager.pack(_asset("a", 20, 10, 60))
    assert (first.x, first.y, first.page_index) == (
        replacement.x,
        replacement.y,
        replacement.page_index,
    )
    assert replacement.reused_slot
    assert _region_bytes(manager, "b") == other_bytes
    assert set(np.frombuffer(_region_bytes(manager, "a"), dtype=np.uint8)) == {60}


def test_page_and_pixel_budgets_raise_atlas_full():
    manager = AtlasManager(
        AtlasBudget(page_size=32, max_pages=1, max_occupied_pixels=1024, padding=0)
    )
    manager.pack(_asset("a", 20, 20, 1))
    with pytest.raises(GlyphForgeError) as caught:
        manager.pack(_asset("b", 20, 20, 2))
    assert caught.value.code is ErrorCode.ATLAS_FULL


def test_fallback_and_delay_are_reproducible():
    manager = AtlasManager(
        AtlasBudget(page_size=32, max_pages=1, max_occupied_pixels=100, padding=0)
    )
    coordinator = AtlasBudgetManager(manager)
    primary = _asset("request|48|nonzero", 12, 12, 5, size=48)
    fallback = _asset("fallback|12", 6, 6, 7, size=12)
    outcome = coordinator.publish(
        primary.key,
        primary,
        fallbacks=(fallback,),
        referenced_keys={primary.key},
    )
    assert outcome.status == "fallback"
    assert outcome.packed_size == 12
    assert manager.regions[primary.key].content_hash == fallback.content_hash
    assert any(item["action"] == "fallback_size" for item in coordinator.decisions)

    second_manager = AtlasManager(
        AtlasBudget(page_size=32, max_pages=1, max_occupied_pixels=100, padding=0)
    )
    second = AtlasBudgetManager(second_manager)
    second_outcome = second.publish(
        primary.key,
        primary,
        fallbacks=(fallback,),
        referenced_keys={primary.key},
    )
    assert second_outcome.to_dict() == outcome.to_dict()
    assert second.ledger() == coordinator.ledger()


def test_recycled_page_repack_keeps_identical_region_pixels():
    manager = AtlasManager(
        AtlasBudget(page_size=64, max_pages=1, max_occupied_pixels=4096, padding=1)
    )
    coordinator = AtlasBudgetManager(manager)
    asset = _asset("a", 20, 12, 42)
    first = coordinator.publish("a", asset, referenced_keys={"a"})
    assert first.status == "packed"
    first_bytes = _region_bytes(manager, "a")
    assert first_bytes == bytes([42]) * (20 * 12)

    coordinator.remove("a")
    recycled = coordinator.recycle(())
    assert recycled == (0,)
    assert not manager.pages
    assert coordinator.outcomes == ()

    second = coordinator.publish("a", asset, referenced_keys={"a"})
    assert second.status == "packed"
    assert _region_bytes(manager, "a") == first_bytes
    assert manager.regions["a"].content_hash == first.region.content_hash


def test_pressure_recycle_invalidates_evicted_outcomes():
    manager = AtlasManager(
        AtlasBudget(page_size=32, max_pages=1, max_occupied_pixels=200, padding=0)
    )
    coordinator = AtlasBudgetManager(manager)
    coordinator.publish("old", _asset("old", 10, 10, 1), referenced_keys={"old"})
    outcome = coordinator.publish(
        "new",
        _asset("new", 12, 12, 2),
        referenced_keys={"new"},
    )
    assert outcome.status == "packed"
    assert outcome.reason == "after_page_recycle"
    old = {item.request_key: item for item in coordinator.outcomes}["old"]
    assert old.status == "delayed"
    assert old.reason == "page_recycled"
    assert "old" not in manager.regions
