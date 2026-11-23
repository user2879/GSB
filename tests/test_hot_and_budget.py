from __future__ import annotations

from spmm_diag.hot import HotBlockTable


def test_hot_table_sort_and_eviction_rules():
    table = HotBlockTable(3)
    table.invalidate(8, "a")
    table.invalidate(2, "b")
    table.invalidate(2, "c")
    table.invalidate(5, "d")
    assert [entry.block_id for entry in table.entries()] == [2, 5, 8]
    table.invalidate(9, "e")
    # Capacity is three: least invalidations first, then ascending block id.
    assert [entry.block_id for entry in table.entries()] == [2, 8, 9]
    table.add_recompute(8, 17)
    entries = table.entries()
    assert [entry.block_id for entry in entries] == [2, 8, 9]
    assert next(entry for entry in entries if entry.block_id == 8).recompute_nnz == 17


def test_hot_table_capacity_zero_is_disabled():
    table = HotBlockTable(0)
    table.invalidate(1, "x")
    table.add_recompute(1, 9)
    assert table.entries() == []
