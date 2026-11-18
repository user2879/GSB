from __future__ import annotations

from graygate.ledger import LedgerStore


def test_running_rebuild_is_marked_unknown_after_restart(tmp_path):
    path = tmp_path / "ledger.db"
    ledger = LedgerStore(path)
    ledger.create_rebuild(
        rebuild_id="rb-crash",
        source_ledger_version=0,
        started_ms=123,
        budget_ms=100,
    )
    ledger.close()

    recovered = LedgerStore(path)
    try:
        row = recovered.get_rebuild("rb-crash")
        assert row is not None
        assert row["status"] == "UNKNOWN"
        assert row["error_code"] == "E_RECOVERY"
        assert recovered.recovered_unknown[0]["rebuild_id"] == "rb-crash"
    finally:
        recovered.close()

