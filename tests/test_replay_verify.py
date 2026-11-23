from __future__ import annotations

from pathlib import Path

import pytest

from walcrash.errors import BusinessError
from walcrash.verification import verify_replay

from .conftest import make_replayed


def test_first_page_version_difference_is_localised(tmp_path: Path) -> None:
    storage = make_replayed(tmp_path)
    with storage.transaction() as conn:
        conn.execute("UPDATE page_versions SET digest=? WHERE kind='wal' AND lsn=5", ("f" * 64,))
    with pytest.raises(BusinessError) as error:
        verify_replay(storage)
    assert error.value.code == "E_REPLAY"
    assert error.value.context["first_mismatch_lsn"] == 5
    assert error.value.context["page_id"] == "p3"
    assert error.value.context["field"] == "content_digest"
    assert error.value.context["ledger_record"]["batch_id"] == 2
    assert error.value.context["ledger_record"]["start_lsn"] == 5


def test_visible_transaction_difference_and_ledger_corruption_are_e_replay(tmp_path: Path) -> None:
    storage = make_replayed(tmp_path, name="visibility")
    with storage.transaction() as conn:
        conn.execute("UPDATE page_versions SET visible=0 WHERE kind='wal' AND txid='tx-001'")
    with pytest.raises(BusinessError) as error:
        verify_replay(storage)
    assert error.value.code == "E_REPLAY"
    assert error.value.context["field"] in {"committed_transactions", "visible"}
    assert error.value.context["ledger_record"] is not None

    ledger_storage = make_replayed(tmp_path, name="ledger")
    with ledger_storage.transaction() as conn:
        row = conn.execute("SELECT event_id, payload_json FROM ledger_events ORDER BY event_id LIMIT 1").fetchone()
        conn.execute(
            "UPDATE ledger_events SET payload_json=? WHERE event_id=?",
            (row["payload_json"] + " ", row["event_id"]),
        )
    assert ledger_storage.verify_ledger()["ok"] is False
    with pytest.raises(BusinessError) as ledger_error:
        verify_replay(ledger_storage)
    assert ledger_error.value.code == "E_REPLAY"
    assert ledger_error.value.context["field"] in {"event_hash", "prev_hash", "batch_ledger_hash"}


def test_budget_tampering_breaks_recovery_semantics(tmp_path: Path) -> None:
    storage = make_replayed(tmp_path)
    with storage.transaction() as conn:
        conn.execute("UPDATE budget_params SET global_slots=99 WHERE singleton=1")
    with pytest.raises(BusinessError) as error:
        verify_replay(storage)
    assert error.value.code == "E_REPLAY"
    assert error.value.context["field"] in {
        "budget_params",
        "budget_digest",
        "budget_params_digest",
    }


def test_batch_row_and_checkpoint_digest_tampering_are_detected(tmp_path: Path) -> None:
    batch_storage = make_replayed(tmp_path, name="batch")
    with batch_storage.transaction() as conn:
        conn.execute("UPDATE replay_batches SET bytes_consumed=bytes_consumed+1 WHERE batch_id=1")
    assert batch_storage.verify_ledger()["ok"] is False
    with pytest.raises(BusinessError) as batch_error:
        verify_replay(batch_storage)
    assert batch_error.value.code == "E_REPLAY"
    assert batch_error.value.context["field"].startswith("batch_")

    checkpoint_storage = make_replayed(tmp_path, name="checkpoint")
    with checkpoint_storage.transaction() as conn:
        conn.execute("UPDATE checkpoints SET state_digest=? WHERE checkpoint_id=1", ("0" * 64,))
    with pytest.raises(BusinessError) as checkpoint_error:
        verify_replay(checkpoint_storage)
    assert checkpoint_error.value.code == "E_REPLAY"
    assert checkpoint_error.value.context["field"] in {
        "checkpoint_state_digest",
        "batch_checkpoint_digest",
    }
