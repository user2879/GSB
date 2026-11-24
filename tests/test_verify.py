from __future__ import annotations

from pathlib import Path

import pytest

from cachelease.errors import DivergenceError
from cachelease.ledger import Ledger
from cachelease.projection import build_projection, compare_projections
from cachelease.spec import load_config, load_events
from cachelease.verify import verify_incremental

from .helpers import write_sample


def test_incremental_and_full_rebuild_are_field_identical(tmp_path: Path) -> None:
    config_path, events_path, checkpoint, _ = write_sample(tmp_path, seed=912)
    config = load_config(config_path)
    events = load_events(events_path, config)
    result = verify_incremental(
        config,
        events,
        checkpoint_seq=checkpoint,
        work_dir=tmp_path / "verify-work",
    )
    assert result["ok"] is True
    assert result["prefix_snapshots_byte_identical"] is True
    assert result["incremental_processed_events"] == result["suffix_events"]
    assert result["full_processed_events"] == len(events)
    assert result["progress_equal"] == {"last_ordinal": True, "last_seq": True}


def test_tampered_incremental_projection_reports_e_diverge(tmp_path: Path) -> None:
    config_path, events_path, checkpoint, _ = write_sample(tmp_path, seed=913)
    config = load_config(config_path)
    events = load_events(events_path, config)
    result = verify_incremental(config, events, checkpoint_seq=checkpoint, work_dir=tmp_path / "verify-work")
    incremental_path = Path(result["incremental"]["ledger"])
    full_path = Path(result["full"]["ledger"])
    connection = Ledger(incremental_path, config)
    try:
        connection.connection.execute(
            "UPDATE watermarks SET watermark_units=watermark_units+1 WHERE ordinal=(SELECT MAX(ordinal) FROM watermarks) AND node_id='node-a'"
        )
    finally:
        connection.close()
    incremental = Ledger(incremental_path, config)
    full = Ledger(full_path, config)
    try:
        with pytest.raises(DivergenceError) as exc:
            compare_projections(build_projection(incremental, config), build_projection(full, config), config)
        assert exc.value.code == "E_DIVERGE"
        assert exc.value.details["node_id"] == "node-a"
        assert exc.value.details["field"].startswith("watermark.")
    finally:
        incremental.close()
        full.close()

