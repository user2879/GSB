from __future__ import annotations

import pytest

from tcp_workbench.config import load_session_config
from tcp_workbench.errors import E_VERSION, BusinessError
from tcp_workbench.models import Event
from tcp_workbench.ui_model import build_ui_model
from tests.conftest import SAMPLES


def test_evolve_replaces_whole_conclusion_set_and_archives_old(
    service_factory, sample_events
):
    service = service_factory()
    for raw in sample_events:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    old_items = dict(service.items)
    new_config, new_script = load_session_config(
        SAMPLES / "versions" / "session_v2.json",
        peer_script_override=SAMPLES / "versions" / "peer_script_v2.json",
    )
    report = service.evolve(new_config, new_script)
    assert report.reason == "version_evolution"
    assert list(report.affected_fields) == sorted(
        report.affected_fields, key=lambda change: change.field
    )
    assert report.old_config_version == 1 and report.new_config_version == 2
    assert report.old_script_version == 1 and report.new_script_version == 2
    assert service.config.config_version == 2
    assert service.script.script_version == 2
    assert all(
        item.config_version == 2 and item.script_version == 2
        for item in service.items.values()
    )
    assert len(service.items) == len(old_items)
    history = service.ledger.conclusion_history()
    assert history
    assert all(item.stale for item in history)
    assert all(
        item.config_version == 1 and item.script_version == 1 for item in history
    )
    assert service.ledger.conclusions() == service.items
    view = build_ui_model(service)
    assert all(
        row["version"] == "C2/S2" and row["stale"] == "-"
        for row in view["conclusion_rows"]
    )
    service.verify_incremental_equivalence()


def test_evolve_rejects_non_monotonic_changed_version(service_factory, sample_events):
    service = service_factory()
    for raw in sample_events[:3]:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    current_config = service.config
    current_script = service.script
    changed_raw = dict(current_config.raw)
    changed_raw["window_limit"] = 15000
    # Same config_version with changed content must be rejected.
    with pytest.raises(BusinessError) as shown:
        from tcp_workbench.config import validate_session_config

        changed, errors = validate_session_config(changed_raw)
        assert not errors
        service.evolve(changed, current_script)
    assert shown.value.code == E_VERSION
    assert shown.value.fields == ["config_version"]


def test_open_with_changed_content_requires_evolve(
    service_factory, sample_events, tmp_path
):
    service = service_factory()
    for raw in sample_events[:3]:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    service.close()
    new_config, new_script = load_session_config(
        SAMPLES / "versions" / "session_v2.json",
        peer_script_override=SAMPLES / "versions" / "peer_script_v2.json",
    )
    with pytest.raises(BusinessError) as shown:
        from tcp_workbench.service import SessionService

        SessionService.open(tmp_path / "ledger.sqlite3", new_config, new_script)
    assert shown.value.code == E_VERSION
