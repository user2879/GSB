from __future__ import annotations

from tcp_workbench.models import Event
from tcp_workbench.projection import compare_items, replay


def _event_dicts(raw_events):
    return [Event.from_dict({**raw, "event_seq": 0}) for raw in raw_events]


def test_incremental_items_equal_full_replay_after_every_event(
    service_factory, sample_events
):
    service = service_factory()
    for index, raw in enumerate(sample_events, start=1):
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
        full = replay(
            service.events, service.config, service.script, capture_checkpoints=False
        )
        assert compare_items(full.items, service.items) == []
        assert full.conclusions_digest == service.full_recompute().conclusions_digest


def test_unaffected_item_row_is_byte_identical(service_factory, sample_events):
    service = service_factory()
    for raw in sample_events[:5]:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    before = dict(
        service.ledger.conn.execute(
            "SELECT * FROM conclusions WHERE item_key = 'connection_state'"
        ).fetchone()
    )
    result = service.ingest_event(Event.from_dict({**sample_events[5], "event_seq": 0}))
    after = dict(
        service.ledger.conn.execute(
            "SELECT * FROM conclusions WHERE item_key = 'connection_state'"
        ).fetchone()
    )
    assert "connection_state" not in result.affected_items
    assert before == after
    assert result.affected_items == tuple(sorted(result.affected_items))


def test_conclusions_record_basis_versions_and_dependencies(
    service_factory, sample_events
):
    service = service_factory()
    for raw in sample_events:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    for key, item in service.items.items():
        assert item.config_version == 1
        assert item.script_version == 1
        assert item.basis_event_seq >= 0
        assert item.last_affected_event_seq <= len(service.events)
        if key not in {"stats", "budget", "transitions"}:
            assert item.depends_on_events == tuple(sorted(item.depends_on_events))
    assert service.items["cwnd"].revision >= 1
    assert service.items["stats"].revision == len(service.events)
    assert service.items["budget"].revision == len(service.events)
