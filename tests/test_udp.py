from __future__ import annotations

import time

import pytest

from tunnel_interop.errors import E_FIELD, E_TIMEOUT, TunnelError
from tunnel_interop.fragmenter import build_bundle
from tunnel_interop.ledger import SessionLedger
from tunnel_interop.spec import load_spec_mapping
from tunnel_interop.workflow import forward_bundle

from conftest import make_spec_data


@pytest.mark.parametrize("scenario", ["normal", "reverse", "duplicate"])
def test_real_loopback_udp_and_reassembly(tmp_path, scenario):
    data = make_spec_data(route_count=1, sample_bytes=620, global_limit=2, per_tunnel_quota=2)
    spec = load_spec_mapping(data, source_path="<udp>")
    bundle = build_bundle(spec)
    ledger_path = tmp_path / f"{scenario}.db"
    report = forward_bundle(
        spec=spec,
        bundle=bundle,
        ledger_path=ledger_path,
        scenario=scenario,
        message_keys=["tun-0"],
    )
    message = report["messages"][0]
    assert report["ok"] is True
    assert message["payload_matches_inner_sample"] is True
    assert message["receipt_count"] == len(bundle.messages[0].fragments) * (2 if scenario == "duplicate" else 1)
    if scenario == "reverse":
        assert message["arrival_indices"] == list(reversed(range(len(bundle.messages[0].fragments))))
    if scenario == "duplicate":
        assert len(message["duplicate_indices"]) == len(bundle.messages[0].fragments)
    with SessionLedger(ledger_path) as ledger:
        opened = ledger.open(spec)
        replay = ledger.replay_committed(spec)
    assert opened.unknown_batches == ()
    assert replay.checked_batches == 1
    assert replay.checked_fragments == len(bundle.messages[0].fragments)
    assert replay.checked_reassemblies == 1


@pytest.mark.parametrize("target_format", ["f1", "f2"])
def test_real_udp_supports_legacy_wire_formats(tmp_path, target_format):
    data = make_spec_data(route_count=1, sample_bytes=540)
    spec = load_spec_mapping(data, source_path="<legacy>")
    bundle = build_bundle(spec, target_format=target_format)
    ledger_path = tmp_path / f"{target_format}.db"
    report = forward_bundle(
        spec=spec,
        bundle=bundle,
        ledger_path=ledger_path,
        message_keys=["tun-0"],
    )
    assert report["messages"][0]["payload_matches_inner_sample"] is True
    assert report["target_format"] == target_format
    with SessionLedger(ledger_path) as ledger:
        ledger.open(spec)
        replay = ledger.replay_committed(spec)
    assert replay.checked_batches == 1


def test_declared_reassembly_timeout_is_enforced_by_real_socket_deadline(tmp_path):
    data = make_spec_data(
        route_count=1,
        timeout_ms=30,
        ack_timeout_ms=250,
        max_retries=0,
    )
    spec = load_spec_mapping(data, source_path="<reassembly-timeout>")
    bundle = build_bundle(spec)
    started = time.monotonic()
    with pytest.raises(TunnelError) as captured:
        forward_bundle(
            spec=spec,
            bundle=bundle,
            ledger_path=tmp_path / "deadline.db",
            message_keys=["tun-0"],
            drop_all=True,
        )
    elapsed = time.monotonic() - started
    assert captured.value.code == E_TIMEOUT
    assert elapsed >= 0.02


def test_real_socket_timeout_is_e_timeout(tmp_path):
    data = make_spec_data(route_count=1, ack_timeout_ms=60, max_retries=0)
    spec = load_spec_mapping(data, source_path="<timeout>")
    bundle = build_bundle(spec)
    started = time.monotonic()
    with pytest.raises(TunnelError) as captured:
        forward_bundle(
            spec=spec,
            bundle=bundle,
            ledger_path=tmp_path / "timeout.db",
            message_keys=["tun-0"],
            drop_all=True,
        )
    elapsed = time.monotonic() - started
    assert captured.value.code == E_TIMEOUT
    assert captured.value.path.startswith("fragments[")
    assert elapsed >= 0.04
    with SessionLedger(tmp_path / "timeout.db") as ledger:
        opened = ledger.open(spec)
    assert opened.unknown_batches
    assert opened.unknown_batches[0]["rerun_required"] is True
    assert opened.can_claim_exit_report is False
    assert opened.exit_report is None


def test_real_ack_missing_field_is_e_field(tmp_path):
    data = make_spec_data(route_count=1)
    spec = load_spec_mapping(data, source_path="<field>")
    bundle = build_bundle(spec)
    with pytest.raises(TunnelError) as captured:
        forward_bundle(
            spec=spec,
            bundle=bundle,
            ledger_path=tmp_path / "field.db",
            message_keys=["tun-0"],
            ack_mode="missing-field",
        )
    assert captured.value.code == E_FIELD
    assert captured.value.path == "ack.received_at_ms"
