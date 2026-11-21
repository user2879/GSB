from __future__ import annotations

import json
from pathlib import Path

from qosfdiag.engine import CascadeEngine
from qosfdiag.generator import DeterministicUdpGenerator, decode_packet, encode_packet
from qosfdiag.policy import load_policy
from qosfdiag.samples import three_stage_generator, three_stage_policy
from qosfdiag.transport import InMemoryTransport


def test_generator_and_codec_are_deterministic() -> None:
    first = DeterministicUdpGenerator(three_stage_generator())
    second = DeterministicUdpGenerator(three_stage_generator())
    assert first.manifest == second.manifest
    assert first.sequence_digest == second.sequence_digest
    for packet in first.iter_packets():
        assert decode_packet(encode_packet(packet)) == packet


def test_queue_fifo_and_drop_statistics() -> None:
    policy = load_policy(three_stage_policy())
    generator = DeterministicUdpGenerator(three_stage_generator())
    transport = InMemoryTransport([shaper.shaper_id for shaper in policy.shapers])
    with CascadeEngine(policy, generator, scale=3, transport=transport) as engine:
        result = engine.run()
    assert result.summary.total_generated_packets == 40
    assert result.summary.final_released_packets == 25
    assert result.summary.final_released_bytes == 21771
    assert result.summary.average_delay_ns > 0
    stage = {item.stage_id: item for item in result.summary.stage_stats}
    assert stage["edge"].queued_packets > 0
    assert stage["edge"].dropped_packets > 0
    assert stage["core"].max_queue_len <= policy.by_id["core"].queue_limit_packets
    assert all(
        event.queue_len_before == 0
        for event in result.events
        if event.kind == "arrival" and event.decision == "PASS"
    )
    releases = [
        event
        for event in result.events
        if event.stage_id == "edge" and event.kind == "release"
    ]
    assert [event.packet_seq for event in releases] == sorted(
        event.packet_seq for event in releases
    )


def test_loopback_udp_reaches_final_sink() -> None:
    policy = load_policy(three_stage_policy())
    raw = three_stage_generator()
    raw["count"] = 10
    generator = DeterministicUdpGenerator(raw)
    with CascadeEngine(
        policy,
        generator,
        scale=3,
        use_loopback=True,
        base_port=56200,
    ) as engine:
        assert all(49152 <= port <= 65535 for port in engine.ports.values())
        result = engine.run()
        sink_datagrams = engine.transport.sink_datagrams
        sink_bytes = engine.transport.sink_bytes
    assert sink_datagrams == result.summary.final_released_packets
    assert sink_bytes == result.summary.final_released_bytes
    assert result.summary.transport_datagrams_sent > 0


def test_unmatched_ingress_has_explicit_decision() -> None:
    policy = load_policy(three_stage_policy())
    raw = three_stage_generator()
    raw["count"] = 2
    raw["dst_port_start"] = 61000
    generator = DeterministicUdpGenerator(raw)
    with CascadeEngine(
        policy,
        generator,
        scale=3,
        transport=InMemoryTransport([s.shaper_id for s in policy.shapers]),
    ) as engine:
        result = engine.run()
    assert result.summary.unmatched_packets == 2
    assert [event.stage_id for event in result.events] == ["<ingress>", "<ingress>"]
    assert all(event.decision == "DROP" for event in result.events)
