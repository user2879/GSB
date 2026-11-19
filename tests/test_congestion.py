from __future__ import annotations

from tcp_workbench.congestion import RttEstimator, grow_cwnd_on_ack
from tcp_workbench.models import CongestionConfig, Event
from tcp_workbench.runner import load_event_script
from tcp_workbench.state_machine import SimulationState
from tests.conftest import SAMPLES


def config() -> CongestionConfig:
    return CongestionConfig(
        initial_cwnd_mss=4,
        initial_ssthresh_bytes=8000,
        rtt_scale=8,
        initial_rto_ms=1000,
        min_rto_ms=1,
        max_rto_ms=8000,
    )


def test_rtt_fixed_point_and_timeout_backoff():
    estimator = RttEstimator(config())
    assert estimator.rto_ms == 1000
    estimator.add_sample(16)
    assert estimator.srtt_fp == 16 * 8
    assert estimator.rttvar_fp == (16 * 8) // 2
    assert estimator.srtt_ms == 16
    assert estimator.rttvar_ms == 8
    assert estimator.rto_ms == 48  # 16 + 4*8, floor at scale 8
    estimator.on_timeout()
    assert estimator.rto_ms == 96
    assert estimator.backoff_count == 1


def test_slow_start_and_congestion_avoidance_are_integer_only():
    assert grow_cwnd_on_ack(4000, 8000, 1000, 1000, 16000) == 5000
    assert grow_cwnd_on_ack(5000, 2000, 1000, 1000, 16000) == 5200
    assert grow_cwnd_on_ack(2000, 2000, 1000, 1000, 16000) == 2500


def test_state_machine_triple_duplicate_timeout_and_karn(sample_config_script):
    config_value, script = sample_config_script
    events = load_event_script(SAMPLES / "events.json")
    state = SimulationState(config_value, script)
    for index, event in enumerate(events[:9], start=1):
        state.apply(Event.from_dict({**event.as_dict(), "event_seq": index}))
    assert state.ssthresh == 2000
    assert state.fast_retransmits == 1
    assert state.cwnd == 5000

    state.apply(Event.from_dict({**events[9].as_dict(), "event_seq": 10}))
    assert state.rtt.sample_count == 1
    assert state.rtt.karn_excluded == 1  # fast retransmit segment acknowledgement
    state.apply(Event.from_dict({**events[10].as_dict(), "event_seq": 11}))

    state.apply(Event.from_dict({**events[11].as_dict(), "event_seq": 12}))
    assert state.timeouts == 1
    assert state.cwnd == 1000
    assert state.rtt.backoff_count == 1
    assert state.rtt.rto_ms == 400  # first RTT sample already clamped RTO to 200

    state.apply(Event.from_dict({**events[12].as_dict(), "event_seq": 13}))
    state.apply(Event.from_dict({**events[13].as_dict(), "event_seq": 14}))
    assert state.rtt.karn_excluded == 2
    assert state.rtt.sample_count == 1
    state.apply(Event.from_dict({**events[14].as_dict(), "event_seq": 15}))
    state.apply(Event.from_dict({**events[15].as_dict(), "event_seq": 16}))
    assert state.rtt.sample_count == 2


def test_transition_log_contains_packet_seq_event_and_monotonic_ms(
    sample_config_script,
):
    config_value, script = sample_config_script
    state = SimulationState(config_value, script)
    events = load_event_script(SAMPLES / "events.json")
    for index, event in enumerate(events, start=1):
        state.apply(Event.from_dict({**event.as_dict(), "event_seq": index}))
    assert [transition["to"] for transition in state.transitions] == [
        "SYN_SENT",
        "ESTABLISHED",
        "FIN_WAIT",
        "CLOSE_WAIT",
        "CLOSED",
    ]
    assert all("trigger_packet_type" in row for row in state.transitions)
    assert all("trigger_seq" in row for row in state.transitions)
    assert all("event_seq" in row and "mono_ms" in row for row in state.transitions)
