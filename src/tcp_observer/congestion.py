from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .fixedpoint import ceil_div, floor_div, saturate, scaled_ms_to_ceil
from .spec import ScriptEntry, SessionSpec

SLOW_START = "slow_start"
CONGESTION_AVOIDANCE = "congestion_avoidance"
FAST_RECOVERY = "fast_recovery"


@dataclass(frozen=True)
class ErrorBounds:
    cwnd_scaled: int
    ssthresh_scaled: int
    srtt_scaled: int
    rttvar_scaled: int
    rto_scaled: int

    def max_scaled(self) -> int:
        return max(
            self.cwnd_scaled,
            self.ssthresh_scaled,
            self.srtt_scaled,
            self.rttvar_scaled,
            self.rto_scaled,
        )

    def to_json(self, scale: int) -> dict[str, Any]:
        return {
            "cwnd_scaled": self.cwnd_scaled,
            "cwnd_ms_unit_packets_ceil": ceil_div(self.cwnd_scaled, scale),
            "ssthresh_scaled": self.ssthresh_scaled,
            "srtt_scaled": self.srtt_scaled,
            "srtt_ms_ceil": ceil_div(self.srtt_scaled, scale),
            "rttvar_scaled": self.rttvar_scaled,
            "rttvar_ms_ceil": ceil_div(self.rttvar_scaled, scale),
            "rto_scaled": self.rto_scaled,
            "rto_ms_ceil": ceil_div(self.rto_scaled, scale),
            "max_scaled": self.max_scaled(),
            "max_physical_micro_units_ceil": ceil_div(self.max_scaled() * 1000, scale),
        }


@dataclass
class CongestionState:
    scale: int
    window_max_scaled: int
    cwnd_scaled: int
    ssthresh_scaled: int
    srtt_scaled: int
    rttvar_scaled: int
    rto_scaled: int
    outstanding_packets: int = 0
    dupack_count: int = 0
    phase: str = SLOW_START
    fast_retransmit_count: int = 0
    timeout_count: int = 0
    saturation_count: int = 0
    bounds: ErrorBounds = field(default_factory=lambda: ErrorBounds(0, 0, 0, 0, 0))

    @classmethod
    def initial(cls, spec: SessionSpec) -> "CongestionState":
        scale = spec.scale
        srtt = spec.initial_rtt_ms * scale
        rttvar = floor_div(srtt, 2)
        rto = srtt + 4 * rttvar + scale
        return cls(
            scale=scale,
            window_max_scaled=spec.window_max * scale,
            cwnd_scaled=spec.initial_cwnd * scale,
            ssthresh_scaled=spec.initial_ssthresh * scale,
            srtt_scaled=srtt,
            rttvar_scaled=rttvar,
            rto_scaled=rto,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "cwnd_scaled": self.cwnd_scaled,
            "cwnd": floor_div(self.cwnd_scaled, self.scale),
            "cwnd_ceil": ceil_div(self.cwnd_scaled, self.scale),
            "ssthresh_scaled": self.ssthresh_scaled,
            "ssthresh": floor_div(self.ssthresh_scaled, self.scale),
            "ssthresh_ceil": ceil_div(self.ssthresh_scaled, self.scale),
            "srtt_scaled": self.srtt_scaled,
            "srtt_ms": floor_div(self.srtt_scaled, self.scale),
            "rttvar_scaled": self.rttvar_scaled,
            "rttvar_ms": floor_div(self.rttvar_scaled, self.scale),
            "rto_scaled": self.rto_scaled,
            "rto_ms": scaled_ms_to_ceil(self.rto_scaled, self.scale),
            "outstanding_packets": self.outstanding_packets,
            "outstanding_bytes": self.outstanding_packets * 0,  # filled by caller with MSS
            "dupack_count": self.dupack_count,
            "fast_retransmit_count": self.fast_retransmit_count,
            "timeout_count": self.timeout_count,
            "saturation_count": self.saturation_count,
        }


def _sat_cwnd(state: CongestionState, bound: int) -> int:
    if state.cwnd_scaled >= state.window_max_scaled:
        state.cwnd_scaled = state.window_max_scaled
        state.saturation_count += 1
        return 0
    return bound


def update_rtt(state: CongestionState, rtt_sample_ms: int) -> dict[str, int]:
    scale = state.scale
    sample_scaled = max(1, rtt_sample_ms) * scale
    old_srtt_bound = state.bounds.srtt_scaled
    old_var_bound = state.bounds.rttvar_scaled
    if state.srtt_scaled == 0:
        state.srtt_scaled = sample_scaled
        state.rttvar_scaled = floor_div(sample_scaled, 2)
        srtt_bound = 0
        var_bound = ceil_div(sample_scaled, 2) - state.rttvar_scaled
    else:
        difference = abs(state.srtt_scaled - sample_scaled)
        new_var = floor_div(3 * state.rttvar_scaled + difference, 4)
        new_srtt = floor_div(7 * state.srtt_scaled + sample_scaled, 8)
        var_bound = ceil_div(3 * old_var_bound + old_srtt_bound, 4) + 1
        srtt_bound = ceil_div(7 * old_srtt_bound, 8) + 1
        state.rttvar_scaled = new_var
        state.srtt_scaled = new_srtt
    state.rto_scaled = saturate(state.srtt_scaled + 4 * state.rttvar_scaled + scale)
    rto_bound = srtt_bound + 4 * var_bound + scale
    state.bounds = ErrorBounds(
        cwnd_scaled=state.bounds.cwnd_scaled,
        ssthresh_scaled=state.bounds.ssthresh_scaled,
        srtt_scaled=srtt_bound,
        rttvar_scaled=var_bound,
        rto_scaled=rto_bound,
    )
    return {"rtt_sample_ms": max(1, rtt_sample_ms), "rtt_sample_scaled": sample_scaled}


def _apply_new_ack(state: CongestionState) -> dict[str, Any]:
    scale = state.scale
    before = state.cwnd_scaled
    rule = ""
    local_error = 0
    if state.cwnd_scaled < state.ssthresh_scaled:
        state.cwnd_scaled = saturate(state.cwnd_scaled + scale, state.window_max_scaled)
        state.phase = SLOW_START
        rule = "slow_start:cwnd_plus_one_mss_per_ack"
    else:
        increment = floor_div(scale * scale, max(scale, state.cwnd_scaled))
        local_error = 1 if scale * scale % max(scale, state.cwnd_scaled) else 0
        state.cwnd_scaled = saturate(state.cwnd_scaled + increment, state.window_max_scaled)
        state.phase = CONGESTION_AVOIDANCE
        rule = "congestion_avoidance:floor(K^2/cwnd_scaled)_per_ack"
    cwnd_bound = state.bounds.cwnd_scaled + local_error
    cwnd_bound = _sat_cwnd(state, cwnd_bound)
    state.bounds = ErrorBounds(
        cwnd_scaled=cwnd_bound,
        ssthresh_scaled=state.bounds.ssthresh_scaled,
        srtt_scaled=state.bounds.srtt_scaled,
        rttvar_scaled=state.bounds.rttvar_scaled,
        rto_scaled=state.bounds.rto_scaled,
    )
    state.dupack_count = 0
    return {"rule": rule, "cwnd_before_scaled": before, "cwnd_after_scaled": state.cwnd_scaled, "local_truncation_units": local_error}


def _reduce_on_loss(state: CongestionState, kind: str) -> dict[str, Any]:
    old_cwnd = state.cwnd_scaled
    new_ssthresh = max(2 * state.scale, floor_div(old_cwnd, 2))
    ssthresh_bound = ceil_div(state.bounds.cwnd_scaled, 2) + 1
    state.ssthresh_scaled = saturate(new_ssthresh, state.window_max_scaled)
    state.cwnd_scaled = state.scale
    state.dupack_count = 0
    state.phase = SLOW_START
    state.bounds = ErrorBounds(
        cwnd_scaled=0,
        ssthresh_scaled=ssthresh_bound,
        srtt_scaled=state.bounds.srtt_scaled,
        rttvar_scaled=state.bounds.rttvar_scaled,
        rto_scaled=state.bounds.rto_scaled,
    )
    if kind == "timeout":
        state.timeout_count += 1
        state.rto_scaled = saturate(state.rto_scaled * 2)
        return {"rule": "timeout:ssthresh=max(2MSS,floor(cwnd/2));cwnd=1MSS;rto_backoff_x2", "old_cwnd_scaled": old_cwnd}
    state.fast_retransmit_count += 1
    return {"rule": "three_dup_ack:ssthresh=max(2MSS,floor(cwnd/2));cwnd=1MSS;slow_start", "old_cwnd_scaled": old_cwnd}


def step_congestion(
    state: CongestionState,
    entry: ScriptEntry,
    spec: SessionSpec,
    response: dict[str, Any] | None = None,
) -> tuple[CongestionState, dict[str, Any]]:
    """Advance one event. The response, when supplied, confirms peer output."""

    propagation: list[str] = []
    if response is not None and response.get("event_seq") != entry.event_seq:
        propagation.append("warning:response_event_seq_does_not_match_script")

    rtt_info = update_rtt(state, entry.delay_ms)
    propagation.append(f"rtt_fixed_point:floor_recurrence_K={spec.scale}")

    state.outstanding_packets += 1
    propagation.append("send:outstanding_plus_one_mss")

    acks_to_process = 0
    dup_events = 0
    event_flags: list[str] = []
    loss_action: dict[str, Any] | None = None

    if entry.type in {"ack", DELAY := "delay", "reorder"}:
        acks_to_process = max(0, entry.ack_count)
    elif entry.type == "dup_ack":
        dup_events = max(0, entry.dup_count)
    elif entry.type == "loss":
        event_flags.append("loss")
        loss_action = _reduce_on_loss(state, "timeout")
        propagation.append(loss_action["rule"])

    acknowledged = min(state.outstanding_packets, acks_to_process)
    state.outstanding_packets -= acknowledged
    if acks_to_process > acknowledged:
        propagation.append("ack_count_exceeded_outstanding:clamped_to_zero")

    ack_steps: list[dict[str, Any]] = []
    for _ack_index in range(acks_to_process):
        step_info = _apply_new_ack(state)
        ack_steps.append(step_info)
        propagation.append(step_info["rule"])

    fast_triggered = False
    for _ in range(dup_events):
        state.dupack_count += 1
        if state.dupack_count >= spec.dupack_threshold and not fast_triggered:
            fast_triggered = True
            event_flags.append("fast_retransmit")
            loss_action = _reduce_on_loss(state, "fast_retransmit")
            propagation.append(loss_action["rule"])
        elif not fast_triggered:
            propagation.append(f"dup_ack:{state.dupack_count}/{spec.dupack_threshold}")

    snapshot = state.snapshot()
    snapshot["outstanding_bytes"] = state.outstanding_packets * spec.mss
    detail = {
        "event_seq": entry.event_seq,
        "entry_type": entry.type,
        "event_flags": event_flags,
        "rtt": rtt_info,
        "acks_processed": acks_to_process,
        "ack_steps": ack_steps,
        "peer_order": list(entry.order),
        "propagation_basis": propagation,
        "loss_action": loss_action,
        "local_quantization_error": {
            "truncation": "floor toward zero for non-negative fixed-point numerators",
            "overflow": f"saturate to [0, {state.window_max_scaled}] for cwnd/ssthresh and int64 max otherwise",
            "single_event_bound_scaled_units": 1,
            "single_event_physical_bound_ms_ceil": ceil_div(1, spec.scale),
        },
        "error_bounds": state.bounds.to_json(spec.scale),
        "state_after": snapshot,
    }
    return state, detail
