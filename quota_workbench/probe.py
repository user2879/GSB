"""Real-socket health probing and the instance health state machine.

Probe verdicts come only from real TCP connects / reads with real socket
timeouts; no verdict is ever inferred from sleeps or process state.
Instances migrate unknown -> healthy | unhealthy -> ejected; every
migration records the probe round, consecutive-failure count and a
monotonic-clock millisecond timestamp.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass

from .spec import Spec
from .slog import StructuredLogger, NULL_LOGGER

STATES = ("unknown", "healthy", "unhealthy", "ejected")


@dataclass(frozen=True)
class Transition:
    instance_id: str
    from_state: str
    to_state: str
    probe_round: int
    consec_failures: int
    mono_ms: int
    virt_ms: int
    reason: str

    def logical_key(self) -> tuple:
        """Clock-independent identity used for replay comparison."""
        return (
            self.instance_id,
            self.from_state,
            self.to_state,
            self.probe_round,
            self.consec_failures,
            self.virt_ms,
            self.reason,
        )


def probe_once(host: str, port: int, timeout_ms: int, probe_round: int) -> tuple:
    """One real probe. Returns (ok, reason). The timeout is produced by the
    socket itself via settimeout."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_ms / 1000.0)
    try:
        try:
            sock.connect((host, port))
        except socket.timeout:
            return False, "connect_timeout"
        except ConnectionRefusedError:
            return False, "connect_refused"
        except OSError as exc:
            return False, f"connect_error:{exc.errno}"
        try:
            sock.sendall(f"PING {probe_round}\n".encode("ascii"))
            data = sock.recv(256)
        except socket.timeout:
            return False, "recv_timeout"
        except OSError as exc:
            return False, f"recv_error:{exc.errno}"
        if data.startswith(b"OK "):
            return True, "ok"
        return False, "bad_response"
    finally:
        sock.close()


def run_probe_rounds(spec: Spec, rounds: int, logger: StructuredLogger = NULL_LOGGER) -> list:
    """Run `rounds` probe rounds sequentially over instances in sorted-id
    order, sleeping the declared interval between rounds. Returns the
    ordered list of health transitions."""
    policy = spec.probe_policy
    states = {iid: "unknown" for iid in spec.instances}
    consec = {iid: 0 for iid in spec.instances}
    transitions: list = []
    ordered_ids = sorted(spec.instances)
    for rnd in range(rounds):
        for iid in ordered_ids:
            inst = spec.instances[iid]
            ok, reason = probe_once(inst.host, inst.port, policy.timeout_ms, rnd)
            old = states[iid]
            if ok:
                consec[iid] = 0
                new = "healthy"
            else:
                consec[iid] += 1
                if consec[iid] >= policy.eject_threshold:
                    new = "ejected"
                elif consec[iid] >= policy.failure_threshold:
                    new = "unhealthy"
                else:
                    new = old
            if new != old:
                transition = Transition(
                    instance_id=iid,
                    from_state=old,
                    to_state=new,
                    probe_round=rnd,
                    consec_failures=consec[iid],
                    mono_ms=int(time.monotonic() * 1000),
                    virt_ms=rnd * policy.interval_ms,
                    reason=reason,
                )
                transitions.append(transition)
                states[iid] = new
                logger.log(
                    "health_transition",
                    instance=iid,
                    from_state=old,
                    to_state=new,
                    probe_round=rnd,
                    consec_failures=consec[iid],
                    reason=reason,
                )
        if rnd < rounds - 1:
            time.sleep(policy.interval_ms / 1000.0)
    return transitions


class HealthTimeline:
    """Virtual-time health view consumed by the arbiter."""

    def __init__(self, instance_ids) -> None:
        self._timeline = {iid: [] for iid in instance_ids}

    def apply(self, transition: Transition) -> None:
        self._timeline[transition.instance_id].append((transition.virt_ms, transition.to_state))

    def state_at(self, instance_id: str, virt_ms: int) -> str:
        state = "unknown"
        for ts, st in self._timeline[instance_id]:
            if ts <= virt_ms:
                state = st
            else:
                break
        return state
