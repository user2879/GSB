"""Simulated convergence executor, run as an independent child process.

Deterministic behaviour: the advance duration and the resulting state
digest are pure functions of (seed, round, target_digest), so identical
pulse sequences always yield byte-identical version trajectories.

Version rule (declared): version advances monotonically to the pulse
round number; the state digest commits to
sha256(scenario_id|executor_id|v<version>|target_digest).
"""

import argparse
import hashlib
import random
import socket
import sys
import time

from . import wire


def advance_ms_for(seed, round_no, base_ms, jitter_ms):
    """Deterministic per-round advance duration in integer ms."""
    rng = random.Random((seed * 2654435761 + round_no * 40503) & 0xFFFFFFFF)
    return base_ms + rng.randint(0, jitter_ms)


def state_digest_for(scenario_id, executor_id, version, target_digest):
    payload = "%s|%s|v%d|%s" % (scenario_id, executor_id, version,
                                target_digest)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Executor:
    def __init__(self, scenario_id, executor_id, port, seed, target_digest,
                 base_ms=1, jitter_ms=4, drop_first_attempt_rounds=()):
        self.scenario_id = scenario_id
        self.executor_id = executor_id
        self.port = port
        self.seed = seed
        self.target_digest = target_digest
        self.base_ms = base_ms
        self.jitter_ms = jitter_ms
        self.drop_first_attempt_rounds = set(drop_first_attempt_rounds)
        self.version = 0
        self.last_round = 0
        self.attempts = {}
        self.receipts = {}

    def handle_pulse(self, msg):
        round_no = msg["round"]
        if round_no < self.last_round:
            return wire.encode_error(self.executor_id, round_no, "E_STALE")
        cached = self.receipts.get(round_no)
        if cached is not None:
            # Idempotent replay: identical result, no extra advancement.
            return cached
        attempt = self.attempts.get(round_no, 0) + 1
        self.attempts[round_no] = attempt
        if attempt == 1 and round_no in self.drop_first_attempt_rounds:
            return None  # deterministic drop -> orchestrator timeout/retry
        advance_ms = advance_ms_for(self.seed, round_no, self.base_ms,
                                    self.jitter_ms)
        if advance_ms > 0:
            time.sleep(advance_ms / 1000.0)  # real lease time elapses
        self.version = round_no  # declared monotonic version rule
        self.last_round = round_no
        obs_mono_ms = msg["send_mono_ms"] + advance_ms
        digest = state_digest_for(self.scenario_id, self.executor_id,
                                  self.version, self.target_digest)
        receipt = wire.encode_receipt(self.executor_id, round_no,
                                      self.version, obs_mono_ms, digest)
        self.receipts[round_no] = receipt
        return receipt

    def serve(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", self.port))
        while True:
            data, addr = sock.recvfrom(4096)
            try:
                kind, msg = wire.decode_datagram(data)
            except wire.WireError:
                continue
            if kind == "ping" and msg["scenario_id"] == self.scenario_id:
                sock.sendto(wire.encode_pong(self.executor_id), addr)
                continue
            if kind != "pulse" or msg["scenario_id"] != self.scenario_id:
                continue
            reply = self.handle_pulse(msg)
            if reply is not None:
                sock.sendto(reply, addr)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="convd.executor")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--executor-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-digest", required=True)
    parser.add_argument("--base-ms", type=int, default=1)
    parser.add_argument("--jitter-ms", type=int, default=4)
    parser.add_argument("--drop-first-attempt-rounds", default="")
    args = parser.parse_args(argv)
    drops = [int(x) for x in args.drop_first_attempt_rounds.split(",") if x]
    Executor(args.scenario_id, args.executor_id, args.port, args.seed,
             args.target_digest, args.base_ms, args.jitter_ms,
             drops).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
