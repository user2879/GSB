"""Deterministic per-service admission rate limiting.

Business admission is an integer policy derived solely from the declaration
and arrival order: within each arrival second (``arrival_seq // rps_limit``
bucketing via a fixed second index) the first ``rps_limit`` requests are
admitted and the rest rejected. A limit of zero rejects every request. This
keeps rate-rejection conclusions byte-identical across runs, thread
interleavings and batch splits. The monotonic clock is still used by the
scheduler to measure real queue waiting time, but never to change a decision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeterministicRateLimiter:
    service_id: str
    rps_limit: int
    _seen_per_second: dict[int, int]

    @classmethod
    def create(cls, service_id: str, rps_limit: int, clock=None):
        return cls(service_id=service_id, rps_limit=int(rps_limit),
                   _seen_per_second={})

    # Arrival "seconds" are derived deterministically from the declaration:
    # each rps_limit consecutive arrival sequences form one admission second,
    # so exactly rps_limit requests are admitted per second and the quota
    # refills at every block boundary without relying on wall-clock jitter.
    def second_index(self, arrival_seq: int) -> int:
        if self.rps_limit <= 0:
            return arrival_seq
        return arrival_seq // max(1, self.rps_limit)

    def try_admit_seq(self, arrival_seq: int) -> bool:
        if self.rps_limit <= 0:
            return False
        second = self.second_index(arrival_seq)
        used = self._seen_per_second.get(second, 0)
        if used >= self.rps_limit:
            return False
        self._seen_per_second[second] = used + 1
        return True


# Backwards-compatible token-bucket name retained for the clock-based wait
# estimator; policy admission uses DeterministicRateLimiter.
TokenBucket = DeterministicRateLimiter
