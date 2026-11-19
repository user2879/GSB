"""Integer-only teaching congestion control and RTT estimation.

RTT fixed-point contract
------------------------
* Scale factor: 8 (three fractional bits).
* All divisions use floor/truncation-toward-zero (all operands are non-negative).
* First sample: SRTT = sample, RTTVAR = sample / 2.
* Later samples: RTTVAR = (3*RTTVAR + |SRTT-sample|) / 4,
  SRTT = (7*SRTT + sample) / 8.
* RTO = SRTT + 4*RTTVAR, clamped to [min_rto_ms, max_rto_ms].
* Before the first sample, RTO is the configured initial_rto_ms.
* A timeout doubles the current RTO, clamped to max_rto_ms.  A later accepted
  sample replaces the backed-off estimate.
* Karn exclusion is enforced by the session engine before add_sample; samples
  that acknowledge a retransmitted segment are never passed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CongestionConfig

RTT_SCALE = 8
RTT_TRUNCATION = "floor"


@dataclass
class RttEstimator:
    config: CongestionConfig
    srtt_fp: int = 0
    rttvar_fp: int = 0
    sample_count: int = 0
    karn_excluded: int = 0
    backoff_count: int = 0

    def __post_init__(self) -> None:
        self.rto_ms = self.config.initial_rto_ms

    def _recompute_rto(self) -> None:
        candidate = (self.srtt_fp + 4 * self.rttvar_fp) // RTT_SCALE
        self.rto_ms = max(
            self.config.min_rto_ms, min(self.config.max_rto_ms, candidate)
        )

    def add_sample(self, sample_ms: int) -> bool:
        if (
            not isinstance(sample_ms, int)
            or isinstance(sample_ms, bool)
            or sample_ms < 0
        ):
            raise ValueError(
                "RTT sample must be a non-negative integer millisecond value"
            )
        sample_fp = sample_ms * RTT_SCALE
        if self.sample_count == 0:
            self.srtt_fp = sample_fp
            self.rttvar_fp = sample_fp // 2
        else:
            deviation = abs(self.srtt_fp - sample_fp)
            self.rttvar_fp = (3 * self.rttvar_fp + deviation) // 4
            self.srtt_fp = (7 * self.srtt_fp + sample_fp) // 8
        self.sample_count += 1
        self.backoff_count = 0
        self._recompute_rto()
        return True

    def exclude_sample(self) -> None:
        self.karn_excluded += 1

    def on_timeout(self) -> None:
        self.backoff_count += 1
        self.rto_ms = min(
            self.config.max_rto_ms, max(self.config.min_rto_ms, self.rto_ms * 2)
        )

    @property
    def srtt_ms(self) -> int:
        return self.srtt_fp // RTT_SCALE

    @property
    def rttvar_ms(self) -> int:
        return self.rttvar_fp // RTT_SCALE

    def as_dict(self) -> dict[str, int | str]:
        return {
            "srtt_fp": self.srtt_fp,
            "srtt_ms": self.srtt_ms,
            "rttvar_fp": self.rttvar_fp,
            "rttvar_ms": self.rttvar_ms,
            "rto_ms": self.rto_ms,
            "sample_count": self.sample_count,
            "karn_excluded": self.karn_excluded,
            "backoff_count": self.backoff_count,
            "scale": RTT_SCALE,
            "truncation": RTT_TRUNCATION,
            "fallback": "initial_rto_ms_before_first_sample",
        }


def grow_cwnd_on_ack(
    cwnd_bytes: int, ssthresh_bytes: int, acked_bytes: int, mss: int, window_limit: int
) -> int:
    """Grow cwnd with byte-granular slow start and conservative avoidance.

    Slow start adds one byte for every acknowledged byte.  Congestion
    avoidance adds MSS*MSS/cwnd per acknowledgement, which yields one MSS per
    congestion window / one window per RTT.  Integer truncation is floor and a
    positive ack grows by at least one byte.
    """

    if acked_bytes <= 0:
        return min(cwnd_bytes, window_limit)
    if cwnd_bytes < ssthresh_bytes:
        grown = cwnd_bytes + acked_bytes
    else:
        increment = max(1, (mss * acked_bytes) // max(1, cwnd_bytes))
        grown = cwnd_bytes + increment
    return min(window_limit, grown)


def threshold_after_loss(bytes_in_flight: int, mss: int) -> int:
    """Loss threshold: half of flight, MSS-aligned, with a two-MSS floor."""

    half_aligned = (max(0, bytes_in_flight) // 2 // mss) * mss
    return max(2 * mss, half_aligned)


def enter_fast_recovery(ssthresh: int, mss: int, window_limit: int) -> int:
    return min(window_limit, ssthresh + 3 * mss)


def timeout_reset_cwnd(mss: int) -> int:
    return mss
