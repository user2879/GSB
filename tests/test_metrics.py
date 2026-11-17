from __future__ import annotations

import unittest

from tunnelvpn.errors import ErrorCode, TunnelError
from tunnelvpn.metrics import Ledger, QueueEvent, SessionRecord, round_half_up


def record(
    session_id: str,
    *,
    route_id: str = "alpha",
    start_ms: int,
    end_ms: int,
    payload: int,
    forwarded: int | None = None,
    queue: tuple[QueueEvent, ...] = (),
    code: str = "OK",
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        route_id=route_id,
        peer_id="peer",
        sequence=0,
        start_ms=start_ms,
        end_ms=end_ms,
        payload_bytes=payload,
        forwarded_bytes=payload if forwarded is None else forwarded,
        wire_bytes=payload + 20,
        queue_events=queue,
        status="COMPLETED" if code == "OK" else "FAILED",
        business_code=code,
        encapsulation_digest="00" * 16,
        fragment_count=1,
    )


class MetricsTests(unittest.TestCase):
    def test_single_write_per_session(self) -> None:
        ledger = Ledger([record("a-1", start_ms=0, end_ms=10, payload=5)])
        with self.assertRaises(TunnelError) as caught:
            ledger.record(record("a-1", start_ms=10, end_ms=20, payload=5))
        self.assertEqual(caught.exception.code, ErrorCode.E_STATE.value)

    def test_fixed_point_statistics(self) -> None:
        ledger = Ledger(
            [
                record(
                    "a-1",
                    start_ms=0,
                    end_ms=1000,
                    payload=1000,
                    queue=(QueueEvent(0, 10),),
                ),
                record(
                    "a-2",
                    start_ms=1000,
                    end_ms=2000,
                    payload=1000,
                    queue=(QueueEvent(1000, 1010),),
                ),
            ]
        )
        stats = ledger.statistics()
        self.assertEqual(stats["scale"], 1000)
        self.assertEqual(stats["quantization_error_max"]["throughput_kbps"], 0.0005)
        self.assertEqual(stats["global"]["forwarded_bytes"], 2000)
        self.assertEqual(stats["global"]["throughput_kbps_fixed"], 977)
        self.assertEqual(stats["global"]["throughput_kbps"], "0.977")
        self.assertEqual(stats["global"]["average_queue_ms_fixed"], 10000)
        self.assertEqual(stats["global"]["average_queue_duration_ms"], "10.000")
        self.assertEqual(stats["routes"]["alpha"]["session_count"], 2)

    def test_failed_session_forwarded_bytes_excluded_from_throughput(self) -> None:
        ledger = Ledger(
            [
                record("a-1", start_ms=0, end_ms=1000, payload=1024, forwarded=0, code="E_PEER"),
                record("a-2", start_ms=1000, end_ms=2000, payload=1024, forwarded=1024),
            ]
        )
        stats = ledger.statistics()
        self.assertEqual(stats["global"]["payload_bytes"], 2048)
        self.assertEqual(stats["global"]["forwarded_bytes"], 1024)
        self.assertEqual(stats["global"]["throughput_kbps_fixed"], 500)
        self.assertEqual(stats["business_code_counts"], {"E_PEER": 1, "OK": 1})

    def test_ledger_digest_is_insertion_order_independent(self) -> None:
        first = record("a-1", start_ms=0, end_ms=10, payload=1)
        second = record("a-2", start_ms=20, end_ms=30, payload=2)
        left = Ledger([first, second])
        right = Ledger([second, first])
        self.assertEqual(left.digest(), right.digest())
        self.assertEqual(left.canonical_bytes(), right.canonical_bytes())

    def test_round_half_up(self) -> None:
        self.assertEqual(round_half_up(1, 2), 1)
        self.assertEqual(round_half_up(5, 2), 3)
        self.assertEqual(round_half_up(2499, 1000), 2)


if __name__ == "__main__":
    unittest.main()
