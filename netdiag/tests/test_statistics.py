import unittest

from netdiag.constants import RATE_SCALE
from netdiag.gaps import analyze_model
from netdiag.merge import merge_sources
from netdiag.statistics import aggregate_session
from netdiag.wire import decode_buffer, encode_frame

FLOW = dict(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=40001,
            dst_port=50001, protocol=6)


def make(seq_ts):
    blob = b"".join(encode_frame(timestamp_us=ts * 1000, seq=seq,
                                 flags=0x02, **FLOW) for seq, ts in seq_ts)
    return decode_buffer(blob)


class StatisticsTests(unittest.TestCase):
    def test_fixed_point_and_error_bound(self):
        cap = make([(1, 1000), (2, 1010), (5, 1020)])  # 缺 3、4
        merged = merge_sources(cap, [])
        conclusions = analyze_model(merged.model)
        stats = aggregate_session(merged.model, conclusions,
                                  window_ms=100, min_samples=10)
        window = stats["flows"][cap.frames[0].flow_key]["windows"][0]
        self.assertEqual(window["expected"], 5)
        self.assertEqual(window["lost"], 2)
        self.assertEqual(window["rate_bps"],
                         (RATE_SCALE * 2) // 5)
        self.assertEqual(window["rounding"], "floor")
        self.assertTrue(window["insufficient_samples"])
        # floor 量化误差上界 ceil(scale/expected)
        self.assertEqual(window["error_bound_bps"],
                         (RATE_SCALE + 4) // 5)

    def test_stable_tie_break_within_error_bound(self):
        cap_a = make([(1, 1000), (3, 1010)])  # rate 1/3
        flow_b = dict(FLOW, src_ip="10.0.0.5", dst_ip="10.0.0.6",
                      src_port=40010, dst_port=50010)
        blob_b = b"".join(encode_frame(timestamp_us=ts * 1000, seq=seq,
                                       flags=0x02, **flow_b)
                          for seq, ts in [(1, 1000), (2, 1005), (4, 1015)])
        cap_b = decode_buffer(blob_b)
        combined = decode_buffer(b"".join([
            b"".join(encode_frame(timestamp_us=ts * 1000, seq=seq,
                                  flags=0x02, **FLOW)
                     for seq, ts in [(1, 1000), (3, 1010)]),
            blob_b]))
        merged = merge_sources(combined, [])
        conclusions = analyze_model(merged.model)
        stats = aggregate_session(merged.model, conclusions,
                                  window_ms=100, min_samples=1)
        ranking = stats["ranking"][0]["order"]
        self.assertEqual(len(ranking), 2)
        # 高比率在前；比率相同时 flow 字典序决胜
        self.assertGreaterEqual(ranking[0]["rate_bps"],
                                ranking[1]["rate_bps"])
        self.assertTrue(any(row["error_bound_decisive"] for row in ranking))


if __name__ == "__main__":
    unittest.main()
