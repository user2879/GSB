import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccobs import fixedpoint as fp
from ccobs.config import SessionConfig

CFG = SessionConfig(initial_seq=0, cwnd_max=65535, mss=1460, scale_factor=3,
                    session_budget=64, batch_size=4, window_batches=2,
                    dup_ack_limit=3)


class FixedPointTest(unittest.TestCase):
    def test_slow_start_grows_one_mss_per_ack(self):
        state = fp.initial_state(CFG)
        self.assertEqual(state.cwnd, 1460 * 1000)
        state = fp.apply_ack(state, CFG, 10)
        self.assertEqual(state.cwnd, 2 * 1460 * 1000)
        state = fp.apply_ack(state, CFG, 10)
        self.assertEqual(state.cwnd, 3 * 1460 * 1000)

    def test_congestion_avoidance_grows_one_mss_per_rt(self):
        unit = CFG.unit
        state = fp.initial_state(CFG)
        state = fp.apply_timeout(state, CFG)  # cwnd=1mss, ssthresh=2mss
        state = fp.apply_ack(state, CFG, 10)  # slow start: cwnd=2mss -> CA
        self.assertEqual(state.mode, fp.CONGESTION_AVOIDANCE)
        before = state.cwnd
        state = fp.apply_ack(state, CFG, 10)
        inc = state.cwnd - before
        # 每确认增加 mss*mss/cwnd (定点截断), 约一往返一个 MSS
        self.assertEqual(inc, (1460 * unit * 1460 * unit) // before)
        self.assertGreater(state.err_cwnd, 0)  # 量化误差上界已登记

    def test_three_dup_acks_trigger_fast_retransmit(self):
        state = fp.initial_state(CFG)
        for _ in range(4):
            state = fp.apply_ack(state, CFG, 10)
        cwnd_before = state.cwnd
        for _ in range(3):
            state = fp.apply_dup_ack(state, CFG)
        self.assertEqual(state.dup_run, 3)
        self.assertEqual(state.ssthresh, max(cwnd_before // 2,
                                             2 * 1460 * CFG.unit))
        self.assertEqual(state.cwnd, state.ssthresh + 3 * 1460 * CFG.unit)

    def test_timeout_returns_to_slow_start(self):
        state = fp.initial_state(CFG)
        for _ in range(3):
            state = fp.apply_ack(state, CFG, 10)
        state = fp.apply_timeout(state, CFG)
        self.assertEqual(state.mode, fp.SLOW_START)
        self.assertEqual(state.cwnd, 1460 * CFG.unit)
        self.assertEqual(state.flight, 0)

    def test_srtt_rttvar_fixed_point_and_rto(self):
        state = fp.initial_state(CFG)
        state = fp.apply_ack(state, CFG, 16)
        self.assertEqual(state.srtt, 16 * CFG.unit)
        self.assertEqual(state.rttvar, 8 * CFG.unit)
        state = fp.apply_ack(state, CFG, 8)
        self.assertEqual(state.srtt, (7 * 16000 + 8000) // 8)
        self.assertEqual(state.rto, state.srtt + 4 * state.rttvar)
        self.assertEqual(fp.item_error_bound(state, fp.ITEM_RTO),
                         state.err_srtt + 4 * state.err_rttvar)


if __name__ == "__main__":
    unittest.main()
