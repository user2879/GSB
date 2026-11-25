import unittest

from rlcb.breaker import STATE_HALF_OPEN, evaluate_service
from rlcb.hashing import canonical_json
from rlcb.spec import RequestSpec, ServiceSpec


def spec(**over):
    base = dict(
        service_id="s", host="127.0.0.1", port=30000, rps_limit=100,
        concurrency_limit=2, failure_threshold_bp=5000, window_length=4,
        degrade_level=1, recovery_windows=1, config_version=1, seed=7,
    )
    base.update(over)
    return ServiceSpec(**base)


def reqs(failed, batch_size=4):
    return [RequestSpec(f"s-R{i:03d}", "s", f"B{i // batch_size}", i, f)
            for i, f in enumerate(failed)]


class BreakerTest(unittest.TestCase):
    def test_opens_streaming_and_degrades(self):
        # 1/1 = 10000bp trips at the very first failure (streaming).
        decisions = evaluate_service(spec(), reqs([1, 0, 1, 1]))
        self.assertEqual(decisions[0].outcome, "degraded")
        self.assertEqual(decisions[0].basis,
                         "streaming_threshold_exceeded_opened")
        # Rest of the tripping window stays open/degraded.
        self.assertTrue(all(d.outcome == "degraded"
                            for d in decisions[1:4]))
        self.assertEqual(decisions[0].triggered_window, 0)
        self.assertEqual(decisions[0].threshold_comparison, "10000>=5000")

    def test_open_persists_to_window_end_then_half_open_recovery(self):
        # Window0 trips; window1 entirely open until boundary -> half_open;
        # window2 is a clean probe that closes the breaker.
        pattern = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        decisions = evaluate_service(spec(), reqs(pattern))
        self.assertEqual([d.outcome for d in decisions[0:4]],
                         ["degraded", "degraded", "degraded", "degraded"])
        self.assertEqual(decisions[4].breaker_state_before, STATE_HALF_OPEN)
        self.assertTrue(all(d.outcome == "executed"
                            for d in decisions[8:12]))
        self.assertEqual(decisions[11].basis,
                         "recovery_window_passed_breaker_closed")
        self.assertEqual(decisions[11].breaker_state_before, STATE_HALF_OPEN)

    def test_failed_probe_reopens(self):
        # Window0 trips; window1 is a probe whose rate crosses threshold and
        # reopens immediately (3/4=7500bp).
        # Window0 trips on its first failure; window1 is a half-open probe.
        # In the probe, 2/4=5000bp is reached only at the last request, which
        # therefore reopens exactly at index 7.
        pattern = [1, 0, 0, 0, 0, 0, 1, 1]
        decisions = evaluate_service(spec(), reqs(pattern))
        self.assertEqual(decisions[6].outcome, "executed")
        self.assertEqual(decisions[7].basis,
                         "probe_threshold_exceeded_reopened")
        self.assertEqual(decisions[7].outcome, "degraded")

    def test_two_recovery_windows_required(self):
        s = spec(recovery_windows=2)
        # Trip window0; window1 clean (1/2 passed); window2 clean -> close.
        pattern = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        decisions = evaluate_service(s, reqs(pattern))
        self.assertEqual(decisions[7].basis,
                         "recovery_window_passed_probe_continues")
        self.assertEqual(decisions[11].basis,
                         "recovery_window_passed_breaker_closed")

    def test_byte_identical_under_batch_resplit_and_dict_order(self):
        pattern = [1, 0, 1, 1, 0, 0, 1, 0]
        first = evaluate_service(spec(), reqs(pattern, batch_size=4))
        resplit = evaluate_service(spec(), reqs(pattern, batch_size=3))
        policy_keys = [k for k in first[0].to_dict() if k != "batch_id"]
        canonical = lambda ds: [
            canonical_json({k: d.to_dict()[k] for k in policy_keys})
            for d in ds]
        self.assertEqual(canonical(first), canonical(resplit))
        # Dict traversal order cannot change anything: recompute via shuffled
        # dict construction and compare.
        shuffled = evaluate_service(
            spec(), sorted(reqs(pattern),
                           key=lambda r: (-r.arrival_seq, r.request_id))[::-1])
        self.assertEqual(canonical(first), canonical(shuffled))

    def test_threshold_zero_trips_on_first_failure(self):
        decisions = evaluate_service(spec(failure_threshold_bp=0),
                                     reqs([0, 0, 1]))
        # Threshold 0 means rate >= 0bp always trips, even on first success.
        self.assertEqual(decisions[0].outcome, "degraded")
        self.assertTrue(all(d.outcome == "degraded" for d in decisions))

    def test_rate_rejected_skips_window(self):
        rejected = frozenset({"s-R000"})
        decisions = evaluate_service(spec(), reqs([1, 0, 0, 0]),
                                     rate_rejected_ids=rejected)
        self.assertEqual(decisions[0].outcome, "rate_rejected")
        self.assertEqual(decisions[0].window_size, 0)


if __name__ == "__main__":
    unittest.main()
