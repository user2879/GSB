import unittest

from src import receipts
from src.normalize import adopt, normalize

D = "a" * 64
E = "b" * 64


def _rec(line, src_file="f", line_no=1):
    return receipts.parse_line(line, line_no=line_no, src_file=src_file)


class TestNormalize(unittest.TestCase):
    def test_r1_fills_with_basis(self):
        rec = _rec(f"r1|batch-0000|req-0000|{D}")
        (view,) = normalize([rec])
        self.assertEqual(view.src_format, 1)
        self.assertEqual(view.filled, frozenset(
            {"param_version", "elapsed_ms", "worker_id", "rank", "dup_of"}))
        self.assertEqual(view.fill_basis["rank"], "derived:request-id-lexicographic")
        self.assertEqual(view.fill_basis["param_version"], "absent-in-format:default-0")
        self.assertEqual(view.real_param_version(), -1)

    def test_r2_fills_rank_and_dup_only(self):
        rec = _rec(f"r2|batch-0000|req-0000|3|9|w0|{D}")
        (view,) = normalize([rec])
        self.assertEqual(view.filled, frozenset({"rank", "dup_of"}))
        self.assertEqual(view.param_version, 3)
        self.assertEqual(view.real_param_version(), 3)

    def test_derived_rank_is_lexicographic(self):
        recs = [_rec(f"r2|batch-0000|req-0003|1|1|w0|{D}"),
                _rec(f"r2|batch-0000|req-0001|1|1|w0|{D}")]
        views = normalize(recs)
        by_id = {v.request_id: v for v in views}
        self.assertEqual(by_id["req-0001"].rank, 0)
        self.assertEqual(by_id["req-0003"].rank, 1)

    def test_filled_never_overrides_real(self):
        old = _rec(f"r1|batch-0000|req-0000|{D}", src_file="a", line_no=1)
        new = _rec(f"r2|batch-0000|req-0000|1|5|w0|{E}", src_file="z", line_no=9)
        result = adopt(normalize([old, new]))
        winner, rejected = result[("batch-0000", "req-0000")]
        self.assertIs(winner.receipt, new)
        self.assertEqual(len(rejected), 1)
        self.assertIn("superseded-by", rejected[0][1])

    def test_param_version_desc_then_format_desc(self):
        r2_old = _rec(f"r2|batch-0000|req-0000|1|5|w0|{D}", src_file="a", line_no=1)
        r3_new = _rec(f"r3|batch-0000|req-0000|2|5|w0|0||{E}", src_file="b", line_no=1)
        r3_old = _rec(f"r3|batch-0000|req-0000|1|5|w0|0||{D}", src_file="c", line_no=1)
        result = adopt(normalize([r3_old, r2_old, r3_new]))
        winner, rejected = result[("batch-0000", "req-0000")]
        self.assertIs(winner.receipt, r3_new)
        self.assertEqual(len(rejected), 2)

    def test_tie_break_stable_by_source(self):
        first = _rec(f"r2|batch-0000|req-0000|1|5|w0|{D}", src_file="a", line_no=1)
        second = _rec(f"r2|batch-0000|req-0000|1|5|w0|{D}", src_file="b", line_no=1)
        for order in ([first, second], [second, first]):
            result = adopt(normalize(list(order)))
            winner, _ = result[("batch-0000", "req-0000")]
            self.assertIs(winner.receipt, first)

    def test_rejection_reason_deterministic(self):
        a = _rec(f"r1|batch-0000|req-0000|{D}", src_file="h", line_no=2)
        b = _rec(f"r3|batch-0000|req-0000|1|5|w0|0||{E}", src_file="w", line_no=1)
        reasons = set()
        for _ in range(5):
            result = adopt(normalize([a, b]))
            _, rejected = result[("batch-0000", "req-0000")]
            reasons.add(rejected[0][1])
        self.assertEqual(len(reasons), 1)


if __name__ == "__main__":
    unittest.main()
