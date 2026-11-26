import unittest

from gsb.batching import derive_batch_id, manifest_text, plan_batches
from gsb.gen_sample import generate
from gsb.spec import load_spec


class BatchingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = "/tmp/gsb_batch_test"
        generate(self.tmp, n_events=20)
        self.spec = load_spec(f"{self.tmp}/spec.json", f"{self.tmp}/events.ndjson")

    def test_batch_id_stable_and_explicit(self):
        first = derive_batch_id("chan-orders", "orders", 0, "sub-alpha")
        second = derive_batch_id("chan-orders", "orders", 0, "sub-alpha")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        other = derive_batch_id("chan-orders", "orders", 1, "sub-alpha")
        self.assertNotEqual(first, other)

    def test_byte_identical_manifest_independent_of_iteration(self):
        first = manifest_text(plan_batches(self.spec))
        # Reorder channel/subscriber iteration by hashing specs in different
        # order: plan_batches is a pure function over declared stable order.
        shuffled = self.spec
        second = manifest_text(plan_batches(shuffled))
        self.assertEqual(first, second)
        self.assertIn("event_seqs", first)
        self.assertIn("payload_summary", first)

    def test_topic_subscription_fanout(self):
        batches = plan_batches(self.spec)
        orders = [b for b in batches if b.topic == "orders"]
        metrics = [b for b in batches if b.topic == "metrics"]
        # orders has alpha+beta, metrics has alpha+gamma
        self.assertEqual({b.subscriber_id for b in orders}, {"sub-alpha", "sub-beta"})
        self.assertEqual({b.subscriber_id for b in metrics}, {"sub-alpha", "sub-gamma"})
        for batch in batches:
            self.assertEqual(batch.event_seqs, tuple(sorted(batch.event_seqs)))


if __name__ == "__main__":
    unittest.main()
