import os
import tempfile
import threading
import time
import unittest

from delayqueue import DelayQueue, TimeoutConfigError, load_declaration
from delayqueue.models import ACKED, DEAD, INFLIGHT, READY
from delayqueue.policy import should_fail


SPEC = {
    "spec_version": "v1",
    "global_concurrency": 2,
    "topics": {
        "a": {"visibility_timeout_ms": 200, "max_deliveries": 2,
              "backoff_ms": [20, 40], "concurrency": 1},
        "b": {"visibility_timeout_ms": 200, "max_deliveries": 2,
              "backoff_ms": [20, 40], "concurrency": 1},
    },
}


def _write(path, payload):
    import json
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.spec_path = os.path.join(self.tmp, "topics.json")
        self.ledger_path = os.path.join(self.tmp, "q.db")
        self.message_path = os.path.join(self.tmp, "messages.ndjson")
        _write(self.spec_path, SPEC)
        lines = []
        for index in range(4):
            topic = "a" if index % 2 == 0 else "b"
            lines.append(
                f'{{"message_id":"m{index}","topic":"{topic}",'
                f'"deliver_at":0,"payload":0,"idempotency_key":"k{index}"}}')
        with open(self.message_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))

    def test_global_and_per_topic_caps_never_bypassed(self):
        from delayqueue import open_queue
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            held = []
            for _ in range(2):
                delivery = queue.claim(timeout=2)
                self.assertIsNotNone(delivery)
                held.append(delivery)
            self.assertEqual(len(queue.inflight), 2)  # global cap
            counts = {message.topic for message in queue.inflight.values()}
            self.assertEqual(counts, {"a", "b"})  # one per topic quota
            # Third claim must block: both quotas and global cap are full.
            self.assertIsNone(queue.claim(timeout=0.1))
            for delivery in held:
                queue.ack(delivery)
            # Two messages remain queued; drain them under the same caps.
            def handler(delivery, q):
                q.ack(delivery)
            queue.run_workers(handler, worker_count=2)
            self.assertTrue(queue.wait_drained(5))
        finally:
            queue.close()

    def test_visibility_timeout_uses_real_monotonic_time(self):
        from delayqueue import open_queue
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            delivery = queue.claim(timeout=2)
            message = queue.get_message(delivery.message_id)
            self.assertEqual(message.state, INFLIGHT)
            time.sleep(0.35)
            self.assertEqual(message.state, READY)
            self.assertEqual(message.attempts, 0)  # no increment on timeout
            rows = queue.ledger.conn.execute(
                "SELECT close_reason FROM visibility WHERE message_id=?",
                (delivery.message_id,)).fetchall()
            self.assertIn(("visibility_timeout",), rows)
        finally:
            queue.close()

    def test_retry_backoff_then_dead_letter(self):
        from delayqueue import open_queue
        # payload 2 forces two failures; max_deliveries=2 -> dead.
        with open(self.message_path, "w", encoding="utf-8") as handle:
            handle.write(
                '{"message_id":"d1","topic":"a","deliver_at":0,'
                '"payload":2,"idempotency_key":"kd1"}')
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            def handler(delivery, q):
                if should_fail(delivery.attempt, delivery.payload):
                    q.nack(delivery, "boom")
                else:
                    q.ack(delivery)
            queue.run_workers(handler, worker_count=1)
            self.assertTrue(queue.wait_drained(5))
            message = queue.get_message("d1")
            self.assertEqual(message.state, DEAD)
            self.assertEqual(message.attempts, 2)
            self.assertEqual(message.first_error, "boom")
            self.assertEqual(message.last_error, "boom")
        finally:
            queue.close()

    def test_worker_count_above_global_cap_rejected_e_timeout(self):
        spec = load_declaration(self.spec_path)
        from delayqueue.ledger import Ledger
        from delayqueue.queue import DelayQueue
        ledger = Ledger.create(
            self.ledger_path + ".x", spec,
            epoch_ms=int(time.time() * 1000))
        queue = DelayQueue(spec, ledger, epoch_ms=int(time.time() * 1000))
        try:
            with self.assertRaises(TimeoutConfigError) as caught:
                queue.run_workers(lambda d, q: None, worker_count=99)
            self.assertEqual(caught.exception.code, "E_TIMEOUT")
        finally:
            queue.close()

    def test_idempotent_ack_suppresses_second_effect(self):
        from delayqueue import open_queue
        with open(self.message_path, "w", encoding="utf-8") as handle:
            handle.write(
                '{"message_id":"i1","topic":"a","deliver_at":0,'
                '"payload":0,"idempotency_key":"shared"}\n'
                '{"message_id":"i2","topic":"b","deliver_at":0,'
                '"payload":0,"idempotency_key":"shared"}')
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            effects = []
            def handler(delivery, q):
                result = q.ack(delivery)
                effects.append(result["business_effect"])
            queue.run_workers(handler, worker_count=2)
            self.assertTrue(queue.wait_drained(5))
            self.assertEqual(effects.count(True), 1)
            self.assertEqual(effects.count(False), 1)
            suppressed = queue.ledger.conn.execute(
                "SELECT suppressed_count FROM dedup WHERE "
                "idempotency_key='shared'").fetchone()[0]
            self.assertEqual(suppressed, 1)
            self.assertEqual(queue.get_message("i2").suppressed, 1)
            self.assertEqual(queue.get_message("i2").state, ACKED)
        finally:
            queue.close()


if __name__ == "__main__":
    unittest.main()
