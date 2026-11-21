import json
import os
import subprocess
import sys
import tempfile
import unittest

from delayqueue import (
    NotReproducibleError,
    ReplayError,
    SnapshotMismatchError,
    SpecError,
    load_declaration,
    open_queue,
    recover_queue,
    replay_ledger,
)
from delayqueue.policy import should_fail

SPEC = {
    "spec_version": "v1",
    "global_concurrency": 3,
    "topics": {
        "a": {"visibility_timeout_ms": 250, "max_deliveries": 3,
              "backoff_ms": [15, 30, 60], "concurrency": 2},
    },
}


def _bootstrap(tmp):
    spec_path = os.path.join(tmp, "topics.json")
    message_path = os.path.join(tmp, "messages.ndjson")
    ledger_path = os.path.join(tmp, "q.db")
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(SPEC, handle)
    with open(message_path, "w", encoding="utf-8") as handle:
        handle.write(
            '{"message_id":"m1","topic":"a","deliver_at":0,'
            '"payload":1,"idempotency_key":"k1"}\n'
            '{"message_id":"m2","topic":"a","deliver_at":0,'
            '"payload":9,"idempotency_key":"k2"}\n'
            '{"message_id":"m3","topic":"a","deliver_at":0,'
            '"payload":0,"idempotency_key":"k3"}')
    return spec_path, message_path, ledger_path


class LedgerRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.spec_path, self.message_path, self.ledger_path = _bootstrap(
            self.tmp)

    def test_atomic_ledger_records_every_migration(self):
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            queue.run_workers(
                lambda d, q: q.nack(d, "x")
                if should_fail(d.attempt, d.payload) else q.ack(d),
                worker_count=2)
            self.assertTrue(queue.wait_drained(5))
            transitions = queue.ledger.conn.execute(
                "SELECT COUNT(*) FROM transitions").fetchone()[0]
            self.assertGreater(transitions, 0)
            progress = queue.ledger.conn.execute(
                "SELECT COUNT(*) FROM progress").fetchone()[0]
            self.assertGreater(progress, 0)
            result = replay_ledger(queue)
            self.assertEqual(result["status"], "consistent")
            self.assertEqual(queue.get_message("m2").state, "dead")
        finally:
            queue.close()

    def test_tampered_digest_raises_snapshot_and_replaces(self):
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        queue.run_workers(lambda d, q: q.ack(d), worker_count=1)
        queue.wait_drained(5)
        queue.ledger.tamper_digest()
        queue.close()
        with self.assertRaises(SnapshotMismatchError) as caught:
            recover_queue(self.ledger_path)
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        self.assertIn("discarded_transitions", caught.exception.report)

        from delayqueue import Ledger, load_stream
        spec = load_declaration(self.spec_path)
        inputs = load_stream(self.message_path, spec)
        backup = os.path.join(self.tmp, "backup.db")
        import time
        report = Ledger.replace(
            self.ledger_path, backup, spec, inputs,
            epoch_ms=int(time.time() * 1000))
        self.assertEqual(report["message_count_new"], 3)
        self.assertEqual(report["message_count_diff"], 0)
        self.assertFalse(os.access(backup, os.W_OK))

    def test_replay_divergence_reports_message_and_field(self):
        # Single-message ledger so the first field divergence is located on
        # that exact message id instead of an earlier aggregate mismatch.
        single_spec = os.path.join(self.tmp, "single_topics.json")
        single_msg = os.path.join(self.tmp, "single_messages.ndjson")
        single_db = os.path.join(self.tmp, "single.db")
        with open(single_spec, "w", encoding="utf-8") as handle:
            json.dump(SPEC, handle)
        with open(single_msg, "w", encoding="utf-8") as handle:
            handle.write(
                '{"message_id":"only","topic":"a","deliver_at":0,'
                '"payload":0,"idempotency_key":"only-k"}')
        queue, _ = open_queue(single_spec, single_msg, single_db)
        try:
            queue.run_workers(lambda d, q: q.ack(d), worker_count=1)
            queue.wait_drained(5)
            queue.ledger.conn.execute(
                "UPDATE messages SET attempts=99 WHERE message_id='only'")
            queue.ledger.conn.commit()
            with self.assertRaises(ReplayError) as caught:
                replay_ledger(queue)
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertEqual(caught.exception.message_id, "only")
            self.assertEqual(caught.exception.field, "attempts")
            self.assertEqual(caught.exception.expected, 0)
            self.assertEqual(caught.exception.actual, 99)
        finally:
            queue.close()

    def test_spec_version_mismatch_is_not_reproducible(self):
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            with queue.ledger.batch() as cur:
                queue.ledger._set_meta(cur, "spec_version", "v999")
            with self.assertRaises(NotReproducibleError) as caught:
                replay_ledger(queue)
            self.assertEqual(caught.exception.code, "E_NOT_REPRODUCIBLE")
            self.assertEqual(caught.exception.details["kind"],
                             "spec_version")
        finally:
            queue.close()

    def test_dedup_version_mismatch_is_not_reproducible(self):
        queue, _ = open_queue(
            self.spec_path, self.message_path, self.ledger_path)
        try:
            with queue.ledger.batch() as cur:
                queue.ledger.bump_dedup_version(cur)
            with self.assertRaises(NotReproducibleError) as caught:
                replay_ledger(queue)
            self.assertEqual(caught.exception.details["kind"],
                             "dedup_version")
        finally:
            queue.close()

    def test_hard_kill_in_subprocess_then_recovery_consistency(self):
        crash_script = os.path.join(self.tmp, "crash.py")
        with open(crash_script, "w", encoding="utf-8") as handle:
            handle.write(
                "import os, sys\n"
                "sys.path.insert(0, %r)\n"
                "from delayqueue import open_queue\n"
                "queue, _ = open_queue(%r, %r, %r)\n"
                "claimed = []\n"
                "for _ in range(2):\n"
                "    d = queue.claim(timeout=2)\n"
                "    if d: claimed.append(d.message_id)\n"
                "sys.stdout.write('claimed=' + ','.join(claimed))\n"
                "sys.stdout.flush()\n"
                "os._exit(42)\n"
                % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   self.spec_path, self.message_path, self.ledger_path))
        proc = subprocess.run(
            [sys.executable, crash_script], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 42)
        self.assertIn("claimed=", proc.stdout)

        queue, report = recover_queue(self.ledger_path)
        try:
            self.assertEqual(report.unknown_messages, 2)
            self.assertEqual(report.recovered_messages, 3)
            self.assertEqual(len(report.redelivered_after_visibility), 2)
            queue.run_workers(
                lambda d, q: q.nack(d, "x")
                if should_fail(d.attempt, d.payload) else q.ack(d),
                worker_count=2)
            self.assertTrue(queue.wait_drained(8))
            result = replay_ledger(queue)
            self.assertEqual(result["status"], "consistent")
            self.assertEqual(result["unknown_redeliveries"], 2)
            # unknown markers persisted on the recovered messages
            unknown = sum(
                message.unknown_count for message in queue.all_messages())
            self.assertEqual(unknown, 2)
        finally:
            queue.close()


if __name__ == "__main__":
    unittest.main()
