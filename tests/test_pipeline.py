import copy
import glob
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest

from rdiq.canonical import canonical_text, digest
from rdiq.engine import Engine, recover_and_replay
from rdiq.errors import ReplayError, SpecError
from rdiq.evidence import arbitrate_round
from rdiq.error_bounds import round_error_attribution
from rdiq.invalidation import propagate_invalidation
from rdiq.ledger import Ledger, expected_from_config
from rdiq.models import load_dataset, write_json
from rdiq.scheduler import Scheduler
from rdiq.synth import generate_sample_set


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clean_glob(pattern):
    for path in glob.glob(pattern):
        if os.path.isfile(path):
            os.chmod(path, 0o644)
            os.remove(path)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.mkdtemp(prefix="rdiq-test-")
        cls.base = os.path.join(cls.temp, "base")
        generate_sample_set(cls.base)
        cls.dataset = load_dataset(cls.base)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp, ignore_errors=True)

    def _engine(self, name, **kwargs):
        return Engine(
            self.dataset,
            os.path.join(self.temp, f"{name}.sqlite"),
            os.path.join(self.temp, f"{name}-backups"),
            **kwargs,
        )

    def test_01_spec_error_locates_first_entry_and_field(self):
        bad_dir = os.path.join(self.temp, "bad-vocab")
        shutil.copytree(self.base, bad_dir)
        with open(os.path.join(bad_dir, "fragments.json"), encoding="utf-8") as handle:
            fragments = json.load(handle)
        fragments["fragments"][0]["terms"] = ["not-in-vocab"]
        write_json(os.path.join(bad_dir, "fragments.json"), fragments)
        with self.assertRaises(SpecError) as caught:
            load_dataset(bad_dir)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.entry, "fragments.json")
        self.assertEqual(
            caught.exception.field,
            ["frag-000", "terms", 0],
        )

        bad_round = os.path.join(self.temp, "bad-round")
        shutil.copytree(self.base, bad_round)
        with open(os.path.join(bad_round, "sessions.json"), encoding="utf-8") as handle:
            sessions = json.load(handle)
        sessions["sessions"][0]["rounds"][1]["round"] = sessions["sessions"][0]["rounds"][0]["round"]
        write_json(os.path.join(bad_round, "sessions.json"), sessions)
        with self.assertRaises(SpecError) as caught:
            load_dataset(bad_round)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.entry, "sessions.json")
        self.assertIn("round", caught.exception.field)

        bad_scale = os.path.join(self.temp, "bad-scale")
        shutil.copytree(self.base, bad_scale)
        with open(os.path.join(bad_scale, "precision.json"), encoding="utf-8") as handle:
            precision = json.load(handle)
        precision["scale"] = 300
        write_json(os.path.join(bad_scale, "precision.json"), precision)
        with self.assertRaises(SpecError) as caught:
            load_dataset(bad_scale)
        self.assertEqual(caught.exception.field, "scale")

    def test_02_byte_identical_across_runs_and_fragment_order(self):
        outputs = []
        for index in range(2):
            db = os.path.join(self.temp, f"det-{index}.sqlite")
            backup = os.path.join(self.temp, f"det-{index}-backups")
            _clean_glob(db + "*")
            engine = Engine(self.dataset, db, backup)
            result = engine.run()
            engine.close()
            outputs.append(canonical_text(result))
        self.assertEqual(outputs[0], outputs[1])

        shuffled_dir = os.path.join(self.temp, "shuffled")
        shutil.copytree(self.base, shuffled_dir)
        fragments_path = os.path.join(shuffled_dir, "fragments.json")
        with open(fragments_path, encoding="utf-8") as handle:
            fragments = json.load(handle)
        fragments["fragments"].reverse()
        write_json(fragments_path, fragments)
        shuffled_dataset = load_dataset(shuffled_dir)
        engine_a = self._engine("order-a")
        engine_b = Engine(
            shuffled_dataset,
            os.path.join(self.temp, "order-b.sqlite"),
            os.path.join(self.temp, "order-b-backups"),
        )
        result_a = engine_a.run()
        result_b = engine_b.run()
        engine_a.close()
        engine_b.close()
        self.assertEqual(
            result_a["graph"]["graph_digest"], result_b["graph"]["graph_digest"]
        )
        self.assertEqual(
            [view["canonical"] for view in self._views(result_a)],
            [view["canonical"] for view in self._views(result_b)],
        )

    def _views(self, result):
        return [session["view"] for session in result["sessions"]]

    def test_03_dedup_weights_and_error_attribution(self):
        session = self.dataset.sessions[0]
        record = arbitrate_round(self.dataset, session["session_id"], session["rounds"][0])
        attribution = round_error_attribution(record)
        self.assertIn("adopted", record)
        self.assertIn("deduped", record)
        self.assertIn("weight_steps", record)
        self.assertEqual(
            len(record["evidence_digests"]),
            len(set(record["evidence_digests"])),
        )
        self.assertEqual(
            attribution["bounds_ulp"]["single_evidence_quantization"], 3
        )
        self.assertIn(attribution["dominant_level"], ("source", "evidence", "round"))
        self.assertTrue(attribution["explicit_threshold"].endswith("50% of total ULP bound"))
        self.assertIsInstance(attribution["dominant_threshold_met"], bool)
        if attribution["dominant_threshold_met"]:
            self.assertGreaterEqual(
                attribution["dominant_observed_basis_points"], 5000
            )
        for step in record["weight_steps"]:
            self.assertIn("truncate", step["basis"])

    def test_04_scheduler_enforces_both_quotas(self):
        lock = threading.Lock()
        active = 0
        per_session = {}
        peak_global = 0
        peak_session = {}

        def job(session_id):
            nonlocal active, peak_global
            with lock:
                active += 1
                per_session[session_id] = per_session.get(session_id, 0) + 1
                peak_global = max(peak_global, active)
                peak_session[session_id] = max(
                    peak_session.get(session_id, 0), per_session[session_id]
                )
            deadline = time.monotonic() + 0.002
            while time.monotonic() < deadline:
                pass
            with lock:
                active -= 1
                per_session[session_id] -= 1

        scheduler = Scheduler(global_limit=3, per_session_limit=2)
        jobs = []
        for session_index in range(3):
            for round_index in range(6):
                session_id = f"s{session_index}"
                jobs.append(
                    scheduler.submit(
                        session_id, round_index, lambda sid=session_id: job(sid)
                    )
                )
        reports = scheduler.shutdown(jobs)
        self.assertLessEqual(peak_global, 3)
        self.assertTrue(all(value <= 2 for value in peak_session.values()))
        self.assertEqual(scheduler.observed_limits()["peak_running"], 3)
        self.assertTrue(
            all(report["wait_ended_at_ms"] >= report["wait_started_at_ms"]
                for report in reports)
        )

    def test_05_invalidation_only_reruns_affected_subset(self):
        edited_dir = os.path.join(self.temp, "edited-weights")
        shutil.copytree(self.base, edited_dir)
        with open(os.path.join(edited_dir, "weights.json"), encoding="utf-8") as handle:
            weights = json.load(handle)
        weights["version"] = 2
        weights["channel_weights"]["feed"] = 5100
        write_json(os.path.join(edited_dir, "weights.json"), weights)
        with open(os.path.join(edited_dir, "sessions.json"), encoding="utf-8") as handle:
            sessions = json.load(handle)
        for session in sessions["sessions"]:
            for request in session["rounds"]:
                request["rule_version_ref"] = 1
        write_json(os.path.join(edited_dir, "sessions.json"), sessions)

        baseline = self._engine("invalidation-base")
        baseline.run()
        before = {
            sid: {idx: record["round_digest"] for idx, record in rounds.items()}
            for sid, rounds in baseline.round_records.items()
        }
        graph = baseline.graph
        baseline.close()

        edited = load_dataset(edited_dir)
        invalidation = propagate_invalidation(self.dataset, edited, graph)
        self.assertTrue(invalidation["invalid_rounds"])
        self.assertEqual(
            invalidation["invalid_rounds"],
            sorted(
                invalidation["invalid_rounds"],
                key=lambda item: (item["session_id"], item["round"]),
            ),
        )
        engine = Engine(
            edited,
            os.path.join(self.temp, "invalidation-base.sqlite"),
            os.path.join(self.temp, "invalidation-base-backups"),
        )
        self.assertIsNotNone(engine.ledger.replacement)
        engine.rerun_invalid(self.dataset, invalidation)
        invalid_keys = {
            (item["session_id"], item["round"])
            for item in invalidation["invalid_rounds"]
        }
        for sid in sorted(before):
            for idx in sorted(before[sid]):
                if (sid, idx) not in invalid_keys:
                    self.assertEqual(
                        engine.round_records[sid][idx]["round_digest"],
                        before[sid][idx],
                    )
        engine.close()

    def test_05b_channel_weight_change_is_a_true_partial_subset(self):
        edited_dir = os.path.join(self.temp, "channel-weight")
        shutil.copytree(
            os.path.join(ROOT, "samples", "edit", "channel-weight"),
            edited_dir,
        )
        baseline = self._engine("partial-base")
        baseline.run()
        before = {
            sid: {idx: record["round_digest"] for idx, record in rounds.items()}
            for sid, rounds in baseline.round_records.items()
        }
        graph = baseline.graph
        baseline.close()

        edited = load_dataset(edited_dir)
        invalidation = propagate_invalidation(self.dataset, edited, graph)
        invalid_keys = {
            (item["session_id"], item["round"])
            for item in invalidation["invalid_rounds"]
        }
        self.assertTrue(invalid_keys)
        self.assertLess(len(invalid_keys), 24)
        engine = Engine(
            edited,
            os.path.join(self.temp, "partial-base.sqlite"),
            os.path.join(self.temp, "partial-base-backups"),
        )
        recompute = engine.rerun_invalid(self.dataset, invalidation)
        self.assertEqual(len(recompute["rerun_jobs"]), len(invalid_keys))
        for sid in sorted(before):
            for idx in sorted(before[sid]):
                if (sid, idx) not in invalid_keys:
                    self.assertEqual(
                        engine.round_records[sid][idx]["round_digest"],
                        before[sid][idx],
                    )
        engine.close()

    def test_06_ledger_whole_snapshot_replacement(self):
        db = os.path.join(self.temp, "replace.sqlite")
        backup = os.path.join(self.temp, "replace-backups")
        ledger = Ledger(db, backup, expected_from_config(self.dataset.config))
        ledger.close()
        edited_config = copy.copy(self.dataset.config)
        edited_config.weights = dict(edited_config.weights, version=99)
        edited_config.versions = dict(edited_config.versions, weight_version=99)
        edited_config.digest = digest({"changed": True})
        replaced = Ledger(db, backup, expected_from_config(edited_config))
        self.assertIsNotNone(replaced.replacement)
        self.assertEqual(replaced.replacement["mode"], "whole-snapshot-replace")
        self.assertTrue(os.path.exists(replaced.replacement["read_only_backup"]))
        mode = os.stat(replaced.replacement["read_only_backup"]).st_mode & 0o777
        self.assertEqual(mode, 0o444)
        self.assertTrue(replaced.replacement["differences"])
        replaced.close()

    def test_07_crash_recovery_and_replay_mismatch(self):
        db = os.path.join(self.temp, "crash.sqlite")
        backup = os.path.join(self.temp, "crash-backups")
        _clean_glob(db + "*")
        ledger = Ledger(db, backup, expected_from_config(self.dataset.config))
        session = self.dataset.sessions[0]
        entries = []
        for request in session["rounds"][:3]:
            record = arbitrate_round(self.dataset, session["session_id"], request)
            entries.append(
                {
                    "session_id": session["session_id"],
                    "round": request["round"],
                    "round_record": record,
                    "error_record": round_error_attribution(record),
                }
            )
        ledger.write_batch(
            entries[:2],
            [
                {
                    "session_id": session["session_id"],
                    "last_committed_round": entry["round"],
                    "total_rounds": len(session["rounds"]),
                }
                for entry in entries[:2]
            ],
            entries[:2],
        )
        ledger.begin_interrupted_batch(entries[2:], [], entries[2:])
        ledger.close()

        reopened = sqlite3.connect(db)
        self.assertEqual(
            reopened.execute("SELECT COUNT(*) FROM batches").fetchone()[0], 1
        )
        self.assertEqual(
            reopened.execute("SELECT COUNT(*) FROM round_records").fetchone()[0], 2
        )
        reopened.close()

        engine, result = recover_and_replay(
            self.dataset, db, backup, global_limit=2, per_session_limit=1
        )
        self.assertEqual(result["recovery"]["committed_rounds"], 2)
        unknown_keys = {
            (item["session_id"], item["round"])
            for item in result["recovery"]["unknown_rerun_rounds"]
        }
        self.assertIn((session["session_id"], 2), unknown_keys)
        engine.close()

        tampered = sqlite3.connect(db)
        row = tampered.execute(
            "SELECT round_json FROM round_records WHERE round_index = 0"
        ).fetchone()
        record = json.loads(row[0])
        record["confidence"] += 1
        tampered.execute(
            "UPDATE round_records SET round_json = ? WHERE round_index = 0",
            (canonical_text(record),),
        )
        tampered.commit()
        tampered.close()
        with self.assertRaises(ReplayError) as caught:
            recover_and_replay(self.dataset, db, backup)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.entry, session["session_id"])
        self.assertIn("confidence", caught.exception.field)


if __name__ == "__main__":
    unittest.main(verbosity=2)
