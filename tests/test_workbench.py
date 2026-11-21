from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tlstrust.canonical import canonical_json, digest_without_field
from tlstrust.certfamily import generate_family, load_family
from tlstrust.demo import run_demo
from tlstrust.errors import E_CHAIN, E_REPLAY, E_REVOKE, E_ROLLBACK, E_SNAPSHOT, E_TRUST, WorkbenchError
from tlstrust.faults import inject_chain, inject_revoke, inject_snapshot
from tlstrust.handshake import capture_real_handshake, load_input_snapshot, write_input_snapshot
from tlstrust.ledger import Ledger
from tlstrust.trust import demo_revisions, parse_trust_document, write_trust_document
from tlstrust.verify import (
    compare_results,
    compute_flips,
    validate_many,
    validate_snapshot,
)
from tlstrust.workspace import prepare_workspace


class CertificateTests(unittest.TestCase):
    def test_deterministic_family_and_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            family1 = generate_family(first, roots=2, intermediate_levels=1, leaves=1, validity_days=7, key_bits=512)
            family2 = generate_family(second, roots=2, intermediate_levels=1, leaves=1, validity_days=7, key_bits=512)
            self.assertEqual(family1.certificate_fingerprints, family2.certificate_fingerprints)
            self.assertEqual(family1.serial_numbers, family2.serial_numbers)
            self.assertEqual(family1.root_fingerprints, family2.root_fingerprints)
            normal = family1.load_chain("leaf-normal-01")
            for child, parent in zip(normal, normal[1:]):
                self.assertTrue(child.verify_signature(parent))
            self.assertTrue(normal[-1].is_self_signed())
            badsig = family1.load_chain("leaf-badsig-01")
            self.assertFalse(badsig[0].verify_signature(badsig[1]))
            overlong = family1.load_chain("leaf-overlong-01")
            self.assertEqual(len(overlong), len(normal) + 1)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.family = generate_family(
            cls.root / "family",
            roots=2,
            intermediate_levels=2,
            leaves=1,
            validity_days=30,
            key_bits=1024,
        )
        cls.document = parse_trust_document(
            {"revisions": [revision.content() for revision in demo_revisions(cls.family)]},
            cls.family,
        )
        cls.snapshot = capture_real_handshake(cls.family)
        cls.results = validate_many(cls.snapshot, cls.document)
        cls.trust_path = cls.root / "trust.json"
        cls.snapshot_path = cls.root / "snapshot.json"
        write_trust_document(cls.trust_path, cls.document.revisions)
        write_input_snapshot(cls.snapshot_path, cls.snapshot)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_real_handshake_capture(self) -> None:
        self.assertEqual(self.snapshot["capture_method"], "stdlib_ssl_tls12_wire_proxy")
        self.assertEqual(self.snapshot["negotiated"]["tls_version"], "TLSv1.2")
        self.assertEqual(self.snapshot["chain_fingerprints"], self.family.leaf()["chain_fingerprints"])
        self.assertEqual(len(self.snapshot["chain"]), 4)
        self.assertEqual(load_input_snapshot(self.snapshot_path), self.snapshot)

    def test_five_categories_and_flips(self) -> None:
        statuses = [(item["revision"], item["conclusion"], item["failure_category"]) for item in self.results]
        self.assertEqual(
            statuses,
            [
                (1, "PASS", None),
                (2, "FAIL", "UNTRUSTED_ANCHOR"),
                (3, "FAIL", "REVOKED"),
                (4, "FAIL", "EXPIRED"),
                (5, "FAIL", "PATH_LEN"),
                (6, "PASS", None),
            ],
        )
        flips = compute_flips(self.results)
        self.assertEqual(
            [(item["from_revision"], item["to_revision"], item["kind"]) for item in flips],
            [
                (1, 2, "CONCLUSION"),
                (2, 3, "CATEGORY"),
                (3, 4, "CATEGORY"),
                (4, 5, "CATEGORY"),
                (5, 6, "CONCLUSION"),
            ],
        )
        for item in self.results:
            if item["conclusion"] == "FAIL":
                self.assertIn(item["failure_category"], {"UNTRUSTED_ANCHOR", "BAD_SIGNATURE", "PATH_LEN", "EXPIRED", "REVOKED"})
                self.assertIsNotNone(item["first_failure"])

    def test_bad_signature_expired_and_overlong(self) -> None:
        cases = {
            "leaf-badsig-01": "BAD_SIGNATURE",
            "leaf-expired-01": "EXPIRED",
            "leaf-overlong-01": "PATH_LEN",
        }
        for leaf_id, category in cases.items():
            snapshot = capture_real_handshake(self.family, leaf_id=leaf_id)
            result = validate_snapshot(snapshot, self.document.revisions[0])
            self.assertEqual(result["conclusion"], "FAIL")
            self.assertEqual(result["failure_category"], category)

    def test_repeatability_and_compare(self) -> None:
        first = validate_snapshot(self.snapshot, self.document.revisions[0])
        second = validate_snapshot(self.snapshot, self.document.revisions[0])
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(first["index"], second["index"])
        comparison = compare_results(first, second)
        self.assertTrue(comparison["equal"])
        other = validate_snapshot(self.snapshot, self.document.revisions[1])
        changed = compare_results(first, other)
        self.assertFalse(changed["equal"])
        self.assertIn("layer", changed)
        self.assertIn("item", changed)
        self.assertIn("expected", changed)

    def test_index_is_stable_across_hash_seeds(self) -> None:
        code = (
            "from tlstrust.certfamily import load_family;"
            "from tlstrust.trust import load_trust_document;"
            "from tlstrust.handshake import load_input_snapshot;"
            "from tlstrust.verify import validate_many;"
            f"f=load_family({str(self.root / 'family' / 'family.json')!r});"
            f"t=load_trust_document({str(self.trust_path)!r},f);"
            f"s=load_input_snapshot({str(self.snapshot_path)!r});"
            "print('|'.join(x['index'] for x in validate_many(s,t)))"
        )
        outputs = []
        for seed in ("1", "2"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(Path(__file__).resolve().parent.parent),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_trust_validation_failures(self) -> None:
        base = [revision.content() for revision in demo_revisions(self.family)]

        def altered(mutator):
            value = copy.deepcopy(base)
            mutator(value)
            return value

        cases = [
            (lambda value: value[1].__setitem__("revision", 3), E_TRUST, "revision"),
            (lambda value: value[1].__setitem__("parent_revision", 99), E_TRUST, "parent_revision"),
            (lambda value: value[0]["trusted_root_fingerprints"].append("sha256:" + "f" * 64), E_TRUST, "trusted_root"),
            (lambda value: value[0]["revoked_serial_numbers"].append("XYZ"), E_TRUST, "revoked_serial"),
            (lambda value: value[0]["revoked_serial_numbers"].append(1234), E_TRUST, "revoked_serial"),
            (lambda value: value[0].__setitem__("validity_evaluation_epoch_ms", -1), E_TRUST, "validity"),
            (
                lambda value: value[0]["path_len_overrides"].append(
                    {"certificate_fingerprint": "sha256:" + "a" * 64, "path_len": 0}
                ),
                E_TRUST,
                "path",
            ),
            (lambda value: value[0]["revoked_serial_numbers"].append("DEADBEEF"), E_REVOKE, "revoked_serial"),
        ]
        for mutator, code, field in cases:
            with self.subTest(code=code, field=field):
                with self.assertRaises(WorkbenchError) as raised:
                    parse_trust_document({"revisions": altered(mutator)}, self.family)
                self.assertEqual(raised.exception.code, code)
                self.assertIsNotNone(raised.exception.revision)
                self.assertIn(field, raised.exception.field_path or "")

    def test_ledger_staleness_rollback_and_replay(self) -> None:
        ledger_path = self.root / "ledger.sqlite3"
        ledger = Ledger(ledger_path)
        try:
            created = ledger.create_trust_version(self.document, source_path=str(self.trust_path))
            run_id = ledger.start_run(created["version"], self.snapshot, self.document.revision_numbers)
            for result in self.results:
                ledger.record_revision_result(run_id, result)
            session = ledger.finalize_run(run_id)["payload"]
            replay = ledger.replay_session(session, self.family)
            self.assertTrue(replay["ok"])
            view1 = ledger.build_current_view()
            self.assertEqual(view1["trust_version"], created["version"])
            self.assertEqual(len(view1["runs"]), 1)
            rollback = ledger.rollback_to_revision(3)
            self.assertEqual(rollback["version"], created["version"] + 1)
            self.assertEqual(rollback["summary"]["old_trust_version"], created["version"])
            all_sessions = ledger.list_session_snapshots(current_only=False)
            stale = [item for item in all_sessions if item["content_digest"] == session["content_digest"]]
            self.assertEqual(len(stale), 1)
            self.assertEqual(stale[0]["stale"], 1)
            view2 = ledger.build_current_view()
            self.assertEqual(view2["trust_version"], rollback["version"])
            self.assertEqual(view2["runs"], [])
            stale_cache = ledger.conn.execute(
                "SELECT stale FROM ui_cache WHERE version = ?",
                (created["version"],),
            ).fetchone()
            self.assertEqual(stale_cache["stale"], 1)
            old_matches = ledger.find_results_by_index(self.results[0]["index"])
            self.assertTrue(old_matches)
            self.assertTrue(all(match["stale"] for match in old_matches))
            self.assertEqual(
                rollback["summary"]["old_failure_categories"],
                ["EXPIRED", "PATH_LEN", "REVOKED", "UNTRUSTED_ANCHOR"],
            )
            with self.assertRaises(WorkbenchError) as raised:
                ledger.rollback_to_revision(99)
            self.assertEqual(raised.exception.code, E_ROLLBACK)
            self.assertTrue(ledger.replay_session(session, self.family)["ok"])

            rolled_document = ledger.current_trust_document()
            self.assertIsNotNone(rolled_document)
            rolled = []
            for revision in rolled_document.revisions:
                rolled.append(validate_snapshot(self.snapshot, revision))
            run2 = ledger.start_run(rollback["version"], self.snapshot, rolled_document.revision_numbers)
            for result in rolled:
                ledger.record_revision_result(run2, result)
            session2 = ledger.finalize_run(run2)["payload"]
            self.assertTrue(ledger.replay_session(session2, self.family)["ok"])

            tampered = copy.deepcopy(session2)
            tampered["results"][0]["checks"][0]["expected"] = "tampered"
            tampered["content_digest"] = digest_without_field(tampered, "content_digest")
            with self.assertRaises(WorkbenchError) as raised:
                ledger.replay_session(tampered, self.family)
            self.assertEqual(raised.exception.code, E_REPLAY)
            self.assertEqual(raised.exception.revision, 1)
            self.assertIsNotNone(raised.exception.field_path)

            semantic_change = copy.deepcopy(session2)
            semantic_change["validator_version"] = 999
            semantic_change["content_digest"] = digest_without_field(semantic_change, "content_digest")
            with self.assertRaises(WorkbenchError) as raised:
                ledger.replay_session(semantic_change, self.family)
            self.assertEqual(raised.exception.code, E_REPLAY)
            self.assertEqual(raised.exception.revision, 1)
            self.assertEqual(raised.exception.field_path, "session.validator_version")

            changed_family = load_family(self.root / "family" / "family.json")
            changed_data = copy.deepcopy(changed_family.data)
            changed_data["inventory"]["certificate_fingerprints"] = []
            from tlstrust.certfamily import Family

            changed_family = Family(root=changed_family.root, data=changed_data)
            with self.assertRaises(WorkbenchError) as raised:
                ledger.replay_session(session2, changed_family)
            self.assertEqual(raised.exception.code, E_REPLAY)
            self.assertEqual(raised.exception.revision, 1)
            self.assertEqual(raised.exception.layer, 0)
            self.assertIn("input_snapshot.chain[0].fingerprint", raised.exception.field_path or "")

            digest_tampered = copy.deepcopy(session2)
            digest_tampered["trust_version"] += 100
            with self.assertRaises(WorkbenchError) as raised:
                ledger.validate_session_payload(digest_tampered)
            self.assertEqual(raised.exception.code, E_SNAPSHOT)
        finally:
            ledger.close()

    def test_incomplete_run_recovery(self) -> None:
        ledger_path = self.root / "crash.sqlite3"
        writer = Ledger(ledger_path, auto_recover=False)
        created = writer.create_trust_version(self.document, source_path=str(self.trust_path))
        run_id = writer.start_run(created["version"], self.snapshot, self.document.revision_numbers)
        writer.record_revision_result(run_id, self.results[0])
        writer.close()
        recovered = Ledger(ledger_path)
        try:
            self.assertEqual(len(recovered.startup_recovery["incomplete_runs"]), 1)
            run = recovered.get_run(run_id)
            self.assertEqual(run["status"], "INCOMPLETE")
            rows = [(item["revision"], item["conclusion"], item["committed"]) for item in run["results"]]
            self.assertEqual(rows[0], (1, "PASS", 1))
            self.assertTrue(all(row[1] == "UNKNOWN" and row[2] == 0 for row in rows[1:]))
            self.assertTrue(
                all(
                    item["content_digest"] != run_id
                    for item in recovered.list_session_snapshots(current_only=False)
                )
            )
        finally:
            recovered.close()

    def test_fault_injections(self) -> None:
        workspace = prepare_workspace(
            self.root / "fault-workspace",
            key_bits=1024,
            fresh=False,
        )
        chain_error = inject_chain(workspace)
        self.assertEqual(chain_error["code"], E_CHAIN)
        revoke_error = inject_revoke(workspace)
        self.assertEqual(revoke_error["code"], E_REVOKE)
        snapshot_error = inject_snapshot(workspace)
        self.assertEqual(snapshot_error["code"], E_SNAPSHOT)


class DemoTests(unittest.TestCase):
    def test_three_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary = run_demo(temp, fresh=True, key_bits=1024)
            self.assertEqual(
                [item["scenario"] for item in summary["scenarios"]],
                [
                    "initial_full_revision_validation",
                    "trust_library_rollback_and_revalidation",
                    "forced_termination_recovery_and_snapshot_replay",
                ],
            )
            scenario3 = summary["scenarios"][2]
            self.assertEqual(scenario3["crash_exit_code"], 90)
            self.assertEqual(scenario3["incomplete_run"]["status"], "INCOMPLETE")
            self.assertEqual(scenario3["incomplete_run"]["unknown_revisions"], [3])
            self.assertTrue(scenario3["replay"]["ok"])
            self.assertTrue(scenario3["historical_stale_snapshot_replay"]["result"]["ok"])
            from tlstrust.ui import render_text_view

            family = load_family(Path(temp) / "family" / "family.json")
            ledger = Ledger(Path(temp) / "ledger.sqlite3")
            try:
                view = ledger.build_current_view()
            finally:
                ledger.close()
            rendered = render_text_view(view, family)
            positions = [
                rendered.index("CERTIFICATE FAMILY"),
                rendered.index("REAL HANDSHAKE CHAINS"),
                rendered.index("REVISION RESULTS AND FLIP POINTS"),
                rendered.index("INDEX COMPARISON"),
                rendered.index("SESSION SNAPSHOTS AND REPLACEMENT AUDIT"),
            ]
            self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
