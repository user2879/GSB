"""全量测试：覆盖确定性身份、强校验、盖章幂等、演进失效、
账本替换、调度预算、崩溃恢复重放、异常码与独立核对。

运行：python -m unittest tests.test_all -v
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_stamper.app import build_identities, run_threaded_demo  # noqa: E402
from inference_stamper.errors import (  # noqa: E402
    AuditError,
    IdentityError,
    ReplayError,
    SnapshotError,
    SpecError,
)
from inference_stamper.ledger import (  # noqa: E402
    LEDGER_FORMAT_VERSION,
    open_ledger,
    tamper_content_digest,
    tamper_stamp_checksum,
)
from inference_stamper.parameters import (  # noqa: E402
    LAYER_SPECS,
    SUPPORTED_QUANTS,
    build_identity,
)
from inference_stamper.recovery import ForcedCrash, recover  # noqa: E402
from inference_stamper.replay import replay_ledger  # noqa: E402
from inference_stamper.scheduler import (  # noqa: E402
    Task,
    plan_virtual_schedule,
)
from inference_stamper.spec import (  # noqa: E402
    deterministic_input_vector,
    load_spec,
    write_sample_spec,
)
from inference_stamper.stamping import Stamper  # noqa: E402

SAMPLE = ROOT / "samples" / "spec.base.json"


class IdentityDeterminismTest(unittest.TestCase):
    def test_bit_identical_across_runs(self) -> None:
        spec = load_spec(SAMPLE)
        first = build_identities(spec)
        second = build_identities(spec)
        self.assertEqual(first.keys(), second.keys())
        for backend_id in first:
            a = first[backend_id]
            b = second[backend_id]
            self.assertEqual(a.identity_digest, b.identity_digest)
            for layer_a, layer_b in zip(a.layers, b.layers):
                self.assertEqual(layer_a.digest, layer_b.digest)
                self.assertEqual(layer_a.shape, layer_b.shape)

    def test_identity_changes_with_dim_seed_quant_version(self) -> None:
        base = build_identity("b", 8, 12, 1001, "fp32", 1)
        variants = [
            build_identity("b", 9, 12, 1001, "fp32", 1),
            build_identity("b", 8, 13, 1001, "fp32", 1),
            build_identity("b", 8, 12, 1002, "fp32", 1),
            build_identity("b", 8, 12, 1001, "fp16", 1),
            build_identity("b", 8, 12, 1001, "int8", 1),
            build_identity("b", 8, 12, 1001, "fp32", 2),
        ]
        for variant in variants:
            self.assertNotEqual(base.identity_digest,
                                variant.identity_digest)

    def test_all_quants_supported_and_layer_count_fixed(self) -> None:
        for quant in SUPPORTED_QUANTS:
            artifact = build_identity("b", 8, 12, 1001, quant, 1)
            self.assertEqual(len(artifact.layers), len(LAYER_SPECS))
            self.assertEqual(len(artifact.composition_order),
                             len(LAYER_SPECS))

    def test_deterministic_input_vectors(self) -> None:
        a = deterministic_input_vector("b", 1, 8, 42)
        b = deterministic_input_vector("b", 1, 8, 42)
        c = deterministic_input_vector("b", 2, 8, 42)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 8)


class SpecValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, payload: dict) -> Path:
        path = self.tmp_path / "spec.json"
        write_sample_spec(path, payload)
        return path

    def _valid(self) -> dict:
        return {
            "backends": [{
                "backend_id": "b1", "model_dim": 4, "hidden_dim": 6,
                "seed": 7, "quant": "fp32", "identity_version": 1,
            }],
            "requests": [{
                "token_id": "t1", "backend_id": "b1",
                "input_vector": [0.1, 0.2, 0.3, 0.4], "request_seq": 1,
            }],
        }

    def test_first_violation_reported_with_id_and_field(self) -> None:
        cases = [
            (lambda p: p["backends"][0].update(model_dim=0),
             "b1", "model_dim"),
            (lambda p: p["backends"][0].update(seed=1.5),
             "b1", "seed"),
            (lambda p: p["backends"][0].update(quant="fp99"),
             "b1", "quant"),
            (lambda p: p["requests"][0].update(backend_id="missing"),
             "t1", "backend_id"),
            (lambda p: p["requests"][0].update(input_vector=[1, 2]),
             "t1", "input_vector"),
        ]
        for mutate, expected_id, expected_field in cases:
            payload = self._valid()
            mutate(payload)
            with self.assertRaises(SpecError) as ctx:
                load_spec(self._write(payload))
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertEqual(ctx.exception.item_id, expected_id)
            self.assertEqual(ctx.exception.field, expected_field)

    def test_duplicate_ids_and_seq_rejected(self) -> None:
        payload = self._valid()
        payload["backends"].append(dict(payload["backends"][0]))
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(payload))
        self.assertEqual(ctx.exception.field, "backend_id")

        payload = self._valid()
        payload["requests"].append(dict(payload["requests"][0]))
        payload["requests"][1]["token_id"] = "t2"
        with self.assertRaises(SpecError) as ctx:
            load_spec(self._write(payload))
        self.assertEqual(ctx.exception.field, "request_seq")

    def test_broken_json_rejected(self) -> None:
        path = self.tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_spec(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "json")


class StampingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger, _ = open_ledger(Path(self.tmp.name) / "ledger.db")
        self.spec = load_spec(SAMPLE)
        self.identities = build_identities(self.spec)
        self.stamper = Stamper(self.ledger, self.identities)

    def tearDown(self) -> None:
        self.ledger.close()
        self.tmp.cleanup()

    def test_idempotent_stamp_and_monotonic_seq(self) -> None:
        first = self.stamper.stamp(self.spec.requests[0])
        second = self.stamper.stamp(self.spec.requests[0])
        self.assertTrue(second.reused)
        self.assertEqual(first.stamp_seq, second.stamp_seq)
        self.assertEqual(first.checksum, second.checksum)

        alpha_requests = [r for r in self.spec.requests
                          if r.backend_id == "backend-alpha"]
        seqs = [self.stamper.stamp(r).stamp_seq for r in alpha_requests]
        self.assertEqual(seqs, [1, 2, 3])
        beta_first = next(r for r in self.spec.requests
                          if r.backend_id == "backend-beta")
        self.assertEqual(self.stamper.stamp(beta_first).stamp_seq, 1)

    def test_verify_baseline_and_stale_evolution(self) -> None:
        for request in self.spec.requests:
            self.stamper.stamp(request)
        results = self.stamper.verify_all(self.spec)
        self.assertTrue(all(r.accepted and r.code == "OK" for r in results))

        evolved_identity = load_spec(
            ROOT / "samples" / "spec.evolved-identity.json")
        stale = Stamper(self.ledger,
                        build_identities(evolved_identity)).verify("token-001")
        self.assertEqual(stale.code, "E_STALE")
        self.assertEqual(stale.field, "identity_digest")
        self.assertFalse(stale.accepted)

        evolved_spec = load_spec(
            ROOT / "samples" / "spec.evolved-token-spec.json")
        stale_spec = Stamper(self.ledger,
                             build_identities(evolved_spec)).verify("token-002")
        self.assertEqual(stale_spec.code, "E_STALE")
        self.assertEqual(stale_spec.field, "token_spec_version")

        baseline = self.stamper.verify_all(self.spec)
        self.assertTrue(all(r.accepted for r in baseline))

    def test_tampered_checksum_identity_error(self) -> None:
        request = self.spec.requests[0]
        self.stamper.stamp(request)
        tamper_stamp_checksum(self.ledger.db_path, request.token_id)
        result = self.stamper.verify(request.token_id)
        self.assertEqual(result.code, "E_IDENTITY")
        self.assertEqual(result.field, "checksum")

    def test_missing_token_raises_identity_error(self) -> None:
        with self.assertRaises(IdentityError):
            self.stamper.verify("does-not-exist")


class LedgerReplacementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "ledger.db"
        self.spec = load_spec(SAMPLE)
        self.identities = build_identities(self.spec)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _stamp_everything(self) -> None:
        ledger, _ = open_ledger(self.db_path)
        stamper = Stamper(ledger, self.identities)
        for request in self.spec.requests:
            stamper.stamp(request)
            stamper.verify(request.token_id)
        ledger.close()

    def test_clean_reopen_keeps_rows(self) -> None:
        self._stamp_everything()
        reopened, report = open_ledger(self.db_path)
        self.assertIsNone(report)
        self.assertEqual(len(reopened.list_stamps()),
                         len(self.spec.requests))
        self.assertEqual(len(reopened.list_verifications()),
                         len(self.spec.requests))
        reopened.close()

    def test_digest_mismatch_raises_or_replaces_with_backup(self) -> None:
        self._stamp_everything()
        tamper_content_digest(self.db_path)
        with self.assertRaises(SnapshotError) as ctx:
            open_ledger(self.db_path, on_mismatch="raise")
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")

        reopened, report = open_ledger(self.db_path, on_mismatch="replace")
        self.assertIsNotNone(report)
        self.assertTrue(report.replaced)
        self.assertEqual(report.reason, "content-digest")
        self.assertEqual(len(report.stamp_token_ids),
                         len(self.spec.requests))
        self.assertEqual(reopened.list_stamps(), [])
        self.assertEqual(reopened.list_verifications(), [])
        backup = Path(report.backup_path)
        self.assertTrue(backup.exists())
        mode = backup.stat().st_mode & 0o777
        self.assertEqual(mode & 0o222, 0)
        with self.assertRaises(glob_permission_error()):
            backup.write_text("x", encoding="utf-8")
        reopened.close()

    def test_format_version_mismatch_replaces(self) -> None:
        self._stamp_everything()
        connection = sqlite3.connect(str(self.db_path))
        connection.execute(
            "UPDATE meta SET value = '999' WHERE key = 'format_version'")
        connection.commit()
        connection.close()
        reopened, report = open_ledger(self.db_path)
        self.assertIsNotNone(report)
        self.assertEqual(report.reason, "format-version")
        self.assertEqual(report.format_version, LEDGER_FORMAT_VERSION)
        reopened.close()


def glob_permission_error():
    return PermissionError


class SchedulerBudgetTest(unittest.TestCase):
    def _tasks(self) -> list[Task]:
        spec = load_spec(SAMPLE)
        tasks = []
        for order, request in enumerate(spec.requests):
            tasks.append(Task(
                task_id=request.token_id,
                backend_id=request.backend_id,
                kind="stamp",
                duration_ms=10 + order * 3,
                order=order,
            ))
        return tasks

    def test_virtual_schedule_respects_budgets_and_is_stable(self) -> None:
        tasks = self._tasks()
        first = plan_virtual_schedule(tasks, global_limit=2,
                                      per_backend_limit=1)
        second = plan_virtual_schedule(list(tasks), global_limit=2,
                                       per_backend_limit=1)
        self.assertEqual(first, second)
        self._assert_invariants(first, 2, 1)
        self.assertTrue(any(entry.wait_ms > 0 for entry in first))

    def test_lower_global_limit_changes_trace(self) -> None:
        tasks = self._tasks()
        tight = plan_virtual_schedule(tasks, 1, 1)
        loose = plan_virtual_schedule(tasks, 4, 4)
        self._assert_invariants(tight, 1, 1)
        self._assert_invariants(loose, 4, 4)
        self.assertGreater(
            max(entry.run_end_ms for entry in tight),
            max(entry.run_end_ms for entry in loose),
        )

    def test_threaded_runner_respects_budgets(self) -> None:
        spec = load_spec(SAMPLE)
        ledger, _ = open_ledger(":memory:")
        try:
            stamper = Stamper(ledger, build_identities(spec))
            trace = run_threaded_demo(stamper, spec, 20261123, 2, 1)
            self.assertEqual(len(trace), 2 * len(spec.requests))
            self._assert_invariants(trace, 2, 1)
        finally:
            ledger.close()

    def _assert_invariants(self, trace, global_limit: int,
                           per_backend_limit: int) -> None:
        events = []
        for entry in trace:
            events.append((entry.run_start_ms, 1, entry.backend_id))
            events.append((entry.run_end_ms, -1, entry.backend_id))
        events.sort(key=lambda event: (event[0], event[1]))
        active = 0
        backend_active: dict[str, int] = {}
        for _, delta, backend_id in events:
            active += delta
            backend_active[backend_id] = \
                backend_active.get(backend_id, 0) + delta
            self.assertLessEqual(active, global_limit)
            self.assertLessEqual(backend_active[backend_id],
                                 per_backend_limit)


class RecoveryReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "crash.db"
        self.spec = load_spec(SAMPLE)
        self.identities = build_identities(self.spec)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_crash_recovery_only_uses_committed_rows(self) -> None:
        from inference_stamper.app import stamp_all

        ledger, _ = open_ledger(self.db_path)
        stamper = Stamper(ledger, self.identities)
        crash_point = self.spec.requests[2].token_id
        with self.assertRaises(ForcedCrash):
            stamp_all(stamper, self.spec,
                      crash_after_token_id=crash_point)
        report = recover(ledger, self.spec)
        self.assertEqual(
            report.committed_token_ids,
            tuple(r.token_id for r in self.spec.requests[:3]),
        )
        self.assertEqual(
            report.unknown_token_ids,
            tuple(r.token_id for r in self.spec.requests[3:]),
        )
        items = replay_ledger(ledger, self.spec)
        self.assertEqual(len(items), 3)

        for request in self.spec.requests[3:]:
            stamper.stamp(request)
        report_after = recover(ledger, self.spec)
        self.assertEqual(report_after.unknown_token_ids, ())
        self.assertEqual(len(replay_ledger(ledger, self.spec)),
                         len(self.spec.requests))
        ledger.close()

    def test_replay_detects_first_mismatch(self) -> None:
        ledger, _ = open_ledger(self.db_path)
        stamper = Stamper(ledger, self.identities)
        for request in self.spec.requests:
            stamper.stamp(request)
        tamper_stamp_checksum(self.db_path,
                              self.spec.requests[1].token_id)
        with self.assertRaises(ReplayError) as ctx:
            replay_ledger(ledger, self.spec)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.item_id,
                         self.spec.requests[1].token_id)
        self.assertEqual(ctx.exception.field, "checksum")
        ledger.close()


class AuditToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "audit.db"
        self.spec = load_spec(SAMPLE)
        ledger, _ = open_ledger(self.db_path)
        stamper = Stamper(ledger, build_identities(self.spec))
        for request in self.spec.requests:
            stamper.stamp(request)
            stamper.verify(request.token_id)
        ledger.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_audit_passes_and_detects_tamper(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import audit as audit_module

        self.assertEqual(
            audit_module.audit(str(SAMPLE), str(self.db_path)), 0)
        tamper_stamp_checksum(self.db_path,
                              self.spec.requests[0].token_id)
        # 篡改后内容摘要也不再匹配，先触发 E_SNAPSHOT 包装为 E_AUDIT
        with self.assertRaises(AuditError) as ctx:
            audit_module.audit(str(SAMPLE), str(self.db_path))
        self.assertEqual(ctx.exception.code, "E_AUDIT")

    def test_audit_detects_content_level_mismatch_with_valid_digest(self) \
            -> None:
        """内容被改且摘要被同步伪造时，核对仍须定位 E_AUDIT。"""
        sys.path.insert(0, str(ROOT / "scripts"))
        import audit as audit_module
        from inference_stamper.ledger import compute_content_digest

        target = self.spec.requests[2].token_id
        connection = sqlite3.connect(str(self.db_path))
        connection.execute(
            "UPDATE verifications SET accepted=0 WHERE token_id=?",
            (target,))
        stamps = connection.execute(
            "SELECT token_id, backend_id, identity_digest, identity_version,"
            " token_spec_version, request_seq, stamp_seq, checksum,"
            " stamped_at_ms FROM stamps ORDER BY token_id").fetchall()
        verifications = connection.execute(
            "SELECT token_id, accepted, code, reason, field, verified_at_ms"
            " FROM verifications ORDER BY token_id").fetchall()
        forged = compute_content_digest(list(stamps),
                                        list(verifications))
        connection.execute(
            "UPDATE meta SET value=? WHERE key='content_digest'",
            (forged,))
        connection.commit()
        connection.close()

        with self.assertRaises(AuditError) as ctx:
            audit_module.audit(str(SAMPLE), str(self.db_path))
        self.assertEqual(ctx.exception.code, "E_AUDIT")
        self.assertEqual(ctx.exception.item_id, target)
        self.assertEqual(ctx.exception.field, "verification.accepted")
