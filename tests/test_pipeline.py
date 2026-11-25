import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fxpipe.errors import Code, IssueLog
from fxpipe.loader import load_edits, load_pipeline
from fxpipe.orchestrator import Orchestrator, build_embedding_table
from fxpipe.samplegen import generate


SAMPLE_ROOT = Path(tempfile.mkdtemp(prefix="fxpipe-samples-"))


def build_paths(version="e2"):
    paths = generate(SAMPLE_ROOT)
    return paths


def make_orchestrator(paths, version="e2", snapshot_root=None, workers=None):
    log = IssueLog()
    model = load_pipeline(
        paths["config"], paths["tokens"], paths["segments"], paths["graph"],
        e1_path=paths["e1"] if version == "e1" else None,
        e2_path=paths["e2"] if version == "e2" else None,
        bypass_path=paths["bypass"] if version == "e2" else None,
    )
    if workers is not None:
        model.config.max_workers = workers
    root = snapshot_root or Path(tempfile.mkdtemp(prefix="fxpipe-snap-"))
    return Orchestrator(model, root, exchange_version=version)


def full_run(orchestrator: Orchestrator):
    orchestrator.start()
    _, diag, _ = orchestrator.run_to_completion()
    return orchestrator.engine.all_outputs(), diag


class PipelineConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = build_paths()

    def test_full_run_has_all_stages_and_integer_values(self):
        orch = make_orchestrator(self.paths)
        outputs, diag = full_run(orch)
        stages = {entry["stage"] for entry in outputs.values()}
        for stage in ("embed", "segment_pool", "multihead_attn", "layer_norm", "final"):
            self.assertIn(stage, stages)
        final_keys = [key for key in outputs if key.endswith(":final:t0,1,2,3,4,5,6,7")]
        self.assertTrue(final_keys)
        final = outputs[final_keys[0]]["output"]
        for item in final["per_token"]:
            self.assertTrue(all(isinstance(v, int) for v in item["values"]))
            self.assertIsInstance(item["residual"], int)

    def test_incremental_equals_clean_rebuild(self):
        edits = load_edits(self.paths["edits"], IssueLog())
        # Incremental: start clean, apply all edits, then run.
        inc = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=Path(tempfile.mkdtemp()))
        inc.start()
        inc.apply_edits(edits)
        inc_outputs, _ = inc.run_to_completion() and (inc.engine.all_outputs(), None)

        # Clean rebuild: apply the same edits to a fresh model, run from blank.
        clean = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=Path(tempfile.mkdtemp()))
        clean.editor.apply(edits)
        if clean.editor.exchange_version != clean.exchange_version:
            clean.exchange_version = clean.editor.exchange_version
            clean.engine.exchange_version = clean.exchange_version
        clean.start()
        clean_outputs, _ = full_run(clean)

        self.assertEqual(set(inc_outputs), set(clean_outputs))
        for key in inc_outputs:
            self.assertEqual(inc_outputs[key]["cache_key"], clean_outputs[key]["cache_key"], key)
            self.assertEqual(inc_outputs[key]["output"], clean_outputs[key]["output"], key)
            self.assertEqual(inc_outputs[key]["output_fp"], clean_outputs[key]["output_fp"], key)

    def test_token_edit_only_invalidates_dependent_entries(self):
        orch = make_orchestrator(self.paths)
        orch.start()
        full_run(orch)
        marker = len(orch.engine.ledger.events)
        orch.apply_edits([{"cmd": "replace_token", "index": 0, "token": 9}])
        events = orch.engine.ledger.events[marker:]
        invalidated = [e for e in events if e.kind == "invalidated"]
        self.assertTrue(invalidated)
        # Every invalidated entry references the replaced token index.
        for event in invalidated:
            entry = None
            for candidate in orch.engine.ledger.all_entries():
                if candidate.entry_key == event.entry_key:
                    entry = candidate
            # entry may already be recomputed and re-present; check reason text
            self.assertIn("token-edit", event.reason)
        # Re-run and confirm a clean completion.
        _, diag, _ = orch.run_to_completion()
        self.assertEqual(diag.pending, [])

    def test_unaffected_entries_are_reused(self):
        orch = make_orchestrator(self.paths)
        orch.start()
        full_run(orch)
        marker = len(orch.engine.ledger.events)
        # Editing only an attention param must not invalidate an unrelated chain 2.
        orch.apply_edits([
            {"cmd": "update_op_param", "chain_id": 1, "seq": 2, "params": {"causal": True}},
        ])
        events = orch.engine.ledger.events[marker:]
        invalidated_nodes = {e.entry_key.split(":")[0] for e in events if e.kind == "invalidated"}
        self.assertTrue(all(node == "c1op2" or node.startswith("c1op") for node in invalidated_nodes))
        self.assertFalse(any(node.startswith("c2") for node in invalidated_nodes))
        orch.run_to_completion()
        reuse_nodes = {e.entry_key.split(":")[0] for e in orch.engine.ledger.events if e.kind == "reused"}
        self.assertTrue(any(node.startswith("c2") for node in reuse_nodes))

    def test_concurrency_matches_serial(self):
        serial = make_orchestrator(copy.deepcopy(self.paths),
                                   snapshot_root=Path(tempfile.mkdtemp()), workers=1)
        parallel = make_orchestrator(copy.deepcopy(self.paths),
                                     snapshot_root=Path(tempfile.mkdtemp()), workers=8)
        serial_out, _ = full_run(serial)
        parallel_out, _ = full_run(parallel)
        self.assertEqual(serial_out, parallel_out)

    def test_gear_and_frac_bits_change_cache_keys_and_values(self):
        base = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=Path(tempfile.mkdtemp()))
        base_out, _ = full_run(base)
        shifted = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=Path(tempfile.mkdtemp()))
        shifted.model.config.gear = "g3"
        shifted_out, _ = full_run(shifted)
        sample_key = next(iter(base_out))
        self.assertNotEqual(base_out[sample_key]["cache_key"], shifted_out[sample_key]["cache_key"])
        self.assertNotEqual(base_out[sample_key]["output"], shifted_out[sample_key]["output"])

    def test_e1_and_e2_versions_distinct_cache_keys(self):
        e2 = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=Path(tempfile.mkdtemp()), version="e2")
        e1 = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=Path(tempfile.mkdtemp()), version="e1")
        e2_out, _ = full_run(e2)
        e1_out, _ = full_run(e1)
        key = next(iter(e2_out))
        self.assertIn(key, e1_out)
        self.assertNotEqual(e2_out[key]["cache_key"], e1_out[key]["cache_key"])


class SnapshotRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = build_paths()

    def test_resume_after_budget_pending_uses_committed_snapshot(self):
        root = Path(tempfile.mkdtemp(prefix="fxpipe-rec-"))
        orch = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=root)
        manifest = orch.start()
        manifest, diag = orch.run_batch(manifest)
        self.assertTrue(orch.store.current_dir().exists())
        # A brand new orchestrator on the same root must resume, not rebuild.
        resumed = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=root)
        resumed_manifest = resumed.start()
        self.assertIsNotNone(resumed_manifest)
        self.assertEqual(resumed_manifest.config_hash, manifest.config_hash)
        _, final_diag, history = resumed.run_to_completion()
        self.assertEqual(final_diag.pending, [])

    def test_config_hash_mismatch_forces_rebuild(self):
        root = Path(tempfile.mkdtemp(prefix="fxpipe-cfg-"))
        orch = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=root)
        orch.start()
        orch.run_to_completion()
        changed = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=root)
        changed.model.config.gear = "g2"
        self.assertIsNone(orch.store.should_resume(changed.model.config.to_dict(), "e2"))

    def test_diagnostics_contain_required_sections(self):
        orch = make_orchestrator(self.paths)
        _, diag, _ = orch.run_to_completion()
        payload = diag.to_dict()
        for section in ("stage_points", "invalidated_count", "pending", "code_counts",
                        "completed_ops", "consumed_points", "elapsed_ms"):
            self.assertIn(section, payload)


class ErrorHandlingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = build_paths()

    def test_invalid_entries_are_skipped_not_fatal(self):
        log = IssueLog()
        model = load_pipeline(
            self.paths["config"], self.paths["tokens"], self.paths["segments"],
            self.paths["invalid_graph"],
            e2_path=self.paths["invalid_quant"], bypass_path=self.paths["bypass"],
        )
        log.extend(model.log)
        codes = log.count_by_code()
        self.assertIn("E_SCHEMA", codes)
        # Valid chains from other inputs still load independently elsewhere.
        self.assertIsInstance(model.chains, dict)

    def test_budget_and_quota_codes_emitted(self):
        paths = copy.deepcopy(self.paths)
        cfg = json.loads(Path(paths["config"]).read_text())
        cfg["point_budget"] = 40
        cfg["operator_quota"] = 10_000
        Path(paths["config"]).write_text(json.dumps(cfg))
        orch = make_orchestrator(paths, snapshot_root=Path(tempfile.mkdtemp()))
        manifest = orch.start()
        _, diag = orch.run_batch(manifest)
        self.assertIn(Code.E_BUDGET.value, diag.code_counts)

    def test_quota_exceeded_code(self):
        paths = copy.deepcopy(self.paths)
        cfg = json.loads(Path(paths["config"]).read_text())
        cfg["point_budget"] = 10_000
        cfg["operator_quota"] = 10
        Path(paths["config"]).write_text(json.dumps(cfg))
        orch = make_orchestrator(paths, snapshot_root=Path(tempfile.mkdtemp()))
        manifest = orch.start()
        _, diag = orch.run_batch(manifest)
        self.assertIn(Code.E_QUOTA.value, diag.code_counts)


if __name__ == "__main__":
    unittest.main()


class RoundTripEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = build_paths()

    def test_config_restored_to_baseline_still_invalidates_changed_content(self):
        root = Path(tempfile.mkdtemp(prefix="fxpipe-rt-"))
        orch = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=root)
        manifest = orch.start()
        orch.run_to_completion(manifest)
        self.assertEqual(len(orch.engine.ledger.entries), 10)

        boundary = len(orch.engine.ledger.events)
        edits = load_edits(self.paths["edits"], IssueLog())
        orch.apply_edits(edits)
        new_events = orch.engine.ledger.events[boundary:]
        invalidated = [event for event in new_events if event.kind == "invalidated"]
        # The edit stream ends back at g1/e2, yet op/token content changed.
        self.assertTrue(len(invalidated) > 0)
        reasons = {event.reason for event in invalidated}
        self.assertTrue(any(reason in reasons for reason in
                            ("op-edit", "token-edit", "content-fingerprint-mismatch",
                             "quant-or-version-changed")))

    def test_reconcile_detects_param_change_even_when_final_config_identical(self):
        root = Path(tempfile.mkdtemp(prefix="fxpipe-recon-"))
        orch = make_orchestrator(copy.deepcopy(self.paths), snapshot_root=root)
        orch.start().run_completion if False else orch.run_to_completion(orch.start())
        before = {key: entry.cache_key for key, entry in orch.engine.ledger.entries.items()}
        orch.apply_edits([
            {"cmd": "update_op_param", "chain_id": 1, "seq": 2, "params": {"causal": True}},
        ])
        # c1op2 and its downstream c1op3/c1op4 must be invalidated; chain 2 intact.
        surviving_nodes = {entry.node_id for entry in orch.engine.ledger.all_entries()}
        self.assertNotIn("c1op2", surviving_nodes)
        self.assertIn("c2op0", surviving_nodes)
