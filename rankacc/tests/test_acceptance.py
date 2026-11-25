"""Acceptance tests: round-trip restore, boundary decisions, incremental vs
full rebuild identity, concurrency identity, budget and fault handling."""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rankacc.core import REF_EXTRA, config_from_dict, half_quantum_ref, validate_request
from rankacc.formats import export_v1, export_v2, parse_v1_line, v1_lines_to_v2
from rankacc.pipeline import compute_all
from rankacc.pipeline import attribute_request
from rankacc.runner import Runner
from rankacc.samplegen import gen_config, gen_edits, gen_faults, gen_requests
from rankacc.store import Store, canon, fp


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rankacc-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def store(self, name):
        return os.path.join(self.tmp, name)


class TestBoundary(Base):
    def test_boundary_marks_and_diffs(self):
        cfg = config_from_dict({
            "topk": 2, "sample_ratio_permille": 500, "quant_tier": "Q12",
            "norm_method": "minmax", "channel_weights": {"news": 1.0},
            "batch_budget": 100, "per_request_quota": 100})
        req_raw = {"request_id": "r1", "candidates": [
            {"candidate_id": i, "score": float(i), "aux": 1, "channel": "news"}
            for i in (1, 2, 3, 4)]}
        req = validate_request(req_raw, cfg)
        out = compute_all(req, cfg, cfg.frac, fp(cfg.channel_weights))
        trunc = out["trunc"]
        self.assertEqual(trunc["selected"], [4, 3])
        half = half_quantum_ref()
        self.assertEqual(half, 1 << (REF_EXTRA - 1))
        # cutoff is the last selected candidate (id 3 at 2/3); only it sits
        # within half a quantum of itself
        self.assertEqual(trunc["boundary"],
                         [{"candidate_id": 3, "diff_ref": 0, "tier": "Q12"}])
        req_raw2 = {"request_id": "r1", "candidates": [
            {"candidate_id": 1, "score": 0.0, "aux": 1, "channel": "news"},
            {"candidate_id": 2, "score": 1.0, "aux": 1, "channel": "news"},
            {"candidate_id": 3, "score": 1.0, "aux": 1, "channel": "news"},
            {"candidate_id": 4, "score": 2.0, "aux": 1, "channel": "news"}]}
        req2 = validate_request(req_raw2, cfg)
        trunc2 = compute_all(req2, cfg, cfg.frac, fp(cfg.channel_weights))["trunc"]
        self.assertEqual(trunc2["selected"], [4, 2])  # id asc breaks the tie
        marked = {b["candidate_id"] for b in trunc2["boundary"]}
        self.assertEqual(marked, {3, 2})
        for b in trunc2["boundary"]:
            self.assertEqual(b["diff_ref"], 0)
            self.assertEqual(b["tier"], "Q12")


class TestRoundTrip(Base):
    def _conclusions(self):
        cfg = config_from_dict(gen_config())
        reqs = gen_requests()[:3]
        config_fp = fp(cfg.channel_weights)
        out = {}
        for raw in reqs:
            req = validate_request(raw, cfg)
            res = compute_all(req, cfg, cfg.frac, config_fp)
            out[req.request_id] = {
                "request_id": req.request_id, "quant_tier": cfg.quant_tier,
                "scores": {str(k): v for k, v in res["scores"].items()},
                "truncated": res["trunc"]["selected"],
                "truncation_order": res["trunc"]["order"],
                "boundary": res["trunc"]["boundary"],
                "layers": res["sample"]["layers"],
                "total_quota": res["sample"]["total_quota"],
                "version": "v2"}
        return out

    def test_v2_v1_v2_restores_boundary_from_sidecar(self):
        conclusions = self._conclusions()
        ledger = []
        v2_records = {rid: export_v2(c) for rid, c in conclusions.items()}
        lines, sidecar = [], {}
        for rid in sorted(conclusions):
            new_lines, sc = export_v1(conclusions[rid])
            lines.extend(new_lines)
            sidecar[rid] = sc
            ledger.append({"type": "downgrade", "direction": "v2->v1",
                           "request_id": rid})
        parsed = parse_v1_line(lines[0])
        self.assertEqual(set(parsed), {"request_id", "ordinal", "candidate_id",
                                       "score_permille", "selection_code"})
        restored = {r["request_id"]: r
                    for r in v1_lines_to_v2(lines, sidecar, ledger)}
        n_boundary = 0
        for rid, rec in restored.items():
            want = {c["candidate_id"]: c for c in v2_records[rid]["candidates"]}
            for got in rec["candidates"]:
                w = want[got["candidate_id"]]
                self.assertEqual(got["boundary"], w["boundary"])
                self.assertEqual(got["boundary_diff_ref"], w["boundary_diff_ref"])
                self.assertEqual(got["score"], w["score"])
                self.assertEqual(got["selection"], w["selection"])
                n_boundary += 1 if got["boundary"] else 0
        self.assertGreater(n_boundary, 0)
        self.assertTrue(any(e["type"] == "downgrade" for e in ledger))
        folded_losses = 0
        for line in lines:
            p = parse_v1_line(line)
            rec = restored[p["request_id"]]
            exact = rec["candidates"][p["ordinal"]]["score"]
            if p["score_permille"] / 1000.0 != exact:
                folded_losses += 1
        self.assertGreater(folded_losses, 0)  # permille folding really loses bits


class TestAttribution(Base):
    def test_boundary_flip_classified_as_norm_quantization(self):
        # Q8: half-quantum window is 0.5 main units; at frac+1 it is 0.25.
        # A candidate 0.0015 below the cutoff flips boundary membership.
        cfg = config_from_dict({
            "topk": 1, "sample_ratio_permille": 500, "quant_tier": "Q8",
            "norm_method": "minmax", "channel_weights": {"news": 1.0},
            "batch_budget": 100, "per_request_quota": 100})
        req = validate_request({"request_id": "r1", "candidates": [
            {"candidate_id": 1, "score": 0.0, "aux": 1, "channel": "news"},
            {"candidate_id": 2, "score": 0.9985, "aux": 1, "channel": "news"},
            {"candidate_id": 3, "score": 1.0, "aux": 1, "channel": "news"}]}, cfg)
        rep = attribute_request(req, cfg, fp(cfg.channel_weights))
        self.assertEqual(rep["boundary_flips"], [2])
        self.assertEqual(rep["flip_causes"], {"2": "norm_quantization"})
        self.assertEqual(rep["first_stage_exceeding"], "truncate")
        self.assertFalse(rep["norm_exceeds_tolerance"])


class TestIncremental(Base):
    def test_edits_then_full_rebuild_identical(self):
        runner_a = Runner(self.store("inc"))
        runner_a.set_inputs(gen_requests(), gen_config())
        runner_a.full_run(note="base")
        for op in gen_edits():
            runner_a.apply_edit(op)
        final = runner_a.store.state
        runner_b = Runner(self.store("full"))
        runner_b.set_inputs(final["requests"], final["config"])
        runner_b.full_run(note="full")
        self.assertEqual(canon(runner_a.store.outputs), canon(runner_b.store.outputs))
        self.assertEqual(canon(runner_a.store.report["cache_keys"]),
                         canon(runner_b.store.report["cache_keys"]))
        reuse_map = runner_a.store.report["reuse_map"]
        reused = sum(1 for v in reuse_map.values()
                     for s in list(v["norm"].values()) + [v["trunc"]]
                     + list(v["layer"].values()) + [v["ser"]] if s == "reuse")
        self.assertGreater(reused, 0)
        arbitrations = runner_a.store.report["arbitrations"]
        self.assertTrue(arbitrations)
        self.assertTrue(all(a["winner"] == "v2" for a in arbitrations))

    def test_layer_level_invalidation_on_candidate_replace(self):
        reqs = [{"request_id": "r1", "candidates": [
            {"candidate_id": i, "score": float(i), "aux": i % 3,
             "channel": "news" if i <= 5 else "ads"} for i in range(1, 11)]}]
        cfg = gen_config()
        cfg["topk"] = 8
        runner = Runner(self.store("layer"))
        runner.set_inputs(reqs, cfg)
        runner.full_run(note="base")
        runner.apply_edit({"op": "replace_candidate", "request_id": "r1",
                           "candidate": {"candidate_id": 2, "score": 99.5,
                                         "aux": 1, "channel": "news"}})
        reuse = runner.store.report["reuse_map"]["r1"]
        self.assertEqual(reuse["norm"]["ads"], "reuse")
        self.assertEqual(reuse["norm"]["news"], "recompute")

    def test_concurrency_identical_to_serial(self):
        cfg = gen_config()
        outs = []
        for threads in (1, 4):
            c = dict(cfg)
            c["threads"] = threads
            runner = Runner(self.store("t%d" % threads))
            runner.set_inputs(gen_requests(), c)
            runner.full_run(note="threads")
            outs.append(canon(runner.store.outputs))
        self.assertEqual(outs[0], outs[1])


class TestBudgetAndFaults(Base):
    def test_budget_pending_then_dedicated_batch(self):
        runner = Runner(self.store("budget"))
        runner.set_inputs(gen_requests(), gen_config())
        runner.full_run(note="budget")
        report = runner.store.report
        pending = [e for e in report["budget_events"]
                   if e["code"] == "E_POINT_BUDGET_PENDING"]
        self.assertEqual([e["request_id"] for e in pending], ["req0007"])
        self.assertIn("req0007", runner.store.outputs["conclusions"])
        self.assertGreater(len(report["batches"]), 1)
        env = report["environment"]
        self.assertTrue(env["threads_ok"] and env["mem_ok"])

    def test_fault_injection_isolated(self):
        runner = Runner(self.store("faults"))
        runner.set_inputs(gen_requests(), gen_config())
        runner.full_run(faults=gen_faults(), note="faults")
        failures = runner.store.outputs["failures"]
        by_req = {f["request_id"]: f["code"] for f in failures}
        self.assertEqual(by_req["req0003"], "E_BAD_SCORE")
        self.assertEqual(by_req["req0005"], "E_UNKNOWN_CHANNEL")
        self.assertIn("req0001", runner.store.outputs["conclusions"])
        self.assertIn("req0002", runner.store.outputs["conclusions"])

    def test_crash_leaves_last_commit_and_recovers(self):
        clean = Runner(self.store("clean"))
        clean.set_inputs(gen_requests(), gen_config())
        clean.full_run(note="clean")
        victim_dir = self.store("victim")
        victim = Runner(victim_dir)
        victim.set_inputs(gen_requests(), gen_config())
        victim.full_run(note="base")
        ledger_before = victim.store.read_ledger()
        victim2 = Runner(victim_dir)
        victim2.set_inputs(gen_requests(), gen_config())
        with self.assertRaises(Exception):
            victim2.full_run(faults=[{"request_id": "req0004",
                                      "stage": "sample", "fault": "crash"}],
                             note="crash")
        self.assertEqual(Store(victim_dir).read_ledger(), ledger_before)
        recovered = Runner(victim_dir)
        recovered.set_inputs(gen_requests(), gen_config())
        recovered.full_run(note="recovered")
        self.assertEqual(canon(recovered.store.outputs), canon(clean.store.outputs))


if __name__ == "__main__":
    unittest.main()
