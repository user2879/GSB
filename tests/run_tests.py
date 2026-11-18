"""Deterministic test suite for the incremental embedding+gather component.

Runs with a plain python interpreter (no pytest required):
    python tests/run_tests.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from embedgather.config import load_config  # noqa: E402
from embedgather.errors import (E_EMBED_INVALID, E_ID_MISSING,  # noqa: E402
                                E_POINT_BUDGET_EXHAUSTED,
                                E_SNAPSHOT_MISMATCH, BusinessError)
from embedgather.fixedpoint import fixed_mean, quantize_scalar  # noqa: E402
from embedgather.gather import GatherRequest, execute_gather  # noqa: E402
from embedgather.injection import Injector  # noqa: E402
from embedgather.pipeline import Pipeline, load_requests  # noqa: E402
from embedgather.table import EmbeddingTable  # noqa: E402
import gen_data  # noqa: E402
from run_incremental import write_final_state  # noqa: E402

RESULT_FIELDS = ["status", "code", "acc", "contributions", "fingerprint",
                 "referenced", "mode", "quant_level", "frac_bits", "ids",
                 "points_used", "error_position"]


# ------------------------------------------------------------------ helpers
def make_workspace(tmp):
    gen_data.main(tmp)
    return load_config(os.path.join(tmp, "configs", "config.json"))


def write_config(tmp, **over):
    base = {
        "block_size": 4096, "rows_per_block": 2, "dim": 3,
        "quant_level": 65536, "frac_bits": 12, "agg_mode": "sum",
        "max_concurrency": 4, "global_point_budget": 1000,
        "per_request_quota": 100,
        "table_path": "table.jsonl", "requests_path": "requests.json",
        "snapshot_dir": "snapshots", "report_path": "report.json",
    }
    base.update(over)
    path = os.path.join(tmp, "config.json")
    with open(path, "w") as fh:
        json.dump(base, fh)
    return load_config(path)


def write_table(config, records):
    with open(config.table_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def write_requests(config, ids_list):
    with open(config.requests_path, "w") as fh:
        json.dump({"requests": [{"ids": ids} for ids in ids_list]}, fh)


def compare_results(inc_report, full_report):
    fi = {q["request_index"]: q for q in inc_report["requests"]}
    ff = {q["request_index"]: q for q in full_report["requests"]}
    assert fi.keys() == ff.keys(), "request index sets differ"
    for idx in sorted(fi):
        for field in RESULT_FIELDS:
            assert fi[idx][field] == ff[idx][field], \
                f"request {idx} field {field} differs"


# --------------------------------------------------------------------- tests
def test_fixedpoint():
    assert fixed_mean(5, 2) == 3
    assert fixed_mean(-5, 2) == -3
    assert fixed_mean(7, 3) == 2
    assert fixed_mean(-7, 3) == -2
    assert fixed_mean(8, 3) == 3
    assert fixed_mean(-8, 3) == -3
    assert quantize_scalar(0.5, 12, 65536) == 2048
    assert quantize_scalar(-0.25, 12, 65536) == -1024
    assert quantize_scalar(100.0, 12, 65536) == 32767      # clamp hi
    assert quantize_scalar(-100.0, 12, 65536) == -32768    # clamp lo


def test_table_validation():
    with tempfile.TemporaryDirectory() as tmp:
        config = write_config(tmp)
        write_requests(config, [[1]])
        write_table(config, [
            {"word_id": 1, "block_index": 0, "vector": [0.1, 0.2, 0.3]},
            {"word_id": 1, "block_index": 0, "vector": [0.1, 0.2, 0.3]},
            {"word_id": 2, "block_index": 0, "vector": [0.1, 0.2]},
            {"word_id": 3, "block_index": 0, "vector": [0.1, float("nan"), 0.3]},
            {"word_id": 4, "block_index": 2, "vector": [0.1, 0.2, 0.3]},
        ])
        table = EmbeddingTable.load(config)
        codes = {f["code"] for f in table.failure_list}
        assert codes == {E_EMBED_INVALID}
        reasons = " ".join(f["reason"] for f in table.failure_list)
        assert "duplicate word_id" in reasons
        assert "vector length" in reasons
        assert "non-finite" in reasons
        assert "not continuous" in reasons
        assert set(table.rows) == {1, 4}
        assert table.failed_word_ids == {1, 2, 3}


def test_gather_sum_mean_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        config = write_config(tmp)
        write_table(config, [
            {"word_id": 1, "block_index": 0, "vector": [0.5, -0.25, 1.0]},
            {"word_id": 2, "block_index": 0, "vector": [0.25, 0.25, -0.5]},
        ])
        table = EmbeddingTable.load(config)
        req = GatherRequest(0, [1, 2])
        res_sum = execute_gather(req, table, "sum", 65536, 12, 100)
        assert res_sum.acc == [3072, 0, 2048]
        assert res_sum.contributions == [[2048, -1024, 4096],
                                         [1024, 1024, -2048]]
        res_mean = execute_gather(req, table, "mean", 65536, 12, 100)
        assert res_mean.acc == [1536, 0, 1024]
        again = execute_gather(req, table, "sum", 65536, 12, 100)
        assert res_sum.to_dict() == again.to_dict()
        assert res_sum.fingerprint == again.fingerprint
        assert res_sum.fingerprint != res_mean.fingerprint


def test_missing_id_records_position():
    with tempfile.TemporaryDirectory() as tmp:
        config = write_config(tmp)
        write_table(config, [
            {"word_id": 1, "block_index": 0, "vector": [0.5, 0.5, 0.5]},
            {"word_id": 2, "block_index": 0, "vector": [0.5, 0.5, 0.5]},
        ])
        table = EmbeddingTable.load(config)
        res = execute_gather(GatherRequest(7, [1, 999]), table, "sum",
                             65536, 12, 100)
        assert res.code == E_ID_MISSING
        assert res.error_position == 1
        assert res.detail["request_index"] == 7
        assert res.detail["word_id"] == 999


def test_invalidation_precision():
    with tempfile.TemporaryDirectory() as tmp:
        config = make_workspace(tmp)
        requests = load_requests(config.requests_path)
        with open(os.path.join(tmp, "data", "edits.json")) as fh:
            commands = json.load(fh)["commands"]
        pipe = Pipeline(config)
        report = pipe.incremental_session(requests, commands)
        steps = report["steps"]
        # Step 0: replace_row(1003) hits only requests referencing 1003.
        expected = sorted(r.index for r in requests
                          if 1003 in r.ids and r.index != 10)
        assert steps[0]["invalidated"] == expected
        # Step 2: append_block retries only the id-missing request 10.
        assert steps[2]["invalidated"] == [10]
        # Steps 3/4: quant level / agg mode switch invalidates gather only.
        assert steps[3]["invalidated"] == list(range(len(requests)))
        assert steps[3]["stage_invalidations"] == {
            "validate": False, "fingerprint": False, "gather": True}
        assert steps[4]["stage_invalidations"] == {
            "validate": False, "fingerprint": False, "gather": True}
        # Step 5: replace_row(1026) hits only request 10.
        assert steps[5]["invalidated"] == [10]
        for step in steps:
            inv, reu = set(step["invalidated"]), set(step["reused"])
            assert not (inv & reu)
            assert inv | reu == set(range(len(requests)))


def test_snapshot_recovery_and_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        config = make_workspace(tmp)
        requests = load_requests(config.requests_path)
        pipe = Pipeline(config)
        pipe.full_run(requests)
        version = pipe.manifest["global_version"]
        # Same config: recovery hits and reports the version.
        pipe2 = Pipeline(config)
        pipe2.full_run(requests)
        assert pipe2.recovery_info["version"] == version
        # Quant level change: refuse to continue from old snapshot.
        cfg_path = os.path.join(tmp, "configs", "config_level.json")
        raw = json.load(open(os.path.join(tmp, "configs", "config.json")))
        raw["quant_level"] = 32768
        json.dump(raw, open(cfg_path, "w"))
        try:
            Pipeline(load_config(cfg_path)).full_run(requests)
            raise AssertionError("expected E_SNAPSHOT_MISMATCH")
        except BusinessError as exc:
            assert exc.code == E_SNAPSHOT_MISMATCH
            assert "quant_level" in exc.detail["mismatched_fields"]
        # Layout change: refuse as well.
        with open(config.table_path, "a") as fh:
            for i in range(config.rows_per_block):
                fh.write(json.dumps({
                    "word_id": 5000 + i, "block_index": 99,
                    "vector": [0.1] * config.dim}) + "\n")
        try:
            Pipeline(config).full_run(requests)
            raise AssertionError("expected E_SNAPSHOT_MISMATCH")
        except BusinessError as exc:
            assert exc.code == E_SNAPSHOT_MISMATCH
            assert "layout_fp" in exc.detail["mismatched_fields"]


def test_incremental_vs_full_equivalence():
    with tempfile.TemporaryDirectory() as tmp:
        config = make_workspace(tmp)
        requests = load_requests(config.requests_path)
        with open(os.path.join(tmp, "data", "edits.json")) as fh:
            commands = json.load(fh)["commands"]
        pipe = Pipeline(config)
        inc_report = pipe.incremental_session(requests, commands)
        # Fresh full rebuild from the final table + final config.
        final_dir = os.path.join(tmp, "final_state")
        final_cfg_path = write_final_state(pipe, final_dir)
        full_pipe = Pipeline(load_config(final_cfg_path))
        full_report = full_pipe.full_run(requests)
        compare_results(inc_report, full_report)
        # Final step invalidation list matches independent expectation.
        last = inc_report["steps"][-1]
        assert last["invalidated"] == [10]
        assert last["reused"] == [i for i in range(len(requests))
                                  if i != 10]


def test_budget_determinism_across_concurrency():
    with tempfile.TemporaryDirectory() as tmp:
        make_workspace(tmp)
        reports = []
        for conc in (1, 4):
            cfg_path = os.path.join(tmp, "configs", f"cfg_c{conc}.json")
            raw = json.load(open(os.path.join(tmp, "configs",
                                              "config.json")))
            raw["max_concurrency"] = conc
            raw["snapshot_dir"] = f"../snapshots_c{conc}"
            json.dump(raw, open(cfg_path, "w"))
            config = load_config(cfg_path)
            pipe = Pipeline(config)
            reports.append(pipe.full_run(load_requests(
                config.requests_path)))
        r1, r4 = reports
        for q1, q4 in zip(r1["requests"], r4["requests"]):
            for field in RESULT_FIELDS:
                assert q1[field] == q4[field]
        assert (r1["budget"]["deduction_sequence"]
                == r4["budget"]["deduction_sequence"])


def test_budget_exhaustion_and_pending():
    with tempfile.TemporaryDirectory() as tmp:
        config = make_workspace(tmp)
        cfg_path = os.path.join(tmp, "configs", "cfg_budget.json")
        raw = json.load(open(os.path.join(tmp, "configs", "config.json")))
        raw["per_request_quota"] = 2
        json.dump(raw, open(cfg_path, "w"))
        config = load_config(cfg_path)
        requests = load_requests(config.requests_path)
        pipe = Pipeline(config)
        report = pipe.full_run(requests)
        # Every request has >= 3 ids but quota is 2: all exhaust budget
        # except request 10 which fails earlier with E_ID_MISSING.
        for q in report["requests"]:
            if q["request_index"] == 10:
                assert q["code"] == E_ID_MISSING
            else:
                assert q["code"] == E_POINT_BUDGET_EXHAUSTED
                assert q["error_position"] == 2
        assert report["pending_requests"] == [
            i for i in range(len(requests)) if i != 10]


def test_fault_injection():
    with tempfile.TemporaryDirectory() as tmp:
        config = make_workspace(tmp)
        injector = Injector.load(os.path.join(tmp, "configs",
                                              "injection.json"))
        pipe = Pipeline(config, injector=injector)
        report = pipe.full_run(load_requests(config.requests_path))
        hits = {h["injection_id"]: h["business_code"]
                for h in report["injections"]}
        assert hits == {
            "INJ-001": E_EMBED_INVALID,
            "INJ-002": E_ID_MISSING,
            "INJ-003": E_SNAPSHOT_MISMATCH,
            "INJ-004": E_POINT_BUDGET_EXHAUSTED,
        }
        by_index = {q["request_index"]: q for q in report["requests"]}
        assert by_index[2]["code"] == E_ID_MISSING
        assert by_index[3]["code"] == E_POINT_BUDGET_EXHAUSTED
        assert by_index[4]["code"] == E_EMBED_INVALID
        # Single faults never interrupt the remaining requests.
        ok = [i for i, q in by_index.items()
              if q["status"] == "ok" or q["code"] == E_ID_MISSING]
        assert 0 in ok and 1 in ok and 5 in ok
        assert any(f["word_id"] == 1005 for f in report["failure_list"])
        assert set(report["business_codes"]) == {
            E_EMBED_INVALID, E_ID_MISSING,
            E_POINT_BUDGET_EXHAUSTED, E_SNAPSHOT_MISMATCH}


def test_observability_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        config = make_workspace(tmp)
        pipe = Pipeline(config)
        report = pipe.full_run(load_requests(config.requests_path))
        for q in report["requests"]:
            assert "max_abs_error" in q
            assert "deviation_positions" in q
            if q["status"] == "ok":
                assert q["max_abs_error"] is not None
                assert q["max_abs_error"] < 1e-3
        assert report["error_summary"]["max_abs_error_overall"] < 1e-3


def test_report_schema():
    with tempfile.TemporaryDirectory() as tmp:
        config = make_workspace(tmp)
        pipe = Pipeline(config)
        report = pipe.full_run(load_requests(config.requests_path))
        for key in ("run_kind", "snapshot_version", "recovery", "config",
                    "layout_fp", "requests", "invalidation", "budget",
                    "error_summary", "injections", "failure_list",
                    "pending_requests", "business_codes"):
            assert key in report, key
        inv = report["invalidation"]
        assert inv["invalidated_count"] == len(inv["invalidated"])
        assert inv["reused_count"] == len(inv["reused"])
        assert report["snapshot_version"] >= 3


def _signal_workspace(tmp):
    make_workspace(tmp)
    rng_values = [round(0.001 * i, 6) for i in range(1, 40)]
    commands = [{"op": "replace_row", "word_id": 1000 + (i % 24),
                 "vector": [v] * 8}
                for i, v in enumerate(rng_values)]
    edits_path = os.path.join(tmp, "data", "many_edits.json")
    with open(edits_path, "w") as fh:
        json.dump({"commands": commands}, fh)
    return edits_path


def test_graceful_exit_and_recovery():
    with tempfile.TemporaryDirectory() as tmp:
        edits_path = _signal_workspace(tmp)
        config_path = os.path.join(tmp, "configs", "config.json")
        env = dict(os.environ)
        cmd = [sys.executable,
               os.path.join(ROOT, "scripts", "run_incremental.py"),
               "--config", config_path, "--edits", edits_path,
               "--delay", "0.1"]
        # Handleable signal: must exit 0 with a progress record.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env)
        time.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, (proc.returncode, out, err)
        progress_path = os.path.join(tmp, "snapshots", "progress.json")
        assert os.path.exists(progress_path)
        progress = json.load(open(progress_path))
        assert progress["snapshot_version"] >= 3
        # Restart: recovers from committed snapshots, reports version.
        proc2 = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=120)
        assert proc2.returncode == 0, proc2.stderr
        report = json.load(open(os.path.join(tmp, "reports",
                                             "report.json")))
        assert report["recovery"] is not None
        assert report["recovery"]["version"] >= 3
        assert report["recovery"]["progress"] is not None


def test_forced_kill_restart():
    with tempfile.TemporaryDirectory() as tmp:
        edits_path = _signal_workspace(tmp)
        config_path = os.path.join(tmp, "configs", "config.json")
        cmd = [sys.executable,
               os.path.join(ROOT, "scripts", "run_incremental.py"),
               "--config", config_path, "--edits", edits_path,
               "--delay", "0.1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        time.sleep(1.0)
        proc.kill()  # SIGKILL: no cleanup possible
        proc.communicate(timeout=30)
        assert proc.returncode != 0
        # Restart must recover solely from fully replaced snapshots.
        proc2 = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=120)
        assert proc2.returncode == 0, proc2.stderr
        report = json.load(open(os.path.join(tmp, "reports",
                                             "report.json")))
        assert report["recovery"]["version"] >= 3


ALL_TESTS = [
    test_fixedpoint,
    test_table_validation,
    test_gather_sum_mean_deterministic,
    test_missing_id_records_position,
    test_invalidation_precision,
    test_snapshot_recovery_and_mismatch,
    test_incremental_vs_full_equivalence,
    test_budget_determinism_across_concurrency,
    test_budget_exhaustion_and_pending,
    test_fault_injection,
    test_observability_metrics,
    test_report_schema,
    test_graceful_exit_and_recovery,
    test_forced_kill_restart,
]


def main():
    failures = []
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(ALL_TESTS) - len(failures)}/{len(ALL_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
