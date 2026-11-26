#!/usr/bin/env python3
"""Run the local end-to-end acceptance record for this delivery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsb.cli import collect_doctor
from gsb.codec import encode_block
from gsb.hashing import canonical_bytes
from gsb.ledger import Ledger, batch_tx_digest
from gsb.pipeline import recover_pipeline, run_pipeline
from gsb.sample import build_changed_sample_spec, build_sample_spec
from gsb.snapshot import open_or_replace_snapshot, write_snapshot
from gsb.spec import load_spec


CRASH_CHILD = """
import sys
from gsb.pipeline import run_pipeline
run_pipeline(spec_path=sys.argv[1], output_dir=sys.argv[2], session_id='acceptance-crash', crash_after_commit=1)
"""


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shuffled_spec() -> dict:
    spec = build_sample_spec()
    spec["block_groups"] = list(reversed(spec["block_groups"]))
    for query in spec["queries"]:
        query["predicates"] = list(reversed(query["predicates"]))
    return spec


def main() -> int:
    started = time.perf_counter()
    evidence_dir = ROOT / "artifacts" / "verification"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="acceptance-run-", dir=evidence_dir))
    checks: dict[str, bool] = {}
    record: dict = {"ok": False, "run_dir": str(run_dir), "checks": checks}

    doctor = collect_doctor(ROOT)
    write_json(evidence_dir / "doctor.json", doctor)
    checks["doctor_ok"] = doctor["ok"]
    record["doctor"] = doctor["checks"]

    test_cmd = [sys.executable, "-m", "unittest", "tests.test_spec_codec_planner", "tests.test_pipeline_runtime", "-v"]
    test_proc = subprocess.run(test_cmd, cwd=ROOT, text=True, capture_output=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    test_output = test_proc.stdout + "\n" + test_proc.stderr
    (evidence_dir / "test-results.txt").write_text(test_output, encoding="utf-8")
    checks["twenty_unittest_cases_pass"] = test_proc.returncode == 0 and "Ran 20 tests" in test_output and "OK" in test_output

    baseline_dir = run_dir / "baseline"
    baseline = run_pipeline(spec=build_sample_spec(), output_dir=baseline_dir, session_id="acceptance")
    baseline_cache = json.loads((baseline_dir / "cache.json").read_text(encoding="utf-8"))
    checks["faulted_sample_false_skip"] = baseline["summary"]["false_skips"] == 1
    checks["full_verification"] = (
        baseline["verification"]["mode"] == "full"
        and baseline["verification"]["actual_decoded_block_groups"] == 4
        and baseline["verification"]["actual_decoded_rows"] == 44
    )
    checks["index_distortion_count"] = baseline["verification"]["index_distorted_block_groups"] == 1
    checks["lease_limits_observed"] = (
        baseline["execution"]["lease_peaks"]["peak_global_in_flight"] == 3
        and max(baseline["execution"]["lease_peaks"]["peak_per_column"].values()) == 2
    )
    ports = [receipt["worker_port"] for receipt in baseline["execution"]["worker_receipts"].values()]
    checks["unique_tcp_ports"] = len(set(ports)) == 4
    checks["baseline_recovery_consistent"] = baseline["ledger"]["recovery"]["status"] == "consistent"

    reordered_dir = run_dir / "reordered"
    run_pipeline(spec=shuffled_spec(), output_dir=reordered_dir, session_id="acceptance")
    reordered_cache = json.loads((reordered_dir / "cache.json").read_text(encoding="utf-8"))
    checks["reordered_plan_digests_identical"] = reordered_cache["query_digests"] == baseline_cache["query_digests"]

    crash_spec_path = run_dir / "spec.json"
    crash_spec_path.write_text(json.dumps(build_sample_spec()), encoding="utf-8")
    crash_dir = run_dir / "crash"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    killed = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD, str(crash_spec_path), str(crash_dir)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    checks["crash_exit_sigkill"] = killed.returncode == -9
    after_kill = recover_pipeline(output_dir=crash_dir, session_id="acceptance-crash", spec_path=crash_spec_path)
    checks["crash_committed_one_unknown_three"] = (
        after_kill["recovery"]["unknown_batches"] == ["BG-A02", "BG-A03", "BG-A04"]
        and len(after_kill["recovery"]["committed_batches_replayed"]) == 1
    )
    resumed = run_pipeline(spec_path=crash_spec_path, output_dir=crash_dir, session_id="acceptance-crash")
    resume_cache = json.loads((crash_dir / "cache.json").read_text(encoding="utf-8"))
    checks["crash_resume_committed_missing_only"] = resumed["execution"]["decode_jobs_this_run"] == ["BG-A02", "BG-A03", "BG-A04"]
    checks["crash_resume_consistent"] = resumed["ledger"]["recovery"]["status"] == "consistent"
    checks["crash_query_digests_identical"] = resume_cache["query_digests"] == baseline_cache["query_digests"]

    invalidation_dir = run_dir / "invalidation"
    run_pipeline(spec=build_sample_spec(), output_dir=invalidation_dir, session_id="invalidation")
    old_cache = json.loads((invalidation_dir / "cache.json").read_text(encoding="utf-8"))
    unaffected_before = canonical_bytes(old_cache["query_payloads"]["q_id_values"])
    changed = run_pipeline(spec=build_changed_sample_spec(), output_dir=invalidation_dir, session_id="invalidation")
    new_cache = json.loads((invalidation_dir / "cache.json").read_text(encoding="utf-8"))
    checks["incremental_only_invalidated_group"] = changed["execution"]["decode_jobs_this_run"] == ["BG-A02"]
    checks["incremental_invalidation_has_predicate_propagation"] = changed["invalidation"]["invalidated_queries"] == [
        "q_amount_high",
        "q_null_amount",
        "q_region_3",
    ]
    checks["incremental_unaffected_query_bytes_identical"] = (
        canonical_bytes(new_cache["query_payloads"]["q_id_values"]) == unaffected_before
    )

    snapshot_path = run_dir / "snapshot.json"
    write_snapshot(snapshot_path, {"stale": True, "snapshot_version": 1, "spec_digest": "right"})
    bad_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    bad_snapshot["spec_digest"] = "wrong"
    snapshot_path.write_text(json.dumps(bad_snapshot), encoding="utf-8")
    _, snapshot_info = open_or_replace_snapshot(
        snapshot_path,
        session_id="s",
        snapshot_version=1,
        spec_digest="right",
        factory=lambda: {"fresh": True},
    )
    checks["snapshot_mismatch_wholesale_replacement"] = snapshot_info["replaced"] and snapshot_info["reason"] == "spec_digest"

    replay_dir = run_dir / "replay-corruption"
    run_pipeline(spec=build_sample_spec(), output_dir=replay_dir, session_id="replay")
    ledger = Ledger(replay_dir / "ledger.sqlite3")
    corrupt_row = dict(ledger.connection.execute("SELECT * FROM batches WHERE block_group_id='BG-A02'").fetchone())
    corrupt_row["content_digest"] = "f" * 64
    corrupt_tx = batch_tx_digest(corrupt_row)
    with ledger.connection:
        ledger.connection.execute(
            "UPDATE batches SET content_digest=?, tx_digest=? WHERE session_id=? AND sequence=?",
            (corrupt_row["content_digest"], corrupt_tx, corrupt_row["session_id"], corrupt_row["sequence"]),
        )
    ledger.close()
    spec = load_spec(build_sample_spec())
    groups = {group["id"]: group for group in spec["block_groups"]}

    def verify_batch(raw, payload):
        rebuilt = encode_block(groups[raw["block_group_id"]], payload["block"]["columns"])
        if rebuilt["content_digest"] != raw["content_digest"]:
            return {"field": "content_digest", "expected": rebuilt["content_digest"], "actual": raw["content_digest"]}
        return None

    ledger = Ledger(replay_dir / "ledger.sqlite3")
    corruption_recovery = ledger.recover("replay", [q["id"] for q in spec["queries"]], verify_batch=verify_batch)
    ledger.close()
    first_problem = corruption_recovery["first_inconsistency"] or {}
    checks["recovery_locates_first_inconsistency"] = (
        corruption_recovery["status"] == "inconsistent"
        and first_problem.get("block_group_id") == "BG-A02"
        and first_problem.get("field") == "content_digest"
    )

    record.update(
        ok=all(checks.values()),
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        baseline={
            "spec_digest": baseline["spec_digest"],
            "query_digests": baseline_cache["query_digests"],
            "diagnostic_digest": baseline["diagnostic_digest"],
            "verification": {k: v for k, v in baseline["verification"].items() if k != "block_groups"},
            "issue": baseline["issues"][0],
            "attribution_dominant": baseline["attribution"][0]["dominant_source"],
        },
        crash={
            "committed_before_kill": 1,
            "unknown_batches": after_kill["recovery"]["unknown_batches"],
            "resume_new_batches": len(resumed["execution"]["decode_jobs_this_run"]),
            "final_replayed_batches": len(resumed["ledger"]["recovery"]["committed_batches_replayed"]),
            "final_replayed_queries": len(resumed["ledger"]["recovery"]["query_results_replayed"]),
        },
        incremental={
            "decode_jobs": changed["execution"]["decode_jobs_this_run"],
            "invalidated_queries": changed["invalidation"]["invalidated_queries"],
            "unaffected_queries": changed["invalidation"]["unaffected_queries"],
        },
        first_inconsistency=first_problem,
        artifacts={"test_results": str(evidence_dir / "test-results.txt"), "doctor": str(evidence_dir / "doctor.json")},
    )
    write_json(evidence_dir / "acceptance.json", record)
    write_json(ROOT / "artifacts" / "acceptance-local.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
