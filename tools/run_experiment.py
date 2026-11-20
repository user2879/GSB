"""Repeatable end-to-end local experiment (no network, no external services).

Stages:
  1. Load + strict schema validation.
  2. Generate v1, v2 and direct-v3 bucket generations.
  3. Upgrade v1 -> v3 with checkpoints and read-only backups.
  4. Byte-identity equivalence: upgraded v3 vs direct v3.
  5. Full hash join (v3) + partition/global stats + per-row error attribution.
  6. Resume join reusing unaffected partitions, cross-check vs full run.
  7. Forced-kill during generation and during upgrade, then checkpoint resume.
  8. Unknown/partial bucket detection and forced rerun.
  9. Snapshot mismatch -> whole-state replacement recovery.
  10. Independent E_VERIFY recomputation for every generation/version.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj import bucket as bucket_mod
from dhj import engine
from dhj import join as join_mod
from dhj import verify as verify_mod
from dhj.errors import DHJError
from dhj.jsonio import write_json
from dhj.schema import query_fingerprint
from dhj.shuffle import bucket_path, generation_fingerprint
from tools.experiment_lib import (
    ROOT,
    assert_byte_identical,
    load_experiment_spec,
    paths,
    stats_calculator,
    write_ledger,
)


def reset_workspace(root):
    p = paths(root)
    for name in ("buckets", "snapshots", "logs"):
        if os.path.isdir(p[name]):
            shutil.rmtree(p[name])
        os.makedirs(p[name])
    os.makedirs(p["artifacts"], exist_ok=True)


def stage_generate(logger, spec, base, version, num_partitions,
                   fraction_bits, phase, callback=None):
    target = os.path.join(base, f"gen-v{version}")
    manifest, checkpoint = engine.generate_version(
        target,
        spec,
        version,
        num_partitions,
        fraction_bits,
        phase=phase,
        logger=logger,
        progress_callback=callback,
    )
    return target, manifest, checkpoint


def run(sample, query, num_partitions, fraction_bits, root=ROOT):
    p = paths(root)
    reset_workspace(root)
    logger = engine.make_logger(p["logs"], "experiment.jsonl")
    report = {"stages": [], "parameters": {
        "num_partitions": num_partitions,
        "fraction_bits": fraction_bits,
    }}

    spec = load_experiment_spec(sample, query)
    report["query_fingerprint"] = query_fingerprint(spec.query)
    report["source_digest"] = spec.source_digest
    logger.event("schema_loaded", query=spec.query.query_id)

    base = p["buckets"]
    v1_dir, v1_manifest, _ = stage_generate(
        logger, spec, base, 1, num_partitions, fraction_bits, "generate-v1"
    )
    v2_dir, _, _ = stage_generate(
        logger, spec, base, 2, num_partitions, fraction_bits, "generate-v2"
    )
    v3_dir, _, _ = stage_generate(
        logger, spec, base, 3, num_partitions, fraction_bits, "generate-v3"
    )
    report["stages"].append({
        "stage": "generation",
        "versions": [1, 2, 3],
        "v1_manifest": v1_manifest,
    })

    upgrade_dir = os.path.join(base, "upgrade-v3")
    upgrade_report, upgrade_checkpoint = engine.upgrade_generation(
        v1_dir,
        upgrade_dir,
        spec,
        num_partitions,
        fraction_bits,
        logger=logger,
    )
    mismatches = assert_byte_identical(
        upgrade_dir, v3_dir, num_partitions
    )
    assert not mismatches, mismatches
    report["stages"].append({
        "stage": "upgrade_v1_to_v3",
        "report": upgrade_report,
        "equivalence": "byte_identical",
        "backups_present": all(
            os.path.exists(
                bucket_mod.backup_path_for(
                    bucket_path(upgrade_dir, side, partition), 3
                )
            )
            for side in ("left", "right")
            for partition in range(num_partitions)
        ),
    })

    calc = stats_calculator(spec, fraction_bits)
    full = join_mod.run_join(
        v3_dir,
        os.path.join(base, "results-full"),
        spec,
        num_partitions,
        fraction_bits,
        calc,
        logger=logger,
    )
    full_ledger = write_ledger(
        os.path.join(base, "results-full"), full, spec,
        num_partitions, fraction_bits,
    )
    report["stages"].append({
        "stage": "full_join",
        "total_rows": full_ledger["total_row_count"],
        "partitions": [e["row_count"] for e in full_ledger["partitions"]],
    })
    return report, spec, logger, base, calc, full


def stage_resume_and_crash(spec, logger, base, calc, full,
                           num_partitions, fraction_bits, report):
    v3_dir = os.path.join(base, "gen-v3")
    resume_result_dir = os.path.join(base, "results-resume")
    reuse = {p: d for p, d in full["documents"].items()}
    resumed = join_mod.run_join(
        v3_dir,
        resume_result_dir,
        spec,
        num_partitions,
        fraction_bits,
        calc,
        reuse=reuse,
        logger=logger,
    )
    join_mod.cross_check_resume(full["documents"], resumed["documents"])
    resumed_ledger = write_ledger(
        resume_result_dir, resumed, spec, num_partitions, fraction_bits
    )
    full_bytes = {p: open(
        os.path.join(base, "results-full", f"join-p{p:05d}.json"), "rb"
    ).read() for p in full["documents"]}
    resume_bytes = {p: open(
        os.path.join(resume_result_dir, f"join-p{p:05d}.json"), "rb"
    ).read() for p in resumed["documents"]}
    assert full_bytes == resume_bytes, "resume output is not byte identical"
    report["stages"].append({
        "stage": "resume_join",
        "reused_partitions": resumed["reused_partitions"],
        "recomputed_partitions": resumed["recomputed_partitions"],
        "byte_identical": True,
        "total_rows": resumed_ledger["total_row_count"],
    })

    kill_dir = os.path.join(base, "gen-v3-kill")
    calls = {"n": 0}

    def kill_after_three(side, partition):
        calls["n"] += 1
        if calls["n"] == 4:
            raise engine.Terminate("simulated forced kill during generation")

    killed = False
    try:
        engine.generate_version(
            kill_dir, spec, 3, num_partitions, fraction_bits,
            phase="generate", logger=logger, progress_callback=kill_after_three,
        )
    except engine.Terminate:
        killed = True
    states_after_kill = engine.scan_bucket_states(
        kill_dir, num_partitions
    )
    unknown = [k for k, v in states_after_kill.items() if v == "unknown"]
    removed = engine.recover_bucket_directory(
        kill_dir, spec, 3, num_partitions, fraction_bits, logger=logger
    )
    manifest_restart, _ = engine.generate_version(
        kill_dir, spec, 3, num_partitions, fraction_bits,
        phase="generate", logger=logger,
    )
    assert generation_fingerprint(
        kill_dir, num_partitions
    ) == generation_fingerprint(v3_dir, num_partitions)
    report["stages"].append({
        "stage": "forced_kill_generation",
        "killed": killed,
        "unknown_after_kill": unknown,
        "reset_buckets": removed,
        "resumed_fingerprint_matches": True,
    })

    upgrade_kill_dir = os.path.join(base, "upgrade-v3-kill")
    os.makedirs(upgrade_kill_dir, exist_ok=True)
    import glob
    import shutil

    for path in glob.glob(os.path.join(
        os.path.join(base, "gen-v1"), "*.dhj"
    )):
        shutil.copy2(
            path, os.path.join(upgrade_kill_dir, os.path.basename(path))
        )
    calls_up = {"n": 0}

    def kill_upgrade(side, partition):
        calls_up["n"] += 1
        if calls_up["n"] == 3:
            raise engine.Terminate("simulated forced kill during upgrade")

    try:
        engine.upgrade_generation(
            os.path.join(base, "gen-v1"),
            upgrade_kill_dir,
            spec,
            num_partitions,
            fraction_bits,
            logger=logger,
            progress_callback=kill_upgrade,
        )
    except engine.Terminate:
        pass
    upgrade_report_restart, upgrade_cp_restart = engine.upgrade_generation(
        os.path.join(base, "gen-v1"),
        upgrade_kill_dir,
        spec,
        num_partitions,
        fraction_bits,
        logger=logger,
    )
    total_entries = len(upgrade_cp_restart["entries"])
    assert total_entries == 2 * num_partitions
    assert not assert_byte_identical(
        upgrade_kill_dir, v3_dir, num_partitions
    )
    report["stages"].append({
        "stage": "forced_kill_upgrade",
        "checkpoint_entries": total_entries,
        "restart_report_count": len(upgrade_report_restart),
        "byte_identical_after_resume": True,
    })

    partial_dir = os.path.join(base, "gen-v3-partial")
    engine.generate_version(
        partial_dir, spec, 3, num_partitions, fraction_bits, logger=logger
    )
    target = bucket_path(partial_dir, "left", 0)
    with open(target, "rb") as handle:
        good = handle.read()
    with open(target, "wb") as handle:
        handle.write(good[: len(good) // 2])
    states = engine.scan_bucket_states(partial_dir, num_partitions)
    assert states["left:0"] == "unknown"
    engine.recover_bucket_directory(
        partial_dir, spec, 3, num_partitions, fraction_bits, logger=logger
    )
    engine.generate_version(
        partial_dir, spec, 3, num_partitions, fraction_bits, logger=logger
    )
    assert generation_fingerprint(
        partial_dir, num_partitions
    ) == generation_fingerprint(v3_dir, num_partitions)
    report["stages"].append({
        "stage": "partial_bucket_recovery",
        "unknown_marked": "left:0",
        "rerun_fingerprint_matches": True,
    })


def stage_snapshot_and_verify(spec, logger, base, calc, full,
                              num_partitions, fraction_bits, report, p):
    v3_dir = os.path.join(base, "gen-v3")
    result_dir = os.path.join(base, "results-full")
    snapshot_dir = p["snapshots"]
    snapshot, state = engine.persist_snapshot(
        spec, snapshot_dir, v3_dir, result_dir,
        num_partitions, fraction_bits,
    )
    reopened = engine.open_or_replace_session(
        spec, snapshot_dir, v3_dir, result_dir,
        num_partitions, fraction_bits, logger=logger,
    )
    assert reopened["action"] == "resumed"

    sandbox = os.path.join(base, "snapshot-sandbox")
    sandbox_buckets = os.path.join(sandbox, "buckets")
    sandbox_results = os.path.join(sandbox, "results")
    sandbox_snapshots = os.path.join(sandbox, "snapshots")
    os.makedirs(sandbox, exist_ok=True)
    engine.generate_version(
        sandbox_buckets, spec, 3, num_partitions, fraction_bits, logger=logger
    )
    import shutil as _shutil

    _shutil.copytree(
        result_dir, sandbox_results,
        ignore=_shutil.ignore_patterns("checkpoint-*.json"),
    )
    sandbox_snapshot, _ = engine.persist_snapshot(
        spec, sandbox_snapshots, sandbox_buckets, sandbox_results,
        num_partitions, fraction_bits,
    )
    target = bucket_path(sandbox_buckets, "right", num_partitions - 1)
    alt_dir = os.path.join(sandbox, "alt-scale-buckets")
    engine.generate_version(
        alt_dir, spec, 3, num_partitions,
        max(1, fraction_bits - 1), logger=logger,
    )
    import shutil as _shutil2

    _shutil2.copy2(
        bucket_path(alt_dir, "right", num_partitions - 1), target
    )
    replaced = engine.open_or_replace_session(
        spec, sandbox_snapshots, sandbox_buckets, sandbox_results,
        num_partitions, fraction_bits, logger=logger,
    )
    assert replaced["action"] == "whole_state_replaced"
    assert os.path.isdir(sandbox_buckets + ".replaced")
    assert os.path.isfile(
        os.path.join(sandbox_snapshots + ".replaced", "snapshot.backup.json")
    ) or os.path.isdir(sandbox_snapshots + ".replaced")
    report["stages"].append({
        "stage": "snapshot_recovery",
        "content_digest": snapshot["content_digest"][:16],
        "clean_reopen": reopened["action"],
        "mismatch_replace": replaced["mismatches"],
        "read_only_backups": replaced["backups"],
    })

    verification = []
    for version, directory in (
        (3, v3_dir),
        (3, os.path.join(base, "upgrade-v3")),
    ):
        result = verify_mod.verify_all(
            spec, directory, result_dir, num_partitions,
            fraction_bits, calc,
        )
        verification.append({
            "directory": os.path.basename(directory),
            "version": version,
            "status": result["status"],
            "total_rows": result["total_rows"],
        })
    verify_mod.verify_bucket_assignment(
        spec, os.path.join(base, "gen-v1"), num_partitions, fraction_bits
    )
    verify_mod.verify_bucket_assignment(
        spec, os.path.join(base, "gen-v2"), num_partitions, fraction_bits
    )
    verification.append({
        "directory": "gen-v1",
        "version": 1,
        "status": "assignment_ok",
    })
    verification.append({
        "directory": "gen-v2",
        "version": 2,
        "status": "assignment_ok",
    })
    report["stages"].append({
        "stage": "independent_verification",
        "results": verification,
    })

    from dhj.stats import merge_partition_stats

    global_stats = merge_partition_stats(
        {pid: doc["stats"] for pid, doc in full["documents"].items()},
        spec.query.projections,
        spec.tables,
        fraction_bits,
    )
    report["global_stats"] = global_stats
    report["sample_error_attribution"] = [
        {
            "row_id": join_mod.output_row_id(row),
            "primary_error_source": row["primary_error_source"],
            "float_columns": {
                ref: {
                    "float_ref": detail["float_ref"],
                    "fixed_point_value": detail["fixed_point_value"],
                    "abs_error": detail["abs_error"],
                    "error_source": detail["error_source"],
                }
                for ref, detail in row["projection"].items()
                if "float_ref" in detail
            },
        }
        for row in list(full["documents"].values())[0]["rows"][:3]
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the DHJ experiment")
    parser.add_argument("--sample", default=os.path.join(ROOT, "data", "sample.json"))
    parser.add_argument("--query", default=None)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--fraction-bits", type=int, default=4)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    try:
        report, spec, logger, base, calc, full = run(
            args.sample,
            args.query,
            args.partitions,
            args.fraction_bits,
        )
        p = paths()
        stage_resume_and_crash(
            spec, logger, base, calc, full,
            args.partitions, args.fraction_bits, report,
        )
        stage_snapshot_and_verify(
            spec, logger, base, calc, full,
            args.partitions, args.fraction_bits, report, p,
        )
    except DHJError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 2
    report_path = args.report or os.path.join(
        paths()["artifacts"], "experiment-report.json"
    )
    write_json(report_path, report)
    print(json.dumps({"status": "ok", "report": report_path,
                      "stages": [s["stage"] for s in report["stages"]]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
