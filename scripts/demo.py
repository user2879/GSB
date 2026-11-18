"""Repeatable end-to-end demonstration with stable, ordered output.

Scenarios, in fixed order:
  1. cold start full pipeline
  2. determinism proof (second run, byte-identical summaries)
  3. evolution: normalization parameter change -> invalidation/replacement
  4. evolution: shard target size change
  5. sample evolution invalidates every stage
  6. forced termination -> UNKNOWN stages -> recovery + replay
  7. shard tamper -> E_SHARD via ledger replay
  8. three injections -> E_DATA / E_PARAM / E_OUTLIER
  9. headless UI view model dump
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from workbench.cli import main as cli_main  # noqa: E402
from workbench.demo_data import generate_fixtures  # noqa: E402
from workbench.errors import WorkbenchError  # noqa: E402
from workbench.ledger import Ledger  # noqa: E402
from workbench.report import render_report  # noqa: E402
from workbench.runner import Runner  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_cli(*args: str) -> int:
    print("$ python -m workbench.cli " + " ".join(args))
    return cli_main(list(args))


def main() -> int:
    demo_root = os.path.join(ROOT, "artifacts", "demo_output")
    workspace = os.path.join(demo_root, "workbench")
    fixture_root = os.path.join(demo_root, "fixtures")
    # Fresh deterministic workspace per demo invocation.
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)
    os.makedirs(fixture_root, exist_ok=True)
    spec_path = generate_fixtures(fixture_root)
    report_path = os.path.join(demo_root, "diagnostic_report.txt")

    section("1. COLD START FULL PIPELINE")
    runner = Runner(workspace)
    first = runner.run_pipeline(spec_path)
    first_report = render_report(runner.ledger, first["run_id"])
    print(first_report, end="")
    with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(first_report)

    section("2. DETERMINISM PROOF (SECOND RUN MUST BE BYTE-IDENTICAL)")
    second = runner.run_pipeline(spec_path)
    first_rows = runner.ledger.stage_rows(first["run_id"])
    second_rows = runner.ledger.stage_rows(second["run_id"])
    identical = all(
        a["output_digest"] == b["output_digest"] and
        a["column_stats_json"] == b["column_stats_json"] and
        a["extra_json"] == b["extra_json"]
        for a, b in zip(first_rows, second_rows))
    shards_a = runner.ledger.shard_rows(first["run_id"])
    shards_b = runner.ledger.shard_rows(second["run_id"])
    shard_identical = all(
        a["shard_fingerprint"] == b["shard_fingerprint"] and
        a["content_digest"] == b["content_digest"]
        for a, b in zip(shards_a, shards_b))
    print(f"stage summaries identical: {identical}")
    print(f"shard fingerprints identical: {shard_identical}")
    print(f"fingerprint versions: run1=v{first_rows[0] and 1} "
          f"run2=v{runner.ledger.get_run(second['run_id'])['fingerprint_version']}")

    section("3. EVOLUTION: NORMALIZATION PARAMETER scale 3 -> 4")
    variant_root = os.path.join(demo_root, "variants", "normalization")
    if os.path.isdir(variant_root):
        shutil.rmtree(variant_root)
    variant_spec = generate_fixtures(variant_root, normalize_scale=4)
    evolved = runner.run_pipeline(variant_spec)
    evolution = evolved["evolution"]
    print("reasons:", evolution["reasons"])
    print("changed stages:", evolution["changed_stage_ids"])
    print("affected stages:", evolution["affected_stage_ids"])
    print("preserved stages:", evolution["unchanged_stage_ids"])
    print("invalidated shards:", evolution["invalidated_shards"])
    for change in evolution["stage_changes"]:
        print(f"CHANGE {change['stage_id']} old={change['old_fingerprint'][:16]}"
              f" new={change['new_fingerprint'][:16]}")
    preserved_ok = all(
        next(c for c in runner.ledger.stage_rows(first["run_id"])
             if c["stage_id"] == sid)["output_digest"] ==
        next(c for c in runner.ledger.stage_rows(evolved["run_id"])
             if c["stage_id"] == sid)["output_digest"]
        for sid in evolution["unchanged_stage_ids"])
    print("preserved upstream digests unchanged:", preserved_ok)

    section("4. EVOLUTION: SHARD TARGET ROWS 4 -> 5")
    shard_root = os.path.join(demo_root, "variants", "shard")
    if os.path.isdir(shard_root):
        shutil.rmtree(shard_root)
    shard_spec = generate_fixtures(shard_root, target_rows=5)
    shard_run = runner.run_pipeline(shard_spec)
    shard_ev = shard_run["evolution"]
    print("reasons:", shard_ev["reasons"])
    print("affected:", shard_ev["affected_stage_ids"])
    print("invalidated shards:", shard_ev["invalidated_shards"])
    latest_shards = runner.ledger.shard_rows(shard_run["run_id"])
    print("new shard row counts:",
          [row["row_count"] for row in latest_shards])
    runner.close()

    section("5. SAMPLE EVOLUTION INVALIDATES EVERY STAGE")
    sample_root = os.path.join(demo_root, "variants", "sample")
    if os.path.isdir(sample_root):
        shutil.rmtree(sample_root)
    sample_spec_path = generate_fixtures(sample_root)
    with open(os.path.join(sample_root, "samples", "train_a.json"),
              "r", encoding="utf-8") as handle:
        sample_doc = json.load(handle)
    sample_doc["rows"][0]["visits"] = 99
    with open(os.path.join(sample_root, "samples", "train_a.json"),
              "w", encoding="utf-8", newline="\n") as handle:
        json.dump(sample_doc, handle, indent=2)
    runner = Runner(workspace)
    sample_run = runner.run_pipeline(sample_spec_path)
    sample_ev = sample_run["evolution"]
    print("sample_changed:", sample_ev["sample_changed"])
    print("affected stages:", sample_ev["affected_stage_ids"])
    print("reasons:", sample_ev["reasons"])
    runner.close()

    section("6. FORCED TERMINATION -> UNKNOWN -> RECOVERY + REPLAY")
    recovery_ws = os.path.join(demo_root, "recovery_workspace")
    if os.path.isdir(recovery_ws):
        shutil.rmtree(recovery_ws)
    runner = Runner(recovery_ws)
    killed = runner.run_pipeline(spec_path, kill_after_ordinal=2)
    print("killed run:", killed["run_id"], killed["status"])
    killed_rows = runner.ledger.stage_rows(killed["run_id"])
    print("post-kill stage statuses:",
          [(row["stage_id"], row["status"]) for row in killed_rows])
    runner.close()
    runner = Runner(recovery_ws)
    recovery = runner.recover(spec_path)
    print("recovery:", recovery)
    replay_result = runner.replay_run(recovery["new_run_id"])
    print("replay:", replay_result)
    runner.close()

    section("7. SHARD TAMPER -> E_SHARD ON LEDGER REPLAY")
    tamper_ws = os.path.join(demo_root, "tamper_workspace")
    if os.path.isdir(tamper_ws):
        shutil.rmtree(tamper_ws)
    runner = Runner(tamper_ws)
    tamper_run = runner.run_pipeline(spec_path)
    target_shard = runner.ledger.shard_rows(tamper_run["run_id"])[0]
    with open(target_shard["file_path"], "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    row = json.loads(lines[0])
    row["visits"] = row["visits"] + 1
    lines[0] = json.dumps(row, sort_keys=True,
                          separators=(",", ":")) + "\n"
    with open(target_shard["file_path"], "w", encoding="utf-8",
              newline="\n") as handle:
        handle.writelines(lines)
    try:
        runner.replay_run(tamper_run["run_id"])
        print("UNEXPECTED: replay passed")
    except WorkbenchError as error:
        print("business code:", error.code)
        print("stage:", error.stage_id, "field:", error.field_path)
        print("message:", error.message)
    runner.close()

    section("8. FAULT INJECTIONS (INPUT/PARAM ONLY) -> BUSINESS CODES")
    for kind in ("data", "param", "outlier"):
        inject_ws = os.path.join(demo_root, "inject_" + kind)
        if os.path.isdir(inject_ws):
            shutil.rmtree(inject_ws)
        code = run_cli("inject", spec_path, "--workspace", inject_ws,
                       "--kind", kind)
        print(f"injection={kind} exit_code={code}")

    section("9. HEADLESS UI VIEW MODEL DUMP")
    dump_path = os.path.join(demo_root, "ui_model.json")
    code = run_cli("ui-dump", "--workspace", workspace, "--out", dump_path)
    with open(dump_path, "r", encoding="utf-8") as handle:
        model = json.load(handle)
    print("ui-dump exit:", code)
    print("selected run:", model["selected_run"])
    print("stages rendered:", [s["stage_id"] for s in model["stages"]])
    print("shards rendered:", len(model["shards"]))
    print("fingerprint versions:",
          [fp["version"] for fp in model["fingerprints"]])

    section("DEMO COMPLETE")
    print("all scenarios executed; reports under", demo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
