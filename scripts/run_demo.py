"""End-to-end demo: deterministic fixed-point PEFT on synthetic local weights.

Steps:
  1. generate the sample directory and fault scenarios (if missing)
  2. full pipeline run (validate -> train -> merge -> diagnose -> export)
  3. rerun -> adapters reused (resume), byte-identical results
  4. base upgrade step 0 -> only adapters on the replaced layer retrain
  5. fault scenarios -> expected error codes
  6. incremental edit sequence vs fresh full rebuild -> field-identical
Writes demo_out/report.json and prints a verification summary.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from peft_fp.pipeline import apply_upgrade_step, run_pipeline  # noqa: E402

OUT = os.path.join(ROOT, "demo_out")
SAMPLES = os.path.join(ROOT, "samples")
FAULTS = os.path.join(ROOT, "faults")


def strip(report):
    r = copy.deepcopy(report)
    r.pop("global_version", None)
    r.pop("resume", None)
    return r


def strip_dispositions(report):
    """Dispositions (trained/reused/retrained) depend on store history; all
    substantive outputs must still be identical."""
    r = strip(report)
    for t in r["tasks"]:
        t.pop("disposition", None)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(SAMPLES):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_samples.py"), SAMPLES], check=True)
    if not os.path.isdir(FAULTS):
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "gen_faults.py"), SAMPLES, FAULTS],
            check=True,
        )

    work = os.path.join(OUT, "samples")
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(SAMPLES, work)
    store = os.path.join(OUT, "store")
    if os.path.isdir(store):
        shutil.rmtree(store)

    print("== 1. initial run ==")
    r1 = run_pipeline(work, store)
    for t in r1["tasks"]:
        print(f"  {t['name']:10s} {t['disposition']:9s} losses[0]={t['losses'][0]:.5f} "
              f"losses[-1]={t['losses'][-1]:.5f} merge={t['merge_status']}")
    print(f"  failures: {r1['failures']}")
    print(f"  merge budget consumed: {r1['merge']['consumed']} / {r1['merge']['budget']}")

    print("== 2. rerun (resume, reuse) ==")
    r2 = run_pipeline(work, store)
    identical = strip_dispositions(r1) == strip_dispositions(r2)
    print(f"  dispositions: {[t['disposition'] for t in r2['tasks']]}")
    print(f"  outputs identical to first run (modulo version/disposition): {identical}")

    print("== 3. base upgrade step 0 (single-layer replacement) ==")
    applied = apply_upgrade_step(work, 0)
    r3 = run_pipeline(work, store)
    disp = {t["name"]: t["disposition"] for t in r3["tasks"]}
    print(f"  replaced layers: {applied}")
    print(f"  dispositions: {disp}")
    mism = [f for f in r3["failures"] if f["code"] == "E_BASE_VERSION_MISMATCH"]
    print(f"  invalidated adapters: {sorted({f['task'] for f in mism})}")

    print("== 4. diagnostics (stage-wise error attribution) ==")
    for d in r3["diagnostics"]:
        flips = {f["sample_id"]: f["cause"] for f in d["flips"]}
        print(f"  {d['task']:10s} first_exceeding={d['first_exceeding_stage']} flips={flips}")
    # tight-tolerance pass to exhibit verdict flips and their attribution
    tight = os.path.join(OUT, "samples_tight")
    if os.path.isdir(tight):
        shutil.rmtree(tight)
    shutil.copytree(work, tight)
    import yaml

    cfg_path = os.path.join(tight, "peft_config.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    cfg["atol"] = 1e-4
    cfg["rtol"] = 1e-3
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=True)
    r_tight = run_pipeline(tight, os.path.join(OUT, "store_tight"))
    n_flips = sum(len(d["flips"]) for d in r_tight["diagnostics"])
    causes = sorted({f["cause"] for d in r_tight["diagnostics"] for f in d["flips"]})
    print(f"  tight-tolerance run: {n_flips} verdict flips, causes={causes}")

    print("== 5. fault scenarios ==")
    for scn in sorted(os.listdir(FAULTS)):
        r = run_pipeline(os.path.join(FAULTS, scn), os.path.join(OUT, f"store_{scn}"))
        codes = sorted({f["code"] for f in r["failures"]})
        print(f"  {scn:22s} -> {codes}")

    print("== 6. incremental edits vs full rebuild ==")
    tasks_path = os.path.join(work, "tasks.json")
    tasks = json.load(open(tasks_path))
    tasks["tasks"] = [t for t in tasks["tasks"] if t["name"] != "task_delta"]
    json.dump(tasks, open(tasks_path, "w"), indent=2, sort_keys=True)
    run_pipeline(work, store)
    r_inc = run_pipeline(work, store)
    full = os.path.join(OUT, "samples_full")
    if os.path.isdir(full):
        shutil.rmtree(full)
    shutil.copytree(work, full)
    r_full = run_pipeline(full, os.path.join(OUT, "store_full"))
    same = strip_dispositions(r_inc) == strip_dispositions(r_full)
    same_order = [t["name"] for t in r_inc["tasks"]] == [t["name"] for t in r_full["tasks"]]
    print(f"  incremental == full rebuild (modulo disposition): {same}, order equal: {same_order}")

    report_path = os.path.join(OUT, "report.json")
    run_pipeline(work, store, report_path)
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
