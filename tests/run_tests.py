#!/usr/bin/env python3
"""Acceptance tests for the video event boundary diagnostic workbench.

Covers: parallel/serial parity, budget exhaustion + resume, force-kill
recovery, graceful signal stop, decode-failure isolation, config-mismatch
refusal, and frame-block residency pressure determinism.
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
PY = os.path.join(ROOT, ".venv", "bin", "python")
ENTRY = os.path.join(ROOT, "run_workbench.py")
SAMPLES = os.path.join(ROOT, "samples")
CONFIGS = os.path.join(ROOT, "configs")

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def run_pipeline(samples, configs, workdir, concurrency=None, env=None):
    cmd = [PY, ENTRY, "run", "--samples", samples, "--configs", configs,
           "--workdir", workdir]
    if concurrency is not None:
        cmd += ["--max-concurrency", str(concurrency)]
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          timeout=300)


def load(workdir, name):
    with open(os.path.join(workdir, name), encoding="utf-8") as fh:
        return json.load(fh)


def write_schedule(configs_dir, **overrides):
    with open(os.path.join(CONFIGS, "schedule.json"), encoding="utf-8") as fh:
        schedule = json.load(fh)
    schedule.update(overrides)
    os.makedirs(configs_dir, exist_ok=True)
    shutil.copy(os.path.join(CONFIGS, "event_params.json"), configs_dir)
    with open(os.path.join(configs_dir, "schedule.json"), "w",
              encoding="utf-8") as fh:
        json.dump(schedule, fh, indent=2, sort_keys=True)


def main():
    tmp = tempfile.mkdtemp(prefix="wb_tests_")
    print(f"test workspace: {tmp}")

    print("== 1. parallel vs serial parity ==")
    serial = os.path.join(tmp, "serial")
    par4 = os.path.join(tmp, "par4")
    r1 = run_pipeline(SAMPLES, CONFIGS, serial, concurrency=1)
    r4 = run_pipeline(SAMPLES, CONFIGS, par4, concurrency=4)
    check("serial run exits 0", r1.returncode == 0, r1.stderr[-300:])
    check("parallel run exits 0", r4.returncode == 0, r4.stderr[-300:])
    seg_s = load(serial, "events.json")["segments"]
    seg_p = load(par4, "events.json")["segments"]
    check("segments field-identical across concurrency", seg_s == seg_p)
    trk_s = load(serial, "tracks.json")["tracks"]
    trk_p = load(par4, "tracks.json")["tracks"]
    check("tracks field-identical across concurrency", trk_s == trk_p)
    rep_s, rep_p = load(serial, "report.json"), load(par4, "report.json")
    check("quota deduction order identical",
          rep_s["budget"]["quota_deduction_order"]
          == rep_p["budget"]["quota_deduction_order"])
    check("parallel thread peak > 1", rep_p["thread_peak"] > 1,
          str(rep_p["thread_peak"]))

    print("== 2. budget exhaustion and resume ==")
    cfg_budget = os.path.join(tmp, "cfg_budget")
    write_schedule(cfg_budget, global_frame_budget=100)
    wd = os.path.join(tmp, "budget")
    rb1 = run_pipeline(SAMPLES, cfg_budget, wd)
    check("budget run 1 exits 0", rb1.returncode == 0, rb1.stderr[-300:])
    rep1 = load(wd, "report.json")
    check("run 1 consumed exactly 100 frames",
          rep1["budget"]["global_consumed_this_run"] == 100,
          str(rep1["budget"]["global_consumed_this_run"]))
    check("E_BUDGET_EXHAUSTED reported",
          "E_BUDGET_EXHAUSTED" in rep1["business_codes"])
    check("unfinished sequences have cursors",
          all(0 < s["processed_frames"] < 40
              for s in rep1["sequences"].values()))
    rb2 = run_pipeline(SAMPLES, cfg_budget, wd)
    check("budget run 2 (resume) exits 0", rb2.returncode == 0,
          rb2.stderr[-300:])
    rep2 = load(wd, "report.json")
    check("resume completed all frames",
          all(s["processed_frames"] == 40
              for s in rep2["sequences"].values()))
    check("resume recorded recovery point",
          rep2["recovered_from"] is not None
          and rep2["recovered_from"]["stage"] in
          ("evidence", "boundary", "trajectory", "summary"))
    seg_budget = load(wd, "events.json")["segments"]
    check("budget-resumed segments equal serial reference", seg_budget == seg_s)
    check("budget-resumed tracks equal serial reference",
          load(wd, "tracks.json")["tracks"] == trk_s)

    print("== 3. force-kill recovery ==")
    cfg_kill = os.path.join(tmp, "cfg_kill")
    write_schedule(cfg_kill, snapshot_commit_interval_frames=8)
    wd_kill = os.path.join(tmp, "kill")
    proc = subprocess.Popen(
        [PY, ENTRY, "run", "--samples", SAMPLES, "--configs", cfg_kill,
         "--workdir", wd_kill],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.6)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    check("killed mid-run", proc.returncode != 0, str(proc.returncode))
    manifest = os.path.join(wd_kill, "snapshots", "current", "manifest.json")
    check("snapshot survived SIGKILL", os.path.isfile(manifest))
    rk = run_pipeline(SAMPLES, cfg_kill, wd_kill)
    check("recovery run exits 0", rk.returncode == 0, rk.stderr[-300:])
    check("recovery printed resume point", "resume: recovered from" in rk.stdout,
          rk.stdout[-200:])
    repk = load(wd_kill, "report.json")
    check("recovery report has cursors",
          repk["recovered_from"] is not None
          and bool(repk["recovered_from"]["cursors"]))
    check("kill-recovered segments equal serial reference",
          load(wd_kill, "events.json")["segments"] == seg_s)
    check("kill-recovered tracks equal serial reference",
          load(wd_kill, "tracks.json")["tracks"] == trk_s)

    print("== 4. graceful signal stop ==")
    cfg_stop = os.path.join(tmp, "cfg_stop")
    write_schedule(cfg_stop, snapshot_commit_interval_frames=8)
    wd_stop = os.path.join(tmp, "stop")
    proc = subprocess.Popen(
        [PY, ENTRY, "run", "--samples", SAMPLES, "--configs", cfg_stop,
         "--workdir", wd_stop],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.4)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=60)
    check("SIGTERM run exits 0", proc.returncode == 0,
          f"rc={proc.returncode} {err[-200:]}")
    check("stop committed progress", "stop requested" in out, out[-200:])
    rs = run_pipeline(SAMPLES, cfg_stop, wd_stop)
    check("post-stop resume exits 0", rs.returncode == 0, rs.stderr[-300:])
    check("post-stop segments equal serial reference",
          load(wd_stop, "events.json")["segments"] == seg_s)

    print("== 5. decode failure isolation ==")
    bad_samples = os.path.join(tmp, "bad_samples")
    shutil.copytree(SAMPLES, bad_samples)
    with open(os.path.join(bad_samples, "seq_01", "frame_0005.png"),
              "wb") as fh:
        fh.write(b"not-a-png-at-all")
    wd_bad = os.path.join(tmp, "bad")
    rbad = run_pipeline(bad_samples, CONFIGS, wd_bad)
    check("decode-fail run exits 0", rbad.returncode == 0, rbad.stderr[-300:])
    repb = load(wd_bad, "report.json")
    check("E_DECODE_FAIL reported",
          "E_DECODE_FAIL" in repb["business_codes"])
    check("seq_01 excluded from boundary output",
          repb["sequences"]["seq_01"]["excluded"] is True
          and repb["sequences"]["seq_01"]["event_segments"] == 0)
    ref_ok = [s for s in seg_s if s["sequence"] == "seq_00"]
    bad_ok = [s for s in load(wd_bad, "events.json")["segments"]
              if s["sequence"] == "seq_00"]
    check("other sequences unaffected", ref_ok == bad_ok)

    print("== 6. config mismatch refusal ==")
    cfg_mm = os.path.join(tmp, "cfg_mm")
    write_schedule(cfg_mm)
    wd_mm = os.path.join(tmp, "mm")
    run_pipeline(SAMPLES, cfg_mm, wd_mm)
    with open(os.path.join(cfg_mm, "event_params.json"),
              encoding="utf-8") as fh:
        params = json.load(fh)
    params["min_event_duration_ms"] += 40
    with open(os.path.join(cfg_mm, "event_params.json"), "w",
              encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
    rmm = run_pipeline(SAMPLES, cfg_mm, wd_mm)
    check("config mismatch exits 2", rmm.returncode == 2,
          f"rc={rmm.returncode}")
    check("E_CONFIG_MISMATCH emitted", "E_CONFIG_MISMATCH" in rmm.stdout)
    check("mismatch recorded in failures.json",
          any(f["code"] == "E_CONFIG_MISMATCH"
              for f in load(wd_mm, "failures.json")["failures"]))

    print("== 7. frame-block residency pressure ==")
    cfg_cache = os.path.join(tmp, "cfg_cache")
    write_schedule(cfg_cache, frame_block_residency_budget=1)
    wd_cache = os.path.join(tmp, "cache")
    rc = run_pipeline(SAMPLES, cfg_cache, wd_cache)
    check("cache-pressure run exits 0", rc.returncode == 0, rc.stderr[-300:])
    repc = load(wd_cache, "report.json")
    check("evictions and recomputes occurred",
          repc["cache"]["evictions"] > 0 and repc["cache"]["recomputes"] > 0,
          str(repc["cache"]))
    check("no recompute mismatch failures",
          not any(f["exception"] == "RecomputeMismatch"
                  for f in repc["failures"]))
    check("cache-pressure segments equal serial reference",
          load(wd_cache, "events.json")["segments"] == seg_s
          if False else load(wd_cache, "events.json")["segments"] == seg_s)
    check("cache-pressure tracks equal serial reference",
          load(wd_cache, "tracks.json")["tracks"] == trk_s)

    print(f"\n{PASS} passed, {FAIL} failed")
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
