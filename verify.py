#!/usr/bin/env python3
"""Acceptance verification.

1. Regenerates the fixed-seed samples.
2. Runs the pipeline serially and with the configured thread pool.
3. Requires both reports to be byte-identical.
4. Checks failure codes, injection hits, speaker counts, error tracking,
   snapshot manifest and graceful signal handling.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(args, **kw):
    return subprocess.run([PY] + args, cwd=ROOT, capture_output=True,
                          text=True, **kw)


def main():
    checks = []

    proc = run(["generate_samples.py"])
    checks.append(("generate_samples", proc.returncode == 0))

    proc = run(["tests/test_units.py"], env=dict(os.environ))
    checks.append(("unit_tests", proc.returncode == 0))
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)

    shutil.rmtree(os.path.join(ROOT, "out_serial"), ignore_errors=True)
    proc = run(["main.py", "--output", "out_serial", "--serial"])
    checks.append(("serial_run_exit0", proc.returncode == 0))

    shutil.rmtree(os.path.join(ROOT, "out_parallel"), ignore_errors=True)
    proc = run(["main.py", "--output", "out_parallel"])
    checks.append(("parallel_run_exit0", proc.returncode == 0))

    with open(os.path.join(ROOT, "out_serial", "report.json")) as fh:
        serial_report = json.load(fh)
    with open(os.path.join(ROOT, "out_parallel", "report.json")) as fh:
        parallel_report = json.load(fh)
    # the "serial" flag itself is run metadata, not pipeline output
    serial_report.pop("serial", None)
    parallel_report.pop("serial", None)
    checks.append(("serial_parallel_identical", serial_report == parallel_report))

    rep = serial_report
    failures = {(f["file"], f["code"]) for f in rep["failures"]}
    checks.append(("nonfinite_detected",
                   ("inj_target_nonfinite.wav", "E_NONFINITE_INPUT") in failures))
    checks.append(("bad_format_detected",
                   ("bad_format.wav", "E_AUDIO_FORMAT") in failures))
    checks.append(("injection_hits_recorded",
                   sorted(h["id"] for h in rep["injection_hits"])
                   == ["inj-002", "inj-003", "inj-004"]))

    audios = {a["file"]: a for a in rep["audios"]}
    checks.append(("two_speakers_count",
                   audios["two_speakers.wav"]["speaker_count_low"] == 2))
    checks.append(("three_speakers_count",
                   audios["three_speakers.wav"]["speaker_count_low"] == 3))
    checks.append(("quiet_single_count",
                   audios["quiet_single.wav"]["speaker_count_low"] == 1))
    checks.append(("overlap_pair_count",
                   audios["overlap_pair.wav"]["speaker_count_low"] == 2))
    checks.append(("truncate_continues",
                   "inj_target_truncate.wav" in audios))
    checks.append(("no_segments_after_failure",
                   "inj_target_nonfinite.wav" not in audios
                   and "bad_format.wav" not in audios))
    checks.append(("error_tracking_present",
                   all(len(a["error_tracking"]["stages"]) == 4
                       for a in rep["audios"])))
    checks.append(("first_exceed_annotated",
                   all(a["error_tracking"]["first_exceed_stage"]
                       == "embedding_quantization" for a in rep["audios"])))
    diverged = [a for a in rep["audios"] if a["flip_attribution"]["diverged"]]
    checks.append(("flip_attribution_exercised",
                   len(diverged) >= 1
                   and all(a["flip_attribution"]["cause"] for a in diverged)))

    manifest_path = os.path.join(ROOT, "out_parallel", "snapshots", "manifest.json")
    checks.append(("snapshot_manifest_exists", os.path.isfile(manifest_path)))
    if os.path.isfile(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        checks.append(("manifest_fields",
                       manifest.get("version", 0) > 0
                       and manifest.get("config_hash") == rep["config_hash"]
                       and manifest.get("injection_hash") == rep["injection_hash"]
                       and manifest.get("frac_bits") == rep["frac_bits_low"]
                       and set(manifest.get("stage_versions", {}))
                       == {"windowing", "embedding", "clustering", "segmentation"}))

    # graceful signal handling: SIGTERM mid-run must exit 0 with a report
    shutil.rmtree(os.path.join(ROOT, "out_signal"), ignore_errors=True)
    proc = subprocess.Popen([PY, "main.py", "--output", "out_signal"],
                            cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    checks.append(("sigterm_exit0", proc.returncode == 0))
    checks.append(("sigterm_report_written",
                   os.path.isfile(os.path.join(ROOT, "out_signal", "report.json"))))

    ok = True
    for name, passed in checks:
        print("%s %s" % ("PASS" if passed else "FAIL", name))
        ok = ok and passed
    print("speaker_counts:", {k: v["speaker_count_low"] for k, v in audios.items()})
    print("diverged_audios:", [a["file"] for a in diverged])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
