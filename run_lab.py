"""一键可复现实验：确定性运行、再分发一致性、崩溃恢复、异常注入。"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(ROOT, "samples", "skew_sample.json")


def run(args, expect=0):
    env = dict(os.environ, PYTHONHASHSEED="42")
    proc = subprocess.run(
        [sys.executable, "-m", "hashlot.cli", *args],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    status = "OK" if proc.returncode == expect else "FAIL"
    print(f"[{status}] {' '.join(args)} -> exit {proc.returncode} (expect {expect})")
    if proc.returncode != expect:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(1)
    return proc


def main():
    shutil.rmtree(os.path.join(ROOT, "out", "lab"), ignore_errors=True)
    if not os.path.exists(SAMPLE):
        run(["--sample", SAMPLE, "--out-dir", "out/init", "validate"], 0)
    base = ["--sample", SAMPLE, "--out-dir", "out/lab", "--virtual-clock"]
    run(base + ["validate"], 0)
    run(base + ["run"], 0)
    run(base + ["replay"], 0)
    crash_dir = os.path.join(ROOT, "out", "lab_crash")
    shutil.rmtree(crash_dir, ignore_errors=True)
    crash_args = ["--sample", SAMPLE, "--out-dir", "out/lab_crash",
                  "--virtual-clock", "--crash-after-batch", "5", "run"]
    run(crash_args, 7)
    run(["--sample", SAMPLE, "--out-dir", "out/lab_crash",
         "--virtual-clock", "recover"], 0)
    for flag, code in (("--inject-borrow-cycle", "E_BORROW"),
                       ("--inject-early-deadline", "E_DEADLINE")):
        out = f"out/inj_{code.lower()}"
        shutil.rmtree(os.path.join(ROOT, out), ignore_errors=True)
        proc = run(["--sample", SAMPLE, "--out-dir", out,
                    "--virtual-clock", flag, "run"], 3)
        assert f'"{code}"' in proc.stdout
    digest_out = "out/inj_e_digest"
    shutil.rmtree(os.path.join(ROOT, digest_out), ignore_errors=True)
    run(["--sample", SAMPLE, "--out-dir", digest_out,
         "--virtual-clock", "--inject-digest-mismatch", "run"], 0)
    proc = run(["--sample", SAMPLE, "--out-dir", digest_out,
                "--virtual-clock", "--inject-digest-mismatch", "replay"], 3)
    assert '"E_DIGEST"' in proc.stdout
    report = json.load(open(os.path.join(ROOT, "out", "lab", "report.json"),
                            encoding="utf-8"))
    print("\n=== 实验摘要 ===")
    print("result_digest:", report["result_digest"])
    print("hot_partitions:", report["mapping_summary"]["hot_partitions"])
    print("peak_active:", report["watermark"]["peak_active_connections"])
    print("borrowed_connections:",
          sum(1 for c in report["watermark"]["connections"] if c["is_borrowed"]))
    print("all lab checks passed")


if __name__ == "__main__":
    main()
