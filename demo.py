#!/usr/bin/env python3
"""End-to-end acceptance demo.

1. generate local samples
2. uninterrupted run (snapshot + state log persisted)
3. log replay vs snapshot rebuild -> identical state digest
4. snapshot resume vs uninterrupted run -> bitwise identical
5. batch-size / out-of-order combinations -> identical results
6. four fault injections -> E_SNAPSHOT / E_LOG / E_BATCH / E_ORDER
7. standalone verifier cross-check
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, "sample", "tables.json")
QUERIES = os.path.join(ROOT, "sample", "queries.json")
RUNS = os.path.join(ROOT, "runs")


def step(title):
    print(f"\n=== {title}")


def cli(*argv, expect=0):
    proc = subprocess.run(
        [sys.executable, "-m", "aggwin.cli", *argv],
        cwd=ROOT, capture_output=True, text=True)
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() \
        else proc.stderr.strip()
    print(f"  {last}")
    assert proc.returncode == expect, (argv, proc.returncode, proc.stdout,
                                       proc.stderr)
    return json.loads(last)


def main():
    step("0. prepare samples")
    subprocess.run([sys.executable, "make_samples.py"], cwd=ROOT,
                   check=True)
    if os.path.isdir(RUNS):
        shutil.rmtree(RUNS)

    step("1. uninterrupted run (batch=7, snapshot at 50%)")
    full = cli("run", "--tables", TABLES, "--queries", QUERIES,
               "--workdir", os.path.join(RUNS, "full"))

    step("2. log replay vs snapshot rebuild (E_RECOVER path)")
    cli("recover-check", "--tables", TABLES, "--queries", QUERIES,
        "--workdir", os.path.join(RUNS, "full"),
        "--snapshot", os.path.join(RUNS, "full", "snapshots",
                                   "manifest-v2-final.json"))

    step("3. resume from mid snapshot (batch=5) -> bitwise identical")
    resumed = cli("run", "--tables", TABLES, "--queries", QUERIES,
                  "--workdir", os.path.join(RUNS, "resumed"),
                  "--resume-from", os.path.join(RUNS, "full", "snapshots",
                                                "manifest-v2-mid.json"),
                  "--batch-size", "5")
    assert resumed["state_digest"] == full["state_digest"]
    cli("diverge-check", "--tables", TABLES, "--queries", QUERIES,
        "--workdir-a", os.path.join(RUNS, "full"),
        "--workdir-b", os.path.join(RUNS, "resumed"))

    step("4. batch/shuffle combinations -> identical results")
    digests = set()
    for tag, extra in (("b1", ["--batch-size", "1"]),
                       ("b3s", ["--batch-size", "3", "--shuffle-window", "4",
                                "--shuffle-seed", "5"]),
                       ("b11s", ["--batch-size", "11", "--shuffle-window",
                                 "6", "--shuffle-seed", "9"])):
        out = cli("run", "--tables", TABLES, "--queries", QUERIES,
                  "--workdir", os.path.join(RUNS, f"combo-{tag}"), *extra)
        with open(os.path.join(RUNS, f"combo-{tag}", "results.json"),
                  encoding="utf-8") as fh:
            results = json.load(fh)
        digests.add(json.dumps({k: results[k] for k in
                                ("aggs", "windows")}, sort_keys=True))
    assert len(digests) == 1, "results differ across combinations"
    print("  results identical across 3 batch/shuffle combinations")

    step("5. fault injections")
    expectations = {"snapshot-digest": "E_SNAPSHOT",
                    "log-truncate": "E_LOG",
                    "batch-mid-window": "E_BATCH",
                    "out-of-order": "E_ORDER"}
    for name, code in expectations.items():
        out = cli("run", "--tables", TABLES, "--queries", QUERIES,
                  "--workdir", os.path.join(RUNS, f"inj-{name}"),
                  "--inject", name, expect=2)
        assert out["code"] == code, (name, out)
        print(f"  {name}: {out['code']} OK")

    step("6. standalone verifier")
    proc = subprocess.run(
        [sys.executable, "verify_tool.py",
         "--tables", TABLES, "--queries", QUERIES,
         "--snapshot", os.path.join(RUNS, "full", "snapshots",
                                    "manifest-v2-final.json"),
         "--statelog", os.path.join(RUNS, "full", "statelog"),
         "--catalog", os.path.join(RUNS, "full", "catalog.db"),
         "--results", os.path.join(RUNS, "full", "results.json")],
        cwd=ROOT, capture_output=True, text=True)
    print(proc.stdout.strip())
    assert proc.returncode == 0

    step("7. query parameter evolution -> stale snapshot replaced")
    evolved = os.path.join(ROOT, "sample", "queries_evolved.json")
    with open(QUERIES, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["session"]["scale"] = 100
    with open(evolved, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    out = cli("run", "--tables", TABLES, "--queries", evolved,
              "--workdir", os.path.join(RUNS, "evolved"),
              "--catalog", os.path.join(RUNS, "full", "catalog.db"),
              "--resume-from", os.path.join(RUNS, "full", "snapshots",
                                            "manifest-v2-mid.json"))
    print("  stale snapshot detected; state replaced wholesale")
    print("\nALL ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
