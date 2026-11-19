"""Reproducible acceptance runner: executes the full experiment matrix and
emits a single structured JSON summary plus writes a Markdown report.

Usage:
    python -m bench.acceptance --workdir run/acceptance
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli(workdir: str, *args: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "bench.mw.cli", "--workdir", workdir, *args],
        cwd=ROOT, capture_output=True, text=True, check=False)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return proc.returncode, payload


def run(workdir: str) -> dict:
    os.makedirs(workdir, exist_ok=True)
    basic_spec = os.path.join(ROOT, "samples", "basic", "spec.json")
    basic_events = os.path.join(ROOT, "samples", "basic", "events.ndjson")
    expected = json.load(open(os.path.join(
        ROOT, "samples", "basic", "expected.json")))
    resume_spec = os.path.join(ROOT, "samples", "resume", "spec.json")
    part1 = os.path.join(ROOT, "samples", "resume", "events_part1.ndjson")
    part2 = os.path.join(ROOT, "samples", "resume", "events_part2.ndjson")

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: dict) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    # 1-3 ingest + finalization + fixed point
    code, ingest = _cli(workdir, "ingest", "--spec", basic_spec,
                        "--events", basic_events, "--drain")
    view = json.loads(ingest["cache_view"])
    actual = {p["partition"]: p for p in view["partitions"]}
    agg_ok = all(
        actual[k]["finalized_count"] == expected[k]["finalized_count"]
        and actual[k]["late_count"] == expected[k]["late_count"]
        and actual[k]["sum_fixed"] == expected[k]["sum_fixed"]
        for k in ("p0", "p1", "p2"))
    record("watermark_finalization_and_fixedpoint_aggregation", agg_ok,
           {"expected": {k: expected[k] for k in ("p0", "p1", "p2")},
            "actual": {k: {f: actual[k][f] for f in
                           ("finalized_count", "late_count", "sum_fixed")}
                       for k in actual}})

    # 4 cache vs authoritative replay byte equality
    code, recompute = _cli(workdir, "recompute", "--spec", basic_spec)
    record("cache_authoritative_byte_identical",
           recompute.get("status") == "identical",
           {"exit_code": code,
            "view_bytes": recompute.get("view_bytes")})

    # 5 latency: normal + over budget
    _, q1 = _cli(workdir, "query", "--spec", basic_spec)
    _, q2 = _cli(workdir, "query", "--spec", basic_spec,
                 "--inject-delay-ms", "80")
    _, latency = _cli(workdir, "latency", "--spec", basic_spec)
    lat_ok = (q2["code"] == "E_LAT"
              and latency["overall"]["samples"] == 2
              and latency["over_budget_count"] == 1)
    record("latency_budget_observation", lat_ok,
           {"normal": {"latency_ms": q1["latency_ms"],
                       "over_budget": q1["over_budget"]},
            "injected": {"latency_ms": q2["latency_ms"],
                         "over_budget": q2["over_budget"]},
            "stats": latency["overall"]})

    # 6 parameter replacement
    new_spec = os.path.join(workdir, "new_spec.json")
    with open(new_spec, "w") as handle:
        json.dump({"out_of_order_ms": 250, "latency_budget_ms": 30,
                   "scale": 4,
                   "partitions": [{"key": k, "aggregations": ["sum", "count"]}
                                  for k in ("p0", "p1", "p2")]}, handle)
    _, params = _cli(workdir, "set-params", "--spec", basic_spec,
                     "--new-spec", new_spec)
    record("parameter_version_replacement",
           params["new_param_version"] == 2 and params["stale_versions"] == [1],
           {"new_param_version": params["new_param_version"],
            "affected_partitions":
                [a["partition"] for a in params["affected_partitions"]]})

    # 7-8 snapshot + restore + resume replay (separate workdir)
    resume_dir = os.path.join(workdir, "resume")
    os.makedirs(resume_dir, exist_ok=True)
    _cli(resume_dir, "ingest", "--spec", resume_spec, "--events", part1)
    _, snap = _cli(resume_dir, "snapshot", "--spec", resume_spec)
    _, restore = _cli(resume_dir, "restore", "--spec", resume_spec)
    code, resumed = _cli(resume_dir, "resume", "--spec", resume_spec,
                         "--events", part2)
    final_view = json.loads(resumed["final_view"])
    full = {p["partition"]: p for p in final_view["partitions"]}
    expected_full = json.load(open(os.path.join(
        ROOT, "samples", "resume", "expected.json")))["full"]
    resume_ok = all(
        full[k]["finalized_count"] == expected_full[k]["finalized_count"]
        and full[k]["sum_fixed"] == expected_full[k]["sum_fixed"]
        for k in ("p0", "p1"))
    record("snapshot_restore_replay_consistency",
           restore.get("replacement_required") is False
           and resumed["snapshot_check"]["status"] == "identical"
           and resume_ok,
           {"snapshot_digest": snap["content_digest"],
            "final": {k: {"finalized_count": full[k]["finalized_count"],
                          "sum_fixed": full[k]["sum_fixed"]}
                      for k in full}})

    # fault injections
    inj_dir = os.path.join(workdir, "injections")
    os.makedirs(inj_dir, exist_ok=True)
    c1, i1 = _cli(inj_dir, "inject", "--spec", basic_spec,
                  "--events", basic_events, "--kind", "corrupt")
    _cli(inj_dir, "ingest", "--spec", resume_spec, "--events", part1)
    _cli(inj_dir, "snapshot", "--spec", resume_spec)
    c2, i2 = _cli(inj_dir, "inject", "--spec", resume_spec,
                  "--events", part2, "--kind", "regression",
                  "--partition", "p0", "--seq", "2", "--ts-ms", "10")
    c3, i3 = _cli(inj_dir, "inject", "--spec", resume_spec,
                  "--kind", "snapshot-digest")
    record("fault_injection_error_codes",
           (i1["error"]["code"] == "E_SPEC" and c1 == 2
            and i2["error"]["code"] == "E_ORDER" and c2 == 5
            and i3["error"]["code"] == "E_SNAPSHOT" and c3 == 6),
           {"E_SPEC_exit": c1, "E_ORDER_exit": c2, "E_SNAPSHOT_exit": c3})

    summary = {
        "generated_ms": int(time.time() * 1000),
        "python": sys.version.split()[0],
        "workdir": os.path.abspath(workdir),
        "total_checks": len(checks),
        "passed": sum(1 for c in checks if c["ok"]),
        "all_passed": all(c["ok"] for c in checks),
        "checks": checks,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=os.path.join("run", "acceptance"))
    parser.add_argument("--json-out")
    args = parser.parse_args()
    summary = run(args.workdir)
    text = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    raise SystemExit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
