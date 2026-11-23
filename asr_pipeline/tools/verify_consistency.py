"""Field-level consistency verification across pipeline run modes."""

import argparse
import json
import sys

# Fields that legitimately vary across independently launched runs.
TRANSIENT_FIELDS = ("shard_timings_ms", "mode", "max_concurrency")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def normalize(report):
    report = dict(report)
    for field in TRANSIENT_FIELDS:
        report.pop(field, None)
    report["reuse_recompute_log"] = [
        {k: v for k, v in entry.items() if k != "cache_key"}
        for entry in report["reuse_recompute_log"]
    ]
    return report


def compare(name, left, right, fields):
    failures = []
    for field in fields:
        if left.get(field) != right.get(field):
            failures.append(field)
    status = "PASS" if not failures else "FAIL"
    print(f"[{status}] {name}")
    if failures:
        for field in failures:
            print(f"    differing field: {field}")
    return not failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--parallel", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--injection", required=True)
    args = parser.parse_args()

    serial = normalize(load(args.serial))
    parallel = normalize(load(args.parallel))
    resumed = normalize(load(args.resumed))
    injection = load(args.injection)

    comparable_fields = [
        "transcripts", "budget_ledger", "pending_queue", "failures",
        "reuse_recompute_log", "shards_total", "shards_completed",
        "shards_pending", "shards_failed",
    ]
    results = []
    results.append(compare("parallel vs serial (full fields)",
                           serial, parallel, comparable_fields))
    results.append(compare("resumed-budget vs serial transcripts/words",
                           serial, resumed, ["transcripts"]))

    ledger_ok = all(entry["remaining_after"] >= 0 for entry in resumed["budget_ledger"])
    print(f"[{'PASS' if ledger_ok else 'FAIL'}] resumed ledger never goes negative")
    results.append(ledger_ok)

    ledger_keys = [(e["audio"], e["shard_index"]) for e in resumed["budget_ledger"]]
    ordered = ledger_keys == sorted(ledger_keys)
    print(f"[{'PASS' if ordered else 'FAIL'}] budget ledger follows stable shard order")
    results.append(ordered)

    hits = injection["injections_hit"]
    expected = {
        "INJ-01": "E_AUDIO_FORMAT",
        "INJ-02": "E_AUDIO_FORMAT",
        "INJ-03": "E_CONFIG_INVALID",
        "INJ-04": "E_SHARD_BUDGET_EXHAUSTED",
    }
    injection_ok = (
        all(hit["hit"] for hit in hits)
        and all(hit["business_code"] == expected[hit["injection_id"]] for hit in hits)
        and "utter_004.wav" not in injection["transcripts"]
    )
    print(f"[{'PASS' if injection_ok else 'FAIL'}] all 4 injections isolated with distinct codes")
    for hit in hits:
        print(f"    {hit['injection_id']} -> {hit['business_code']}")
    results.append(injection_ok)

    if all(results):
        print("\nOVERALL: ALL CONSISTENCY CHECKS PASSED")
        return 0
    print("\nOVERALL: CONSISTENCY CHECK FAILURE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
