"""Deterministic three-scenario demo with stable-ordered summary output.

Scenario 1: small scale, single-batch-per-backend run.
Scenario 2: enlarged scale, multi-backend, forced termination + recovery +
            replay verification.
Scenario 3: canary weight change -> config version bump, stale marking,
            wholesale view replacement, replay on the new version.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gateway.engine import run
from gateway.ledger import Ledger
from gateway.loader import load_declaration
from gateway.replay import recovery_scan, replay_verify

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "demo_ledger.db")


def p(path: str) -> str:
    return os.path.join(HERE, path)


def print_run(title: str, result) -> None:
    print(f"== {title} ==")
    d = result.decision
    print(f"strategy={d.strategy} tie_threshold={d.tie_threshold} "
          f"estimates={d.estimates}")
    print(f"reason={d.reason}")
    print(f"config_version={result.config_version} "
          f"hash={result.config_hash} stale={result.stale_versions}")
    print(f"peak_running={result.peak_running} "
          f"backend_peak={dict(sorted(result.backend_peak.items()))}")
    for model_id in sorted(result.fingerprints):
        print(f"fingerprint {model_id} {result.fingerprints[model_id]}")
    for rule_id in sorted(result.canary_distribution):
        dist = result.canary_distribution[rule_id]
        for variant in sorted(dist):
            r = dist[variant]
            print(f"canary {rule_id} {variant} "
                  f"{r.value_bp}/10000 ({r.numerator}/{r.denominator}) "
                  f"bound=+/-{r.error_bound_bp}bp {r.rounding}")
    for batch_id in sorted(result.batch_stats):
        s = result.batch_stats[batch_id]
        print(f"batch {batch_id} stable={s['stable_hits']} "
              f"canary={s['canary_hits']} "
              f"canary_bp={s['canary_ratio'].value_bp} "
              f"avg_ms={s['avg_duration'].value}")
    print()


def main() -> None:
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(DB + suffix):
            os.remove(DB + suffix)

    # Scenario 1: small scale.
    ledger = Ledger(DB)
    r1 = run(p("backends.json"), p("rules.json"), p("requests_small.ndjson"),
             ledger, global_limit=2, forward_repeat=50)
    print_run("scenario 1: small scale, single wave", r1)
    ledger.close()

    # Scenario 2: enlarged scale with a forced termination, then recovery.
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(DB + suffix):
            os.remove(DB + suffix)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "crash_runner.py"), "5", DB])
    print(f"== scenario 2: forced termination, exit={proc.returncode} ==")
    ledger = Ledger(DB)
    decl = load_declaration(p("backends.json"), p("rules.json"),
                            p("requests_large.ndjson"))
    version = ledger.current_version()
    scan = recovery_scan(decl, ledger, version)
    print(f"committed={scan['committed']}")
    print(f"unknown(need rerun)={scan['unknown']}")
    print(f"cursor={scan['cursor']}")
    r2 = run(p("backends.json"), p("rules.json"), p("requests_large.ndjson"),
             ledger, global_limit=4, forward_repeat=200)
    print(f"resumed: skipped_committed={r2.skipped_committed}")
    report = replay_verify(decl, ledger, r2.config_version)
    print(f"replay={report}")
    print_run("scenario 2: enlarged scale after recovery", r2)
    ledger.close()

    # Scenario 3: weight change -> version bump, stale view, replay.
    ledger = Ledger(DB)
    r3 = run(p("backends.json"), p("rules_weight_change.json"),
             p("requests_large.ndjson"), ledger,
             global_limit=4, forward_repeat=200)
    print_run("scenario 3: weight change, stale protection", r3)
    report = replay_verify(r3.decl, ledger, r3.config_version)
    print(f"replay_current_version={report}")
    stale_rows = ledger.ledger_rows(r3.config_version - 1)
    print(f"stale_version_rows_excluded={len(stale_rows)} "
          f"(v{r3.config_version - 1}, not merged into current report)")
    for snap in ledger.snapshots():
        print(f"snapshot v{snap['config_version']} hash={snap['config_hash']}")
    ledger.close()


if __name__ == "__main__":
    main()
