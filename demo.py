"""Deterministic three-scenario demo (run from the workspace root):

  python demo.py

Scenario 1: cold start processing on a fresh ledger.
Scenario 2: forced termination mid-stream, then recovery to completion.
Scenario 3: subscriber-set version bump triggers wholesale ledger replacement.

Each scenario ends with a stable-ordered verification summary produced by the
independent tool tools/verify.py.
"""

import json
import os
import shutil
import stat
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VAR = os.path.join(ROOT, "var")
LEDGER = os.path.join(VAR, "ledger.db")
LOG = os.path.join(VAR, "eventbus.log")
SOURCES = os.path.join(VAR, "sources.json")
SUBSCRIBERS = os.path.join(VAR, "subscribers.json")


def run_cli(arguments, expect_codes=(0,)):
    command = [sys.executable, os.path.join(ROOT, "main.py")] + arguments
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in expect_codes:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("unexpected exit code %d for %s" % (result.returncode, command))
    return result


def verify():
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "verify.py"),
         "--ledger", LEDGER, "--sources", SOURCES],
        cwd=ROOT, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        raise SystemExit("verification failed")


def common_args():
    return ["--sources", SOURCES, "--subscribers", SUBSCRIBERS,
            "--ledger", LEDGER, "--log", LOG]


def reset_var():
    if os.path.isdir(VAR):
        for name in os.listdir(VAR):
            path = os.path.join(VAR, name)
            os.chmod(path, 0o644)
            os.remove(path)
    else:
        os.makedirs(VAR)
    shutil.copyfile(os.path.join(ROOT, "samples", "sources.json"), SOURCES)
    shutil.copyfile(os.path.join(ROOT, "samples", "subscribers.json"), SUBSCRIBERS)


def scenario_cold_start():
    print("=== scenario 1: cold start ===")
    result = run_cli(["run"] + common_args())
    print(result.stdout.strip())
    verify()


def scenario_kill_recover():
    print("=== scenario 2: forced termination and recovery ===")
    for suffix in ("", "-wal", "-shm", ".backup-f1-s1"):
        if os.path.exists(LEDGER + suffix):
            os.chmod(LEDGER + suffix, 0o644)
            os.remove(LEDGER + suffix)
    killed = run_cli(["run", "--kill-after", "5"] + common_args(), expect_codes=(3,))
    print("forced termination exit code: %d (no exit report expected)" % killed.returncode)
    recovered = run_cli(["run"] + common_args())
    print(recovered.stdout.strip())
    verify()


def scenario_version_replacement():
    print("=== scenario 3: subscriber-set version bump -> wholesale replacement ===")
    with open(SUBSCRIBERS, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    doc["set_version"] = doc["set_version"] + 1
    with open(SUBSCRIBERS, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
    result = run_cli(["run"] + common_args())
    print(result.stdout.strip())
    backup = LEDGER + ".backup-f1-s1"
    if not os.path.exists(backup):
        raise SystemExit("expected read-only backup missing: %s" % backup)
    mode = stat.S_IMODE(os.stat(backup).st_mode)
    if mode & 0o222:
        raise SystemExit("backup is not read-only: %s" % backup)
    print("read-only backup confirmed: %s" % backup)
    verify()


def main():
    reset_var()
    scenario_cold_start()
    scenario_kill_recover()
    scenario_version_replacement()
    print("=== demo complete: all scenarios verified ===")


if __name__ == "__main__":
    main()
