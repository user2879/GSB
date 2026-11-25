"""End-to-end byte-identity and three-class conclusion verification."""
import glob
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(patterns):
    for pattern in patterns:
        for path in glob.glob(os.path.join(ROOT, pattern)):
            if os.path.isfile(path):
                os.chmod(path, 0o644)
                os.remove(path)


def run(*args):
    subprocess.check_call([sys.executable, "-m", "rdiq.cli", *args],
                          cwd=ROOT, stdout=subprocess.DEVNULL)


def sha(path):
    with open(os.path.join(ROOT, path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


clean(["ledger/rdiq.sqlite*", "ledger/backups/*"])
run("run", "--sample-dir", "samples/base", "--output", "samples/tmp/verify-1.json",
    "--ledger-db", "ledger/rdiq.sqlite", "--backup-dir", "ledger/backups")
clean(["ledger/rdiq.sqlite*", "ledger/backups/*"])
run("run", "--sample-dir", "samples/base", "--output", "samples/tmp/verify-2.json",
    "--ledger-db", "ledger/rdiq.sqlite", "--backup-dir", "ledger/backups")
first = sha("samples/tmp/verify-1.json")
second = sha("samples/tmp/verify-2.json")
assert first == second, (first, second)

import json
baseline = json.load(open(os.path.join(ROOT, "data/baseline.json"), encoding="utf-8"))
assert baseline["graph"]["node_count"] > 0
for session in baseline["sessions"]:
    assert session["view"]["view_digest"]
    for attribution in session["error_attribution"]:
        assert attribution["dominant_level"] in {"source", "evidence", "round"}
recovery = json.load(open(os.path.join(ROOT, "data/recovery.json"), encoding="utf-8"))
assert recovery["committed_batches"] == 1
assert recovery["unknown_rerun_rounds"]
assert "matched ledger" in recovery["replay_status"]
print({
    "byte_identical_runs": True,
    "baseline_sha256": first,
    "graph_digest": baseline["graph"]["graph_digest"],
    "recovery_committed_batches": recovery["committed_batches"],
    "recovery_unknown_rounds": len(recovery["unknown_rerun_rounds"]),
})
