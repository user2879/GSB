#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo "== 0. environment =="
"$PY" -m blockledger doctor
"$PY" -m blockledger gui-check || true

echo "== 1. generate deterministic samples =="
"$PY" -m blockledger generate-samples

reset_paths() {
  "$PY" - "$@" <<'PYRESET'
from pathlib import Path
import sys
for arg in sys.argv[1:]:
    base = Path(arg)
    for path in [base, Path(str(base) + "-wal"), Path(str(base) + "-shm")]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
PYRESET
}

echo "== 2. strong spec validation and per-volume summary =="
"$PY" -m blockledger validate --spec samples/spec.json --out artifacts/validation.json

echo "== 3. automated tests =="
"$PY" -m unittest discover -s tests

echo "== 4. deterministic replay =="
"$PY" -m blockledger replay --spec samples/spec.json --out artifacts/replay-effects.json

echo "== 5. invalid subset recomputation vs full rebuild =="
"$PY" -m blockledger invalidate \
  --spec samples/spec.json \
  --change samples/change-write.json \
  --out artifacts/invalidation-write.json
"$PY" -m blockledger invalidate \
  --spec samples/spec.json \
  --change samples/change-delete.json \
  --out artifacts/invalidation-delete.json

echo "== 6. authoritative rollback run and independent verify =="
reset_paths ledger/blockledger.sqlite ledger/session.json
"$PY" -m blockledger run \
  --spec samples/spec.json \
  --report artifacts/run-report.json
"$PY" -m blockledger verify \
  --spec samples/spec.json \
  --out artifacts/verify-report.json

echo "== 7. real subprocess SIGKILL, unknown then rerun =="
reset_paths ledger/kill-demo.sqlite ledger/kill-demo-session.json
"$PY" -m blockledger run \
  --spec samples/spec.json \
  --ledger ledger/kill-demo.sqlite \
  --session ledger/kill-demo-session.json \
  --report artifacts/kill-run-report.json \
  --kill-block vol-beta:1
set +e
"$PY" -m blockledger verify \
  --spec samples/spec.json \
  --ledger ledger/kill-demo.sqlite \
  --out artifacts/kill-verify-before-rerun.json
KILL_VERIFY_BEFORE=$?
set -e
echo "verify_before_rerun_exit=${KILL_VERIFY_BEFORE} (expected 4)"
"$PY" -m blockledger recover \
  --spec samples/spec.json \
  --ledger ledger/kill-demo.sqlite \
  --session ledger/kill-demo-session.json \
  --rerun \
  --report artifacts/kill-recovery-report.json
"$PY" -m blockledger verify \
  --spec samples/spec.json \
  --ledger ledger/kill-demo.sqlite \
  --out artifacts/kill-verify-after-rerun.json

echo "== 8. SIGKILLed process lease, invalidation unfinished set, restart recovery =="
reset_paths ledger/crash-demo.sqlite ledger/crash-demo-session.json
set +e
"$PY" -m blockledger run \
  --spec samples/spec.json \
  --ledger ledger/crash-demo.sqlite \
  --session ledger/crash-demo-session.json \
  --only-rollback rb-alpha-11 \
  --crash-after-lease-batch rb-alpha-11:a01-b000
CRASH_EXIT=$?
set -e
echo "crash_exit=${CRASH_EXIT} (expected 137)"
"$PY" - <<'PY'
import sqlite3
conn = sqlite3.connect("ledger/crash-demo.sqlite")
print("durable_running_batches=", conn.execute("SELECT batch_id,status FROM batches").fetchall())
PY
"$PY" -m blockledger invalidate \
  --spec samples/spec.json \
  --change samples/change-delete.json \
  --ledger ledger/crash-demo.sqlite \
  --out artifacts/invalidation-delete-unfinished.json
"$PY" -m blockledger recover \
  --spec samples/spec.json \
  --ledger ledger/crash-demo.sqlite \
  --session ledger/crash-demo-session.json \
  --rerun \
  --report artifacts/crash-recovery-report.json
"$PY" -m blockledger run \
  --spec samples/spec.json \
  --ledger ledger/crash-demo.sqlite \
  --session ledger/crash-demo-session.json \
  --attempt 2 \
  --report artifacts/crash-full-run-report.json
"$PY" -m blockledger verify \
  --spec samples/spec.json \
  --ledger ledger/crash-demo.sqlite \
  --out artifacts/crash-verify-report.json

echo "== 9. corruption E_CORRUPT and E_VERIFY =="
reset_paths ledger/corrupt-demo.sqlite ledger/corrupt-demo-session.json
set +e
"$PY" -m blockledger run \
  --spec samples/spec.json \
  --ledger ledger/corrupt-demo.sqlite \
  --session ledger/corrupt-demo-session.json \
  --report artifacts/corrupt-run-report.json \
  --corrupt-block vol-beta:1
CORRUPT_RUN_EXIT=$?
"$PY" -m blockledger verify \
  --spec samples/spec.json \
  --ledger ledger/corrupt-demo.sqlite \
  --out artifacts/corrupt-verify-report.json
CORRUPT_VERIFY_EXIT=$?
set -e
echo "corrupt_run_exit=${CORRUPT_RUN_EXIT} (expected 3)"
echo "corrupt_verify_exit=${CORRUPT_VERIFY_EXIT} (expected 4)"

echo "== 10. session identity mismatch replaces whole ledger =="
reset_paths ledger/replacement-demo.sqlite ledger/replacement-demo-session.json
"$PY" - <<'PY'
from pathlib import Path
for path in Path("ledger/backups").glob("replacement-demo.replaced-*"):
    if path.is_file():
        path.unlink()
PY
"$PY" -m blockledger run \
  --spec samples/spec.json \
  --ledger ledger/replacement-demo.sqlite \
  --session ledger/replacement-demo-session.json \
  --report artifacts/replacement-original-report.json >/dev/null
"$PY" - <<'PY'
import json
from pathlib import Path
spec = json.loads(Path("samples/spec.json").read_text(encoding="utf-8"))
spec["budget"]["verification_sample_permille"] = 1000
Path("artifacts/spec-changed-sample-rate.json").write_text(
    json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
PY
"$PY" -m blockledger run \
  --spec artifacts/spec-changed-sample-rate.json \
  --ledger ledger/replacement-demo.sqlite \
  --session ledger/replacement-demo-session.json \
  --report artifacts/replacement-changed-report.json
"$PY" -m blockledger verify \
  --spec artifacts/spec-changed-sample-rate.json \
  --ledger ledger/replacement-demo.sqlite \
  --out artifacts/replacement-verify.json
"$PY" - <<'PY'
import json
from pathlib import Path
report = json.load(open("artifacts/replacement-changed-report.json"))
diff = report["session_replacement"]
print("replacement_differences=", [item["field"] for item in diff["differences"]])
print("backup_database=", diff["backup_database"])
mode = Path(diff["backup_database"]).stat().st_mode & 0o777
print("backup_mode=", oct(mode))
assert mode == 0o444
PY

echo "== ACCEPTANCE OK =="
