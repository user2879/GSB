#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
STATE_DIR="${INS_DIAG_CRASH_STATE:-$(mktemp -d /tmp/ins-diag-crash.XXXXXX)}"
REPORT_DIR="${INS_DIAG_REPORT_DIR:-$ROOT/artifacts}"
mkdir -p "$REPORT_DIR"

echo "state_dir=$STATE_DIR"

python3 -m ins_diag run \
  --config "$ROOT/examples/config.yaml" \
  --state-dir "$STATE_DIR" \
  --output "$REPORT_DIR/crash-baseline.json" >/dev/null
baseline_exit=$?

set +e
python3 -m ins_diag run \
  --config "$ROOT/examples/config.yaml" \
  --state-dir "$STATE_DIR" \
  --crash-after after_lease >/dev/null 2>&1
killed_exit=$?
set -e

python3 -m ins_diag run \
  --config "$ROOT/examples/config.yaml" \
  --state-dir "$STATE_DIR" \
  --output "$REPORT_DIR/crash-recovered.json" >/dev/null
recovered_exit=$?

python3 - "$REPORT_DIR/crash-recovered.json" "$REPORT_DIR/crash-report.json" \
  "$baseline_exit" "$killed_exit" "$recovered_exit" <<'PY'
import json
import sys

recovered_path, output_path, baseline_exit, killed_exit, recovered_exit = sys.argv[1:]
report = json.load(open(recovered_path, encoding="utf-8"))
summary = {
    "status": (
        "ok"
        if int(baseline_exit) == 0
        and int(killed_exit) in {-9, 137}
        and int(recovered_exit) == 0
        and report["recovery"]["resume_scope"] == "cache_only_superseded"
        else "failed"
    ),
    "baseline_exit": int(baseline_exit),
    "forced_sigkill_exit": int(killed_exit),
    "restart_exit": int(recovered_exit),
    "restart_status": report["status"],
    "recovery": report["recovery"],
    "cache": {
        "reused_count": report["cache"]["reused_count"],
        "recomputed_count": report["cache"]["recomputed_count"],
    },
    "snapshot_commit": report["snapshot"]["commit"],
    "error_codes": report["error_codes"],
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, sort_keys=True, indent=2)
    handle.write("\n")
print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
raise SystemExit(0 if summary["status"] == "ok" else 3)
PY

