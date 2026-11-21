#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/work/acceptance"
mkdir -p "$WORK/samples"
PY="${PYTHON:-python3}"
CLI=(-m emergency_drill.cli --log "$WORK/drill.log.jsonl")

"$PY" "${CLI[@]}" generate-samples --sample-dir "$WORK/samples"
"$PY" "${CLI[@]}" run --plan "$WORK/samples/plan_valid.json" \
  --events "$WORK/samples/events_valid.json" --ledger "$WORK/drill.sqlite3" \
  --output "$WORK/run.json"
cat > "$WORK/changeset.json" <<'JSON'
{"operations":[
  {"op":"update_duration","stage_id":"recover","duration_ms":1700},
  {"op":"rewrite_criterion","stage_id":"recover","criterion":{"mode":"all_of","required_actions":["a_service_restore"]}}
]}
JSON
"$PY" "${CLI[@]}" edit --ledger "$WORK/drill.sqlite3" \
  --changeset "$WORK/changeset.json" --output "$WORK/edit.json"
"$PY" "${CLI[@]}" verify-incr --ledger "$WORK/drill.sqlite3"
"$PY" "${CLI[@]}" snapshot --ledger "$WORK/drill.sqlite3" --output "$WORK/snapshot.json"
"$PY" "${CLI[@]}" replay --ledger "$WORK/drill.sqlite3" --snapshot "$WORK/snapshot.json"
"$PY" "${CLI[@]}" tamper-snapshot --snapshot "$WORK/snapshot.json" --output "$WORK/snapshot.bad.json"
"$PY" "${CLI[@]}" open-snapshot --ledger "$WORK/drill.sqlite3" --snapshot "$WORK/snapshot.bad.json" || true
"$PY" "${CLI[@]}" view --ledger "$WORK/drill.sqlite3" --text
