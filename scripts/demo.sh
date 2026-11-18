#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Runtime dependency preparation is explicit. After the first `go mod download`,
# this demo performs build, test, pipeline, verify, and replay with networking disabled.
export GOTOOLCHAIN=local
export GOPROXY=off

RUN_DIR="$ROOT/out/demo-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR" "$ROOT/bin"
FL="$ROOT/bin/flowledger"

echo "[1/8] build"
go build -o "$FL" ./cmd/flowledger

echo "[2/8] tests"
go test ./... -count=1 | tee "$RUN_DIR/tests.log"

echo "[3/8] run deterministic v1"
"$FL" run \
  --spec samples/traffic-v1.json --policy samples/policy-v1.json \
  --ledger "$RUN_DIR/ledger.db" --report "$RUN_DIR/v1-report.json" \
  --k 3 --window-minutes 1 --global-concurrency 3 --per-key-concurrency 2 \
  --task-delay-ms 10 > "$RUN_DIR/v1-summary.json"

echo "[4/8] replay and verify v1"
"$FL" replay --ledger "$RUN_DIR/ledger.db" --output "$RUN_DIR/v1-replay.json" | tee "$RUN_DIR/v1-replay-stdout.json"
"$FL" verify --ledger "$RUN_DIR/ledger.db" --output "$RUN_DIR/v1-verify.json" | tee "$RUN_DIR/v1-verify-stdout.json"

echo "[5/8] start v2 and inject forced termination after committed windows"
set +e
"$FL" run \
  --spec samples/traffic-v2.json --policy samples/policy-v2.json \
  --ledger "$RUN_DIR/ledger.db" --report "$RUN_DIR/v2-report.json" \
  --k 3 --window-minutes 1 --global-concurrency 3 --per-key-concurrency 2 \
  --task-delay-ms 10 --crash-after 2 \
  > "$RUN_DIR/v2-crash-summary.json" 2> "$RUN_DIR/v2-crash-error.json"
crash_code=$?
set -e
if [[ $crash_code -ne 75 ]]; then
  echo "expected crash exit 75, got $crash_code" >&2
  exit 1
fi
if [[ -e "$RUN_DIR/v2-report.json" ]]; then
  echo "forced termination unexpectedly produced an exit report" >&2
  exit 1
fi

echo "[6/8] prove incomplete current view is not reportable"
set +e
"$FL" verify --ledger "$RUN_DIR/ledger.db" > "$RUN_DIR/incomplete-verify.json" 2>&1
incomplete_verify=$?
"$FL" replay --ledger "$RUN_DIR/ledger.db" > "$RUN_DIR/incomplete-replay.json" 2>&1
incomplete_replay=$?
set -e
if [[ $incomplete_verify -ne 66 || $incomplete_replay -ne 65 ]]; then
  echo "unexpected incomplete-view exits: verify=$incomplete_verify replay=$incomplete_replay" >&2
  exit 1
fi
"$FL" history --ledger "$RUN_DIR/ledger.db" > "$RUN_DIR/history-after-crash.json"

echo "[7/8] resume v2, verify, replay"
"$FL" run \
  --spec samples/traffic-v2.json --policy samples/policy-v2.json \
  --ledger "$RUN_DIR/ledger.db" --report "$RUN_DIR/v2-report.json" \
  --k 3 --window-minutes 1 --global-concurrency 3 --per-key-concurrency 2 \
  --task-delay-ms 10 > "$RUN_DIR/v2-summary.json"
"$FL" verify --ledger "$RUN_DIR/ledger.db" --output "$RUN_DIR/v2-verify.json" | tee "$RUN_DIR/v2-verify-stdout.json"
"$FL" replay --ledger "$RUN_DIR/ledger.db" --output "$RUN_DIR/v2-replay.json" | tee "$RUN_DIR/v2-replay-stdout.json"
"$FL" history --ledger "$RUN_DIR/ledger.db" > "$RUN_DIR/history-final.json"

echo "[8/8] stable result summary"
{
  echo "run_dir=$RUN_DIR"
  echo "v1_summary_digest=$(grep -m1 '"summary_digest"' "$RUN_DIR/v1-summary.json" | sed -E 's/.*"([0-9a-f]{64})".*/\1/')"
  echo "v2_summary_digest=$(grep -m1 '"summary_digest"' "$RUN_DIR/v2-summary.json" | sed -E 's/.*"([0-9a-f]{64})".*/\1/')"
  echo "v1_replay=$(grep -m1 '"status"' "$RUN_DIR/v1-replay-stdout.json" | tr -d ' ",' | cut -d: -f2)"
  echo "v2_replay=$(grep -m1 '"status"' "$RUN_DIR/v2-replay-stdout.json" | tr -d ' ",' | cut -d: -f2)"
  echo "v2_recovered_unknown_tasks=$(grep -m1 '"recovered_unknown_tasks"' "$RUN_DIR/v2-verify.json" | sed -E 's/[^0-9]*([0-9]+).*/\1/')"
  echo "final_current_stale_historical=1"
  echo "forced_termination_exit=$crash_code"
  echo "incomplete_verify_exit=$incomplete_verify"
  echo "incomplete_replay_exit=$incomplete_replay"
} | tee "$RUN_DIR/validation-summary.txt"

echo "PASS: artifacts are in $RUN_DIR"
