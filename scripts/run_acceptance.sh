#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
CLI=("$PY" -m feature_backfill)
OUT="$ROOT/results/acceptance"

if [[ -d "$OUT" ]]; then
  mkdir -p "$ROOT/results/acceptance-archive"
  mv "$OUT" "$ROOT/results/acceptance-archive/$(date +%Y%m%dT%H%M%S)-$$"
fi
mkdir -p "$OUT"

run_capture() {
  local name="$1"
  shift
  set +e
  "$@" >"$OUT/$name.stdout.json" 2>"$OUT/$name.stderr.txt"
  local code=$?
  set -e
  printf '%s\n' "$code" >"$OUT/$name.exit_code"
  return 0
}

set -e

run_capture validate-valid \
  "${CLI[@]}" validate \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/default/config.yaml \
  --plan data/default/plan.json

for name in duplicate_row nonfinite_row timestamp_reversal; do
  run_capture "validate-$name" \
    "${CLI[@]}" validate \
    --samples "data/illegal/$name/samples.jsonl" \
    --features "data/illegal/$name/features.json" \
    --config "data/illegal/$name/config.yaml" \
    --plan "data/illegal/$name/plan.json"
done

for name in bad_dag missing_upstream missing_raw_field; do
  run_capture "validate-$name" \
    "${CLI[@]}" validate \
    --samples "data/illegal/$name/samples.jsonl" \
    --features "data/illegal/$name/features.json" \
    --config "data/illegal/$name/config.yaml" \
    --plan "data/illegal/$name/plan.json"
done

for name in bad_config_quota bad_config_lease_ttl bad_plan; do
  run_capture "validate-$name" \
    "${CLI[@]}" validate \
    --samples "data/illegal/$name/samples.jsonl" \
    --features "data/illegal/$name/features.json" \
    --config "data/illegal/$name/config.yaml" \
    --plan "data/illegal/$name/plan.json"
done

run_capture lease-demo "${CLI[@]}" lease-demo --ttl-ms 60

run_capture run-incremental \
  "${CLI[@]}" run \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/default/config.yaml \
  --plan data/default/plan.json \
  --snapshot-dir "$OUT/snap-incremental" \
  --report "$OUT/incremental-report.json"

run_capture run-final-full \
  "${CLI[@]}" run \
  --samples data/final/samples.jsonl \
  --features data/final/features.json \
  --config data/final/config.yaml \
  --plan data/final/plan.json \
  --snapshot-dir "$OUT/snap-final-full" \
  --report "$OUT/final-full-report.json"

run_capture compare-incremental-final \
  "${CLI[@]}" compare "$OUT/incremental-report.json" "$OUT/final-full-report.json"

run_capture run-serial \
  "${CLI[@]}" run \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/scenarios/serial/config.yaml \
  --plan data/default/plan.json \
  --snapshot-dir "$OUT/snap-serial" \
  --report "$OUT/serial-report.json"

run_capture compare-parallel-serial \
  "${CLI[@]}" compare "$OUT/incremental-report.json" "$OUT/serial-report.json" --execution

run_capture run-quota \
  "${CLI[@]}" run \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/scenarios/quota/config.yaml \
  --plan data/default/plan.json \
  --snapshot-dir "$OUT/snap-quota" \
  --report "$OUT/quota-report.json"

for name in nonfinite timestamp_reversal lease_ttl_zero cache_budget_zero; do
  run_capture "run-injection-$name" \
    "${CLI[@]}" run \
    --samples data/default/samples.jsonl \
    --features data/default/features.json \
    --config data/default/config.yaml \
    --plan data/scenarios/empty_plan.json \
    --injections "injections/$name.json" \
    --snapshot-dir "$OUT/snap-injection-$name" \
    --report "$OUT/injection-$name-report.json"
done

set +e
"${CLI[@]}" run \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/default/config.yaml \
  --plan data/default/plan.json \
  --snapshot-dir "$OUT/snap-crash" \
  --crash-before-commit-batch 3 \
  >"$OUT/crash.stdout.json" 2>"$OUT/crash.stderr.txt"
CRASH_CODE=$?
set -e
printf '%s\n' "$CRASH_CODE" >"$OUT/crash.exit_code"

run_capture run-resume \
  "${CLI[@]}" run \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/default/config.yaml \
  --plan data/default/plan.json \
  --snapshot-dir "$OUT/snap-crash" \
  --resume \
  --report "$OUT/resume-report.json"

run_capture compare-resume-final \
  "${CLI[@]}" compare "$OUT/incremental-report.json" "$OUT/resume-report.json"

set +e
"$PY" -m unittest discover -s tests -t . -v >"$OUT/unit-tests.log" 2>&1
TEST_CODE=$?
set -e
printf '%s\n' "$TEST_CODE" >"$OUT/unit-tests.exit_code"

"$PY" "$ROOT/scripts/summarize_acceptance.py"

echo "Acceptance artefacts: $OUT"
