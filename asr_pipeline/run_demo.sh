#!/usr/bin/env bash
# End-to-end acceptance demo: budget scheduling, incremental reuse,
# serial/parallel consistency and injection isolation. Fully local.
set -euo pipefail
cd "$(dirname "$0")"

# Unique run root so repeated demos never reuse a prior run's cache and the
# tight-budget stage genuinely exhausts and resumes.
RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
OUT=demo_output/$RUN_ID
mkdir -p "$OUT"
SAMPLES=samples
WAV=$SAMPLES/wav
echo "Writing artifacts to $OUT"

echo "== 1. Baseline parallel run (generous budget) =="
python3 -m asr_pipeline.cli --sample-dir "$WAV" \
  --score-config "$SAMPLES/score_params.json" \
  --schedule-config "$SAMPLES/schedule.json" \
  --phoneme-map "$SAMPLES/phoneme_map.json" \
  --workdir "$OUT/work_parallel" \
  --report "$OUT/report_parallel.json" --jitter-ms 0.8

echo "== 2. Serial reference run =="
python3 -m asr_pipeline.cli --sample-dir "$WAV" \
  --score-config "$SAMPLES/score_params.json" \
  --schedule-config "$SAMPLES/schedule.json" \
  --phoneme-map "$SAMPLES/phoneme_map.json" \
  --workdir "$OUT/work_serial" \
  --report "$OUT/report_serial.json" --serial

echo "== 3. Tight global budget, auto-resume until complete (parallel) =="
python3 - "$OUT/schedule_tight.json" <<'PY'
import json, sys
cfg=json.load(open("samples/schedule.json"))
cfg.update(global_point_budget=2500, shard_point_quota=700, shards_per_batch=3)
with open(sys.argv[1], "w") as handle:
    json.dump(cfg, handle, indent=2)
PY
# Isolated workdir so the tight-budget run really exhausts and resumes,
# rather than reusing artifacts produced by the generous-budget runs.
python3 -m asr_pipeline.cli --sample-dir "$WAV" \
  --score-config "$SAMPLES/score_params.json" \
  --schedule-config "$OUT/schedule_tight.json" \
  --phoneme-map "$SAMPLES/phoneme_map.json" \
  --workdir "$OUT/work_tight_isolated" \
  --report "$OUT/report_tight_partial.json" || true
python3 -m asr_pipeline.cli --sample-dir "$WAV" \
  --score-config "$SAMPLES/score_params.json" \
  --schedule-config "$OUT/schedule_tight.json" \
  --phoneme-map "$SAMPLES/phoneme_map.json" \
  --workdir "$OUT/work_tight_isolated" \
  --report "$OUT/report_tight_final.json" --max-runs 8

echo "== 4. Injection isolation =="
python3 -m asr_pipeline.cli --sample-dir "$WAV" \
  --score-config "$SAMPLES/score_params.json" \
  --schedule-config "$SAMPLES/schedule.json" \
  --phoneme-map "$SAMPLES/phoneme_map.json" \
  --injections "$SAMPLES/injections.example.json" \
  --workdir "$OUT/work_injection" \
  --report "$OUT/report_injection.json" || true

echo "== 5. Field-level consistency assertion =="
python3 tools/verify_consistency.py \
  --serial "$OUT/report_serial.json" \
  --parallel "$OUT/report_parallel.json" \
  --resumed "$OUT/report_tight_final.json" \
  --injection "$OUT/report_injection.json"

echo "All demo artifacts written under $OUT/"
