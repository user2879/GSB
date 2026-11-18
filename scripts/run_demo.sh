#!/usr/bin/env bash
# Deterministic end-to-end demo: synth -> run -> resume -> incremental -> boundary.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
export PYTHONPATH=voiceprint_pipeline

echo "== 1. generate deterministic synthetic WAV dataset =="
$PY -m voiceprint.cli synth --input-dir data/samples \
    --injection configs/injection_manifest.json >/tmp/synth.json

echo "== 2. full pipeline run (fixed + reference width) =="
$PY -m voiceprint.cli run \
    --input-dir data/samples --config configs/voiceprint_config.json \
    --pairs configs/speaker_pairs.json --injection configs/injection_manifest.json \
    --work-dir data/demo_work --report data/demo_report.json

echo "== 3. resume: expect zero recompute and field-identical output =="
$PY -m voiceprint.cli run --resume \
    --input-dir data/samples --config configs/voiceprint_config.json \
    --pairs configs/speaker_pairs.json --injection configs/injection_manifest.json \
    --work-dir data/demo_work --report data/demo_resume.json

echo "== 4. threshold-only change: only similarity recomputes =="
$PY -m voiceprint.cli run --resume --allow-config-change \
    --input-dir data/samples --config configs/cfg_threshold.json \
    --pairs configs/speaker_pairs.json --injection configs/injection_manifest.json \
    --work-dir data/demo_work --report data/demo_threshold.json

echo "== 5. boundary width-flip scenario =="
$PY -m voiceprint.cli run \
    --input-dir data/samples --config configs/cfg_boundary.json \
    --pairs configs/speaker_pairs.json --injection configs/injection_manifest.json \
    --work-dir data/demo_boundary --report data/demo_boundary.json
echo "demo complete; reports under data/"
