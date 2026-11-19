#!/usr/bin/env bash
# Deterministic end-to-end acceptance: generate -> incremental -> rebuild -> verify.
set -euo pipefail
cd "$(dirname "$0")"

DATASET="${1:-samples/dataset}"
WS=".tmp/demo_ws"
python3 -m scripts.generate_samples --out "$DATASET"
mkdir -p "$WS"

# Clean command stream excludes the deliberately out-of-range injected command.
python3 - "$DATASET/commands.json" "$WS/commands_clean.json" <<'PY'
import json, sys
stream = json.load(open(sys.argv[1]))
clean = {"commands": [c for c in stream["commands"] if "injection_id" not in c]}
json.dump(clean, open(sys.argv[2], "w"), sort_keys=True, indent=2)
PY

echo "== incremental run =="
python3 -m speaker_pipeline.cli incremental \
  --dataset "$DATASET" --config "$DATASET/config.json" \
  --workspace "$WS" --commands "$WS/commands_clean.json"

echo "== full rebuild on final audio =="
python3 -m speaker_pipeline.cli rebuild-final \
  --dataset "$DATASET" --config "$DATASET/config.json" --workspace "$WS"

echo "== parity check =="
python3 -m speaker_pipeline.cli verify \
  --full "$WS/canonical_full.json" \
  --incremental "$WS/canonical_incremental.json"

echo "== injection isolation =="
python3 -m speaker_pipeline.cli incremental \
  --dataset "$DATASET" --config "$DATASET/config.json" \
  --workspace "$WS/injection" \
  --commands "$DATASET/commands.json" \
  --injections "$DATASET/injections.json"

echo "== unit tests =="
python3 -m unittest tests.test_pipeline -v
