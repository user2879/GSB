#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m graymirror.samplegen samples
python3 -m graymirror.cli run --rules samples/rules.json --stream samples/stream.ndjson \
  --scale-exponent 2 --snapshot samples/snapshots/session-k2.json \
  --report samples/report-k2.json
python3 -m graymirror.cli restore --rules samples/rules.json \
  --stream samples/stream.ndjson --snapshot samples/snapshots/session-k2.json
for kind in weight_out_of_range timestamp_regression snapshot_scale_missing; do
  # injection is expected to exit 2 with its business code
  python3 -m graymirror.cli inject --kind "$kind" || true
done
python3 -m unittest discover -s tests
