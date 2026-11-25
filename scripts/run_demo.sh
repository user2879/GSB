#!/usr/bin/env bash
# Repeatable end-to-end demo: generate, run, rebuild, recover.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Configs use relative snapshot/ledger dirs; resolve them to the project root by
# copying into runtime configs with absolute storage paths.
python3 - "$ROOT" <<'PY'
import json, os, sys
root = sys.argv[1]
for name in ("vad_config", "vad_config_injection"):
    src = os.path.join(root, "config", name + ".json")
    raw = json.load(open(src))
    suffix = "_inj" if "injection" in name else ""
    raw["snapshot_dir"] = os.path.join(root, "snapshots" + suffix)
    raw["ledger_dir"] = os.path.join(root, "ledger" + suffix)
    # Demo tuning: cheap per-segment cost keeps incremental appends fresh,
    # while global parameter/quantization changes (which recompute every
    # window) exceed the budget and are returned as E_STALE_RESULT.
    raw["segment_cost_ms"] = 1
    raw["window_cost_ms"] = 1
    raw["latency_budget_ms"] = 60
    if raw.get("injection_manifest"):
        raw["injection_manifest"] = os.path.join(
            root, "config", "injection_manifest.json")
    os.makedirs(os.path.join(root, "artifacts", "runtime_config"), exist_ok=True)
    json.dump(raw, open(os.path.join(
        root, "artifacts", "runtime_config", name + ".json"), "w"), indent=2)
PY

python3 -m vad_diag.cli generate --out-dir samples/demo --seed 20261126 --segments 20 --segment-samples 3200
python3 -m vad_diag.cli run \
  --config artifacts/runtime_config/vad_config.json \
  --manifest samples/demo/segments_manifest.json \
  --commands config/command_stream.json \
  --report artifacts/run_report.json
python3 -m vad_diag.cli run \
  --config artifacts/runtime_config/vad_config_injection.json \
  --manifest samples/demo/segments_manifest.json \
  --commands config/command_stream_injection.json \
  --report artifacts/injection_report.json
python3 -m vad_diag.cli rebuild \
  --config artifacts/runtime_config/vad_config.json \
  --manifest samples/demo/segments_manifest.json \
  --report artifacts/rebuild_report.json
python3 -m vad_diag.cli recover --config artifacts/runtime_config/vad_config.json
