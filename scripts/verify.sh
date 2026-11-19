#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
dotnet build LevelWorkbench.sln
dotnet run --project src/LevelWorkbench.SampleGen -- --output sample-data
dotnet run --project tests/LevelWorkbench.SelfTest

run_dir="$(mktemp -d)"
trap 'rm -rf "$run_dir"' EXIT
dotnet run --project src/LevelWorkbench.Cli -- validate \
  --scene sample-data/scene.json --records sample-data/records/mixed.json
dotnet run --project src/LevelWorkbench.Cli -- replay \
  --scene sample-data/scene.json --records sample-data/records/mixed.json \
  --state "$run_dir/state" --report "$run_dir/replay-report.json"
dotnet run --project src/LevelWorkbench.Cli -- pick \
  --scene sample-data/scene.json --script sample-data/pick-script.json \
  --state "$run_dir/pick-state" --out "$run_dir/picked-records.json"
dotnet run --project src/LevelWorkbench.Cli -- replay \
  --scene sample-data/scene.json --records "$run_dir/picked-records.json" \
  --state "$run_dir/picked-replay-state"
