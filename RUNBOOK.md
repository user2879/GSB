# Runbook

All paths below are relative to the repository root. Commands were verified in
the environment recorded in `ENVIRONMENT.md`.

## 1. Generate the deterministic sample workspace

```bash
python3 -m remote_seg generate \
  --out sample_data \
  --width 48 --height 48 \
  --rows 2 --cols 3 \
  --overlap 8 \
  --seed 20261122 \
  --patch-batches 2
```

This writes RGBA PNG tiles (semantic bands: red, green, NIR, elevation),
`index.json`, `fusion_versions.json`, `patch_plan.json`, and the extra pool tile
used by the add-tile edit. Width, height, rows, columns, overlap, seed and the
number of patch batches are configurable. The checked-in `sample_data/` was
generated with the command above.

## 2. Run the core engine

Initial run, draining all budget batches:

```bash
python3 -m remote_seg run \
  --index sample_data/index.json \
  --config configs/segmentation.json \
  --fusion-versions sample_data/fusion_versions.json \
  --state-dir runtime/state \
  --report runtime/report.json \
  --drain
```

Run one budget batch only (exit code `3` means `E_RECOMPUTE_BUDGET_PENDING`):

```bash
python3 -m remote_seg run \
  --index sample_data/index.json \
  --config configs/segmentation.json \
  --fusion-versions sample_data/fusion_versions.json \
  --state-dir runtime/state
```

Run again with the same state directory to process the pending queue first.
`--full` ignores prior reusable state for this invocation. Normal incremental
runs intentionally accept edited inputs; use `--strict-recovery` when an input
hash mismatch must produce `E_SNAPSHOT_STALE` and force a blank rebuild.

The default report is `<state-dir>/report.json`. It contains the class-grid
summary, failures and injection IDs, budget deduction sequence, version
convergence records, final reuse/recompute ledger, snapshot version and fusion
version.

## 3. Apply a patch batch

```bash
python3 -m remote_seg patch \
  --index sample_data/index.json \
  --plan sample_data/patch_plan.json \
  --batch 1
```

The command validates the tile row/column and patch rectangle before replacing
the PNG bytes. An out-of-range rectangle raises `E_PATCH_BOUNDS`.

## 4. Run the full acceptance example

```bash
python3 -m remote_seg demo --out runtime/demo
```

The demo copies the sample workspace, applies all operations in
`configs/edit_sequence.json` one by one, runs incremental inference after each
operation, performs a blank full rebuild for every edit prefix, and then performs
one final full rebuild. `runtime/demo/verification.json` contains the per-edit
convergence history, all prefix comparisons, and the final state comparison. A
passing run prints `"status": "pass"`.

## 5. Materialize and run fault injections

First create a mutable source workspace:

```bash
mkdir -p runtime/injection-source
cp -R sample_data/tiles sample_data/pool \
  sample_data/index.json sample_data/fusion_versions.json \
  sample_data/patch_plan.json runtime/injection-source/
cp configs/segmentation.json runtime/injection-source/segmentation.json
```

Choose one profile, for example:

```bash
python3 -m remote_seg inject \
  --source runtime/injection-source \
  --out runtime/injection-workspace \
  --plan configs/injections.json \
  --profile combined_tile_faults

python3 -m remote_seg run \
  --index runtime/injection-workspace/index.json \
  --config runtime/injection-workspace/segmentation.json \
  --fusion-versions runtime/injection-workspace/fusion_versions.json \
  --state-dir runtime/injection-state \
  --report runtime/injection-report.json \
  --drain
```

Available profiles:

- `truncate_png` -> `E_TILE_DECODE`
- `band_count_mismatch` -> `E_TILE_SHAPE_MISMATCH`
- `patch_bounds` -> `E_PATCH_BOUNDS`
- `unknown_band_version` -> `E_BAND_VERSION_UNKNOWN`
- `unknown_fusion_version` -> `E_FUSION_VERSION_UNKNOWN`
- `combined_tile_faults` -> all four tile-local faults at once

Each item ID is copied into `injection_manifest.json` and appears in
`injection_hits` / `failures` in the run report.

## 6. Recover and compare snapshots

Inspect the only recoverable snapshot point:

```bash
python3 -m remote_seg recover --state-dir runtime/state
```

Strict recovery check against current inputs:

```bash
python3 -m remote_seg recover \
  --state-dir runtime/state \
  --config configs/segmentation.json \
  --index sample_data/index.json \
  --strict
```

Compare two state directories field by field:

```bash
python3 -m remote_seg compare --left runtime/inc-state --right runtime/full-state
```

The comparison covers per-pixel global classes, tile classes/scores, feature and
score summaries, fusion region votes, postprocess backfills, and the canonical
reuse ledger.

## 7. Graceful termination

Send `SIGINT` or `SIGTERM` to a running `run` command. The engine stops adding
new tile tasks, waits for in-flight tile tasks, commits the complete snapshot,
reports `"status": "graceful_stop"`, and exits with code `0`.

## 8. Tests

```bash
pytest -q
```

The suite covers PNG round trips, loader error codes, deterministic
classification and ties, weighted fusion, budget resumption, serial/parallel
equivalence, version invalidation, snapshot recovery/staleness, partial
generation safety, SIGTERM handling, all injection profiles, and incremental
versus full rebuild equality through the complete edit sequence.
