# Technical Design

## Module map

- `remote_seg/pngio.py`: dependency-free 8-bit PNG reader/writer with all PNG
  row filters supported on read.
- `remote_seg/inputs.py`: config/index/fusion-registry validation, tile decode,
  band/shape checks, and cache-key construction.
- `remote_seg/classify.py`: deterministic sliding-window feature extraction,
  scoring, classification, summaries, and window task keys.
- `remote_seg/fusion.py`: pair-region fingerprints/traces, global weighted vote,
  tie resolution, connected-component filtering, and backfill audit records.
- `remote_seg/snapshot.py`: full-generation persistence, checksum verification,
  and atomic `CURRENT` pointer replacement.
- `remote_seg/engine.py`: exact invalidation, pending queue, budget scheduler,
  thread pool, failure isolation, graceful stop, and reports.
- `remote_seg/patches.py`, `edits.py`, `injections.py`: deterministic patch and
  edit semantics plus fault-profile materialization.
- `remote_seg/demo.py`, `verify.py`: end-to-end edit sequence and canonical
  incremental/full comparison.

## Input contracts

### Tile index

The index declares `index_version`, `band_order_version`, the fixed band order
`["red", "green", "nir", "elevation"]`, `tile_size`, and a list of tiles. Each
tile has a unique `(row, col)`, a unique `tile_id`, bounds with exactly the
declared size, a relative PNG path, a band order version, and symmetric
neighbors. Structural defects produce `E_INDEX_INVALID`.

PNG failures are isolated per tile:

- decode/truncation -> `E_TILE_DECODE`
- band count or decoded/index size mismatch -> `E_TILE_SHAPE_MISMATCH`
- undeclared band order version -> `E_BAND_VERSION_UNKNOWN`

Failed tiles are removed before classification and fusion. Remaining valid tiles
continue through the batch.

### Segmentation config

The config contains window size, stride, context pixels, four feature
thresholds, per-class scoring weights, minimum connected area, requested fusion
version, local recompute budget, and maximum workers. An undeclared fusion
version produces `E_FUSION_VERSION_UNKNOWN`.

### Fusion registry

The registry declares band-order versions and all historical fusion strategy
versions in effective order. A strategy defines distance weight power, scale,
and tie policy. The strategy version and parameters are part of every fusion
region fingerprint.

## Deterministic classification

Window origins traverse rows then columns. The last origin is pinned to the tile
edge so every pixel is covered. Each window expands by `context_overlap` for
feature computation, then contributes only its central area:

- vegetation: NDVI `(NIR-R)/(NIR+R)`
- water: NDWI `(G-NIR)/(G+NIR)`
- edge density: gradient magnitude of mean red/green luminance
- slope: elevation gradient magnitude

Thresholds convert raw features into clipped strengths. Per-class scores are the
bias plus weighted strengths. Window scores are accumulated per pixel and divided
by coverage. `argmax` therefore resolves equal scores by ascending class code.

The tile cache key is:

```text
SHA256(content fingerprint, band-order version, classification config fingerprint)
```

Tile summaries include raw/strength feature means and ranges, class histogram,
score distribution, and every window's stable task key and feature means.

## Fusion and tracing

Every adjacent tile pair with non-empty geometric overlap becomes a dependency
region. Its fingerprint contains both tile cache keys, overlap bounds, fusion
version, weight power, scale, and tie policy.

For each covered pixel:

- a single tile supplies its class directly;
- two or more tiles contribute weights based on the minimum distance to their
  own tile border;
- votes are summed by class;
- equal vote totals choose the class supported by the lexicographically first
  `(row, col, tile_id)` source; remaining ties use class code.

The snapshot stores both pair-region traces and global overlap-pixel traces.
Each overlap record lists source tiles, classes, weights, vote distribution,
pre-postprocess winner, and final class.

Connected components use four-neighbor connectivity. Components below
`min_connected_area` are scanned in row/column order and filled from stable
adjacent classes. Choice order is neighbor count, stable target area, then class
code. If no stable adjacent class exists, the globally dominant stable class is
used. Every action is recorded with original/target class and neighbor counts.

## Exact invalidation and budget scheduling

- A changed tile content fingerprint, band version, or classification config
  changes its tile cache key and dirties that tile.
- A fusion-version switch changes every region fingerprint.
- A tile change dirties only pairs containing that tile; adjacency edits add or
  remove only the affected regions.
- Postprocess-only changes reuse tiles and regions but rebuild the global result.

Dirty tasks are ordered by row/column (regions by participating tile order and
pair ID). Previous pending tasks are filtered to still-dirty items and placed
before newly dirty tasks. Exactly `recompute_budget` tasks are selected per
batch. Unselected tasks remain pending and set
`E_RECOMPUTE_BUDGET_PENDING`. No dirty task is silently skipped.

## Two ledgers

The final snapshot includes a canonical `reuse_recompute_ledger`. It records the
dependency state of every final tile and region (`cache_key`, `region_version`,
`state=valid`), not the historical order of cache hits. Therefore an incremental
result and a blank full rebuild from identical final inputs have byte-identical
canonical ledgers.

Actual execution events (`recompute`, `deferred_by_signal`) and the budget
deduction order are reported separately in `execution_ledger`; they are an audit
log and are not treated as final inference output.

## Snapshot protocol and recovery

Each commit writes a complete generation under `state/generations/`. All data
files are written first, their SHA-256 checksums and every tile/region version
are placed in `manifest.json`, and only then is `CURRENT` atomically replaced.
Partial generation directories are never selected.

`CURRENT` and the manifest must agree on the state fingerprint. Every referenced
file must exist and match its checksum. Corrupt or incomplete snapshots produce
`E_SNAPSHOT_STALE` and cause a blank rebuild.

Normal incremental runs deliberately accept edited configuration/index files.
For recovery expectations after a crash, `--strict-recovery` or
`recover --strict` compares config and index hashes and reports
`E_SNAPSHOT_STALE` on mismatch. This distinction preserves exact local
invalidation for legitimate edits while keeping recovery checks strict.

## Concurrency, stop and failure isolation

Tile classification uses `ThreadPoolExecutor(max_workers=config.max_workers)`.
At most `max_workers` tasks are in flight. Results are committed in sorted task
order, so worker count cannot affect outputs, budget order, or failures.

The SIGINT/SIGTERM handler is installed at process entry, before heavy imports.
On signal, no new tile tasks are submitted; in-flight tiles finish and commit,
unstarted tasks remain pending, the snapshot is published, status is
`graceful_stop`, and the process exits `0`.

Each tile task has its own exception boundary. Failures record row, column,
stage, business code, and injection ID, then the rest of the batch continues.

## Snapshot artifact layout

```text
state/
  CURRENT
  generations/g000000NN-<fingerprint>/
    manifest.json
    tiles/<tile>.npz
    tiles/<tile>.json
    regions/<pair>.json
    global.npz
    global.json
  report.json
```
