# PageAgg

Deterministic columnar-page grouped aggregation for a local Windows environment. The project models fixed-size local pages, executes two explicit aggregation routes under an integer cache-slot budget, persists every committed batch to SQLite, verifies replay consistency, and includes a Tkinter dashboard.

No container runtime, network service, or external dataset is required at runtime.

## Environment

Required:

- Python 3.11 or later
- `numpy`
- Standard-library `sqlite3` and `tkinter`

Windows quick start:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import numpy, sqlite3, tkinter; print(numpy.__version__, sqlite3.sqlite_version, tkinter.TkVersion)"
```

Run tests and the complete acceptance suite:

```powershell
python -m pytest -q
python scripts/acceptance.py
```

The acceptance run writes:

- `artifacts/acceptance_report.json`
- `artifacts/VALIDATION.md`
- `artifacts/acceptance/` (databases, samples, fault copies, raw CLI responses)

## Quick Commands

Generate the deterministic sample:

```powershell
python -m pageagg generate --out data/sample --force
```

Run resident aggregation and persist a ledger snapshot:

```powershell
python -m pageagg aggregate `
  --catalog data/sample/catalog.json `
  --db artifacts/resident.db `
  --budget 4 `
  --route resident
```

Force the rescan route and compare byte-level outputs:

```powershell
python -m pageagg compare-routes `
  --catalog data/sample/catalog.json `
  --budget 4 `
  --out-dir artifacts/results/compare
```

Replay committed batches and independently validate the ledger:

```powershell
python -m pageagg replay --db artifacts/resident.db
python -m pageagg validate --db artifacts/resident.db
```

Open the Tkinter dashboard:

```powershell
python -m pageagg ui --db artifacts/resident.db
```

## Sample Dataset

The generator writes eight pages with 24 rows each. Columns are:

- `group_id`: integer, never null, used as the group key
- `value_int`: integer with deterministic nulls
- `value_float`: finite float64 with deterministic nulls
- `flag`: integer with deterministic nulls

The 12-access scan sequence is:

```text
[1, 2, 3, 1, 4, 2, 3, 5, 6, 1, 7, 8]
```

With slot budget 4, LRU produces 3 cache hits, 9 physical scans, and 5 evictions. The route score is `375000 ppm`, inside the declared boundary window `[350000, 450000]`; the stable tie rule keeps `RESIDENT`.

## Page Format

Each `.cpg` page is a deterministic little-endian binary columnar file:

```text
magic "CPAGE001" | u8 version | i64 page_id | u32 rows | u16 ncols
for each column:
  u16 name_len | utf-8 name | u8 type
  rows * 8 bytes values (int64 or float64)
  rows bytes null bitmap (each byte exactly 0 or 1)
  u32 CRC32 of the null bitmap
```

`catalog.json` declares the shared column schema, fixed rows per page, group key, batch size, route thresholds, scan sequence, page paths, row counts, and SHA-256 of each complete page file. `dataset_manifest.json` records the catalog digest and page digests.

Validation rejects the entire load at the first violating page:

- duplicate or malformed page IDs
- missing/unreadable/trailing page content
- page header/catalog row or column mismatch
- non-finite float values
- group-key nulls
- page content digest mismatch
- scan references to undeclared pages
- unsafe page paths outside the catalog directory

Business codes:

| Code | Meaning | Exit |
|---|---|---:|
| `E_PAGE` | catalog/page/layout/digest failure | 10 |
| `E_NULLMAP` | null bitmap length, byte, or CRC failure; category is still `E_PAGE` | 11 |
| `E_BUDGET` | slot budget `< 1`; includes minimum feasible slots | 12 |
| `E_ROUTE` | resident/rescan/eviction plan conflict | 13 |
| `E_REPLAY` | ledger replay or first-field mismatch | 14 |

Every business error is appended as one JSON object per line to `artifacts/logs/business.jsonl` unless `--log-path` is supplied.

## Route Semantics

### Resident route

The route uses a deterministic LRU slot plan. On a miss with a full cache, the victim is selected by `(last_access_seq, page_id)`, with `page_id` as the stable tie-breaker. Each eviction records event sequence, victim page ID, incoming page ID, slot, and access sequence.

A repeated access while the page remains resident is a cache hit and does not reread the page file.

### Rescan route

The rescan route never retains a physical resident page and always reports zero cache hits. Every scan access reads the page file and emits a rescan event with page ID, event sequence, and an explicit reason.

Both routes share the same canonical LRU slot plan. The resident route realizes it physically; the rescan route emits the identical eviction-event bytes as a shadow plan for cross-route consistency checks. Rescan page outcomes are labeled with `occupancy_kind = "shadow_plan"`, while resident outcomes use `"actual"`. This makes the required eviction sequence byte-identical without counting any rescan page as resident or as a hit.

### Route rule

All values are integers; no floating-point decision is used.

```text
overflow_pages      = max(0, distinct_pages - budget)
overflow_ratio_ppm  = overflow_pages * 1_000_000 // distinct_pages
pressure_ppm        = overflow_ratio_ppm * rows_per_page // rows_reference

if pressure_ppm > threshold_ppm: RESCAN
else: RESIDENT
```

At exact equality, the stable tie rule selects `RESIDENT`. The output includes the reason, all key scale parameters, the threshold, lower/upper boundary, delta, and whether the decision sits inside the declared boundary margin.

## Fixed-Point Aggregation

Rules are versioned as `fixed-s64-scale1e6-trunc-toward-zero-v1`:

- Scale: `1_000_000` fixed-point units per unit of data.
- Integer values: exact after scaling.
- Float values: truncated toward zero at the scale boundary.
- Per-value float quantization error: strictly less than one fixed-point unit.
- Sum: sticky signed-64-bit saturation.
- Mean: integer division truncated toward zero.
- Accumulation order: the declared page scan sequence, independent of cache hit/rescan timing.

Each group and numeric column reports row count, non-null value count, null count, scaled sum/decimal sum, scaled mean/decimal mean, saturation status, per-page quantization terms in scan order, sum error bound, mean error bound, and explicit truncation/saturation/error-growth rules.

`mean_error_bound_units = ceil(float_values / value_count) + 1`. No binary float is serialized into the result view. Canonical JSON bytes and SHA-256 digests are used for byte-level comparison.

## Evolution and Whole-View Replacement

`aggregate` compares the active snapshot configuration with current disk state. A changed catalog digest, row count, group key, route, batch size, or budget creates a new monotonic plan version and deletes the old snapshot's aggregate/event state in the replacement transaction, then persists the new view from the beginning.

`evolve` additionally reports stable propagation steps, affected pages/groups, route-only affected pages/groups, added/removed/changed/byte-unchanged/unaffected groups, and old/new per-group digests for unaffected groups.

An unaffected group whose bytes change is treated as a propagation defect and reported as `E_REPLAY`.

```powershell
python -m pageagg evolve `
  --db artifacts/resident.db `
  --catalog data/sample/catalog.json `
  --budget 2
```

## SQLite Ledger and Crash Recovery

The ledger has an integer format version (`1`) and a SHA-256 content digest over all ledger state. Each batch commit is one SQLite transaction containing page outcomes, eviction/rescan events, cumulative per-group fixed-point state, fixed-point version, progress cursor, and batch/state digests.

Before each batch transaction, a separate committed `STARTED` attempt row is written. The batch data and `COMMITTED` status then commit atomically. If the process is killed inside the transaction, SQLite rolls back the data and the next open marks the leftover attempt `UNKNOWN`. Recovery trusts only committed batches and returns `forced_exit_report: null`, `forced_exit_report_available: false`, and `unknown_batches_require_rerun: true`.

The test hook `--crash-during-batch N` uses `os._exit(137)` inside the named uncommitted transaction. A subsequent `aggregate` verifies the committed prefix, resumes at the committed cursor, and completes remaining batches.

## Dashboard

`python -m pageagg ui --db ...` displays page outcomes, actual/shadow slot occupancy, group sums/means and a quantization-error heatmap, route decision details, evictions/rescans, ledger format/version/status/cursor/content digest, and the latest structured business error code.

## Repository Layout

```text
pageagg/
  catalog.py        catalog + page validation
  page_format.py    binary page reader/writer
  planner.py        LRU slot plan and route decision
  fixedpoint.py     fixed-point estimates and error bounds
  engine.py         resident/rescan execution and batching
  ledger.py         SQLite schema, batch commits, replacement
  replay.py         first-difference replay and validator
  service.py        aggregate/resume/evolve orchestration
  ui.py             Tkinter dashboard
  cli.py            command-line interface
scripts/
  acceptance.py     one-command end-to-end evidence run
tests/              22 automated tests
data/sample/        deterministic local sample
artifacts/          generated verification evidence (not required source)
```

See `docs/DESIGN.md` for invariants, transaction boundaries, and replay rules.

## Verified Results

Last local run in this workspace:

- `python -m pytest -q`: **22 passed**
- `python scripts/acceptance.py`: **12/12 checks passed**
- Resident/rescan canonical result SHA-256: `888a8c2187e5d5af03c5b7cc0efc719f27b5b2b389e96f48ca9d4b542de300db`
- Shared eviction sequence digest: `79d80c4663ff74135c12d5f0f0a82669b6b073ccb32381f716845bb2a8ec3b91`
- Resident: 3 cache hits, 9 physical scans, 5 evictions
- Rescan: 0 cache hits, 12 physical scans, 12 rescan events
- Snapshot evolution: plan version `1 -> 2`, whole replacement, unaffected group digests unchanged
- Forced exit: exit code `137`, recovered cursor `3`, unknown batch recorded, resume verified, validator passed
- Anomaly codes: `E_PAGE`, `E_NULLMAP`, `E_BUDGET`

Exact evidence is in `artifacts/acceptance_report.json` and `artifacts/VALIDATION.md`.
