# Deterministic Time-Series Backtest Pipeline

This workspace contains a local, deterministic time-series forecasting pipeline and result-verification tool. It is designed for Windows command-line execution with Python 3.10+ and NumPy. No container, service, network call, database, or third-party YAML package is used.

The implementation covers two coupled concerns:

- reproducible windowed backtesting, baseline training, fixed-point metrics, incremental recomputation, and crash-safe snapshots;
- isolated real-input fault injection, stable failure ordering, and field-level incremental/full parity evidence.

## Deliverables

- `ts_pipeline/` — pipeline implementation.
- `tests/test_pipeline.py` — standard-library `unittest` suite.
- `data/` — deterministic JSONL samples, YAML config, and injection manifest.
- `reports/full_report.json` — a full-run report for `data/config.yaml`.
- `verification/` — recorded demo summary, parity reports, and unittest output.
- `scripts/run_windows_checks.bat` — Windows one-command test/demo entry point.
- `requirements.txt` — NumPy dependency only.

## Environment

Verified locally with Python `3.11.0rc1` and NumPy `1.26.4`. The code targets Python 3.10 or newer.

Windows `cmd.exe`:

```bat
python --version
python -m pip install -r requirements.txt
python -c "import numpy; print(numpy.__version__)"
```

## Quick Start

Generate deterministic fixtures:

```bat
python -m ts_pipeline generate --out data --seed 20261122 --force
```

Run a clean backtest:

```bat
python -m ts_pipeline run ^
  --config data\config.yaml ^
  --state-dir .pipeline_state ^
  --report reports\full_report.json ^
  --reset
```

Run all verification scenarios, including incremental edits, a fresh full rebuild, repeat-cache reuse, snapshot interruption/resume, concurrency 1 vs 4, quota exhaustion, and budget exhaustion:

```bat
python -m ts_pipeline demo --root demo_workspace --seed 20261122 --force
```

Run tests:

```bat
python -m unittest discover -s tests -v
```

Or run both:

```bat
scripts\run_windows_checks.bat
```

`run` exits `0` for a completed run, including runs that contain isolated window/model failures. Fatal load or configuration errors return `2`. A handled interrupt commits the current stage and returns `0` with `status: interrupted`.

## Input Formats

### Samples (`data/samples.jsonl`)

One JSON object per line. Duplicate IDs are rejected. Values must be finite numbers or a missing marker (`null` or `"MISSING"`), and timestamps must be strictly increasing with a constant interval.

```json
{"sequence_id":"s-alpha","timestamps_ms":[1700000000000,1700000001000],"values":[13.7,null],"missing":[false,true]}
```

Invalid lines emit `E_SAMPLE_INVALID`; valid sequences continue to be processed.

### Configuration (`data/config.yaml`)

The bundled YAML loader supports the subset used here: nested mappings, block/inline lists, quoted scalars, numbers, booleans, null, and comments. JSON is also accepted.

```yaml
schema_version: 1
samples: samples.jsonl
injection_manifest: injections.json
backtest:
  window_length: 12
  step: 4
  forecast_horizon: 3
  error_tolerance: 0.65
models:
  enabled: [naive_persistence, moving_average, autoregressive]
  moving_average_window: 3
  feature_lag_order: 3
  autoregressive_regularization: 1.0e-6
quantization:
  fixed_decimal_places: 3
  quantization_level: 1
scheduler:
  concurrency: 4
  batch_point_budget: 500000
  per_series_point_quota: 500000
snapshot:
  commit_interval_windows: 2
  version: 1
```

Allowed values:

- `fixed_decimal_places`: `0..6`
- `quantization_level`: `1, 2, 5, 10, 20, 50, 100`
- all three baseline models must be enabled
- `step <= window_length`

Core configuration errors emit `E_CONFIG_INVALID`. Invalid injection-list entries also emit `E_CONFIG_INVALID` and are skipped while valid injections continue.

### Injection Manifest (`data/injections.json`)

Injection is independent from the sample file. A valid item targets an existing sequence/window and is recorded by ID in the report.

```json
{
  "injections": [
    {"id":"INJ-001","kind":"non_finite_observation","sequence_id":"s-beta","window_index":1,"position":4,"value":"NaN"},
    {"id":"INJ-002","kind":"irregular_timestamp","sequence_id":"s-gamma","window_index":2,"position":5,"offset_ms":17},
    {"id":"INJ-003","kind":"window_too_long","sequence_id":"s-alpha","window_index":3,"window_length":99},
    {"id":"INJ-004","kind":"quantization_out_of_range","sequence_id":"s-delta","window_index":1,"model":"moving_average","quantization_level":999}
  ]
}
```

| Kind | Mapped business code |
|---|---|
| `non_finite_observation` | `E_SAMPLE_INVALID` |
| `irregular_timestamp` | `E_SAMPLE_INVALID` |
| `window_too_long` | `E_CONFIG_INVALID` |
| `quantization_out_of_range` | `E_CONFIG_INVALID` |

A window fault affects only its target window. A quantization fault can target one model or all models. Failures carry sequence ID, window index, model name, stage, code, and injection ID. Failed windows are excluded from metric aggregation.

## Pipeline Semantics

### Windows and imputation

For window index `i`, training indices are `[i*step, i*step+window_length)` and forecast indices follow immediately for `forecast_horizon` points. Only windows with a complete forecast range are scheduled.

Missing training values are filled deterministically inside the same window:

- linear interpolation when observations exist on both sides;
- carry-forward or carry-backward at an edge;
- fail the window when the window contains no observed training value.

Every fill records its position, method, source positions, source window, and filled value. Window construction never reads training data beyond the current training window.

### Baseline models

All models implement the same `fit(...) -> artifact` / `predict(artifact, horizon) -> list[float]` interface. Each artifact explicitly contains:

- the complete training window and its content fingerprint;
- fitted parameters;
- the residual sequence and residual fingerprint.

Models:

- `naive_persistence` — repeats the last training value.
- `moving_average` — uses `moving_average_window` from config.
- `autoregressive` — uses `feature_lag_order`, intercept and lag coefficients. The fixed normal equations include the configured ridge regularization. Singular systems fall back deterministically to a pseudo-inverse rather than throwing.

No model accesses global random state.

### Quantization and metrics

Predictions and actual values are converted to fixed-point buckets using:

```text
step = quantization_level / 10^fixed_decimal_places
bucket = round_half_up(value / step)
```

MAE, RMSE, and MAPE are computed from fixed-point integer errors, then scaled by `step`. Window traversal is stable: series ID ascending, window index ascending, step index ascending. Each window records the first step exceeding tolerance and its absolute, squared, and percentage error components. `metric_matrix` presents all three models side by side for each window.

### Cache keys and invalidation

The key dependency chain is:

```text
series content
  -> window content + window spec + window fault fingerprint
  -> per-model training key (model parameter fingerprint)
  -> prediction key (horizon + quantization fingerprint)
  -> metric key (actual fingerprint + tolerance)
```

Changing a sequence invalidates only its windows. Changing one model's hyperparameters leaves the other models' training artifacts reusable. Changing quantization reuses all training artifacts and recomputes predictions/metrics. Changing window length or step invalidates window splitting and downstream artifacts. `incremental_audit` records invalidated/unchanged tasks and reused/computed layers.

### Scheduler, budget, and quota

A task is identified by the stable key:

```text
<sequence_id>#W<eight-digit window index>#<model name>
```

Each model/window task logically costs `window_length + forecast_horizon` observation points. Budget decisions are made in stable series/task order before threads run, so cache hits do not alter budget outcomes and thread timing cannot change results.

- Series allocation proceeds in stable sequence-ID order.
- Exceeding `per_series_point_quota` aborts the rest of that series with `E_SERIES_QUOTA_EXHAUSTED`.
- Exhausting `batch_point_budget` marks unfinished tasks with `E_POINT_BUDGET_EXHAUSTED` and stores `pending_cursor`.
- Training and prediction execute in a bounded `ThreadPoolExecutor`; results are collected and sorted by stable task key.

### Snapshots and interruption

Stages are `load`, `windows`, `models`, and `metrics`. Each checkpoint is written into a temporary generation directory and then published by atomically replacing the manifest. The manifest records:

- snapshot schema/version and generation;
- config, injection, and sample hashes;
- completed series/windows/tasks and pending cursor;
- consumed budget;
- stage product versions;
- committed cache inventory.

At startup, snapshot version or config-hash mismatch produces `E_SNAPSHOT_STALE` and rebuilds stage products. Committed content artifacts whose keys are unchanged are still reusable.

Only artifacts listed in the last fully published generation inventory are read. A forced process kill can leave orphan temp/cache files, but they are ignored on restart. A handled `SIGINT`/`SIGTERM` finishes the current stage, commits it, writes an interrupted report, and exits `0`.

## Report

Reports are JSON and include:

- per-model window predictions, training windows, fitted parameters, residual sequences, quantized points, MAE/RMSE/MAPE, tolerance flags and first-exceeded details;
- global imputation records and fixed-point window artifacts;
- `metric_matrix` and over-tolerance window lists;
- injection manifest items, hit IDs, and business-code mapping;
- stable failure list and scheduler status;
- cache/invalidation audit and cache plan;
- snapshot version/status and per-stage timings;
- sample/config/injection hashes.

`python -m ts_pipeline compare --left A.json --right B.json` compares result-bearing fields. Execution-only fields such as timings, generation IDs, snapshot status, concurrency, and incremental history are intentionally excluded. The comparison covers predictions, metrics, imputation, failure order, cache plan, injections, and scheduler outcomes.

## Actual Verification

Recorded in `verification/test_results.txt`:

```text
Ran 9 tests in about 13.6s
OK
```

Clean run on `data/config.yaml` (`reports/full_report.json`):

- exit code `0`, status `completed`;
- 4 sequences, 25 valid windows, 75 model tasks, 74 successful model windows;
- 4 injection hits, no unhit injections;
- failures: two `E_SAMPLE_INVALID` and two `E_CONFIG_INVALID`, sorted by stable key;
- aggregate MAE/RMSE/MAPE:
  - naive persistence: `1.692284 / 2.051337 / 7.721815%`
  - moving average: `2.100028 / 2.408785 / 9.887435%`
  - autoregressive: `0.979203 / 1.293960 / 4.697976%`

End-to-end demo (`verification/demo_summary.json`):

- incremental edit sequence vs fresh full rebuild: 0 differences;
- repeat incremental vs full: 0 differences;
- interrupted after `windows`, resumed, vs full: 0 differences;
- concurrency 1 vs 4: 0 differences;
- quota run emitted `E_SERIES_QUOTA_EXHAUSTED`;
- budget run emitted `E_POINT_BUDGET_EXHAUSTED` and preserved a pending cursor;
- quant-only edit reused all 75 existing training artifacts and recomputed 0 trainings;
- sequence replacement invalidated 18 tasks and reused 57 unchanged tasks;
- removing one committed cache-inventory entry caused exactly that window to recompute while the rest were reused;
- single-model hyperparameter edit invalidated only `moving_average` and `autoregressive` tasks while reusing `naive_persistence`.
