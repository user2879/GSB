# Architecture

## 1. Layers

| Layer | Modules | Responsibility |
|---|---|---|
| Declaration | `loader.py`, `models.py` | Parse local JSON, validate all-or-nothing, compile a deterministic stage chain |
| Identity | `canonical.py` | Canonical JSON, dataset/pipeline digests, composite fingerprint, stage/shard digests |
| Computation | `compute.py` | Integer fixed-point transformations and batch observations |
| Ledger | `ledger.py` | SQLite schema, transactions, spec-version allocation, view replacement, audit reads |
| Execution | `engine.py` | Run lifecycle, batch commits, stage commits, shard binding, evolution report |
| Recovery | `recovery.py` | Interrupted-run inspection, committed-prefix revalidation, resumption, full replay |
| Evolution | `evolution.py` | Dependency impact, first-change detection, shard invalidation/reuse |
| Interfaces | `cli.py`, `ui.py`, `demo.py` | Repeatable CLI/demo and Tkinter diagnostic application |

The runtime dependency graph is one-directional: `ui/cli/demo -> engine/recovery -> ledger/compute/evolution/loader -> models/canonical/errors`.

## 2. Dataset model

`mlpipe.dataset.v1` contains ordered `sources`. Each source has an ID, explicit `rows`, and ordered columns. A column has:

- `name`;
- `type`: `int` or `float`;
- `values`: explicit JSON integers, finite numbers, or decimal strings;
- `nulls`: an equal-length bitmap of `0` / `1`.

Validation rejects booleans as integers, duplicate source/column IDs, unsupported types, value length mismatches, non-finite decimals, out-of-range signed 64-bit integers, malformed bitmaps, and invalid JSON. The first violation is surfaced as `E_SPEC` before any spec/run row is allocated.

Sources are concatenated in declaration order. A source-local column is padded with nulls outside its global row interval. This gives every stage column the same global row count while preserving source identity in the qualified key (`source.column`).

The canonical dataset digest masks no information: it includes ordered sources/columns, canonical values, and null bitmaps. Decimal spellings such as `1.0` and `1.00` canonicalise to `1`; masked runtime values are marked by the bitmap.

## 3. Pipeline compilation

`mlpipe.pipeline.v1` declares:

- `fixed_scale`: integer decimal exponent `0..6`;
- `sharding.target_rows` and `sharding.remainder_policy`;
- a stage array.

Every stage has an ID, supported type, positive integer version, null/string upstream, and an exact integer-only params object. Compilation:

1. verifies references and stage-ID uniqueness;
2. detects cycles with a deterministic DFS over sorted stage IDs;
3. enforces one root, one successor per stage, and a single connected chain;
4. enforces phase order `canonicalize -> fill_missing -> normalize -> clip_outliers -> shard_pack`;
5. applies type-specific parameter bounds;
6. assigns chain ordinals and per-stage config fingerprints.

The stage array order in the JSON file is not semantic. The unique valid chain determines the compiled order. Each stage config fingerprint includes ID, type, version, upstream, and params, so a version or upstream change is observable even when output happens to remain equal.

## 4. Fingerprint recipe

The byte contract is implemented in `canonical.py` and documented there:

```text
canonical(x) = UTF-8 JSON, sorted object keys, compact separators, sequence order preserved

dataset_digest  = SHA256("MLPIPE-DATASET1\0"  || canonical(dataset))
pipeline_digest = SHA256("MLPIPE-PIPELINE1\0" || canonical(pipeline))
fingerprint     = SHA256("MLPIPE-SPEC1\0"
                         || frame("dataset",  dataset_digest)
                         || frame("pipeline", pipeline_digest))

frame(name, value) = len(name):u32be || name || len(value):u64be || value
```

`pipeline` in this recipe is semantic: ordered compiled stages with normalised integer params plus fixed scale and sharding. It is not the raw file byte stream. Formatting, key order, stage-array order, and dictionary hash seed therefore cannot affect a fingerprint.

## 5. Fixed-point computation

`compute.apply_stage` receives the current table, one compiled stage, fixed scale, sharding declaration, and batch size. Before mutating rows it computes any global statistics. It then processes fixed row intervals and returns one `BatchObservation` per interval.

Rules:

- `canonicalize`: integer value `v -> v * 10**scale`; decimal `d -> trunc_toward_zero(d * 10**scale)`; nulls remain null. Signed 64-bit overflow saturates when `saturate=1`, otherwise raises `E_PARAM`.
- `fill_missing`: replaces every null bit and value with `fill_fixed`.
- `normalize`: for target column min/max, `out = ((value - min) * 10**normalize_scale) // (max - min)`; a constant column maps to zero and the exponent is a session integer in `0..6`. The value is stored on the run and conclusion view and is reused by recovery/replay. It is execution context, not a declared stage parameter, so it does not enter the specification fingerprint.
- `clip_outliers`: integer mean `sum // n`; variance `sum((x-mean)^2) // n`; standard deviation `isqrt(variance)`; bounds `mean ± floor(z_limit_permille * std / 1000)`. If outlier ratio exceeds `max_outlier_permille`, execution stops with `E_OUTLIER` before mutating that stage. Otherwise values are clamped.
- `shard_pack`: does not transform bytes; it generates ordered shards from the final table.

No Python float participates in execution or statistical decisions. Source decimals are parsed as `Decimal`, then converted to integers.

Batch observations contain content-only row-interval digests. Because statistics are globally computed before row batches, outputs and final stage digests are independent of `batch_size`.

## 6. Shards

For every shard:

```text
rows_digest       = digest of columns and values in [start_row, end_row)
shard_fingerprint = SHA256(canonical({
  stage_id, source_stage_version, start_row, end_row, row_count, rows_digest
}))
```

The DB record also stores the source stage's full output digest, current spec version, and current spec fingerprint. A shard is stale if its source stage ID/version or bound spec fingerprint differs from the current compiled plan. Old-spec rows remain historical and are not exposed in the current view.

Evolution can reuse an unchanged shard's row bytes and fingerprint while rebinding it to the new spec. Reuse requires equal stage ID, stage version, row interval, row count, and shard fingerprint. Any change to a shard's data changes `rows_digest` and therefore its shard fingerprint.

Remainder semantics are explicit:

- `keep_partial`: retain the final shorter shard;
- `drop_partial`: omit it.

## 7. SQLite schema

Primary tables:

| Table | Purpose |
|---|---|
| `specs` | Monotonic revision, fingerprint, canonical dataset/pipeline snapshots, compiled plan |
| `runs` | Run status, spec binding, base-view binding, batch size, normalize scale, final digest |
| `stage_runs` | Per-stage attempt, status, versions, input/output digests, row count |
| `stage_batches` | Per-batch row interval, input/output digests, spec version, fingerprint, trusted flag |
| `stage_progress` | Committed cursor for each stage attempt |
| `shards` | Ordered shard records, fingerprints, stage/spec binding, reuse source |
| `conclusion_views` | Single current conclusion (`singleton = 1`) |
| `conclusion_history` | Append-only replaced views |
| `evolution_reports` | Old/new identity, affected stages, shard invalidation |
| `business_errors`, `events` | Structured business failures and audit events |
| `reports` | Inspection, recovery, and replay reports |

SQLite uses WAL, `synchronous=FULL`, a busy timeout, and `BEGIN IMMEDIATE` for spec-version allocation and conclusion replacement. Batch insertion and cursor update are one transaction. Stage finalisation and shard insertion are one transaction. Run completion and singleton-view replacement are one transaction.

## 8. Evolution and invalidation

When a run is created, the ledger's current conclusion is recorded as its base view. A new spec is compared with the base spec snapshot:

1. dataset digest and fixed scale affect the whole data path;
2. sharding parameters affect the packing stage;
3. changed stage type/version/params/upstream affects that stage;
4. any downstream stage inherits `upstream_affected`.

The report lists config-changed stages with old/new config fingerprints separately from all affected stages. After execution, stage output digests are compared to the old view. The shard matcher computes a multiset of old shard identities and marks a new shard as preserved only on an exact interval/version/fingerprint match.

`enrich_stage_outputs` explicitly checks that every logically unaffected stage retains its old output digest. A mismatch is recorded in `unaffected_output_mismatches` and would be a correctness defect.

## 9. Recovery state machine

```text
RUNNING
  |-- all stages COMMITTED + completion commit ---> COMPLETE / RECOVERED_COMPLETE
  |-- abrupt termination -------------------------> RUNNING on disk
  |-- stage business error -----------------------> FAILED

inspect(): RUNNING lost stage attempts -> UNKNOWN, batches trusted=0, run INTERRUPTED
recover(): verify committed prefix -> rerun from first unknown attempt -> finalize -> replay
```

An interrupted stage never contributes a partial output to the recovered run. Its attempt rows remain for audit but `trusted=0`; the resumed run writes a new attempt. A stage absent from the ledger at inspection time is materialised as `UNKNOWN` so the report has a complete stage status set.

Recovery compares committed stage observations and batches while rebuilding the in-memory prefix. Full replay then compares:

1. each stage's status, version, input digest, output digest, row count;
2. each expected batch's interval, row count, input/output digest;
3. every shard's index, interval, row count, row digest, shard fingerprint, source output digest, source stage version;
4. the run's final data digest.

The first mismatch becomes `E_REPLAY` with `first_stage_id`, `first_field`, `expected`, and `actual`.

## 10. UI rendering contract

The Tkinter UI has three tabs:

- **管线总览**: compiled dependency chain, stage version/status/row count/output digest, and shard distribution/reuse/state.
- **SQLite 账本**: runs, business errors, and the selected run's latest batch observations.
- **执行 / 恢复报告**: append-only operation summaries including recovered/unknown/rerun lists and replay results.

Rows are always sorted by stage ordinal, shard index, descending run sequence, or error ID. Tree item IDs are stable stage IDs/run IDs where applicable. All execution runs on a worker thread; Tk widget updates are marshalled back to the main thread.
