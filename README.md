# dbml: local SQLite migration and fixed-point precision ledger

`dbml` is a pure-Python, locally executed database migration component. It uses
one SQLite file as both the application store and the transactional source of
truth for a hash-chained migration ledger. No database service, Docker, or
network access is required.

The package covers:

- strict JSON migration manifests with schema-aware validation;
- integer fixed-point quantization and error-bound derivation;
- derived-column dependency chains with stable error paths;
- transactional schema migration and hard-kill recovery;
- before/after row-set drift verification with `E_DRIFT`;
- versioned session snapshots with whole-session replacement on mismatch;
- deterministic replay with per-step, per-column `E_REPLAY` diagnostics.

## Verified environment

- Python `3.11.0rc1` (requires Python 3.11 or newer)
- standard-library SQLite `3.37.2`
- NumPy `1.26.4`, pinned in `requirements.txt` and `pyproject.toml`
- Linux/WSL execution validated; all path handling uses `pathlib` and the code
  is intended for local Windows Python as well

## Install and run

POSIX shell:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_sample_db.py --output artifacts/sample_base.sqlite3
python3 scripts/demo.py
python3 -m unittest discover -v
```

PowerShell:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 scripts/build_sample_db.py --output artifacts/sample_base.sqlite3
py -3.11 scripts/demo.py
py -3.11 -m unittest discover -v
```

The package can also be installed as an editable integration package:

```bash
python3 -m pip install -e .
dbml-demo
```

The demo always deletes and recreates `artifacts/demo/ledger`,
`artifacts/demo/snapshots`, and `artifacts/demo/work` before running. It writes:

- `artifacts/demo/report.json`: stable, sorted end-to-end report;
- `artifacts/demo/ledger/ledger.json`: final authoritative ledger export;
- `artifacts/demo/snapshots/session-original.json`: post-migration snapshot;
- `artifacts/demo/snapshots/session-perturbed.json`: post-perturbation snapshot;
- deterministic SQLite copies under `artifacts/demo/work/`.

## Sample database

`dbml.sample.build_sample_database` creates 48 `orders` rows from fixed seed
`20261118` using `numpy.random.default_rng`. It never uses wall-clock time or
system randomness. The sample schema is:

```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL,
    tax_cents INTEGER NOT NULL,
    total_cents INTEGER NOT NULL,
    net_cents INTEGER NOT NULL,
    discount_rate REAL NOT NULL,
    note TEXT NOT NULL
);
```

The pre-migration `total_cents` and `net_cents` values are derived from
scale-4-truncated discounts. This makes the sample stable while still carrying
nonzero integer-grid error bounds through the migration chain.

The generated seed record contains the schema, row count, and semantic digest:

```text
semantic_digest: 8265aaa17976c82ee07ae1c4c26b24a6d2af14da28a1f1cc25f8ce3176bfaff3
table_content_digest: 01cdee9e2f425e4461ae52441fca1ad1628336190ad23ec674f8f267157f6bac
```

## Manifest

The executable sample is `data/migrations/orders_migration.json`. It performs:

1. `INTEGER -> DECIMAL` widening of `unit_price_cents` at scale 2;
2. backfill of `total_cents` from
   `unit_price_cents * quantity * (1 - discount_rate)`;
3. backfill of `net_cents` from `total_cents - tax_cents`;
4. creation of `ix_orders_customer`;
5. addition of `CHECK (quantity >= 1)`.

`data/expressions/orders_chain.json` is the reviewable expression-chain
description. It records the expected dependency edge
`orders.total_cents -> orders.net_cents` and the stable topological order.

A step has the following core shape:

```json
{
  "step_id": "backfill-total",
  "version": 2,
  "depends_on": ["widen-unit-price"],
  "operation": "backfill",
  "table": "orders",
  "column": "total_cents",
  "expression": "unit_price_cents * quantity * (1 - discount_rate)",
  "quantization": {
    "scale": 2,
    "truncation": "toward_zero",
    "saturation": "reject"
  }
}
```

Supported operations are `create_column`, `widen_column`, `rename_column`,
`backfill`, `add_constraint`, and `create_index`. Supported declared types are
`INTEGER`, `REAL`, `DECIMAL`, and `TEXT`; fixed-point operations require a
numeric type.

Validation rejects the whole manifest before any write when it finds:

- duplicate `step_id`, non-monotonic versions, self/unknown predecessor refs;
- cycles in the step DAG;
- unsupported types or scale outside `0..9`;
- invalid truncation/saturation declarations;
- expression syntax errors or references absent at that step's simulated schema;
- derived references not ordered by a declared predecessor;
- unsupported widenings or missing numeric calibrations;
- invalid index/constraint names or columns.

Failures use `E_MIGRATE` and carry `step_id` plus a JSON field path such as
`steps[1].quantization.scale`. Validation is read-only, so rejection cannot
partially execute the manifest.

## Fixed-point rules

Numeric values are represented in integer grid units:

- `INTEGER` and `DECIMAL` columns physically store scaled integer units;
- `REAL` values are read through their shortest decimal representation and then
  quantized, avoiding binary-float-dependent ledger output;
- each column declares one `QuantizationSpec`: scale, truncation, saturation,
  and optional integer saturation bounds.

Supported truncation directions are `toward_zero`, `floor`, `ceil`, and
`nearest_half_even`. Supported overflow rules are `reject` and `clamp`.
Multiplication and division in expression analysis use integer products,
integer division, and explicit rounding; no float arithmetic is used to derive
ledger bounds.

Three different quantities are recorded:

- single-value quantization error: the integer-grid bound around one value;
- column comparison bound: `before_error + after_error` on a common grid;
- expression bound: interval arithmetic plus sorted source-path contributions,
  with a synthetic `rounding` contribution when interval width exceeds the sum
  of named paths.

Error contributions always sum to the reported bound. Paths are normalized and
sorted lexicographically, not by dictionary or hash iteration. A row is
precision-equivalent iff:

```text
difference == 0 OR difference < before_error + after_error
```

Equality between a nonzero difference and the bound is conservatively reported
as drift because the contract is strictly `<`.

## Ledger and recovery

Internal ledger tables live in the same SQLite file:

- `_mig_run`: manifest digest, stable plan, base schema digest, current attempt;
- `_mig_ledger`: step id/version/operation, row count, before/after column
  calibrations, per-column error reports, content digest, and hash-chain links;
- `_mig_column_meta`: current calibration and calibration version per column;
- `_mig_row_errors`: row-level bound and error-path payload for derived values;
- `_mig_constraints`: package-managed checks and unique constraints;
- `_mig_recalibrations`: scale perturbations and recomputed dependent backfills.

Each migration step is a single `BEGIN IMMEDIATE` transaction containing DDL,
data changes, row-error rows, and the committed ledger record. A hard kill
therefore either leaves the entire step committed or leaves none of it. The
next process only trusts ledger rows:

- committed steps are skipped;
- missing plan steps are returned as `unknown_steps` with
  `requires_rerun: true`;
- an attempt interrupted inside a step simply rolls back and is rerun.

The test harness kills a worker with `os._exit(77)` after `backfill-total`. The
restart sees two committed steps and three unknown steps, resumes with no
duplicate execution, and completes all five steps. No completion report is
written by the killed process.

## Drift verification

`verify_drift` compares ordered row sets by primary key or a supplied row key.
For every numeric column it converts before/after values to a common integer
grid, sums the two error bounds, and records:

- compared rows and row-set mismatches;
- nonzero deviation rows and out-of-bound rows;
- maximum absolute difference and maximum comparison bound;
- first nonzero deviation and first out-of-bound primary key;
- row-level error paths and the step that produced them.

By default an out-of-bound result raises `E_DRIFT`; pass
`raise_on_drift=False` to receive the full report instead. The report is a pure
function of row keys, calibrations, and ledger errors, so changing execution
order or batch size does not change `report_digest` (covered by tests).

## Snapshots and restore

Snapshots store:

- schema version;
- per-table row counts and content digests;
- all column calibration versions;
- the derivation graph and its version;
- the complete ledger/recalibration view and ledger version;
- all supplied verification conclusions and a conclusion version;
- the embedded manifest/plan and the base database fingerprint.

On restore the snapshot versions are compared with the current database. A full
match restores the saved session. Any mismatch replaces the entire session and
ledger view; old verification conclusions are not merged. Differences are
reported as stable sorted version names, affected `table.column` names, and
affected step ids.

## Replay

`replay_snapshot` copies the base database to a fresh local file, re-executes
the embedded manifest, reapplies any recorded scale recalibrations, recomputes
drift conclusions, and compares:

- step identity, version, row count, content digest, and operation metadata;
- per-step before/after column calibrations;
- per-step per-column bounds and error paths;
- recalibration events;
- final drift conclusions per column.

The first mismatch raises `E_REPLAY` with the first differing `step_id` and
column. A base database whose schema/data digest differs from the snapshot is
reported as step `__base__`, column `*`.

## Python API

```python
from pathlib import Path
from dbml import (
    MigrationRunner,
    SnapshotManager,
    compute_error_report,
    load_manifest,
    perturb_scale,
    replay_snapshot,
    restore_snapshot,
    verify_drift,
)

base = Path("artifacts/sample_base.sqlite3")
working = Path("artifacts/orders.sqlite3")
manifest = load_manifest("data/migrations/orders_migration.json", working)

result = MigrationRunner(working, manifest, batch_size=64).run()
report = compute_error_report(working, table="orders")
drift = verify_drift(
    base,
    working,
    table="orders",
    columns=["total_cents", "net_cents"],
    before_calibrations=manifest,
    row_key=["id"],
    raise_on_drift=False,
)

SnapshotManager(
    working,
    manifest,
    "artifacts/snapshots",
    base_db_path=base,
    verification_config={
        "table": "orders",
        "columns": ["total_cents", "net_cents"],
        "row_key": ["id"],
        "batch_size": 64,
    },
).save(name="session.json", conclusions={"drift": drift}, base_db_path=base)

restore = restore_snapshot("artifacts/snapshots/session.json", working)
replay = replay_snapshot(
    "artifacts/snapshots/session.json", base, "artifacts/replay-work"
)
```

Public error codes are `E_MIGRATE`, `E_DRIFT`, and `E_REPLAY`.

## Recorded verification

The complete suite currently passes:

```text
Ran 25 tests in 16.221s
OK
```

The final demo run reports:

```text
initial drift: precision_equivalent (48 rows)
scale perturbation: orders.discount_rate 4->2
affected steps: backfill-total, backfill-net
perturbed drift: precision_equivalent; max diff units=206
snapshot restore: replaced=True
affected columns: orders.discount_rate, orders.net_cents, orders.total_cents
affected steps: backfill-net, backfill-total
replay: original=True perturbed=True
```

Ledger bounds in the demo include `total_cents=3` and `net_cents=3` integer grid
units after the initial migration. Their source path is the scale-4
`orders.discount_rate` quantization contribution, propagated through
`derive:orders.total_cents` and then `derive:orders.net_cents`.

## Implementation limits

- Backfill expressions intentionally support a restricted numeric SQL-like
  grammar: column identifiers, numeric literals, unary `+/-`, binary `+ - * /`,
  and parentheses. Functions and subqueries are not evaluated by the fixed-point
  analyzer.
- `NULL` is rejected as an input to a numeric backfill expression.
- Table rebuild operations preserve ordinary columns, primary keys, NOT NULL,
  defaults, SQLite foreign-key declarations, explicit indexes, and package
  managed checks. Exotic SQLite features such as triggers, generated columns,
  and `WITHOUT ROWID` layouts should be reviewed before use.
- The ledger intentionally stores no timestamps, preserving byte-stable
  deterministic reports for the same input data and manifest.

## File map

```text
src/dbml/errors.py        stable error types
src/dbml/fixedpoint.py    quantization, rounding, integer grid helpers
src/dbml/expression.py    restricted SQL numeric parser/evaluator
src/dbml/database.py      SQLite introspection, rebuilds, digests
src/dbml/manifest.py      validation, schema simulation, stable planning
src/dbml/ledger.py        hash-chained ledger and row errors
src/dbml/analysis.py      backfill execution, error paths, report APIs
src/dbml/migrations.py    transactional runner, recovery, scale perturbation
src/dbml/verify.py        before/after drift verification
src/dbml/snapshot.py      snapshots, version comparison, replacement restore
src/dbml/replay.py        fresh-copy replay and first-mismatch reporting
src/dbml/sample.py        deterministic sample builder
src/dbml/demo.py          complete deterministic demonstration
scripts/                  shell entry points
data/migrations/          executable manifest
data/expressions/         derived-chain review sample
tests/                    25 unittest cases
artifacts/demo/           generated ledger, snapshots, SQLite copies, report
```
