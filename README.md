# GSB fixed-point shard diagnostics

`gsb-diagnostics` is a Python 3.10+ package for diagnosing fixed-point quota
watermarks, dependency invalidation and durable scheduling in a local
multi-node shard workload. Every node is a real child process listening on a
literal `127.0.0.1` TCP socket. Requests and replies use newline-delimited JSON.
All quota, watermark and arbitration decisions use Python integers or NumPy
integer arrays; no float is used as a decision path.

## Delivered components

- Strict local JSON loader with stable first-error paths (`E_SPEC`, `E_DEPEND`).
- Deterministic samples derived from node seed and shard id, with NumPy `int64`
  grouped aggregation and SHA-256 content/aggregate digests.
- Real child-process TCP runtime with startup, handshake and read deadlines
  (`E_PORT`, `E_TIMEOUT`, `E_PROTO`).
- Fixed-point watermarks, explicit truncation/saturation policy, three-part
  error accounting and stable arbitration evidence.
- Dependency-aware invalidation, incremental recomputation, byte-preservation
  checks and first-field divergence reporting (`E_DIVERGE`).
- Thread-pool scheduler enforcing global units, per-node lease count and
  per-node concurrency, with monotonic lease renewal/expiry and wait logs.
- SQLite atomic outcome ledger, whole-snapshot replacement and forced-kill
  recovery with quota reclamation (`E_SNAPSHOT` for invalid snapshots).

## Environment and installation

Validated locally on:

- Linux/WSL Ubuntu, Python `3.11.0rc1` (the package requires Python `>=3.10`)
- NumPy `1.26.4`
- SQLite `3.37.2` from the standard library

The acceptance demo is headless and does not import Tkinter. The current
development image has no `python3-tk`/`libtk8.6` and `sudo` requires a
password, so no GUI is required or claimed. If a local Tk UI is wanted, install
it separately with `sudo apt install python3-tk`.

```bash
python3 -m pip install .
# or, without installation (and for source-tree edits):
PYTHONPATH=src python3 -m gsb --help
```

The wheel build was also verified directly on the supplied old pip 20.3.4
toolchain. If a very old pip fails its editable-install bootstrap, use the
regular wheel install above or `PYTHONPATH=src`.

## Repeatable acceptance demo

```bash
PYTHONPATH=src python3 -m gsb demo \
  --spec examples/spec.json \
  --evolved-spec examples/spec_evolved.json \
  --workdir artifacts/demo
```

The command exits with status `0` and prints `PASS`. It writes the complete
evidence to `artifacts/demo/report.json`, including:

- integer water vectors and error bounds for an indistinguishable water case;
- an exact residual tie in the budget arbiter;
- actual lease issue/renew/release events and `global_budget` queue waits;
- quota-evolution invalidation with only `raw-a`, `join-ab`, `report` rerun;
- byte-preserved results for `raw-b` and `raw-d`;
- field-by-field consistency against a blank full rebuild;
- whole snapshot restoration over a stale session version;
- a real `SIGKILL`-equivalent child exit after a committed lease reservation,
  followed by recovery and quota reclamation;
- live coverage of `E_SPEC`, `E_DEPEND`, `E_PORT`, `E_TIMEOUT`, `E_PROTO` and
  `E_DIVERGE`.

Individual commands:

```bash
PYTHONPATH=src python3 -m gsb validate examples/spec.json
PYTHONPATH=src python3 -m gsb run \
  --spec examples/spec.json --ledger artifacts/run.db --session-version 1
PYTHONPATH=src python3 -m gsb recover \
  --ledger artifacts/run.db --session-version 1
```

## Spec format

See `examples/spec.json` for a complete declaration. Unknown fields are
rejected. The root fields are:

| Field | Meaning |
| --- | --- |
| `format_version` | Integer schema version; currently exactly `1`. |
| `capacity_version` | Opaque non-empty capacity generation identifier. |
| `fixed_scale` | Session fixed scale; one of `100`, `1000`, `10000`, `100000`. |
| `global_budget` | Global resource-unit budget, positive integer. |
| `max_workers` | Worker-pool size. |
| `lease_ttl_ms` | Lease TTL measured with `time.monotonic_ns()`. |
| `handshake_timeout_ms` | TCP connect/handshake deadline. |
| `io_timeout_ms` | Per-request socket read deadline. |
| `poll_interval_ms` | Scheduler wake-up interval. |
| `nodes` | Non-empty node declarations. |
| `shards` | Non-empty shard declarations. |

Each node declares `id`, literal `host: "127.0.0.1"`, globally unique `port`,
integer `seed`, integer `concurrency`, integer `lease_limit`, positive integer
`quota`, non-negative integer `used <= quota`, and a supported `fixed_scale`.

Each shard declares `id`, existing `node_id`, existing upstream
`dependencies`, positive integer `demand`, `sample_count`, `group_count` and
`batch_size`. Node ids, shard ids and ports are unique. The dependency graph is
checked for missing references and cycles in declaration order.

## Package integration

```python
from gsb import ClusterRuntime, Ledger, load_spec, plan_invalidation

old_spec = load_spec("examples/spec.json")
new_spec = load_spec("examples/spec_evolved.json")

with Ledger("artifacts/live.db", session_version=1) as ledger:
    with ClusterRuntime(old_spec, ledger=ledger, session_version=1) as runtime:
        initial = runtime.execute()

# A new session/ledger is used for an evolved declaration.  Only the invalid
# closure is recomputed; records outside it are imported and byte-checked.
plan = plan_invalidation(old_spec, new_spec)
```

Useful modules: `gsb.spec`, `gsb.fixed`, `gsb.arbitration`,
`gsb.invalidation`, `gsb.runtime`, `gsb.scheduler`, `gsb.ledger`,
`gsb.samples`, `gsb.client`.

## Fixed-point policy

For non-negative integer units `U`, quota `Q`, lease limit `L`, and session
scale `S`:

```text
used_fp      = U * S
quota_fp     = Q * S
lease_fp     = L * S
utilization  = floor(U * S / Q)
headroom     = floor((Q - U) * S / Q)
```

The intermediate products are exact Python integers. A final quotient above
signed 64-bit `MAX_I64` saturates at `MAX_I64`; truncation is floor division for
the non-negative domain. Ratios retain their remainder, so each ratio has a
strict truncation error below one fixed unit.

For `n` nodes, the declared decomposition is:

- quantization truncation: `2n` fixed units (`utilization` + `headroom`);
- comparison decision: `2` fixed units for two candidates;
- aggregation accumulation: `n` fixed units when summing node watermarks;
- total: `3n + 2` fixed units.

The dominant segment is the largest component; ties prefer quantization, then
comparison, then aggregation. See `docs/DESIGN.md` for the full derivation.

## Stability and durability rules

- Ready shards are ordered by stable Kahn topological rank, then shard id.
- Equal or error-indistinguishable water is decided by the same stable key and
  marked `error_affected: true`; the complete evidence record is returned.
- Integer budget residuals are exact resource units. Ambiguity at the resource
  boundary therefore occurs at exact residual ties unless the caller supplies
  a valid fixed-bound conversion.
- A stopped lease transitions accounting at most once. Expiry of a still-running
  worker places its capacity in quarantine until worker exit, preventing
  overcommit and double reclamation.
- Every committed shard record carries `format_version`, `content_digest`,
  aggregate/sample digests, lease attribution, units, water, error attribution
  and an integer progress cursor in one SQLite transaction.
- Snapshot restoration is always a whole-file replacement, never an incremental
  merge. Recovery only trusts complete record transactions and explicitly
  returns `exit_report_available: false` after forcible termination.

## Tests and actual results

```bash
PYTHONPATH=src python3 -m pytest -q
```

Result on 2026-11-27: `37 passed`. The demo result is `PASS`; captured evidence
is in `artifacts/demo/report.json` and summarized in
`docs/VALIDATION.md`. No Docker, external account, remote target or network
site is used.
