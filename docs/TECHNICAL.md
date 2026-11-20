# Technical Notes

## Architecture

```text
JSON rules + requests
        |
        v
strict loader / SHA-256 rule version
        |
        v
attempt-layer deterministic SWRR plan
        |
        v
Scheduler ---- CircuitManager ---- HTTP loopback upstreams
    |               |
    +---- ConcurrencyGate (global + service permits)
    |
    +---- SQLite WAL ledger + append-only JSONL
    |
    +---- complete immutable snapshot -> atomic latest.json
    |
    +---- verifier -> stable first deviation / E_VERIFY
    |
    +---- stable view model -> Tkinter
```

## Rule version and validation

The version is `sha256:` plus the SHA-256 of the exact UTF-8 JSON bytes. Any file change, including a whitespace change, produces a different version. The parsed round is immutable. At round end the source file is hashed again; a mismatch sets `rules_changed_during_round=true` and records `observed_rule_version`. Result rows remain bound to the version that produced them; a later round using the changed file publishes a new complete snapshot.

Validation walks the JSON with explicit dotted/array paths and fails the whole document at the first invalid field. The loader also rejects duplicate object keys so a later silent override cannot invalidate an audit trail. Booleans are rejected where integers are required.

## Deterministic weighted selection

Instances are sorted by `(port, name)`. Zero-weight instances are removed from the selector. The selector uses Smooth Weighted Round-Robin:

```text
for each eligible instance: current[i] += weight[i]
chosen = first maximum in sorted order
current[chosen] -= sum(weight)
```

Every possible request attempt is planned before worker threads start. The planning order is:

1. service in stable name order;
2. attempt layer (`1`, then `2`, ...);
3. request input index within that service.

This avoids both hash/dict order and a bias where request-major planning puts every first attempt on the first weighted slot. Each actual attempt records its target, per-service selection ordinal and algorithm name. Snapshot instance rows include declared weight, service weight sum, targeted count and actual selected count.

## Scheduler and budgets

A logical request has `max_retries + 1` attempt slots. Each real upstream attempt performs one atomic acquisition of:

- one global permit if `global_active < global_concurrency`;
- one service permit if `service_active < service_quota`.

If both are unavailable the worker waits; the elapsed interval from enqueue/attempt start to acquisition is `queue_ms`. Circuit-fast-failed attempts do not acquire permits. Attempt concurrency is bounded by both counters even with retries.

Permit accounting is deliberately not coupled to SQLite latency. A coalescing daemon reporter writes the latest occupancy snapshot to `runtime_kv`. The gate returns exact monotonic acquisition/release timestamps, persisted as `permit_start_mono_ms` and `permit_end_mono_ms`. The verifier uses these intervals, not inference from request completion times.

## Circuit state machine

```text
CLOSED -- failure_threshold reached --> OPEN
OPEN -- monotonic now >= cooldown_until --> HALF_OPEN (one probe)
HALF_OPEN -- probe success --> CLOSED
HALF_OPEN -- probe failure --> OPEN (new cooldown)
```

`admit()` is thread-safe and returns the exact admission timestamp. During OPEN, a selected target is recorded as a targeted but unselected `E_CIRCUIT` attempt. It consumes retry budget but not concurrency budget. A late in-flight success from a request admitted before OPEN does not close the breaker.

## SQLite ledger

The ledger uses WAL, `synchronous=FULL`, foreign keys and a serialized write lock. Main tables:

| Table | Purpose |
|---|---|
| `rounds` | rule version, status, limits, observed version, summary |
| `requests` | logical request, budget, queue/start/finish, final result |
| `attempts` | target, selection ordinal, queue/start/end/permit clocks, result/code |
| `request_checkpoints` | crash recovery budget/attempt progress |
| `circuit_events` | open/half-open/reopen/close evidence |
| `circuit_state` | durable latest instance breaker state |
| `runtime_kv` | coalesced live global/service occupancy |
| `snapshots` | immutable snapshot path/hash/version/round binding |

Each attempt start and completion is a committed transaction. A process killed mid-request therefore leaves a `complete=0` attempt and queued/running checkpoint. `recover` never invents an outcome: completed rows are listed as recovered, anything unfinished is listed as `unknown_requires_rerun`, and the report states `exit_report_supported=false`.

## Snapshot publication

`build_snapshot` reads only complete attempt rows and attaches the round rule version to requests, attempts, services, budget rows and circuit events. The snapshot contains the full rule document, stable service/instance statistics, budget evidence and the verifier-ready trajectory.

Publication writes an immutable per-round file first (read-only on POSIX and metadata-marked read-only in SQLite), then atomically replaces `latest.json` with the entire new document. It never applies a patch or merges fields from the old snapshot. A different snapshot ID for an already-published round is rejected with `E_SNAPSHOT`.

## Verifier checks

- Final status matches the terminal attempt; success stops at the last successful attempt; failure code matches the terminal failure.
- Attempt numbers are contiguous from 1 and never exceed `max_retries + 1`; `attempts_used` matches trace length.
- Permit interval sweep stays within global limit and service quota; recorded occupancy at attempt start is also checked.
- No dispatched attempt was admitted inside an `open`/`reopened` cooldown interval; `E_CIRCUIT` rows are never selected.
- Instance weights are non-negative, weight sums match, zero-weight targets are rejected, and target references exist.
- Every displayed request/attempt/service/event row binds to the snapshot's single rule version.

Violations are sorted by request index, request ID, attempt number and check name. The output includes `code=E_VERIFY`, a check matrix, all violations, and a stable `first_deviation` object.

## Fault injection

| CLI value | Business code | Exit | Path exercised |
|---|---|---:|---|
| `rule_corruption` / `E_RULE` | `E_RULE` | 2 | malformed in-memory rules, field/line error path |
| `upstream_unreachable` / `E_UPSTREAM` | `E_UPSTREAM` | 3 | injected transport refusal, retries/circuit still audited |
| `snapshot_write_failure` / `E_SNAPSHOT` | `E_SNAPSHOT` | 4 | round completes, snapshot writer fails before publication |

All three paths write structured JSONL events with the business code.

## Reproducible acceptance

`python -m meshgov acceptance` runs the unit suite plus isolated end-to-end rounds and writes:

- `artifacts/reports/acceptance.json`
- `artifacts/reports/acceptance.md`
- `artifacts/acceptance/<run-id>/...` complete per-scenario evidence

The report records return codes, verification results, deterministic target counts, snapshot hash preservation, injection log codes, crash/recovery unknown items and view-model ordering checks.

