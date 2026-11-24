# receipt-diag

`receipt-diag` is a standalone Python 3.10+ package for auditing inference receipts produced by local batch workers. It normalizes the historical `r1`, `r2`, and `r3` receipt formats into one reviewable SQLite ledger, derives deterministic batch ownership, propagates invalidation from worker parameters to batches and requests, and supports crash recovery, replay, snapshots, and reproducible fault injection.

The implementation is local-only. Workers bind the literal address `127.0.0.1` on declared, globally unique ports and perform deterministic NumPy forward computation. There is no external network access, container runtime, cloud account, or paid service.

## Delivered Pieces

- Strict loaders for `request_stream.json`, `worker_spec.json`, and `merge_rules.json`.
- `r1`/`r2`/`r3` codec with field count, per-field byte length, rank, duration and duplicate-reference validation.
- Format-independent ownership normalization with `source_format`, `filled_fields`, and `fill_basis`.
- Stable batch IDs derived only from worker ID, batch sequence and arrival sequences.
- Real loopback TCP worker subprocesses with a thread-pool scheduler honoring global and per-worker quotas.
- Deterministic parameter/output/receipt digests and per-batch audit artifacts.
- Dependency invalidation with direct and indirect propagation paths and subset-only recomputation.
- SQLite ledger with one transaction per completed batch and a versioned logical content digest.
- Whole-snapshot replacement when snapshot version or content digest differs.
- Forced-termination recovery, unknown-batch reporting, full replay and field-level `E_REPLAY` diagnostics.
- CLI fault injection for receipt corruption, dangling dependency references and ledger digest mismatch.
- Optional Tkinter ledger viewer (`receipt_diag.ui`).
- Fixed-seed sample generator, unit/acceptance tests, and repeatable verification script.

## Repository Layout

```text
receipt_diag/          Package implementation
tests/                 Unit and acceptance tests (30 tests)
scripts/verify.py      Repeatable end-to-end verification and evidence generator
samples/basic/         Fixed-seed baseline sample (ports 43171/43172)
samples/changed/       Baseline with w-alpha seed/param_version changed
docs/DESIGN.md         Invariants and design notes
docs/VERIFICATION.md   Latest generated verification record
artifacts/verification Machine-readable verification evidence
```

## Windows Setup

Use the official CPython installer for Python 3.10 or newer. The installer supplies `sqlite3`, `tkinter`, and the standard networking stack. Install NumPy and this package from the repository root:

```powershell
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install numpy
py -3.10 -m pip install -e .
```

Running from the source tree without installation also works:

```powershell
py -3.10 -m receipt_diag.cli validate --sample-dir samples/basic
```

The current verification was executed in the supplied Linux/WSL environment. The code uses Windows-compatible subprocess, socket, SQLite and Tkinter APIs; no POSIX shell commands are required at runtime.

## JSON Contracts

All three files have integer `format_version: 1`. Unknown extra keys are tolerated, but all required fields are strictly typed.

### request_stream.json

```json
{
  "format_version": 1,
  "requests": [
    {
      "request_id": "req-0000",
      "arrival_seq": 0,
      "input": [3, -2, 7, 1],
      "duplicate_refs": []
    }
  ]
}
```

- `request_id` is unique and non-empty.
- `arrival_seq` is a non-negative integer unique across the stream. Booleans are rejected.
- `input` is a finite numeric vector whose length must equal the assigned worker's `model_dim`.
- `duplicate_refs` is optional and must contain existing request IDs; self references and duplicated entries are rejected.

### worker_spec.json

```json
{
  "format_version": 1,
  "workers": [
    {
      "worker_id": "w-alpha",
      "host": "127.0.0.1",
      "port": 43171,
      "model_dim": 4,
      "hidden_dim": 6,
      "seed": 17,
      "param_version": 3,
      "batch_size": 2,
      "concurrency": 2
    }
  ]
}
```

- `host` must be the literal string `127.0.0.1`.
- Ports are integers in `[1, 65535]` and globally unique across workers.
- `model_dim` and `hidden_dim` are positive integers.
- `seed` is any Python integer; `param_version` is a non-negative integer.
- `batch_size` and optional `concurrency` (default `1`) are positive integers.

### merge_rules.json

```json
{
  "format_version": 1,
  "global_concurrency": 2,
  "socket_timeout_ms": 5000,
  "worker_start_timeout_ms": 10000,
  "receipt_format": "r3",
  "routes": [
    {"worker_id": "w-alpha", "request_ids": ["req-0000", "req-0002"]},
    {"worker_id": "w-beta", "request_ids": ["req-0001", "req-0003"]}
  ]
}
```

- Every request must be assigned exactly once; assignments to unknown workers/requests are rejected.
- `receipt_format` is `r1`, `r2`, or `r3`. The engine can also override it with `--receipt-format`.
- `socket_timeout_ms` is the real client socket timeout used for inference calls.
- `worker_start_timeout_ms` is used only while waiting for a subprocess to bind and answer a loopback ping.

Validation reports the first invalid field with `E_SPEC`, for example:

```json
{"ok": false, "code": "E_SPEC", "path": "worker_spec.workers[0].host", "detail": "expected the literal string 127.0.0.1"}
```

## Receipt Formats and Normalization

Each receipt is canonical JSON with an explicit field count and byte length for every value:

```json
{
  "format": "r2",
  "field_count": 6,
  "fields": [
    {"name": "batch_id", "length": 66, "value": "B_..."},
    {"name": "request_id", "length": 8, "value": "req-0000"}
  ]
}
```

For a string, `length` is the UTF-8 byte length of the value. For arrays and other JSON values it is the byte length of canonical JSON. Fields appear in fixed schema order. Every envelope carries the source format version.

| Format | Fields |
| --- | --- |
| `r1` | `batch_id`, `request_id`, `output_digest` |
| `r2` | `r1` + `param_version`, `duration_ms`, `worker_id` |
| `r3` | `r2` + `rank`, `duplicate_refs`, `param_digest` |

Decoding rejects with `E_FIELD`:

- invalid JSON/envelope shape or unsupported format;
- declared field count or byte length mismatch;
- non-integer, boolean, negative or out-of-range `duration_ms`, `param_version`, or `rank`;
- duplicate/negative ranks; rank must be within `[0, batch_size)` when decoded as a batch; a complete r3 batch must cover every rank exactly once;
- duplicate request IDs or inconsistent batch/output digests in one batch;
- duplicate delivery references that are malformed, self-referential, repeated, or point to a non-existent request.

Normalization always emits `source_format`, `filled_fields`, and `fill_basis`. Missing r1/r2 fields are filled from the authoritative current worker/batch context when one is available and marked `authoritative_batch_context`; otherwise they remain null/empty with an explicit absence basis. The ownership conclusion excludes wall-clock duration and source metadata, so the same batch/request produces identical `output_digest`, `param_version`, `worker_id`, rank, references, parameter digest and attribution digest under r1/r2/r3. This is verified by real runs of all three formats.

## Deterministic Batching

Workers are traversed by `worker_id`, assigned requests by `(arrival_seq, request_id)`, and chunks are formed by `batch_size`. Batch sequence starts at 1 per worker.

```text
payload = canonical_json({
  "schema": "receipt-diag-batch-v1",
  "worker_id": worker_id,
  "batch_seq": batch_seq,
  "arrival_seqs": [arrival_seq, ...]
})
batch_id = "B_" + sha256(payload)
```

No JSON object traversal order, route order, request input order, or dispatch order can affect the plan bytes. Tests load reversed request, route and object-key orders and compare the full plan JSON and fingerprint byte-for-byte.

## Inference and Digests

Parameters are generated from `SeedSequence([seed_low32, seed_high32, param_version, model_dim, hidden_dim])` using deterministic integer-derived float64 weights. The forward pass is:

```text
h = tanh(X @ W1.T + b1)
y = h @ W2.T + b2
```

The parameter digest binds worker ID, seed, parameter version, dimensions and the four parameter array digests. The output digest binds:

- `<batch_id>`;
- the sorted set of participating request IDs;
- parameter version and parameter digest;
- every request's vector digest.

Workers sort requests by arrival sequence before matrix construction. Reversing the request array on the wire does not change the output digest. Each batch artifact records the actual worker host/port, peer socket address, worker PID, transport, parameter digest path, output summary path, receipt digest and request vector digests.

## Invalidation

Changing any of `param_version`, `seed`, `model_dim`, `hidden_dim`, or `batch_size` invalidates every current batch of that worker. New/removed workers, changed routes, request payloads and batch membership are also accounted for. The report is ordered by `worker_id`, then `batch_seq`, then `batch_id`, and includes:

- batches to rerun, their requests and direct/indirect markers;
- removed old batches;
- unaffected batch IDs;
- invalid request IDs;
- a dependency path and explicit propagation edges `worker -> batch -> request`.

Invalidation prunes only removed/invalid rows. Unaffected rows keep their exact batch version, receipt digest and other stored values. Rerun batches increment `batch_version` (starting at 1 for new IDs); only the reported subset is sent to workers again. See `docs/VERIFICATION.md` for the observed 3-rerun / 2-unaffected case.

## Scheduling and Loopback Workers

The scheduler uses a real `ThreadPoolExecutor`. A global semaphore and one semaphore per worker enforce `global_concurrency` and per-worker `concurrency`. Queue wait start/end are captured with `time.monotonic_ns()` relative to scheduler start and persisted with elapsed inference milliseconds.

Every batch opens a new TCP connection to `127.0.0.1:<declared-port>` and sends an inference request to a persistent worker subprocess. Client timeouts use `socket.settimeout`; there is no thread timeout or artificial delay. On Linux a parent-PID monitor stops workers when the coordinator disappears; on Windows the worker uses `OpenProcess`/`WaitForSingleObject` for the same purpose.

## Ledger and Snapshots

SQLite tables:

- `meta`: integer ledger format version, content digest, snapshot version, plan/spec fingerprints, receipt format;
- `session`: running/completed state, run ID and exit report JSON;
- `progress`: completed/total batch cursor;
- `batches`: ownership, parameter/output/receipt digests, monotonic wait/run clocks, batch version, status;
- `assignments`: request rank, references, vector/conclusion digests, normalized receipt metadata;
- `traces`: per-batch execution/invalidation propagation records.

Each completed batch atomically inserts its batch ownership, assignments, receipt summary, invalidation/recovery trace and updated progress cursor in one `BEGIN IMMEDIATE` transaction. The logical content digest is recomputed in that transaction. Volatile timings, worker PIDs and the exit-report envelope are excluded from content identity; ownership, digests, traces, progress and session state are included.

Snapshots are JSON documents containing snapshot schema version, snapshot version, ledger format version, content digest, session, progress, batches and traces. On restore:

1. snapshot schema version and the snapshot's own content digest are verified;
2. the snapshot version and digest are compared with the current ledger;
3. equal version+digest is a no-op;
4. any difference causes one all-or-nothing delete-and-insert replacement, never a merge;
5. the sorted union of current/incoming batch IDs is returned as the replaced set.

## Crash Recovery and Replay

The session begins as `running` and becomes `completed` only after a graceful finalization. A running session on reopen means the previous process did not finish: only fully committed batches are trusted, all planned-but-uncommitted batch IDs are reported as unknown, and no previous exit report is claimed.

`recover` reruns only unknown batches, verifies all resulting ownership records against the deterministic plan, then performs an internal read-only full replay comparison before finalizing. The standalone `replay` command performs the same full comparison explicitly and writes a replay report. The first mismatch is reported deterministically with all three required locators:

```json
{
  "code": "E_REPLAY",
  "path": "B_.../req-0000/output_digest",
  "detail": "actual='...', expected='...'"
}
```

## CLI

All commands print JSON. Run from the repository root, or replace `python` with `py -3.10` on Windows.

```powershell
# Generate the fixed-seed samples
python -m receipt_diag.cli samples --output samples/basic --preset basic --seed 20261125 --port-base 43171
python -m receipt_diag.cli samples --output samples/changed --preset changed --seed 20261125 --port-base 43171

# Validate and inspect the plan
python -m receipt_diag.cli validate --sample-dir samples/basic
python -m receipt_diag.cli plan --sample-dir samples/basic --out plan.json

# Run, inspect, replay
python -m receipt_diag.cli run --sample-dir samples/basic --db ledger.db --artifacts artifacts/run
python -m receipt_diag.cli status --sample-dir samples/basic --db ledger.db
python -m receipt_diag.cli replay --sample-dir samples/basic --db ledger.db --artifacts artifacts/run

# Invalidate only changed dependency slices
python -m receipt_diag.cli invalidate --old-sample-dir samples/basic --new-sample-dir samples/changed --db ledger.db --artifacts artifacts/run --report invalidation.json

# Snapshots
python -m receipt_diag.cli snapshot-export --db ledger.db --out session.snapshot.json
python -m receipt_diag.cli snapshot-restore --db other.db --snapshot session.snapshot.json

# Fault injection
python -m receipt_diag.cli run --sample-dir samples/basic --db bad.db --artifacts artifacts/bad --inject receipt_corruption
python -m receipt_diag.cli run --sample-dir samples/basic --db bad.db --artifacts artifacts/bad --inject dependency_dangling
python -m receipt_diag.cli run --sample-dir samples/basic --db bad.db --artifacts artifacts/bad --inject ledger_digest_mismatch

# Forced-termination drill and recovery
python -m receipt_diag.cli run --sample-dir samples/basic --db crash.db --artifacts artifacts/crash --crash-after-commits 1
python -m receipt_diag.cli status --sample-dir samples/basic --db crash.db
python -m receipt_diag.cli recover --sample-dir samples/basic --db crash.db --artifacts artifacts/crash
python -m receipt_diag.cli replay --sample-dir samples/basic --db crash.db --artifacts artifacts/crash

# Optional Windows Tkinter ledger browser
python -m receipt_diag.ui --db ledger.db
```

On Windows, `--crash-after-commits` produces process exit code `137` through `os._exit` after the selected committed batch. It is a deliberate acceptance drill, not a simulated recovery.

## Error Codes

| Code | CLI exit | Meaning |
| --- | ---: | --- |
| `E_SPEC` | 2 | Invalid local JSON/spec; path points to first invalid field |
| `E_FIELD` | 3 | Receipt decode/rank/reference/field-length failure |
| `E_LEDGER` | 4 | Ledger/snapshot version or content-digest failure |
| `E_REPLAY` | 5 | Recomputed ledger content differs; path is `batch/request/field` |
| `E_WORKER` | 6 | Worker startup, protocol, loopback connection or real socket timeout |
| `E_IO` | 7 | Required local file cannot be read |
| `E_CONFIG` | 8 | Unsupported CLI/injection combination |

The three fault injections map stably to `E_FIELD`, `E_FIELD`, and `E_LEDGER` respectively. Unknown unexpected exceptions are wrapped by the CLI as `E_INTERNAL` (exit 1) without changing the business codes above.

## Tests and Reproducible Verification

```powershell
python -m unittest discover -v
python scripts/verify.py
```

The verification script uses dynamically allocated loopback ports, runs all 30 tests, then independently exercises three receipt formats, invalidation, full replay, field-level replay mismatch, snapshot replacement, all three injections, and a real forced-termination recovery. It writes:

- `artifacts/verification/report.json` — machine-readable checks and evidence;
- `artifacts/verification/test.log` — complete unittest log;
- `docs/VERIFICATION.md` — generated human-readable record.

The latest result is PASS; exact observed digests, batch counts, injection locations and crash-recovery counts are recorded in `docs/VERIFICATION.md`.

## Notes and Boundaries

- Historical r1/r2 receipts cannot reconstruct worker/rank/parameter metadata by themselves. Normalization therefore requires the authoritative current batch context to produce full cross-format equality; without context it explicitly records missing fields rather than inventing values.
- `duration_ms` is real measured worker compute time and is persisted, but it is excluded from attribution and logical ledger identity so unaffected receipts remain byte-stable across reruns.
- SQLite content identity intentionally excludes volatile wall-clock/monotonic timings and process IDs while retaining all ownership and digest fields.
- The optional GUI requires a display and the official Windows CPython Tk installation; all diagnostic and acceptance behavior is available through the CLI and APIs.
