# Design Notes

## Invariants

1. Input order must not determine output bytes. Every collection that can affect a digest is sorted by an explicit business key.
2. Same request + same batch context + same parameters must yield the same ownership conclusion under r1, r2 and r3.
3. A batch ID depends only on worker ID, per-worker batch sequence and participating arrival sequences.
4. Worker parameter changes propagate through batches to request ownership; only invalidated batches are executed again.
5. A batch commit is atomic across ownership, receipt summary, invalidation trace and progress cursor.
6. A hard-terminated process never claims an exit report; only committed batches are trusted.
7. Snapshot restoration is whole replacement, never row-by-row merge.
8. Replay reports the first difference with batch ID, request ID and field.

## Canonicalization

`canonical_json` uses UTF-8, sorted keys, compact separators, ASCII escapes and `allow_nan=False`. Digests never use Python `hash()`. NumPy arrays are hashed as a canonical metadata header plus little-endian float64 bytes in C order.

JSON object insertion order is therefore irrelevant. Lists that represent sets are sorted before hashing. Lists that represent ordered requests are sorted by `(arrival_seq, request_id)` before batching and inside workers.

## Batch Identity and Plan

```text
canonical source = {
  "schema": "receipt-diag-batch-v1",
  "worker_id": <string>,
  "batch_seq": <1-based integer>,
  "arrival_seqs": [<sorted by arrival>]
}
batch_id = "B_" + SHA-256(canonical source)
```

Parameter version is deliberately not part of batch identity. A parameter version change keeps the same plan/batch ID and increments `batch_version` in the ledger; a batch-size or routing change normally creates different batch IDs and both rerun and removal records.

## Receipt Codec

The wire envelope is:

```json
{"format":"r1|r2|r3","field_count":N,"fields":[{"name":"...","length":N,"value":...}]}
```

Field order is fixed by schema. `length` is the UTF-8 byte length for strings and canonical JSON byte length for other values. Decode validates the envelope before interpreting values. Cross-receipt checks (same batch ID, same output digest, unique request IDs, unique/complete rank set, known duplicate references) are performed by `decode_receipt_batch`.

`NormalizedReceipt` separates:

- source provenance (`source_format`, `filled_fields`, `fill_basis`, measured duration);
- ownership attribution (batch, request, worker, parameter version/digest, rank, references, output digest).

Only the attribution is used for cross-format equality. The `fill_basis` map exists even for r3, where the filled-field list is empty.

## Parameter and Output Digests

Parameter generation uses `numpy.random.SeedSequence` seeded from platform-independent integer components. Weight values are generated as integers and divided by 100 for deterministic float64 inputs.

The output digest includes the batch ID, sorted request ID set, parameter version/digest and per-request vector digests. It is independent of request wire order because workers sort by arrival sequence and the digest builder sorts request IDs/digests.

## Scheduler State

The scheduler has four observable intervals for each batch:

1. queue wait start (before any quota acquisition);
2. queue wait end (after global and worker quotas are both acquired);
3. run start (immediately before the first socket request);
4. run end (after the worker response, digest reconstruction, receipt normalization and optional ledger commit).

The thread pool size is bounded by the global limit, and two semaphores provide the hard runtime quotas. Observed maxima are checked against declarations and included in the exit report.

Inference timeout comes exclusively from `socket.create_connection`/`settimeout` on a real loopback client socket. No `Future.result(timeout=...)` or worker-side artificial sleep is used for termination behavior. A dedicated test opens a silent TCP listener and verifies `E_WORKER` from a real socket timeout.

## Ledger Logical Identity

The content digest covers:

- ledger format version;
- plan/spec fingerprints and receipt format;
- session run ID and state;
- completed/total progress cursor;
- sorted batches: ownership, version, request/arrival sequences, parameter/output/receipt digests, status;
- sorted assignments: rank, references, vector digest and normalized conclusion digest;
- sorted traces: reasons, dependency paths, request IDs and stable metadata.

It excludes:

- wall-clock start/finish timestamps;
- monotonic wait/run milliseconds;
- worker PIDs and local ephemeral ports;
- the exit-report JSON envelope.

These exclusions make repeated logical sessions reproducible while preserving complete audit details in ordinary columns/files. After a graceful finalize, the final report is written back into the session envelope so its reported content digest matches the completed ledger state; the envelope itself remains outside content identity.

## Crash State Machine

```text
empty --run start--> running --every committed batch--> running
running --graceful finalize--> completed
running --process death--> running (on reopen: unclean, unknown = plan - committed)
running --recover--> running until missing committed --finalize--> completed
```

`status` is read-only and reports whether a previous exit report can be claimed. A running session never exposes one, even if an old notebook or log file exists. Recovery executes only unknown batches, validates all ownership rows against the plan, then performs an internal read-only full replay before finalizing. The standalone `replay` operation exposes the same comparison for an already completed ledger.

## Snapshot Comparison

Snapshot integrity has two layers:

1. the snapshot's declared content digest must match its own logical content;
2. current ledger snapshot version + content digest are compared with the incoming snapshot.

The second comparison controls replacement. Different version or digest causes the entire table contents to be replaced in one transaction. The returned replaced set is sorted and contains the union of pre-existing and incoming batch IDs. An identical version+digest is a no-op with an empty replaced set.

## Transaction Boundaries

`Ledger.commit_batch` uses one SQLite `BEGIN IMMEDIATE` transaction to insert:

- the batch ownership row;
- all per-request assignment rows;
- one or more execution/invalidation traces;
- the progress cursor;
- the recomputed logical content digest.

`Ledger.prune_for_rebuild` deletes invalid/removed rows, inserts removal traces, updates metadata and rewrites the cursor in one transaction. Snapshot restore likewise deletes and reinserts all data in one transaction.

## Fault Injection Boundaries

- Receipt corruption modifies one encoded field's declared byte length before decode.
- Dangling dependency appends a non-existent reference to an r3 `duplicate_refs` field and recomputes the field length.
- Ledger digest mismatch deliberately replaces the metadata digest with zeros after a complete run and invokes integrity verification.

Each injection is routed through the same production validation path as ordinary malformed input; no special error bypass exists.

## Extension Points

- Add new receipt versions by extending `FIELD_NAMES` and normalization fill rules; keep attribution format-independent.
- Add a new worker backend by preserving the `forward_batch` response contract and real-transport evidence.
- Add ledger invariants by extending the logical projection and bumping `LEDGER_FORMAT_VERSION` only for breaking schema changes.
- Add invalidation causes by appending explicit propagation edges; retain worker/sequence ordering.
