# Design and Invariants

## Goals

PageAgg separates four concerns: immutable local page bytes and catalog validation, deterministic route planning and two execution strategies, route-independent fixed-point aggregate results, and an atomic replayable SQLite ledger.

The central invariant is:

> Given the same catalog bytes, scan sequence, budget, route-independent fixed-point version, and batch size, the resident and rescan routes must produce identical aggregate result bytes and identical eviction-event bytes.

## Data Invariants

- Page IDs are unique in the catalog.
- Every page has the catalog's fixed row count.
- The embedded page header and catalog column order/name/type agree.
- Each column has exactly `rows * 8` value bytes.
- Each null bitmap has exactly `rows` bytes, each equal to 0 or 1.
- The null-map CRC32 agrees with the stored bitmap.
- Every float64 value is finite.
- The group-key column is integer and has no nulls.
- Every scan-sequence entry names a declared page.
- The complete page SHA-256 agrees with the catalog.

Null-map validation runs before the page SHA-256 check. A shifted bitmap can therefore produce `E_NULLMAP` with a precise column/row/CRC diagnosis even if the injector also updates the catalog content digest.

## Slot Plan

The canonical planner is a sequential LRU simulation over the scan sequence:

- empty slots are filled at the lowest slot index;
- a resident access is a hit and updates recency;
- a miss with a full cache evicts the resident with the smallest `(last_access_seq, page_id)`;
- the incoming page occupies the victim's slot;
- eviction event sequence numbers start at 1 and are contiguous.

The plan yields per-access hit/miss and slot, eviction events, final occupancy, and hit/miss counts.

### Resident realization

The execution engine keeps a real `slot -> PageData` array. A hit uses the cached page. A miss rereads and validates the file. The engine compares each actual slot transition with the canonical plan and raises `E_ROUTE` with the first page/event/field conflict.

### Rescan realization

The rescan engine keeps no resident page array. Every access rereads and validates the page file, reports `cache_hit = false`, reports no physical resident pages, and emits a rescan event. It still consumes the canonical slot plan as a shadow event plan, so eviction bytes equal the resident route. The UI labels these occupancy rows `shadow_plan`.

## Aggregation Invariant

Aggregation is driven only by the scan sequence, never by cache residency timing. For each access, rows are grouped by the group key in ascending key order. Within a page, rows retain their physical order.

The fixed-point result includes per-page terms. Term order is the global access sequence. Therefore changing slots, hits, or rescans cannot reorder accumulation.

## Error Bound

Let `K` be the number of non-null float values contributing to a group/column. Every float is truncated toward zero at `1e-6` scale, so each contributes error strictly less than one fixed-point unit:

```text
|sum_reported - sum_exact| <= K units
mean_reported = trunc_toward_zero(sum_reported / value_count)
|mean_reported - mean_exact| <= ceil(K / value_count) + 1 units
```

The extra unit bounds mean truncation. If a sum saturates, the sum and mean bounds become `INT64_MAX` and the metric is marked saturated.

## Route Decision Boundary

```text
overflow_ratio_ppm = (max(0, pages - budget) * 1_000_000) // pages
pressure_ppm       = (overflow_ratio_ppm * rows_per_page) // rows_reference
boundary_affected  = abs(pressure_ppm - threshold_ppm) <= margin_ppm
```

`pressure_ppm > threshold_ppm` selects rescan; all other cases select resident, including exact ties. A forced route records that it was forced and still emits automatic decision context.

## Ledger Transaction Protocol

For batch `i`:

1. transaction A inserts an attempt row with status `STARTED` and commits;
2. transaction B inserts the batch, page outcomes, eviction/rescan events, cumulative group state, page/group references, advances the snapshot cursor, and changes the attempt to `COMMITTED`;
3. transaction B recomputes and stores the ledger content digest;
4. transaction B commits.

If the process dies in transaction B, SQLite rolls back all data and the cursor remains at the prior batch. On the next open, the committed `STARTED` row becomes `UNKNOWN`. No forced-exit report is fabricated.

## Snapshot Replacement

An active snapshot is identified by catalog digest, rows per page, group key, slot budget, selected route, and batch size. Any difference triggers non-incremental replacement. The old snapshot row is deleted, cascading to all aggregate/event state. A new monotonic plan version is allocated and execution is written from cursor zero. The replacement audit records old/new catalogs, result digests, reasons, and propagation detail.

`evolve` distinguishes value-affected pages/groups from route-affected pages/groups. A budget-only change can affect every page's cache outcome while affecting zero group estimates. Unaffected group payload digests are compared before and after; any mismatch is a replay error.

## Replay Localization

Replay rebuilds the same execution from the snapshot route/budget/batch size, then checks committed state in stable order:

1. batch index and cursor;
2. batch/group-state digests;
3. page outcomes by access sequence;
4. eviction events by event sequence;
5. rescan events by event sequence;
6. cumulative group states by group key;
7. final result bytes and digest.

The first mismatch produces `E_REPLAY` with `page_id`, `event_seq`, `field`, expected value, and actual value. The validator runs route self-consistency before aggregate replay so a bad eviction/rescan event reports `E_ROUTE` rather than a generic replay failure.

## Determinism

- Canonical JSON uses sorted keys, compact separators, UTF-8, and no NaN/Infinity.
- Results store no binary floats.
- Group keys and columns are sorted.
- Event sequence, scan sequence, and row order are explicit.
- Route decisions use integer ppm arithmetic.
- Selection ties use documented stable rules.
- SQLite writes use WAL, full synchronous mode, and explicit transactions.

