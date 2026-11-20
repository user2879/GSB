# Design and Invariants

## State model

The durable state is a sequence of immutable stage artifacts plus one mutable commit pointer:

```text
load -> binning -> woe -> monotonic -> evaluation
```

`manifest.json` is the only authoritative commit pointer. A stage artifact is first written under `tmp/`, flushed, and atomically renamed to `stages/<stage>/vNNNNNNNN/`. The manifest is then replaced atomically. If the process dies before the manifest replacement, the new stage directory is unreferenced and cannot affect recovery. Every manifest write truncates all downstream stage pointers, so a new upstream artifact can never be combined with an old downstream artifact.

Per-column cache files are content-addressed and immutable. They are written by temp file + `os.replace`, and validated by payload SHA-256 before reuse. This lets an interrupted run reuse a completed column even if it died before the next stage manifest commit, while still preventing partial JSON from being used.

## Recovery identity

The manifest records:

- snapshot format version and algorithm version,
- global monotonically increasing version,
- canonical experiment fingerprint,
- canonical data-description + generated-table fingerprint,
- last completed stage,
- per-stage artifact version and directory,
- completed columns and cache keys,
- stable failure records.

Startup checks format, algorithm and the optional expected global version. It then compares the current experiment and generated-data fingerprints. A mismatch without `--revision` returns `E_SNAPSHOT_STALE` without touching the manifest. A deliberate live-session edit or explicit `--revision` writes a new manifest revision, clears all stage pointers, and may reuse only cache entries whose content-derived keys still match.

## Cache-key derivation

All keys use canonical JSON (`sort_keys=True`, compact separators, UTF-8) and SHA-256.

```text
content = H(column name, type, values, target values, target name, positive value)
bin_params = H(max_bins, min_bin_ratio, prebins)
direction_priority = H(direction, conflict_priority, max_rounds)

binning_key = H("binning", content, bin_params)
woe_key = H("woe", binning_key, smoothing)
monotonic_key = H("monotonic", woe_key, direction_priority)
evaluation_key = H("evaluation", data_fp, folds, fold_seed, target metadata, all final monotonic keys)
```

Category columns and `direction=none` use a constant non-applicable direction key, so a global priority edit does not recompute them.

## Exactness rules

- Every output float is normalized through `round(value, 12)` with negative zero collapsed to `0.0`.
- All collections that reach a report or artifact are sorted by explicit keys: column name, numeric lower boundary, bin index, monotonic round, and failure tuple.
- The WOE denominator is recomputed from adjusted per-bin counts before computing shares. A JSON cache round trip therefore returns exactly the same binary64 values.
- Evaluation always consumes the same persisted monotonic artifacts used by the report. It never mixes fresh in-memory WOE with cache-loaded WOE.
- A single-column replacement preserves the target and all unaffected columns because the target and every column use independent derived RNG streams.

## Concurrency

Each column-stage task is submitted with the stable key `stage:column`. Submission is in column-name order. Results are collected in a map and emitted in sorted order, independent of completion order. Exceptions are converted to deterministic failure records and do not escape the task. Stage payloads and manifest completed-column lists are sorted.

`max_workers` from the experiment controls binning, WOE and monotonic pools. Evaluation is a single deterministic task because it is a downstream aggregate.

## Graceful stop

`SIGINT`/`SIGTERM` sets an event. Pending futures are cancelled; already-running column tasks finish their compute and atomic cache write. If the current stage is incomplete, no stage manifest is published; progress records completed columns and keys. The process writes an interrupted report and returns integer exit code 0. A hard kill may leave only complete cache files, a complete stage directory, an old manifest, and/or a complete progress file; all partial writes are outside the recovery contract and are ignored or replaced.

## Consistency proof sketch for edits

For any edit, the cache key of a column stage changes only if one of its declared inputs changes. Unaffected keys therefore resolve to the previous immutable payload rather than executing the compute function. The downstream evaluation key includes every final column key, so it changes whenever any scored column changes. Reports are built from persisted stage payloads, not from edit-history-dependent in-memory state. Consequently an edit sequence and a fresh build of the final inputs consume the same key payloads and sort into the same fields.
