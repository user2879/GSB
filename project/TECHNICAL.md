# Technical Design

## 1. Deterministic numeric model

Let `d` be `fixed_decimal_places`, `S = 10**d`, and `Q = quantization_units`. All scoring-stage arithmetic uses `Decimal` with high precision and is written to integer minor units by:

```text
quantize(x) = round_half_up(x * S / Q) * Q
```

For each ticket, `t_min` is the earliest evidence timestamp and `H` is the configured half-life:

```text
decay_i = quantize(0.5 ** ((t_i - t_min) / H))
score_i = quantize(weight(type_i) * trust(role_i) * (decay_i / S))
```

The operation order is fixed. Outputs are sorted by evidence sequence; evidence ranking uses descending score, descending decay, then ascending sequence.

## 2. Relations, fusion and decision

Every relation selector matches `(evidence_type, allowed_roles)`. Matched evidence pairs are generated in sequence order, excluding self-pairs. Relation contributions are applied in `(relation_id, left_seq, right_seq)` order.

- Exclusive pair: configured strength is split with the odd minor unit assigned to the left side. Each side is separately clamped at zero. The contribution detail records both nominal and actually applied deltas.
- Support pair: relation bonus (or configured default) is split and added. Its actual pair delta is recorded.
- Fusion: `sum`, `max`, `min`, `mean`, or `top2_sum`. `mean` rounds in integer minor-unit space, never multiplies by `S` again.
- Upgrade: `score > threshold` upgrades; equality follows `threshold_equality_upgrades` and always sets `boundary=true`.
- Priority: choose the band with the greatest minimum score not above the ticket score.
- Reason codes are sorted by a fixed schema order, then priority/relation codes.

## 3. Stage cache and precise invalidation

Every stage receives a SHA-256 key derived from:

- algorithm/codec version,
- stage name and ticket ID,
- upstream content fingerprint,
- upstream cache key (so a changed upstream key invalidates all downstream stages even if the output value is unchanged),
- ticket-scoped rule parameter fingerprint,
- ticket-scoped resource-pack fingerprint.

The scope is deliberately narrow:

- score: evidence content, weights/trusts actually used by the ticket, half-life, fixed-point settings;
- relations: score cache key/content, matching relation definitions, exclusion/support parameters, fusion;
- decision: relation cache key/content, threshold, fusion, equality policy, priority bands;
- convert: decision cache key/content, target version, codec and scale.

Reference indexes map each evidence type, role and relation entry to affected ticket IDs. Editing one type/role/relation therefore recomputes only referencing tickets and their downstream stages. Unaffected tickets use stored stage values without invoking scoring/relation/decision/conversion functions.

## 4. Exchange codecs

### v1 fixed width

One line contains five `|`-separated fixed fields: 16-char ASCII ticket ID, 1-digit upgrade code, 2-digit priority level, 9-digit truncated per-mille score, 1-digit missing flag. The optional sidecar records the v1 payload SHA-256, original fixed score, truncation residual, precision, reason codes, named priority, boundary marker and unrepresentable fields. A sidecar is accepted only when its payload hash and per-record line fields match.

### v2 JSON

Records contain a floating-point score, named priority and level, precision tag, reason-code list, missing flag and boundary marker. The file also records `fixed_scale` and quantization so reading can reconstruct exact integer scores.

### v3 binary

Header fields:

```text
magic | version/flags | feature_count | fixed_scale | record_count
feature_name_table | offset/length table | int64 score block | metadata payloads
```

Offsets are absolute, ranges must be in-bounds and non-overlapping. Metadata JSON is mapped by the header feature-name table; `score_fixed` is carried only in the dedicated int64 block. Conversion to v1 emits `E_VERSION_LOSSY` with per-ticket detail when per-mille truncation or non-v1 metadata cannot be represented.

## 5. Snapshot protocol

Snapshots use a complete-generation contract, not in-place artifact mutation:

1. Create a staging directory under `temp_dir` on the same filesystem volume as `snapshot_dir`.
2. Write cache, decisions, failures, summary, conversions and manifest; fsync each file.
3. Rename the complete staging directory to `snapshots/generations/gen-<version>-<hash>`.
4. Atomically replace the single `CURRENT` pointer file with `os.replace`.

If the process is killed before step 4, only an unreferenced orphan exists. Recovery reads only the generation named by `CURRENT`, verifies the manifest fingerprint and artifact hashes, and reports the recovery point. A config-hash, rule-fingerprint or expected-version mismatch emits `E_SNAPSHOT_STALE` and disables cache resume.

## 6. Concurrency and graceful stop

`ThreadPoolExecutor` never exceeds `worker_threads`. Each ticket gets stable keys derived from `(ticket_id, stage)`. Completion order is observed only for diagnostics; final tickets, contributions, failures and exchange outputs are re-sorted deterministically.

Stage work is exception-isolated per ticket. A graceful stop sets an event: already-running tickets finish their pipeline, queued tickets observe the event and skip, a partial snapshot is published, and the CLI exits 0. A force kill is recovered from the last fully published `CURRENT` generation.

## 7. Interaction budget

An edit is validated and applied synchronously, then arbitration runs in a serialized background thread. The caller waits only for the remaining configured budget. If the result is not ready, it receives `E_INTERACT_BUDGET_PENDING` with the previous completed summary and affected-ticket list. The background result is retained and later becomes the latest summary.

## 8. Platform notes

The implementation avoids POSIX-only file APIs. `os.replace` is used for file publication, and directory publication targets a non-existent generation name. The same-volume check uses `st_dev`; on platforms with weaker device semantics the paths are still confined to the configured workspace. Windows should run the same `scripts/run_acceptance.py`; the current recorded run was on Linux/WSL2, so Windows-only behavior is not claimed as empirically verified here.

