# Verification Record

Run command:

```bash
.venv/bin/python scripts/run_acceptance.py
```

Latest execution result: **PASS**. Python `3.11.0rc1`, numpy `1.26.4`, WSL2 Linux. The same CLI and `pathlib`-based I/O are intended for Windows local use; directory `fsync` is skipped on `os.name == "nt"`.

## Acceptance matrix

| Area | Evidence | Result |
|---|---|---|
| Fixed environment | Python 3.11+, numpy 1.26.4 | PASS |
| Deterministic samples | seed 20261118, 8 pages x 12 rows, v1/v2/v3 files | PASS |
| Multi-version decode | v1/v2/v3 manifests and segments read | PASS |
| v1 upgrade | summaries + payload checksum + v3 fields filled | PASS |
| v2 upgrade | compression flag + overflow reference filled | PASS |
| Interrupted upgrade | stop after checkpoint 2, resume, byte-equal to one-shot | PASS |
| Spill determinism | repeated small-budget sequences equal | PASS |
| Aggregate parity | full-resident and 3-frame outputs byte-identical | PASS |
| Snapshot restore | wholesale state replacement, stable segment diff | PASS |
| Replay | 7 remaining records reproduced | PASS |
| Independent verifier | 12/12 checks | PASS |
| Tail truncation | `E_FORMAT`, record index 7 | PASS |
| Segment checksum | `E_INTEGRITY`, page `p0001` | PASS |
| Zero budget | `E_BUDGET` | PASS |
| Unit tests | 22 tests | PASS |

## Actual deterministic values

- Full-resident budget `65536` / page size `512` gives `128` frames and no eviction.
- Small budget `1536` gives `3` frames.
- Small-budget overflow sequence:

```text
p0005, p0004, p0002, p0001, p0006, p0003, p0005, p0008
```

- Small-budget swap-in sequence:

```text
p0005, p0001, p0003, p0005, p0002, p0007, p0001
```

- Swap-in count: `7`.
- Aggregation output SHA-256:

```text
2eb1921f7bba435935c26ce8c72377f58d1715a4be77169947da32ee427e0f5d
```

- Full-resident output, small-budget output, and repeated small-budget output are byte-identical.
- Mean absolute error bound: `3/2000000 = 0.0000015`.

## Upgrade evidence

- Interrupted after checkpoint `2`.
- Resume processed indices `[0,1,2,3,4,5,6,7]` without duplicates.
- Checkpoint final stage: `complete`.
- Backup mode: read-only (`0o444`).
- Resumed and one-shot manifests have identical SHA-256:

```text
ffb9c2a655b46e355b78e9226fdfbcc28e965a2912f6c1d3a8010e34ded81960
```

## Snapshot evidence

- Snapshot position: after access `5`.
- Replay compared `7` future records.
- Replay status: `reproducible`.
- Lost/new overflow segments in unchanged workspace: empty.
- Final output hash equals the reference output hash.
- Independent verifier: `PASS`, `12` checks passed, `0` failed.

## Structured injection log

`artifacts/acceptance/structured.log` contains, in order:

```text
E_FORMAT   manifest_tail_truncate      records[7].record_length
E_INTEGRITY segment_checksum_mismatch  p0001 trailer checksum
E_BUDGET   memory_budget_zero          $cache.budget_bytes
```

Additional negative tests cover unsupported record versions, bad record CRC, compression flag mismatch (`E_PAYLOAD`), replay trace field mismatch, manifest version change, fixed-point version change, missing overflow segment, and verifier detection of corrupted mean/error-bound fields.

## Reproduction artifacts

```text
artifacts/acceptance/acceptance-report.json   full machine-readable result
artifacts/acceptance/acceptance-summary.md    concise result summary
artifacts/acceptance/structured.log           JSONL anomaly logs
artifacts/acceptance/unit-tests.log           unittest output
artifacts/acceptance/runs/                    isolated run directories
```
