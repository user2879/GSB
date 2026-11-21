# Validation Result

Generated: `2026-11-21T01:44:45.391230+00:00`
Python: `3.11.0rc1` on `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`

## Checks

| Check | Result | Evidence |
|---|---|---|
| Unit/integration tests | PASS | `23` tests |
| Deterministic loopback scenario | PASS | `9` transactions, `18` pulls |
| Forced-kill recovery | PASS | no forced exit report claimed; incomplete query requeued |

## Scenario Ledger

- Pull samples: `18`; mean `141800` / P95 `318701` / max `318701` ms x1000.
- Query samples: `6`; mean `56148` / P95 `304020` / max `304020` ms x1000.
- Structured error codes observed: `E_CONFLICT, E_DUP, E_LATENCY, E_QUERY, E_STATE`.
- Independent replay: `True`; ledger-file verification: `True`.

## Recovery Ledger

- Forced exit report created: `False`.
- Incomplete queries marked unknown and requeued: `['crash-incomplete-query']`.
- Tail incomplete records quarantined: `1`.
- Recovery error codes: `E_CONFLICT, E_DUP, E_LOG, E_QUERY`.

The machine-readable truth source is `artifacts/validation/acceptance-report.json`; per-interaction latency rows are in the generated SQLite ledgers.
