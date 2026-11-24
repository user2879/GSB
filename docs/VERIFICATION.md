# Verification Record

- Generated (UTC): `2026-11-24T16:33:53.975794+00:00`
- Python: `3.11.0rc1`
- Platform: `linux`
- Overall: **PASS**

## Checks

| Check | Result |
| --- | --- |
| `unit_tests_passed` | PASS |
| `three_formats_equal` | PASS |
| `invalidation_subset_only` | PASS |
| `replay_mismatch_is_e_replay` | PASS |
| `snapshot_whole_replace` | PASS |
| `injection_codes` | PASS |
| `crash_recovery_complete` | PASS |

## Unit Tests

- Exit code: `0`
- Summary: `OK`
- Log: `/home/zsl/feedback/uidfc5909d3-default-230/artifacts/verification/test.log`

## Plan

- Batches: `5`
- Plan fingerprint: `7e802a020f9b741d5c09ab769e69fbd0308194e5114a2d14d625804ec27521f4`

## Receipt Formats

- r1/r2/r3 digest equality: `True`
- `r1`: batches `5`, ledger digest `359ad5a4e62c81a21ec06ba9c6213a0fb721b1802e8f5b04e06b4cdbbefdcf12`
- `r2`: batches `5`, ledger digest `b9e278d06bfed810ecd4727fb5b16e0f2bae0e77811dc786b86232faef7f90a2`
- `r3`: batches `5`, ledger digest `8a501576f783f2d5a09797721d11e64ec389549df47c2026b837d8061aa8705e`

## Invalidation

- Rerun batches: `3`
- Unaffected batches: `2`
- Removed batches: `0`
- Unaffected rows byte-identical: `True`
- Replay verified batches: `5`

## Snapshot

- Source digest: `8a501576f783f2d5a09797721d11e64ec389549df47c2026b837d8061aa8705e`
- Restored digest: `8a501576f783f2d5a09797721d11e64ec389549df47c2026b837d8061aa8705e`
- Replaced batch IDs: `5`
- Repeated restore replaced IDs: `0`

## Injections

| Injection | Code | Path |
| --- | --- | --- |
| `receipt_corruption` | `E_FIELD` | `receipt[0].fields[0].length` |
| `dependency_dangling` | `E_FIELD` | `receipt[0].duplicate_refs[0]` |
| `ledger_digest_mismatch` | `E_LEDGER` | `ledger.content_digest` |

## Replay Mismatch

- Code: `E_REPLAY`
- First location: `B_99304807e73b328e000fe8e3545d77de8ea94181d7afde673e1433f3051795b4/*/output_digest`

## Crash Recovery

- Forced exit code: `137`
- Exit report existed after crash: `False`
- Committed before recovery: `1`
- Unknown before recovery: `4`
- Recovered batches rerun: `4`
- Internal recovery replay verified batches: `5`
- Replay verified batches: `5`
