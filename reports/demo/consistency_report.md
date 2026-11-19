# Consistency Verification Report

Overall: **PASS**

- Generated: `2026-11-19T17:57:23.997789+00:00`
- Python: `3.11.0rc1 (main, Jun 29 2026, 00:05:30) [GCC 11.4.0]`
- NumPy: `1.26.4`
- Index fingerprint: `cf376554f344598ec00f0c9278764bea4d75097e9b58f2cd79e140a9466a3557`
- Logical digest: `7a68d967bf42a83924612497ff5b2c6a96e97c60e9ef5c1e269318b9243a106c`
- Valid documents / queries: `26` / `8`
- Blocks: `147`
- Final format / tier / positions: `v2` / `2` / `True`
- Failures: `5`
- Max fixed error (units): `0.534527342490037`

## Checks

| Check | Result |
|---|---|
| `incremental_equals_full_rebuild` | PASS |
| `dual_tier_lossless` | PASS |
| `position_switch_structural_and_lossless` | PASS |
| `v1_v2_roundtrip_lossless` | PASS |
| `concurrency_stable_order` | PASS |
| `validation_failure_codes` | PASS |
| `interactive_budget_pending` | PASS |
| `snapshot_recovery_cleanup` | PASS |
| `resume_config_mismatch_rejected` | PASS |

## Edit Replay

| Command | Op | Status | Version | Reused blocks | Rebuilt blocks | Reused queries | Rebuilt queries |
|---|---|---|---:|---:|---:|---:|---:|
| `E001` | `add_doc` | `OK` | 2 | 109 | 14 | 2 | 6 |
| `E002` | `fix_doc` | `OK` | 3 | 53 | 69 | 1 | 7 |
| `E003` | `set_config` | `OK` | 4 | 0 | 145 | 8 | 0 |
| `E004` | `set_config` | `OK` | 5 | 0 | 145 | 8 | 0 |
| `E005` | `set_config` | `OK` | 6 | 145 | 0 | 0 | 8 |
| `E006` | `set_config` | `OK` | 7 | 0 | 145 | 8 | 0 |
| `E007` | `add_doc` | `OK` | 8 | 135 | 12 | 6 | 2 |
| `E008` | `set_config` | `OK` | 9 | 0 | 147 | 8 | 0 |

## Incremental vs Full Rebuild

- Index fingerprint equal: `True`
- Decoded posting digest: `1658c2c5c7365e803ac7ffabcaacc76dbcd0f77a7bcf7923be40014dae1d4145`
- Mismatch count: `0`

## Additional Diagnostics

- Tier 1/Tier 2 decoded digest equal: `True`
- Position-off payloads omit positions: `True`
- v2→v1→v2 digest sequence: `1658c2c5c7365e803ac7ffabcaacc76dbcd0f77a7bcf7923be40014dae1d4145` / `1658c2c5c7365e803ac7ffabcaacc76dbcd0f77a7bcf7923be40014dae1d4145` / `1658c2c5c7365e803ac7ffabcaacc76dbcd0f77a7bcf7923be40014dae1d4145`
- Concurrency 1 vs 4 results equal: `True`
- Budget pending status: `E_INTERACT_BUDGET_PENDING` after `command_received`
- Interrupted staging/orphan cleanup: `True`
- Resume mismatch code: `E_CONFIG_MISMATCH`

Observed wall-clock query times are retained in `diagnostics.json` as observations; deterministic task order, logical cost and candidate counts are the concurrency equality fields.
