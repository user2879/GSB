# Multimodal Batch Ledger Diagnostic Report

- State hash: `1e48d6eeeae5149ff2fb9f8ca51c9168615d5559bf525671bede48efe65d785e`
- Receipts accepted: 15
- Failures: 4
- Authoritative requests: 12
- Snapshot version: None
- Restore decision: `cold_full_rebuild`

## Invalidation boundary

- Parameter-version changes observed: 2
- Normalization computed batches: 5
- Normalization reused batches: 0
- Resource computed batches: 5
- Resource reused batches: 0

## Multimodal quality summary

- `image_text`: count=8, duration_observed=6, duration_missing=2, sum_ms=60
- `image_only`: count=4, duration_observed=4, duration_missing=0, sum_ms=44

## Failures

- `E_DUPLICATE_BATCH_ID` fault_receipts/injected_duplicate_batch.json#1: duplicate delivery identity for batch batch-001 injection=INJ-DUPLICATE-BATCH
- `E_DURATION_NONFINITE` fault_receipts/injected_nonfinite_duration.json#0: duration_ms is not finite injection=INJ-NONFINITE-DURATION
- `E_TRUNCATED_RECEIPT` fault_receipts/injected_truncated.json#1: invalid JSON receipt: Expecting value injection=INJ-TRUNCATED
- `E_UNKNOWN_PARAM_VERSION` fault_receipts/injected_unknown_parameter.json#0: unknown parameter_version 99 injection=INJ-UNKNOWN-PARAM

## Fault-injection hits

- `INJ-DUPLICATE-BATCH` hit=True code=E_DUPLICATE_BATCH_ID source=fault_receipts/injected_duplicate_batch.json
- `INJ-NONFINITE-DURATION` hit=True code=E_DURATION_NONFINITE source=fault_receipts/injected_nonfinite_duration.json
- `INJ-TRUNCATED` hit=True code=E_TRUNCATED_RECEIPT source=fault_receipts/injected_truncated.json
- `INJ-UNKNOWN-PARAM` hit=True code=E_UNKNOWN_PARAM_VERSION source=fault_receipts/injected_unknown_parameter.json
