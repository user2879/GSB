# Spectral diagnostics verification

Generated: `2026-11-19T06:38:13.789Z`
Workspace: `/home/zsl/feedback/uidfc5909d3-default-257/demo_workspace`

| Check | Result | Evidence |
|---|---|---|
| Deterministic scene bytes | PASS | `09f68b08ca4ba6c921e2b26d608ffd3407c53693df7451171b74a3e9e8253980` |
| Deterministic pipeline result | PASS | `29507aa02aefa82e8ad421d0cdf7b821d68987e95c76ea1e545e823d13b9430a` |
| Exact matrix round-trip | PASS | `max error 1.4259426972529354e-15` |
| Ill-conditioned pruning | PASS | `removed [3]` |
| Attribution additivity | PASS | `sum=0.004871839248 total=0.004871839248` |
| Convention invalidation closure | PASS | `restored cache hits 10` |
| Anomaly detection and first location | PASS | `E_MATRIX_SINGULAR, E_WHITE_POINT, E_QUANT_RANGE, E_OBSERVER_INVALID, E_INTEGRATION_DENSITY` |
| Forced termination recovery | PASS | `orphans=1` |

## Error codes

`E_MATRIX_SINGULAR`, `E_QUANT_RANGE`, `E_WHITE_POINT`, `E_OBSERVER_INVALID`, `E_INTEGRATION_DENSITY`, `E_ATTRIB_MISMATCH`, `E_SNAPSHOT_VERSION`.

## Attribution method

Three-factor Shapley allocation makes the quantisation, sample-distribution and integration contributions add exactly to `combined - baseline`. Standalone deltas and the interaction allocation are retained in the ledger for audit.
