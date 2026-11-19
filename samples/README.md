# Deterministic samples

All sample channels bind only to the literal `127.0.0.1` and use unprivileged
high ports. The simulator reads fixed sequence/timestamp values from
`simulator.json`; it never reads a cluster or any remote service.

| Directory | Purpose |
| --- | --- |
| `divergence` | Three nodes, majority arbitration, latest tie-break, duplicate observations, one missing key and a non-numeric string expectation |
| `missing` | Missing key, real socket timeout and a quorum that is met despite one missing node |
| `string_expected` | Valid non-numeric (string) expected values |
| `invalid_expected` | Invalid boolean/array expectations; the loader must reject the whole spec with `E_SPEC` |

Run all demonstration scenarios (including the anomaly injections) from the
repository root:

```bat
python -m config_ledger demo --runtime runtime\demo --all-anomalies
```

On a Windows Python installation, use `py -3.11` instead of `python` if needed.
