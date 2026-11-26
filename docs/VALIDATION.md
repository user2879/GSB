# Validation record

Date: 2026-11-27 (Asia/Shanghai)

## Environment

```text
Python 3.11.0rc1
NumPy 1.26.4
SQLite 3.37.2
Linux/WSL Ubuntu
```

`tkinter` is not imported by the package or acceptance path. In this image
`_tkinter` cannot load `libtk8.6.so`; `python3-tk` is not installed and `sudo`
requires a password. No Docker or external target was used.

## Automated tests

Command:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Observed result:

```text
37 passed
```

Coverage includes:

- `E_SPEC`, `E_DEPEND` paths and first illegal entry;
- integer water floor, remainders, saturation and error decomposition;
- ambiguous water/budget arbitration and stable topo/id evidence;
- deterministic sample bytes and aggregate digest under batching/concurrency;
- a real node child process and real TCP replies;
- `E_PORT`, `E_TIMEOUT`, `E_PROTO`;
- global budget, node quota, lease count, concurrency and worker-pool waits;
- monotonic expiry, renewal, quarantine and one-time quota reclamation;
- quota-change invalidation closure and preserved-byte checks;
- incremental versus blank full-rebuild semantic equality;
- atomic ledger fields, `E_DIVERGE`, snapshot whole replacement and `E_SNAPSHOT`;
- a real forced child termination after reservation, followed by committed-only
  recovery and quota reclamation.

Packaging was verified as a real wheel:

```text
gsb_diagnostics-0.1.0-py3-none-any.whl (40,442 bytes)
installed import: gsb/__init__.py
installed console entry point: bin/gsb-demo
```

## Acceptance demo

Command:

```bash
PYTHONPATH=src python3 -m gsb demo \
  --spec examples/spec.json \
  --evolved-spec examples/spec_evolved.json \
  --workdir artifacts/demo
```

Observed final status: `PASS`.

Key observed evidence:

- equal water values `333` versus `333`: difference `0`, comparison bound `2`,
  decision `raw-a` by stable topological/id key, `error_affected: true`;
- exact budget residual tie: both residual values `0`, `raw-a` selected, marked
  affected;
- real queue wait: `raw-d`, start `11 ms`, end `22 ms`, duration `11 ms`, reason
  `global_budget`;
- node-a quota evolution invalidated `raw-a`, `join-ab`, `report`; preserved
  `raw-b`, `raw-d` byte-for-byte;
- incremental and blank full rebuild compared consistent across all five shards;
- stale target session version `9` replaced wholesale by snapshot version `1`
  (`version_mismatch: true`, `digest_mismatch: true`);
- forced crash recovery: committed shards retained, `crash-shard` marked
  unknown/requires rerun, `reclaimed_units: 6`, `exit_report_available: false`.

The complete machine-readable result is `artifacts/demo/report.json`.
