# Actual verification record

Date: 2026-11-18

## Environment

```text
Platform: local Linux validation container (Windows target)
Python: 3.11.0rc1
NumPy: 2.4.6
Tkinter: module unavailable in this container (`_tkinter`/Tk libraries absent)
Network/GPU: not used
```

The UI source was compiled with `compileall`. The missing-Tkinter path was executed and returned `E_IO` with field path `$.tkinter` and exit code `6`. Real window interaction must be performed on the Windows desktop command shown in the README.

## Commands executed

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q mixedprec_diag tests scripts
python3 scripts/demo.py --expected-manifest configs/digest_manifest.json
python3 -m mixedprec_diag.cli validate-config --config configs/fixed.json
python3 -m mixedprec_diag.cli ui --config configs/float32.json
```

## Results

- Unit tests: `Ran 34 tests ... OK`.
- Compile check: `ALL_OK`.
- Valid config: status `VALID`, 64 blocks of 16 KiB for a 1 MiB budget.
- Demo digest check: all three runs matched the manifest.
- Artifact digests:
  - data: `36c21d83c44e75425469d80ef8c4eb541a4b7ba17abecf29baaaa47f9d3d2842`
  - params: `bdb82888a59179f08d2c2388cddf7bf051a03f016303aff5f9b32f455d5c3ad8`
- Version sequence: `1 -> 2 -> 3`.
- Changed fields at v2: `preallocation`, `precision`.
- Changed fields at v3: `batch_size_limit`, `memory_budget_mib`, `quant_scale_power`.
- All memory logs replay-verified.
- All runs reported whole replacement (`replaced_whole=true`).
- Demo result statuses: PASS, PASS, PASS.
- Peak blocks: 7, 9, 9.
- Output digests:
  - v1: `9c5795a72a1ccfa40e74cd7da46e304240e7812ce0b84696d6eb3cb026499b9e`
  - v2: `441ad585bbcd9376e94fd661ac19c78fcf92babe225bf182c4118edec3c0d872`
  - v3: `66dd630be6de3323dd530a5cc66de878c85e3b04a2850a20922d0523a29d9dbd`
- Final v3 UI-query aggregate observed here: mean `1.938 ms`, P95 `4.185 ms`, max `4.185 ms`, 3 raw records, fixed-point 0.001 ms, no latency violation under 16 ms. The UI view query uses a compact read-only projection rather than copying every full diagnostic array.
- `reports/version_0001.json` and `reports/version_0002.json` are physically rewritten with `stale.marker=STALE`; `reports/version_0003.json` and `current.json` are the current unmarked view.
- `configs/oom_probe.json`: report status `FAIL`, codes `E_OOM,E_PRESSURE`, zero batches, one resident model block, no inference.
- `configs/pressure_probe.json`: report status `PASS`, code `E_PRESSURE`, 3 batches, 48 samples, peak 2 blocks.
- `configs/latency_strict.json`: report status `PASS`, code `E_LATENCY`, 2 batches, 16 samples, observed UI-query P95 3.873 ms against a 0.001 ms budget.

## Scenario coverage confirmed by tests

- Same seed and generation arguments produce bitwise-identical inputs, weights and biases.
- Digest mismatch aborts with `E_DATA` and writes an aborted report with `inference_started=false`.
- Invalid/duplicate/unknown config fields are rejected with `E_CONFIG` and the exact JSON field path.
- Contiguous, single-block and non-contiguous allocation; no double ownership; derived fragmentation; OOM details; event replay.
- Explicit half-away-from-zero rounding and int32 saturation.
- Fixed and float paths repeat with identical output digests.
- Attribution is independent of batch size and batch position.
- Capacity split queues residual samples, never exceeds budget, and derives `E_PRESSURE` from event intervals.
- Model-weight OOM is reported without running inference.
- Version increments, affected/stale batch lists are stable, stale snapshots are marked, and current reports exclude old data.
- UI latency aggregates are integer microseconds with nearest-rank P95 and `E_LATENCY` on budget overrun.
