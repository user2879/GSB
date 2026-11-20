# QuantCal Saturation

Deterministic per-tensor quantisation-range calibration, affine scale/zero-point
derivation, fixed-point requantisation, saturation, dequantisation and accuracy
measurement. The repository contains a reusable Python package, a complete local
sample package, an end-to-end verifier and tests. No container or network access is
required at runtime.

## Quick Start

Requirements: Python 3.11+ and NumPy 1.24+.

Windows Command Prompt:

```bat
py -3.11 -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

PowerShell:

```powershell
py -3.11 -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

The convenience wrappers `run_demo.cmd` and `run_demo.ps1` run the verifier and
propagate its integer exit code.

Run one pipeline explicitly:

```bat
python -m quantcal ^
  --input-dir sample_data ^
  --work-dir work_v2 ^
  --exchange-version v2 ^
  --report work_v2_diagnostics.json
echo Exit code: %ERRORLEVEL%
```

PowerShell line continuation uses the backtick instead of `^`. The process returns
an integer exit code and writes canonical UTF-8 JSON reports.

## Repository Layout

```text
quantcal/
  format_v1.py       fixed-width v1 text codec
  format_v2.py       v2 JSON codec and loss-aware v1/v2 conversion
  calibration.py     deterministic quantiles, scale and zero point
  quantize.py        fixed-point kernel, clamp, dequantisation and metrics
  cache.py           content-addressed per-stage cache
  snapshot.py        staged publication, intent and stale-snapshot checks
  pipeline.py        orchestration, scheduling, reports and outputs
  injections.py      real-file mutation helpers for acceptance scenarios
  cli.py             command line entry point
sample_data/         fixed-seed UTF-8 demo input package
scripts/             deterministic sample generator
tests/               format, kernel and integration tests
run_demo.py          end-to-end acceptance verifier
demo_artifacts/      machine-readable verification result
```

All sample paths are ASCII and contain no spaces. Files can be moved to another
directory as a unit.

## Input Package

`sample_data` contains:

- `tensors.json`: tensor names, shapes, distributions and special-value positions.
- `samples.jsonl`: one calibration sample per line, with a stable `sample_id` and a
  map of tensor name to fixed-length feature vector. A tensor may be absent from a
  sample; its referenced sample-ID set is recorded explicitly.
- `calibration_v1.txt`: v1 exchange package.
- `calibration_v2.json`: v2 exchange package.
- `calibration_v2_to_v1_sidecar.json`: generator-side reference loss sidecar.
- `config.json`: quantile, width, fractional-width, rounding, saturation,
  concurrency, budget and commit-interval settings.
- `injections.json`: independent injection manifest. `INJ-001` to `INJ-003` mutate
  real input files; `INJ-004` tampers a real committed snapshot.

The two exchange packages are parsed and cross-validated on every start. Calibration
results are then rebuilt from the calibration samples, because the samples are the
source of truth. The selected output exchange version (`v1` or `v2`) determines how
the calibrated result is represented before fixed-point execution.

## v1 Fixed-Width Format

The first line declares all field widths and the shared quantisation width:

```text
# QSCALE-V1;name_width=24;scale_milli_width=12;zero_point_width=8;saturation_max_width=8;bits=8
attn.k.weight           |          18|       0|     127
```

Each data row has exactly four fields separated by `|`:

1. Tensor name, left justified, fixed width.
2. Positive integer milli-scale, right justified. `18000` means `18.0`.
3. Integer zero point, right justified.
4. Positive symmetric saturation maximum, right justified.

The parser enforces exact row length and separator positions. v1 is inherently
symmetric and cannot carry a named rounding mode, precision tag or quantile range.
Bit width is in the header so `v1 -> v2 -> v1` preserves it without adding a fifth
data field.

## v2 JSON Format

Every tensor carries `name`, `scale`, `zero_point`, `scheme` (`symmetric` or
`asymmetric`), `rounding`, `precision_tag`, `quantization_bits`,
`saturation_max`, `quantile_range` and optional `source` metadata. Tensor order is
preserved by parsers and both renderers.

Strict v2 validation requires `quantile_range[0] < quantile_range[1]`. A measured
zero-span range from a constant or all-zero tensor is retained in `result.json`;
the exchange representation uses a fixed `0.001` half-width guard window and marks
that expansion in `source`. The measured range remains available in
`source.measured_quantile_range`.

## Validation and Business Codes

| Code | Meaning |
|---|---|
| `E_FORMAT_INVALID` | malformed fields, duplicate names, manifest mismatch, bad enum |
| `E_ZERO_POINT_INVALID` | zero point outside the scheme/width integer range |
| `E_QUANT_RANGE_INVALID` | inverted/equal exchange range, bad width, non-positive saturation or scale, invalid per-tensor config |
| `E_NON_FINITE_SAMPLE` | NaN or infinity reached a tensor stage |
| `E_CALIB_BUDGET_PENDING` | stable-queue remainder did not fit the batch point budget |
| `E_SNAPSHOT_STALE` | tampered seal, incompatible resume fingerprint, or unfinished run with changed inputs |
| `E_TENSOR_FAILED` | reserved for a tensor-level failure without a more specific code |

Tensor names must be unique and the exchange tensor set must exactly match the
manifest set. Tensor errors are isolated; a failed tensor never enters the precision
summary.

## Calibration Rules

- Values are sorted with a stable ascending sort. Duplicate multiplicities are
  retained; no jitter or de-duplication is applied.
- Quantiles use NumPy `linear` interpolation, equivalent to type 7:
  `rank = (n - 1) * p`.
- Symmetric:
  `scale = max(abs(q_low), abs(q_high)) / (2^(bits-1) - 1)`, `zero_point = 0`.
- Asymmetric:
  `scale = (q_high - q_low) / (2^bits - 1)` and
  `zero_point = nearest_even(0 - q_low / scale)`, clamped to the integer range.
- Symmetric denominator zero uses `scale = 0.001`.
- Asymmetric zero span with a non-zero absolute level uses
  `scale = max(abs(q_low), abs(q_high)) / (2^bits - 1)`; an all-zero level uses
  `scale = 0.001`.
- Every guard and the selected tie rule are written to result and stage metadata.

## Fixed-Point Kernel

The configured `fixed_fractional_bits` is the resolution used to represent the
scale itself:

```text
quantum         = 2^(-fixed_fractional_bits)
coefficient     = round_half_away(calibrated_scale / quantum)
effective_scale = max(coefficient, 1) * quantum
```

Then:

```text
shifted   = x / effective_scale + zero_point
rounded   = apply_named_rounding(shifted)
clamped   = clamp(rounded, clamp_min, clamp_max)
dequant   = (clamped - zero_point) * effective_scale
```

Supported rounding modes are `nearest_even`, `nearest_away`, `floor`, `ceil` and
`truncate`. `saturate` uses the scheme's full range. `saturate_symmetric` narrows
both schemes to `[- (2^(bits-1)-1), 2^(bits-1)-1]`.

Every clipped element records its stable flat index, source sample ID, source
element index, direction and pre/post-clamp integer. The list is sorted by flat
index. Result metrics include clipped ratio, maximum absolute error, mean absolute
error, MSE and an error-attribution stage.

## Exchange Conversion and Loss Sidecar

`v1 -> v2`:

- Converts milli-scale exactly with `scale = scale_milli / 1000`.
- Emits precision tag `v1_milli_exact_1e-3`.
- Explicitly records that asymmetric scheme, named rounding and quantile range were
  unavailable in v1 and were normalised to symmetric/`nearest_even`/unavailable.

`v2 -> v1`:

- Rounds `scale * 1000` half away from zero.
- Forces the tensor to symmetric scheme and zero point 0 when needed.
- Sorts loss rows by tensor name.
- Writes `exchange_sidecar.json` with scale rounding/truncation residuals, named
  rounding, asymmetric zero-point offset, discarded quantile range, original
  precision tag and saturation bounds.

For a symmetric v1 source, `v1 -> v2 -> v1` restores tensor order, bit width, zero
points and milli-scale exactly.

## Cache Graph, Reuse and Invalidation

Each stage has a content-addressed persistent cache entry:

```text
quantile  = f(tensor values, referenced sample content, quantile parameters, version)
scale     = f(quantile result hash, scale parameters, version)
quantize  = f(scale result hash, tensor values hash, kernel parameters, version)
precision = f(quantize result hash, metric parameters, version)
```

Every cache key includes the selected exchange version. Sample references include
both the tensor's values and the complete content fingerprint of each referenced
sample. Therefore replacing any referenced sample invalidates every tensor that
references it, even when that tensor's own vector happened to remain unchanged.

The ledger records per-stage keys, content fingerprints, `reused`/`computed` status
and the stable `<tensor>:<stage>` task sort keys. The sample edit in the demo
recomputes only `attn.q.weight` and `norm.weight`; deleting `cal-0001` recomputes
only `attn.k.weight` and `zero.weight`; changing `probe.range` quantiles recomputes
only that tensor.

## Scheduling and Budget

All tensor names are processed in lexicographic order. The batch budget is measured
in calibration float elements. Cache hits cost zero. Budget costs for misses are
deducted serially in tensor-name order before the thread pool runs, so the deduction
sequence is independent of thread completion order. The first miss that cannot fit
closes the batch; all later uncached calibrations are queued with
`E_CALIB_BUDGET_PENDING`. Cached tensors after that point can still reuse results.

The thread pool is bounded by `max_concurrency`; results, clip lists, reuse lists
and budget sequences are merged in stable order. Serial and parallel runs produce
byte-identical `result.json` and exchange files for the same final inputs.

## Snapshots and Crash Recovery

Stages are `parse`, `calibrate`, `quantize` and `precision`. Each stage is written
to a temporary directory, renamed as an immutable versioned stage, and published by
an atomic pointer replacement. The snapshot manifest records:

- global and previous version;
- config and exchange-version fingerprints;
- stage artifact versions and fingerprints;
- bypass sidecar hash;
- completed, failed and pending tensor lists;
- result fingerprint and summary.

Before work starts, `run_intent.json` is written atomically. A normal incremental run
may change config without being stale. `--resume` performs strict continuation: a
different config or exchange version, an invalid seal, or an unfinished intent with
changed inputs returns `E_SNAPSHOT_STALE`. Snapshot commit checkpoints are written
every `snapshot_commit_interval` tensors.

## Outputs and Diagnostics

`<work-dir>/output` contains:

- `calibration_v1.txt` or `calibration_v2.json`: selected exchange representation.
- `exchange_sidecar.json`: downgrade losses or v2 preservation/upgrade-loss metadata.
- `result.json`: stable per-tensor fields, quantized arrays, dequantized arrays,
  clip positions, metrics and summary. This file excludes cache status and run
  counters, so incremental and full rebuild can be compared byte-for-byte.
- `ledger.json`: cache/reuse/invalidation/budget ledger.
- `diagnostics.json`: diagnostic report with input interop, scheduling, injection
  hits, errors, per-tensor metrics and snapshot view.

`--report` copies the diagnostics to a separate path. Exit codes are `0` for full
success, `2` for isolated tensor/input errors, `15` for a pending budget batch and
`16` for a stale snapshot; fatal format/config errors use their business-code exit
map.

## Injection Scenarios

The demo applies every injection to real files, not mocks:

| ID | Real target | Expected code |
|---|---|---|
| `INJ-001` | existing JSONL feature element replaced with `"NaN"` | `E_NON_FINITE_SAMPLE` |
| `INJ-002` | real per-tensor config quantile bounds inverted | `E_QUANT_RANGE_INVALID` |
| `INJ-003` | real per-tensor quantisation width set to 1 | `E_QUANT_RANGE_INVALID` |
| `INJ-004` | committed snapshot `global_version` changed, old seal preserved | `E_SNAPSHOT_STALE` |

Each hit records the injection ID, resolved path, JSON pointer and expected/actual
code. Failed tensors are omitted from the precision summary while remaining tensors
continue.

## Verification Results

Run `python run_demo.py` to regenerate `demo_artifacts/demo_report.json`. The checked
run in this repository passed:

- v1 -> v2 -> v1 tensor order, width, zero points and scales exact.
- All 9 incremental edit steps byte-identical to clean full rebuild for result,
  exchange and sidecar.
- Sample replacement: 2 affected tensors recomputed, 7 reused.
- Sample deletion: 2 affected tensors recomputed, 7 reused.
- One tensor quantile edit: 1 tensor recomputed, 8 reused.
- Bit-width and rounding changes: quantile results reused where parameters did not
  affect them; downstream results changed.
- Serial and parallel budget runs produce identical result and deduction sequence.
- 13 unit/integration tests pass.
- All four injection IDs hit their expected business codes.

See `DEMO_RESULTS.md` for the recorded hashes, counts and exit codes.
