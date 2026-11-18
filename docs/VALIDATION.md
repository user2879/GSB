# Validation record

Date: 2026-11-18 (Asia/Shanghai)

## Environment check

```text
Python 3.11.0rc1
/usr/bin/python3
numpy 2.4.6
yaml 5.4.1
```

## Generated inputs

`python3 scripts/generate_inputs.py`:

```text
noise_table=6 guidance_table=6 initial=16x16
```

`python3 scripts/validate_inputs.py` reports `VALIDATION_OK`, three valid prompts
and three seeds (`101, 107, 113`), with both tables at length six for six
denoise steps.

## Automated tests

Command:

```bash
python3 -m unittest discover -s tests -v
```

Result: `Ran 22 tests in 11.667s` / `OK`.

Coverage includes:

- all requested validation business codes;
- invalid seed/prompt exclusion without poisoning valid items;
- prompt-bag and initial-latent repeatability;
- PNG round-trip and byte determinism;
- stable adaptive strategy selection;
- LRU lexical tie-break, eviction, recompute match and mismatch;
- guidance-only and prompt-only invalidation boundaries;
- edit-sequence canonical equality with a cold rebuild;
- concurrent versus serial frame output equality;
- preview budget pending/stale/current behavior.

## Full sample incremental/cold acceptance

Command:

```bash
python3 -m renderlab demo \
  --config configs/workbench.yaml \
  --edits configs/edit_sequence.yaml \
  --batch-size 2 \
  --incremental-dir .renderlab/demo-incremental \
  --fresh-dir .renderlab/demo-fresh \
  --report-dir reports
```

Result from `reports/equivalence.json`:

```text
status: ok
canonical_equal: true
all_png_bytes_equal: true
frames compared: 18
incremental sequence runs: 7
incremental canonical fingerprint: 57faa6f52d72a747984db5fccf731af87f254c7618d5d4c56ef78cc83fe6e522
cold rebuild canonical fingerprint: 57faa6f52d72a747984db5fccf731af87f254c7618d5d4c56ef78cc83fe6e522
final incremental reuse: 179 reused / 1 computed
final invalidation summary:
  encode: 18 reused / 0 computed
  img2img_encode: 18 reused / 0 computed
  init: 17 reused / 1 computed
  step: 108 reused / 0 computed
  decode: 18 reused / 0 computed
```

The one computed init is the frame selected by the local latent patch. It and
all of its downstream stages were correctly invalidated; all unaffected frame
chains were reused.

LRU diagnostics from the final incremental run:

```text
accesses: 450
hits/reuses: 120
evictions: 846
recomputes: 330
recompute digest matches: 330
recompute digest mismatches: 0
```

The full sample selected `S1-balanced`: `tier32`, `tier64`, batch prompt count
two, six denoise steps, chunking enabled, on `strategy-table-v1`.

## Preview budget probe

Command:

```bash
python3 -m renderlab preview-probe \
  --config configs/workbench.yaml \
  --temp-dir .renderlab/preview-probe \
  --batch-size 2 \
  --budget-ms 0 \
  --report reports/preview_probe.json
```

Observed:

- initial preview status: `pending`;
- initial business code: `E_PREVIEW_BUDGET_PENDING`;
- no blocking beyond the zero-millisecond budget;
- final render status: `ok`;
- final preview status: `ok`, tier `tier32`, prompt `p0-dawn`, seed `101`.

## One-run report

Command:

```bash
python3 -m renderlab run \
  --config configs/workbench.yaml \
  --temp-dir .renderlab/run-report \
  --batch-size 2 \
  --report reports/run_report.json
```

Result: `status=ok`, 18 frames, selected policy `S1-balanced`, concurrency
configured four and observed four, no business-code failure. The run had 31
within-run stage reuses and 149 computed stages (shared image encoding plus
cache-cold stages), 120 LRU hits, 330 recomputations and no digest mismatch.
