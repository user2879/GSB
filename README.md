# Deterministic render diagnostics workbench

`renderlab` is a local, CPU-only, deterministic multi-stage text-to-image and
image-to-image diagnostic harness. It uses a local YAML config, a local prompt
sample, a SHA-256 random stream and NumPy only. There is no network access, GPU
requirement, model download or external weight file.

The pipeline is deliberately explicit:

1. strict config/sample validation;
2. normalized restricted-vocabulary bag-of-words encoding;
3. seeded per-tier latent initialisation and optional img2img projection;
4. ordered denoise steps with noise and guidance terms;
5. fixed linear decoding, clipping and deterministic PNG encoding;
6. content-addressed stage persistence, bounded LRU latent blocks and edit-time
   invalidation;
7. thread-pool execution with stable task keys and a non-blocking preview budget.

## Environment

Confirmed in this workspace:

- Python `3.11.0rc1` at `/usr/bin/python3`
- NumPy `2.4.6`
- PyYAML `5.4.1`

```bash
cd /home/zsl/feedback/uid201
python3 --version
python3 -c 'import numpy, yaml; print(numpy.__version__, yaml.__version__)'
python3 scripts/generate_inputs.py
python3 scripts/validate_inputs.py
```

The generator writes `samples/noise_table.yaml`, `samples/guidance_table.yaml`
and `samples/initial.ppm`. It asserts that both tables have exactly six entries,
matching `render.denoise_steps: 6` in `configs/workbench.yaml`.

## Quick start

```bash
# One cold/current-cache render. PNGs and stage metadata stay under .renderlab.
python3 -m renderlab run \
  --config configs/workbench.yaml \
  --temp-dir .renderlab/run-report \
  --batch-size 2 \
  --report reports/run_report.json

# Seven-edit incremental sequence, followed by an independent cold rebuild.
python3 -m renderlab demo \
  --config configs/workbench.yaml \
  --edits configs/edit_sequence.yaml \
  --batch-size 2 \
  --incremental-dir .renderlab/demo-incremental \
  --fresh-dir .renderlab/demo-fresh \
  --report-dir reports

# The first request is expected to return E_PREVIEW_BUDGET_PENDING with budget 0.
python3 -m renderlab preview-probe \
  --config configs/workbench.yaml \
  --temp-dir .renderlab/preview-probe \
  --budget-ms 0 \
  --report reports/preview_probe.json

# Full local test suite.
python3 -m unittest discover -s tests -v
```

JSON reports include the strategy table version, metrics, candidate scores,
selected policy and reason, frame-level embedding/initial-latent/step summaries,
PNG hashes, stage cache keys, invalidation/reuse events, concurrency order, LRU
diagnostics, snapshots and preview state.

## Configuration

`configs/workbench.yaml` controls the pipeline; `samples/prompts.yaml` contains
the restricted vocabulary, embedding dimension, seeds and prompt token bags.
Unknown keys and invalid structural values are rejected rather than silently
ignored.

Important business codes:

- `E_CONFIG_INVALID`: unknown key, dimension/vocabulary mismatch, seed outside
  range, non-increasing tier area, zero cache budget and other structural input
  errors;
- `E_NOISE_TABLE_MISMATCH`: declared or actual noise/guidance table length does
  not equal `render.denoise_steps`;
- `E_IMG2IMG_STEP_INVALID`: img2img replacement count exceeds configured or
  selected denoise steps;
- `E_PREVIEW_BUDGET_PENDING`: no current-parameter preview is ready within the
  budget; a previously completed low-tier preview is returned when available.

Invalid individual seeds or prompts are removed from the render set and appear
in the failure list. Structural failures prevent the run from starting.

## Determinism and cache model

- Prompt embeddings are normalized bag-of-words vectors; canonical token sorting
  makes token order irrelevant.
- Latent noise comes from a SHA-256 counter stream derived from seed, prompt
  fingerprint, tier geometry and fixed initializer version. Python/NumPy global
  RNG state is never used.
- Every stage key is derived from upstream content fingerprint, a parameter
  fingerprint and `strategy-table-v1`.
- Stage artifacts are validated before reuse; malformed or partial entries are
  misses. A cache hit does not execute the stage again.
- Latent blocks use bounded LRU. Equal access-sequence ties evict the lexical
  smallest block key. Evicted blocks requested again are recomputed through a
  pure replay function and compared by digest.
- Thread-pool results are assembled by `(tier rank, prompt index, step, frame
  index)`, so output order is independent of completion order.

## Actual validation result

Full details are in `reports/equivalence.json` and `docs/VALIDATION.md`.

- 22 unit tests passed (`python3 -m unittest discover -s tests -v`).
- The edit sequence made seven incremental runs and then a cold rebuild.
- Both runs produced 18 frames across `tier32` and `tier64`.
- All 18 PNG files compared byte-for-byte equal.
- Incremental and cold canonical fingerprints both equal
  `57faa6f52d72a747984db5fccf731af87f254c7618d5d4c56ef78cc83fe6e522`.
- Final incremental execution reused 179 stage artifacts, computed one patched
  initialization, and performed 120 LRU block hits, 330 deterministic
  recomputations, and zero recompute digest mismatches.
- The preview probe returned `E_PREVIEW_BUDGET_PENDING` immediately, then the
  completed `tier32` first frame after the render finished.

One diagnostic distinction is intentional: actual `computed_stages` /
`reused_stages` execution telemetry depends on whether a persistent cache was
present. The canonical equivalence view contains the final pixels, latent
summaries, strategy decision, snapshots and logical stage keys; those fields are
identical between the incremental and cold rebuild. See `docs/DESIGN.md` for the
cache and invalidation contract.

## Project layout

```text
renderlab/        package: config, engine, strategy, cache, preview, CLI, PNG
configs/          workbench config and edit sequence
samples/          generated tables, prompt description and starter image
scripts/          deterministic input generator and input validator
tests/            unit and integration tests (stdlib unittest)
reports/          generated verification evidence
docs/             design and validation notes
```
