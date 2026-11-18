# Design notes

## Stage graph

The render graph is a strict chain per frame:

```text
encode(prompt)
  -> init(tier, seed, patch)
  -> step(0) -> ... -> step(N-1)
  -> decode
```

An enabled img2img source adds a tier-level `img2img_encode` artifact. For steps
below `replace_early_steps`, the step result is the encoded initial image. From
the first remaining step onward, normal denoise updates continue from that
latent.

Every key is `SHA256(stage, upstream content fingerprint, parameter fingerprint,
strategy version)`. Step keys additionally encode the step index, selected step
count, guidance, both table entries, seed, tier and img2img mode. This makes
invalidations exact: changing guidance reuses encode/init and invalidates every
step and decode; replacing one prompt only invalidates that prompt's chain;
adding a tier leaves the existing tier chain reusable.

## Prompt and latent determinism

`HashRandom` is a counter-mode SHA-256 stream. It is local to a calculation and
does not share process-global state. Every vocabulary token receives a stable
normal vector derived from the token and embedding dimension. A prompt is the
sum of sorted-token vectors, followed by L2 normalization.

Initial latent values come from a SHA-256 stream derived from initializer
version, seed, prompt content fingerprint, tier name and tier geometry. Local
patches are applied to flattened latent positions after initialization and are
part of the init parameter fingerprint.

## Denoise and decode operation order

For a normal step the engine computes:

```text
after_noise = previous + noise_table[s] * noise_field(s)
after_condition = after_noise + guidance_scale * guidance_table[s] * condition_field(s)
```

The two additions are separate NumPy operations in the order shown. Noise and
condition fields are derived by SHA-256 streams, not runtime RNG calls. Image
decoding is a fixed latent-to-RGB dot product, then center/bias add, clip to
`[0,255]`, add `0.5`, floor, and cast to `uint8`. A dependency-free RGB PNG
encoder uses a fixed zlib compression level and filter 0, so the output bytes
are reproducible.

## Strategy table

`strategy-table-v1` contains five fixed policies. Selection metrics include:

- prompt count and seed count;
- requested batch size and all configured batch-size options;
- each tier's estimated single-frame cost;
- cache budget in latent blocks;
- full denoise step count.

Each policy is scored from quality, latency, memory and scale-overrun components.
Ties are resolved by policy id. The decision record serializes the metric values,
every candidate, component scores, total score, selected tier set, selected
prompt-batch size, selected step count, chunking flag and human-readable reason.
For a given metric tuple the result is byte-stable.

## Persistent artifacts and LRU

`ArtifactStore` writes arrays, PNGs, pixels and metadata atomically. A cache hit
requires matching format version, key, kind, shape, dtype and content digest.
Decoded cache hits also verify the PNG and pixel digests.

The in-memory latent block cache is capacity-bounded. It exposes hits, reuses,
misses, evictions, recomputations, recompute matches and mismatches. Eviction
chooses the smallest tuple `(last_access, block_key)`, making equal access
sequence ties deterministic. A block requested after eviction is replayed from
the deterministic initial/image latent and step fields; its digest is compared
with the first value.

The residency exercise walks the newest blocks first. This deliberately exposes
both hits from blocks still resident after a full wave and recomputations for
blocks displaced by a budget smaller than the wave working set.

## Concurrency and preview

Denoise and decode work is submitted to `ThreadPoolExecutor` with at most
`runtime.concurrency` workers. Each task carries a stable key formed from tier,
prompt index, step and frame index. Results are collected into keyed maps and
the final report is emitted in stable serial order. A short startup gate lets the
first worker cohort overlap so the observed concurrency diagnostic is meaningful.

`PreviewBroker` publishes the first frame of the lowest enabled tier. A preview
request waits no longer than its budget for the active parameter signature. If
that signature is not ready, it returns status `pending`,
`E_PREVIEW_BUDGET_PENDING`, progress and the most recent completed preview (with
`stale: true`). It never blocks final rendering outside the requested budget.

## Incremental versus cold equivalence

The final content hashes, latent summaries, PNG bytes, strategy record and
logical stage keys are functions only of the final input state and execution
semantics. They are therefore equal between a seven-step incremental run and a
cold rebuild from an empty temporary directory.

Actual stage execution status is cache-state telemetry. A cold rebuild must
report stages as computed; an incremental run with valid persisted artifacts
must report the unaffected stages as reused. Reporting those runtime statuses as
identical would contradict the cache contract. The workbench keeps both views:

- `invalidation.computed_stages` / `reused_stages`: actual execution telemetry;
- canonical report projection: deterministic output plus the complete logical
  stage-key plan for equivalence tests.

The acceptance comparison in `tests/test_engine_incremental.py` and the CLI demo
uses the canonical projection and additionally compares every PNG byte.
