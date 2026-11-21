# Verification Record

Verification was performed offline on 2026-11-22 with Python `3.11.0rc1`,
NumPy `1.26.4`, and pytest `8.3.3`.

## Automated test suite

Command:

```bash
pytest -q
```

Result:

```text
19 passed
```

Covered behaviors:

- dependency-free PNG encode/decode for 1-4 bands;
- duplicate index cells and asymmetric adjacency -> `E_INDEX_INVALID`;
- unknown band version -> `E_BAND_VERSION_UNKNOWN`;
- truncated PNG -> `E_TILE_DECODE`;
- RGB tile in a four-band index -> `E_TILE_SHAPE_MISMATCH`;
- deterministic tile scores, windows and tie-to-lowest-class behavior;
- weighted fusion tie -> row/column lexicographic source;
- pair region source and vote traces;
- budget exhaustion, pending marker and next-batch pending priority;
- serial (`max_workers=1`) versus parallel (`max_workers=4`) exact equality;
- snapshot checksum recovery and `E_SNAPSHOT_STALE` on hash mismatch;
- interrupted `.building-*` generation never replaces `CURRENT`;
- SIGTERM commits in-flight results and exits `0`;
- each independent injection profile and combined tile-fault isolation;
- complete edit sequence incremental result equals blank full rebuild.

## Acceptance demo

Command:

```bash
python3 -m remote_seg demo --out runtime/demo
```

The demo executes ten edits covering:

1. two local multi-band PNG patch batches;
2. classification weight changes;
3. fusion strategy switches `f2 -> f1 -> f3`;
4. tile deletion and a new overlapping tile addition;
5. adjacency removal;
6. minimum-connected-area change.

After every edit (all ten prefixes), the demo performs a fresh blank full
rebuild from that step's final inputs and compares the complete canonical state.
All ten step comparisons report `equal: true`; the final per-step state-view hash
is `5374d2881db92f058128b6dadd19473f8fc81880cc1309da9287f8326c15a912`.

Recorded comparison categories all pass:

```text
global_classes               true
global_trace                 true
postprocess_backfills        true
tile_summaries               true
tile_pixels                  true
fusion_region_votes          true
reuse_recompute_ledger       true
equal                        true
```

The incremental and full snapshots also have the same canonical snapshot state
fingerprint:

```text
09f46709319c63bfdebd428bfeb66991b6b4803d37f273d698a2f640f452019f
```

The final 88x168 class grid had 6,433 farmland, 1,172 water, 824 building and
3,155 road pixels, with 2,048 overlap pixels and 289 audited postprocess
backfills. Detailed per-edit versions, recomputed tiles/regions and final class
summary are written to `runtime/demo/verification.json`.

## Injection demo

The combined tile-local profile was materialized from `configs/injections.json`
and produced four isolated failures while three unaffected tiles still fused:

```text
INJ-TRUNC-101    E_TILE_DECODE
INJ-BANDS-101    E_TILE_SHAPE_MISMATCH
INJ-PATCH-101    E_PATCH_BOUNDS
INJ-BANDVER-101  E_BAND_VERSION_UNKNOWN
```

The `unknown_fusion_version` profile failed before inference with
`E_FUSION_VERSION_UNKNOWN` and still reported the injection item ID.

## Graceful stop demo

A 6x6-tile, single-worker run received SIGTERM during classification. It exited
with code `0`, reported `graceful_stop`, published a complete generation, and
recovered the same global version and fusion strategy (`f2`).
