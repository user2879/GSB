# Declaration schema

## Dataset (`mlpipe.dataset.v1`)

```json
{
  "format": "mlpipe.dataset.v1",
  "generator": {"name": "numpy.default_rng", "seed": 20261122},
  "sources": [
    {
      "id": "events",
      "rows": 83,
      "columns": [
        {"name": "event_id", "type": "int", "values": [1000], "nulls": [0]},
        {"name": "amount", "type": "float", "values": ["27.125"], "nulls": [0]}
      ]
    }
  ]
}
```

The example row count is illustrative; `values` and `nulls` must both have exactly `rows` entries. Float samples may be finite JSON numbers or decimal strings. Decimal strings are preferred because they preserve an exact declaration.

The `generator` object is provenance metadata. It is not part of the content digest; changing only the declared seed while leaving canonical sample content unchanged does not create a new sample.

## Pipeline (`mlpipe.pipeline.v1`)

```json
{
  "format": "mlpipe.pipeline.v1",
  "fixed_scale": 3,
  "sharding": {"target_rows": 32, "remainder_policy": "keep_partial"},
  "stages": [
    {"id": "canonicalize", "type": "canonicalize", "version": 1, "upstream": null, "params": {"saturate": 1}},
    {"id": "fill", "type": "fill_missing", "version": 1, "upstream": "canonicalize", "params": {"fill_fixed": 0}},
    {"id": "norm", "type": "normalize", "version": 1, "upstream": "fill", "params": {"column": 1}},
    {"id": "clip", "type": "clip_outliers", "version": 1, "upstream": "norm", "params": {"column": 1, "z_limit_permille": 2200, "max_outlier_permille": 100}},
    {"id": "pack", "type": "shard_pack", "version": 1, "upstream": "clip", "params": {"remainder_policy": 1}}
  ]
}
```

### Parameter contract

| Type | Required params | Bounds / meaning |
|---|---|---|
| `canonicalize` | `saturate` | `0` or `1`; `1` saturates signed 64-bit overflow, `0` raises `E_PARAM` on overflow |
| `fill_missing` | `fill_fixed` | `[-1_000_000_000, 1_000_000_000]` fixed-point integer |
| `normalize` | `column` | global zero-based column index |
| `clip_outliers` | `column`, `z_limit_permille`, `max_outlier_permille` | column index; z limit `1..100000`; max ratio `0..1000` permille |
| `shard_pack` | `remainder_policy` | `1` = keep partial, `0` = drop partial; must match top-level policy |

Every stage parameter must be a JSON integer (booleans rejected). Unknown or missing parameters fail `E_SPEC`.

### Chain constraints

- exactly one root (`upstream: null`);
- each stage has at most one upstream and one successor;
- all stage IDs are unique and all referenced upstream IDs exist;
- no cycles, no disconnected components, no branches;
- phase ranks never decrease;
- first stage `canonicalize`, last stage `shard_pack`;
- exactly one `canonicalize`, exactly one `shard_pack`;
- at least one `fill_missing`, one `normalize`, one `clip_outliers`.

### Configuration example: evolved spec

The demo evolved spec changes one mid-range sample amount by `+0.125` and advances `fill_missing` to version 2. Because only the row in global interval `[32, 64)` changes, only shard 1 is polluted; four shards retain their byte-identical fingerprints.
