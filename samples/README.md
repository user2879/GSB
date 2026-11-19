# VecFlow local samples

All files are generated locally by `python -m vecflow generate-samples` or
`python tools/generate_samples.py`.

- `data_blocks.json`: deterministic columnar blocks with null bitmaps.
- `tasks.json`: arithmetic, comparison, logic, ternary, explicit and generated
  shards, plus sum/concat/count aggregation.
- `inject_range/`: shard end beyond `row_count` (must yield `E_RANGE`).
- `inject_nullmap/`: shortened null bitmap (must yield `E_NULLMAP`).
- `inject_budget/`: global budget 0 (must yield `E_BUDGET`).

Runtime files (`ledger.sqlite`, `snapshots/`, `logs/`) are created by the app
and can be cleared before a repeat demonstration.
